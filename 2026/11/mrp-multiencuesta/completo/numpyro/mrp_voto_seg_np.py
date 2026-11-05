# Port de completo/stan/mrp_voto_seg.stan a NumPyro (decisión de José,
# 2026-11-03: en Stan iba a ~25 s por iteración con max_treedepth = 10; el
# ajuste en Stan, otro día).
#
# Mismo modelo y misma parametrización que el .stan:
#   - voto + simpatía, BLANCO aparte (15 categorías);
#   - todos los efectos por niveles suman cero entre niveles (base ortonormal
#     Q del subespacio de suma cero, como sum_to_zero_vector de Stan) y se
#     centran entre partidos (centra);
#   - todos los efectos en forma no centrada, como en el (II) (--centrados
#     para probar otra cosa);
#   - dudoso frente a seguro: d = b_dud + a_dud_edad[edad] + a_dud_rec[rec];
#   - filas sin provincia o sin seg: mezcla en escala log de sus entradas
#     provincia x seg (pesos fijos de 10_datos_voto_seg.R);
#   - encuestas sin AA: verosimilitud condicionada a no-AA;
#   - NUTS con target_accept_prob 0.95 y max_tree_depth 8.
# Sin postestratificación: se guardan los draws de los efectos para hacerla en
# R con la participación de stan/mrp_part.stan.
#
# Uso (desde la raíz del repo):
#   ~/.virtualenvs/r-mrp-numpyro/bin/python -I \
#     2026/11/mrp-multiencuesta/completo/numpyro/mrp_voto_seg_np.py --modo nuts
#   --modo prueba: 1 cadena, 150 + 150
#
# Entrada: completo/datos/voto_seg_numpyro.npz (exporta_voto_seg_npz.R)
# Salida:  completo/datos/voto_seg_numpyro_draws_<modo>.npz

import argparse
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--modo", default="prueba", choices=["prueba", "nuts"])
ap.add_argument("--warmup", type=int, default=None)
ap.add_argument("--samples", type=int, default=None)
ap.add_argument("--chains", type=int, default=None)
ap.add_argument(
    "--target-accept",
    type=float,
    default=0.95,
    help="adapt_delta de Stan; con otro valor que 0.95, la salida lleva el sufijo _ta<valor>",
)
ap.add_argument(
    "--centrados",
    default="",
    help="efectos en forma centrada, separados por comas (p. ej. edad,rec); por defecto "
    "ninguno. 2026-11-03: con edad y rec centrados, una cadena se atascó en sd_edad de BNG "
    "cerca de 0 (Rhat 1,59; salida con sufijo _cedad_rec)",
)
ap.add_argument(
    "--territorio",
    action="store_true",
    help="efectos de provincia y CCAA de los partidos regionales solo dentro de su territorio "
    "(fuera, offset -20: sin datos). Con un solo nivel, sin efecto ni sd. Sufijo _terr",
)
ap.add_argument("--sufijo", default="", help="texto añadido al nombre de la salida")
args = ap.parse_args()
CENTRADOS = {k for k in args.centrados.split(",") if k}
SUF = "" if args.target_accept == 0.95 else f"_ta{args.target_accept:g}"
if CENTRADOS:
    SUF += "_c" + "_".join(sorted(CENTRADOS))
if args.territorio:
    assert not CENTRADOS & {"prov", "ccaa"}
    SUF += "_terr"
SUF += args.sufijo

DIR = "2026/11/mrp-multiencuesta/completo/datos"
CHAINS = args.chains or (1 if args.modo == "prueba" else 4)

import jax

jax.config.update("jax_enable_x64", True)
import numpyro

if CHAINS > 1:
    numpyro.set_host_device_count(CHAINS)
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_feasible

# ---------------------------------------------------------------- datos ----
z = dict(np.load(f"{DIR}/voto_seg_numpyro.npz"))
esc = lambda k: float(z[k][0])  # noqa: E731
ent = lambda k: int(z[k][0])  # noqa: E731
i0 = lambda k: z[k].astype(np.int64) - 1  # 1-based de R a 0-based  # noqa: E731

y = z["y"].astype(np.float64)
off = z["off"]
N, J = y.shape
n_casa, n_mes, n_edad, n_edu = ent("n_casa"), ent("n_mes"), ent("n_edad"), ent("n_edu")
n_rec, n_ccaa, n_prov = ent("n_rec"), ent("n_ccaa"), ent("n_prov")
J_AA = ent("j_aa") - 1
voto_lv = [str(v) for v in z["voto_lv"]]

casa, mes, edad, edu, rec, ccaa = (i0(k) for k in ["casa", "mes", "edad", "edu", "rec", "ccaa"])
casa_rec = casa * n_rec + rec

# filas con provincia y seg conocidos
fila1, prov1, seg1 = i0("fila1"), i0("prov1"), i0("seg1")  # seg: 0 dudoso, 1 seguro
prov_fila = np.zeros(N, dtype=np.int64)
dud_fila = np.zeros(N)
prov_fila[fila1] = prov1
dud_fila[fila1] = (seg1 == 0).astype(float)

# filas con mezcla: matriz densa (N2 x L) con máscara
mix = i0("fila2")
ini, lens = i0("ini"), z["len"].astype(np.int64)
N2, E = len(mix), len(z["ent_w"])
L = int(lens.max())
fila_e = np.repeat(np.arange(N2), lens)
pos_e = np.arange(E) - np.repeat(ini, lens)
ent_prov = np.zeros((N2, L), dtype=np.int64)
ent_dud = np.zeros((N2, L))
ent_w = np.ones((N2, L))
ent_mask = np.zeros((N2, L), dtype=bool)
ent_prov[fila_e, pos_e] = i0("ent_prov")
ent_dud[fila_e, pos_e] = (i0("ent_seg") == 0).astype(float)
ent_w[fila_e, pos_e] = z["ent_w"]
ent_mask[fila_e, pos_e] = True
assert np.allclose(np.where(ent_mask, ent_w, 0).sum(1), 1)


def q_zero_sum(n):
    # base ortonormal (n x n-1) del subespacio de suma cero
    M = np.zeros((n, n - 1))
    M[np.arange(n - 1), np.arange(n - 1)] = 1.0
    M[-1, :] = -1.0
    return np.linalg.qr(M)[0]


n_niv = dict(
    casa=n_casa, mes=n_mes, edad=n_edad, edu=n_edu, rec=n_rec, ccaa=n_ccaa,
    prov=n_prov, casa_rec=n_casa * n_rec, dud_edad=n_edad, dud_rec=n_rec,
)
Q = {k: jnp.asarray(q_zero_sum(n)) for k, n in n_niv.items()}
Q_J = jnp.asarray(q_zero_sum(J))
ESC = {k: float(np.sqrt(n / (n - 1.0))) for k, n in n_niv.items()}
SD_PRIOR = dict(
    casa=esc("s_sd_main"), mes=esc("s_sd_main"), edad=esc("s_sd_main"),
    edu=esc("s_sd_main"), rec=esc("s_sd_rec"), ccaa=esc("s_sd_main"),
    prov=esc("s_sd_main"), casa_rec=esc("s_sd_int"), dud_edad=esc("s_sd_dud"),
    dud_rec=esc("s_sd_dud"),
)
assert CENTRADOS <= set(n_niv), CENTRADOS

# --territorio: niveles de provincia y CCAA donde cada partido puede tener votos
# (offset 0 en alguna fila o entrada de mezcla). Máscara n x J, sacada de los datos
disponible = off == 0
MASCARA = {"prov": np.zeros((n_prov, J), dtype=bool), "ccaa": np.zeros((n_ccaa, J), dtype=bool)}
MASCARA["ccaa"][ccaa[:, None].repeat(J, 1)[disponible], np.nonzero(disponible)[1]] = True
for j in range(J):
    MASCARA["prov"][prov1[disponible[fila1, j]], j] = True
    fila_d = disponible[mix, j]
    MASCARA["prov"][ent_prov[fila_d][ent_mask[fila_d]], j] = True


def efecto_territorio(nombre):
    # como efecto(), pero la suma cero y el efecto de cada partido solo existen en sus
    # niveles; fuera, 0. Con un nivel, el efecto se confunde con alpha: ni efecto ni sd
    n = n_niv[nombre]
    masc = MASCARA[nombre]
    m = masc.sum(0)
    con_sd = np.nonzero(m >= 2)[0]
    libre = numpyro.sample(
        f"sd_{nombre}_libre", dist.HalfNormal(SD_PRIOR[nombre]).expand([len(con_sd)]).to_event(1)
    )
    sd = numpyro.deterministic(f"sd_{nombre}", jnp.zeros(J).at[con_sd].set(libre))
    a = jnp.zeros((n, J))
    nacional = np.nonzero(m == n)[0]
    if len(nacional):
        u = numpyro.sample(f"u_{nombre}", dist.Normal(0.0, 1.0).expand([len(nacional), n - 1]).to_event(2))
        a = a.at[:, nacional].set(ESC[nombre] * (u @ Q[nombre].T).T * sd[nacional][None, :])
    for j in con_sd:
        if m[j] == n:
            continue
        niv = np.nonzero(masc[:, j])[0]
        u = numpyro.sample(f"u_{nombre}_{voto_lv[j]}", dist.Normal(0.0, 1.0).expand([m[j] - 1]).to_event(1))
        a = a.at[niv, j].set(float(np.sqrt(m[j] / (m[j] - 1.0))) * jnp.asarray(q_zero_sum(int(m[j]))) @ u * sd[j])
    return numpyro.deterministic(f"a_{nombre}", centra(a))


# ---------------------------------------------------------------- modelo ----
def centra(a):
    # suma 0 entre partidos (última dimensión)
    return a - a.mean(axis=-1, keepdims=True)


def efecto(nombre):
    # (n x J): columna k = esc * sd[k] * Q u_k, suma 0 entre niveles; luego
    # centra entre partidos. Centrado: u_k ~ N(0, esc * sd[k]); sin centrar:
    # u_k ~ N(0, 1) y se escala
    n = n_niv[nombre]
    sd = numpyro.sample(f"sd_{nombre}", dist.HalfNormal(SD_PRIOR[nombre]).expand([J]).to_event(1))
    if nombre in CENTRADOS:
        u = numpyro.sample(
            f"u_{nombre}",
            dist.Normal(0.0, ESC[nombre] * sd[:, None] * jnp.ones((J, n - 1))).to_event(2),
        )
        a = (u @ Q[nombre].T).T
    else:
        u = numpyro.sample(f"u_{nombre}", dist.Normal(0.0, 1.0).expand([J, n - 1]).to_event(2))
        a = ESC[nombre] * (u @ Q[nombre].T).T * sd[None, :]
    return numpyro.deterministic(f"a_{nombre}", centra(a))


def vector_j(nombre, s):
    u = numpyro.sample(f"u_{nombre}", dist.Normal(0.0, s).expand([J - 1]).to_event(1))
    return numpyro.deterministic(nombre, Q_J @ u)


def model(Y, MUJER, OFF, CASA, MES, EDAD, EDU, REC, CCAA, CASA_REC, PROV_FILA, DUD_FILA,
          MIX, ENT_PROV, ENT_DUD, LOG_W, MASK, SIN_AA):
    alpha = vector_j("alpha", esc("s_alpha"))
    b_mujer = vector_j("b_mujer", esc("s_b_mujer"))
    b_dud = vector_j("b_dud", esc("s_b_dud"))
    a = {k: efecto_territorio(k) if args.territorio and k in MASCARA else efecto(k) for k in n_niv}

    eta = (alpha[None, :] + MUJER[:, None] * b_mujer[None, :] + a["casa"][CASA]
           + a["mes"][MES] + a["edad"][EDAD] + a["edu"][EDU] + a["rec"][REC]
           + a["ccaa"][CCAA] + a["casa_rec"][CASA_REC] + OFF)
    d = b_dud[None, :] + a["dud_edad"][EDAD] + a["dud_rec"][REC]

    lp = jax.nn.log_softmax(eta + a["prov"][PROV_FILA] + DUD_FILA[:, None] * d)
    x = eta[MIX][:, None, :] + a["prov"][ENT_PROV] + ENT_DUD[..., None] * d[MIX][:, None, :]
    lpe = jnp.where(MASK[..., None], LOG_W[..., None] + jax.nn.log_softmax(x, axis=-1), -jnp.inf)
    lp = lp.at[MIX].set(logsumexp(lpe, axis=1))
    # sin AA en el cuestionario: renormalizar sin AA (sus y son 0)
    pen = jnp.where(SIN_AA == 1, Y.sum(-1) * jnp.log1p(-jnp.exp(lp[:, J_AA])), 0.0)
    numpyro.factor("ll", jnp.sum(Y * lp) - jnp.sum(pen))


ji = lambda v: jnp.asarray(v, jnp.int32)  # noqa: E731
datos = dict(
    Y=jnp.asarray(y), MUJER=jnp.asarray(z["mujer"]), OFF=jnp.asarray(off),
    CASA=ji(casa), MES=ji(mes), EDAD=ji(edad), EDU=ji(edu), REC=ji(rec), CCAA=ji(ccaa),
    CASA_REC=ji(casa_rec), PROV_FILA=ji(prov_fila), DUD_FILA=jnp.asarray(dud_fila),
    MIX=ji(mix), ENT_PROV=ji(ent_prov), ENT_DUD=jnp.asarray(ent_dud),
    LOG_W=jnp.asarray(np.log(ent_w)), MASK=jnp.asarray(ent_mask), SIN_AA=ji(z["sin_aa"]),
)

warmup = args.warmup or (150 if args.modo == "prueba" else 1000)
samples = args.samples or (150 if args.modo == "prueba" else 1000)
print(f"modo: {args.modo}  N: {N}  J: {J}  N2 (mezcla): {N2}  E: {E}  L: {L}")
if args.territorio:
    print("niveles por partido (prov, ccaa):",
          {p: (int(MASCARA["prov"][:, j].sum()), int(MASCARA["ccaa"][:, j].sum()))
           for j, p in enumerate(voto_lv)})
print(f"iteraciones: {warmup} + {samples}, {CHAINS} cadena(s) en paralelo, "
      f"target_accept_prob {args.target_accept}, centrados: {sorted(CENTRADOS) or 'ninguno'}",
      flush=True)

mcmc = MCMC(
    NUTS(model, target_accept_prob=args.target_accept, max_tree_depth=8,
         init_strategy=init_to_feasible()),
    num_warmup=warmup, num_samples=samples, num_chains=CHAINS,
    progress_bar=True,  # una barra por cadena en el log (stderr); ver con tail
)
t0 = time.time()
print("inicio:", time.strftime("%H:%M:%S"), flush=True)
mcmc.run(jax.random.PRNGKey(2026), **datos, extra_fields=("num_steps", "diverging"))
jax.block_until_ready(mcmc.get_samples())
minutos = (time.time() - t0) / 60
ex = mcmc.get_extra_fields()
print(f"\nfin: {time.strftime('%H:%M:%S')}  minutos (compilación incluida): {minutos:.1f}")
print(f"divergencias: {int(ex['diverging'].sum())} de {CHAINS * samples}")
print(f"pasos por iteración: mediana {int(np.median(ex['num_steps']))}, "
      f"en el tope (255): {int((ex['num_steps'] >= 255).sum())}")

# ------------------------------------------------- guardar, diagnóstico ----
# Primero se guardan los draws: si algo falla en el diagnóstico, el ajuste no
# se pierde (2026-11-03: arviz 1.3 cambió from_dict y se perdió un ajuste)
s = mcmc.get_samples(group_by_chain=True)
guardar = ["alpha", "b_mujer", "b_dud"] + [f"a_{k}" for k in n_niv] + [f"sd_{k}" for k in n_niv]
flat = {k: np.asarray(s[k]).reshape((-1,) + s[k].shape[2:]) for k in guardar}
f_out = f"{DIR}/voto_seg_numpyro_draws_{args.modo}{SUF}.npz"
np.savez_compressed(
    f_out, **flat, num_steps=np.asarray(ex["num_steps"]), diverging=np.asarray(ex["diverging"]),
    voto_lv=np.asarray(voto_lv), minutos=minutos, n_chains=CHAINS,
    target_accept=args.target_accept,
)
print(f"guardado {f_out}  (draws en orden cadena a cadena: {CHAINS} x {samples})", flush=True)

if CHAINS > 1:
    try:
        import arviz as az

        idata = az.from_dict({"posterior": {k: np.asarray(s[k]) for k in guardar}})
        sm = az.summary(idata, kind="diagnostics", round_to="none")
        print(f"Rhat máx: {sm['r_hat'].max():.3f}  ESS bulk mín: {sm['ess_bulk'].min():.0f}"
              f"  (variable: {sm['ess_bulk'].idxmin()})"
              f"  ESS bulk mín por minuto: {sm['ess_bulk'].min() / minutos:.1f}")
        print(f"parámetros con Rhat > 1,01: {int((sm['r_hat'] > 1.01).sum())} de {len(sm)}")
        print("ESS bulk más bajos:")
        print(sm.sort_values("ess_bulk").head(8)[["ess_bulk", "ess_tail", "r_hat"]].round(3))
    except Exception as e:  # noqa: BLE001
        print("(diagnóstico de arviz falló:", repr(e), ")")

print("\n== b_dud: dudoso frente a seguro, log-odds (media y p5-p95) ==")
for j, p in enumerate(voto_lv):
    v = flat["b_dud"][:, j]
    print(f"{p:>8}: {v.mean():+.2f}  [{np.quantile(v, 0.05):+.2f}, {np.quantile(v, 0.95):+.2f}]")
for k in ["sd_dud_edad", "sd_dud_rec"]:
    print(f"{k}: " + ", ".join(f"{p} {m:.2f}" for p, m in zip(voto_lv, flat[k].mean(0))))
