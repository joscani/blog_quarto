# Port de mrp_part.stan a numpyro. Mismo modelo y misma parametrización:
#   - binomial de decir "10, seguro que voto" (k de n por celda), con efecto
#     casa (solo las casas que lo preguntan), mes, sexo, edad, estudios,
#     recuerdo, CCAA y provincia.
#   - centrados, como en Stan: casa, mes, edad, estudios y recuerdo
#     (a ~ N(0, sd)). Sin centrar: CCAA y provincia (a = sd * z).
#   - casa sum-to-zero: base ortonormal del subespacio (Q). Stan evalúa
#     normal(0, esc * sd) sobre los n_casa_t componentes aunque solo haya
#     n_casa_t - 1 libres; para reproducir su densidad se añade -log(esc * sd)
#     (ver PLAN.md, 2026-10-26).
#   - filas sin provincia: mezcla en escala log,
#       logit p = logsumexp(log w + log_sigmoid(x)) - logsumexp(log w + log_sigmoid(-x)),
#     como mrp_part.stan.
#   - priors como stan/ajusta.R: alpha_t N(1, 1.5), b_mujer_t N(0, 1),
#     sd_rec N+(0, 2), el resto N+(0, 0.5).
#   - part_bruta y delta (calibra) se calculan fuera del modelo, en numpy,
#     como en las generated quantities de mrp_part.stan.
#
# Uso (desde la raíz del repo, con el venv r-mrp-numpyro; antes,
# Rscript 2026/11/mrp-multiencuesta/numpyro/exporta_npz.R <modo>):
#   ~/.virtualenvs/r-mrp-numpyro/bin/python -u \
#     2026/11/mrp-multiencuesta/numpyro/mrp_part_np.py --modo nuts
#
# Salida: datos/part_numpyro_draws_<modo>.npz (efectos, part_bruta y delta
# por muestra, con los nombres de Tp en stan/ajusta.R).

import argparse
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--modo", default="prueba", choices=["prueba", "nuts"])
ap.add_argument("--warmup", type=int, default=None)
ap.add_argument("--samples", type=int, default=None)
ap.add_argument(
    "--chains",
    type=int,
    default=None,
    help="por defecto: 1 en prueba, 4 en nuts (en paralelo, un dispositivo XLA por cadena)",
)
args = ap.parse_args()
npz = f"2026/11/mrp-multiencuesta/datos/part_numpyro_{args.modo}.npz"

import jax

jax.config.update("jax_enable_x64", True)
import numpyro

CHAINS = args.chains or (1 if args.modo == "prueba" else 4)
if CHAINS > 1:
    numpyro.set_host_device_count(CHAINS)
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.nn import log_sigmoid
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_feasible

# ---------------------------------------------------------------- datos ----
z = np.load(npz)
NT = int(z["NT"])
n_casa_t, n_mes, n_edad, n_edu = (
    int(z["n_casa_t"]),
    int(z["n_mes"]),
    int(z["n_edad"]),
    int(z["n_edu"]),
)
n_rec, n_ccaa, n_prov = int(z["n_rec"]), int(z["n_ccaa"]), int(z["n_prov"])
m_alpha_t, s_alpha_t = float(z["m_alpha_t"]), float(z["s_alpha_t"])
s_b_mujer_t = float(z["s_b_mujer_t"])
s_sd_rec, s_sd_main = float(z["s_sd_rec"]), float(z["s_sd_main"])
esc_casa_t = float(np.sqrt(n_casa_t / (n_casa_t - 1.0)))
mes_post = int(z["mes_post"]) - 1
part_obj = float(z["part_obj"])

idx = {k: z[f"t_{k}"].astype(np.int64) - 1 for k in
       ["casa", "mes", "edad", "edu", "rec", "ccaa", "fila1", "prov1", "fila2", "ini"]}
len_fila = z["t_len"].astype(np.int64)

# entradas de la mezcla, en una matriz densa (N2 x L) con máscara
N2 = len(idx["fila2"])
E = int(z["t_ent_prov"].shape[0])
L = int(len_fila.max())
row_of_entry = np.repeat(np.arange(N2), len_fila)
pos_of_entry = np.arange(E) - np.repeat(idx["ini"], len_fila)
ent_pad_prov = np.zeros((N2, L), dtype=np.int32)
ent_pad_logw = np.zeros((N2, L))
ent_mask = np.zeros((N2, L), dtype=bool)
ent_pad_prov[row_of_entry, pos_of_entry] = z["t_ent_prov"] - 1
ent_pad_logw[row_of_entry, pos_of_entry] = np.log(z["t_ent_w"])
ent_mask[row_of_entry, pos_of_entry] = True

# postestratificación
P_N, P_mujer = z["P_N"], z["P_mujer"]
P_edad, P_edu, P_rec = z["P_edad"] - 1, z["P_edu"] - 1, z["P_rec"] - 1
P_ccaa, P_prov = z["P_ccaa"] - 1, z["P_prov"] - 1


def q_zero_sum(n):
    # base ortonormal del subespacio de suma cero (como en mrp_voto_np.py)
    M = np.zeros((n, n - 1))
    M[np.arange(n - 1), np.arange(n - 1)] = 1.0
    M[-1, :] = -1.0
    Q, _ = np.linalg.qr(M)
    return Q


Q_casa_t = jnp.asarray(q_zero_sum(n_casa_t))


# ---------------------------------------------------------------- modelo ----
def model(K, NN, MUJER, CASA, MES, EDAD, EDU, REC, CCAA, FILA1, PROV1, MIX,
          ENT_PROV, ENT_LOGW, MASK):
    alpha_t = numpyro.sample("alpha_t", dist.Normal(m_alpha_t, s_alpha_t))
    b_mujer_t = numpyro.sample("b_mujer_t", dist.Normal(0.0, s_b_mujer_t))
    sd = {
        k: numpyro.sample(f"sd_{k}", dist.HalfNormal(s_sd_rec if k == "rec" else s_sd_main))
        for k in ["casa", "mes", "edad", "edu", "rec", "ccaa", "prov"]
    }

    esc = esc_casa_t * sd["casa"]
    u_casa = numpyro.sample(
        "u_casa", dist.Normal(0.0, esc).expand([n_casa_t - 1]).to_event(1)
    )
    numpyro.factor("casa_como_stan", -jnp.log(esc))
    a_casa = Q_casa_t @ u_casa
    a = {"casa": a_casa}
    for k, n in [("mes", n_mes), ("edad", n_edad), ("edu", n_edu), ("rec", n_rec)]:
        a[k] = numpyro.sample(f"a_{k}", dist.Normal(0.0, sd[k]).expand([n]).to_event(1))
    for k, n in [("ccaa", n_ccaa), ("prov", n_prov)]:
        zk = numpyro.sample(f"z_{k}", dist.Normal(0.0, 1.0).expand([n]).to_event(1))
        a[k] = sd[k] * zk

    eta = (alpha_t + MUJER * b_mujer_t + a["casa"][CASA] + a["mes"][MES]
           + a["edad"][EDAD] + a["edu"][EDU] + a["rec"][REC] + a["ccaa"][CCAA])
    logit1 = eta[FILA1] + a["prov"][PROV1]
    x = eta[MIX][:, None] + a["prov"][ENT_PROV]  # (N2, L)
    l1 = logsumexp(jnp.where(MASK, ENT_LOGW + log_sigmoid(x), -jnp.inf), axis=1)
    l0 = logsumexp(jnp.where(MASK, ENT_LOGW + log_sigmoid(-x), -jnp.inf), axis=1)
    logit = jnp.zeros(NT).at[FILA1].set(logit1).at[MIX].set(l1 - l0)
    numpyro.sample("k", dist.Binomial(total_count=NN, logits=logit), obs=K)


# ------------------------------------------------------------------ ajuste ----
ji = lambda a: jnp.asarray(a, jnp.int32)  # noqa: E731
datos_jax = dict(
    K=jnp.asarray(z["t_k"], jnp.float64),
    NN=jnp.asarray(z["t_n"], jnp.float64),
    MUJER=jnp.asarray(z["t_mujer"], jnp.float64),
    CASA=ji(idx["casa"]), MES=ji(idx["mes"]), EDAD=ji(idx["edad"]), EDU=ji(idx["edu"]),
    REC=ji(idx["rec"]), CCAA=ji(idx["ccaa"]), FILA1=ji(idx["fila1"]),
    PROV1=ji(idx["prov1"]), MIX=ji(idx["fila2"]), ENT_PROV=ji(ent_pad_prov),
    ENT_LOGW=jnp.asarray(ent_pad_logw), MASK=jnp.asarray(ent_mask),
)

warmup = args.warmup or (150 if args.modo == "prueba" else 1000)
samples = args.samples or (150 if args.modo == "prueba" else 1000)
print(f"modo: {args.modo}  filas: {NT}  filas mezcla: {N2}  entradas: {E}")
print(f"iteraciones: {warmup} warmup + {samples} muestreo, {CHAINS} cadena(s) en paralelo")

mcmc = MCMC(
    NUTS(model, target_accept_prob=0.95, max_tree_depth=8,
         init_strategy=init_to_feasible()),
    num_warmup=warmup,
    num_samples=samples,
    num_chains=CHAINS,
    progress_bar=False,
)
t0 = time.time()
mcmc.run(jax.random.PRNGKey(2026), **datos_jax, extra_fields=("num_steps",))
jax.block_until_ready(mcmc.get_samples())
t_mcmc = time.time() - t0
extra = mcmc.get_extra_fields()
print(f"\nMCMC (compilación incluida): {t_mcmc:.1f} s = {t_mcmc / 60:.2f} min")
print(f"divergencias: {int(extra['diverging'].sum())} de {CHAINS * samples}")
print(f"pasos por árbol (mediana): {int(np.median(extra['num_steps']))}")


# ------------------------------------------ cantidades generadas (numpy) ----
s = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=False).items()}
nd = s["alpha_t"].shape[0]
Tp = {
    "alpha": s["alpha_t"], "b": s["b_mujer_t"],
    "mes": s["a_mes"], "edad": s["a_edad"], "edu": s["a_edu"], "rec": s["a_rec"],
    "ccaa": s["sd_ccaa"][:, None] * s["z_ccaa"],
    "prov": s["sd_prov"][:, None] * s["z_prov"],
}


def calibra(eta, N, obj):
    # delta tal que sum(N * inv_logit(eta + delta)) = obj * sum(N), por Newton,
    # con el paso limitado a +-2, como calibra() en funciones.stan; por filas
    delta = np.zeros(eta.shape[0])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(eta + delta[:, None])))
        f = (p * N).sum(axis=1) / N.sum() - obj
        df = (p * (1 - p) * N).sum(axis=1) / N.sum()
        delta -= np.clip(f / df, -2, 2)
        if np.all(np.abs(f) < 1e-10):
            break
    return delta


t0 = time.time()
part_bruta = np.zeros(nd)
delta = np.zeros(nd)
CH = 500
for a0 in range(0, nd, CH):
    sl = slice(a0, min(a0 + CH, nd))
    eta_pt = (Tp["alpha"][sl, None] + P_mujer[None, :] * Tp["b"][sl, None]
              + Tp["mes"][sl, mes_post][:, None] + Tp["edad"][sl][:, P_edad]
              + Tp["edu"][sl][:, P_edu] + Tp["rec"][sl][:, P_rec]
              + Tp["ccaa"][sl][:, P_ccaa] + Tp["prov"][sl][:, P_prov])
    part_bruta[sl] = 100 * (P_N / (1 + np.exp(-eta_pt))).sum(axis=1) / P_N.sum()
    delta[sl] = calibra(eta_pt, P_N, part_obj)
Tp["part_bruta"], Tp["delta"] = part_bruta, delta
print(f"\npart_bruta y delta: {nd} muestras, {time.time() - t0:.1f} s")

if CHAINS > 1:
    try:
        import arviz as az

        summ = az.summary(az.from_numpyro(mcmc), var_names=[f"sd_{k}" for k in
                          ["casa", "mes", "edad", "edu", "rec", "ccaa", "prov"]])
        print(f"Rhat máximo de las sd: {summ['r_hat'].max():.3f}"
              f"  ESS bulk mínimo: {summ['ess_bulk'].min():.0f}")
    except Exception as e:  # noqa: BLE001
        print("(arviz no disponible:", e, ")")


def resumen(v):
    q05, q95 = np.quantile(v, [0.05, 0.95])
    return f"media {v.mean():.3f}  q05 {q05:.3f}  q95 {q95:.3f}"


print("\n== participación ==")
print("part_bruta:", resumen(part_bruta))
print("delta:     ", resumen(delta))
print("sd (mediana):", ", ".join(
    f"{k} {np.median(s[f'sd_{k}']):.3f}"
    for k in ["casa", "mes", "edad", "edu", "rec", "ccaa", "prov"]))

f_out = f"2026/11/mrp-multiencuesta/datos/part_numpyro_draws_{args.modo}.npz"
np.savez_compressed(f_out, **Tp)
print(f"\nguardado {f_out}")
