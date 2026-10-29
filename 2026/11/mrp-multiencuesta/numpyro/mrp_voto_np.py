# Port de mrp_voto.stan a numpyro, con bench contra la prueba de Stan
# (stan/ajusta.R prueba). Mismo modelo, misma parametrización:
#   - alpha y b_mujer sum-to-zero: base ortonormal del subespacio (Q del
#     centro de masas), como el sum_to_zero_vector de Stan.
#   - efectos casa: z_casa sum-to-zero por partido sobre las casas, escala
#     esc_casa * sd_casa y centra() por filas (suma cero entre partidos).
#   - el resto de efectos: z ~ N(0,1), centra() por filas.
#   - priors "nuevo": alpha N(0,1), b_mujer N(0,1), sd_rec N+(0,2),
#     principales N+(0,0.5), interacciones N+(0,0.25). HalfNormal(0,s) es el
#     mismo kernel que normal(0,s) truncada a >0: constante de normalización
#     aparte, irrelevante para MCMC.
#   - verosimilitud: multinomial logit por fila; las filas sin provincia como
#     mezcla en escala log (logsumexp sobre sus entradas), como voto_parcial().
#   - v_casa (postestratificación) se calcula fuera del modelo, en numpy,
#     igual que la versión en R de stan/ajusta.R.
#
# Uso (desde la raíz del repo, con el venv r-mrp-numpyro):
#   ~/.virtualenvs/r-mrp-numpyro/bin/python \
#     2026/11/mrp-multiencuesta/numpyro/mrp_voto_np.py --modo prueba
#
# Por defecto 64 bits (como Stan); --x32 para medir en 32.

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
ap.add_argument("--x32", action="store_true", help="float32 en vez de 64")
ap.add_argument("--npz", default=None)
ap.add_argument(
    "--salida",
    default=None,
    help="ruta del npz de salida (por defecto datos/v_casa_numpyro_<modo><suf>.npz)",
)
ap.add_argument(
    "--celdas",
    default=None,
    help="npz de salida con la postestratificación por celda de la casa media "
    "(p: draws x P x J, pv: draws x P, voto_lv) para los escaños de 08_escanos.R",
)
ap.add_argument(
    "--sin-mezcla",
    action="store_true",
    help="quita las filas sin provincia conocida: prueba de si la mezcla causa la saturación",
)
ap.add_argument(
    "--sd-fija",
    default="",
    help="sd fijadas a un valor (bloque=valor, separadas por coma; ablation de un bloque: "
    "prov, ccaa_rec, casa_rec, ...); la dinámica de esa sd desaparece",
)
ap.add_argument(
    "--suma0",
    default="",
    help="efectos cuyos niveles suman 0 (lista separada por comas: mes,edad,edu,rec,"
    "ccaa,prov), como el de casa; sin ella, el modelo de mrp_voto.stan",
)
args = ap.parse_args()

SD_FIJA = {}
for t in [x for x in args.sd_fija.split(",") if x]:
    k, v = t.split("=")
    SD_FIJA[k] = float(v)
BLOQUES_SD = [
    "casa",
    "mes",
    "edad",
    "edu",
    "rec",
    "ccaa",
    "prov",
    "ccaa_rec",
    "casa_rec",
]
assert all(k in BLOQUES_SD for k in SD_FIJA), SD_FIJA

if args.npz is None:
    args.npz = f"2026/11/mrp-multiencuesta/datos/voto_numpyro_{args.modo}.npz"

import jax

jax.config.update("jax_enable_x64", not args.x32)
import numpyro

CHAINS = args.chains or (1 if args.modo == "prueba" else 4)
if CHAINS > 1:
    numpyro.set_host_device_count(CHAINS)
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_feasible

# ---------------------------------------------------------------- datos ----
z = dict(np.load(args.npz))
y = z["y"]
mujer = z["mujer"]
off = z["off"]
N, J = y.shape
n_casa, n_mes, n_edad, n_edu = (
    int(z["n_casa"]),
    int(z["n_mes"]),
    int(z["n_edad"]),
    int(z["n_edu"]),
)
n_rec, n_ccaa, n_prov = int(z["n_rec"]), int(z["n_ccaa"]), int(z["n_prov"])
P = int(z["P"])
mes_post = int(z["mes_post"]) - 1
s_alpha, s_b_mujer = float(z["s_alpha"]), float(z["s_b_mujer"])
s_sd_rec, s_sd_main, s_sd_int = (
    float(z["s_sd_rec"]),
    float(z["s_sd_main"]),
    float(z["s_sd_int"]),
)
esc_casa = float(np.sqrt(n_casa / (n_casa - 1.0)))
# AA: filas de encuestas que no la ofrecían (00_partidos.R); su verosimilitud
# se condiciona a no-AA: sus ceros de AA son de diseño del cuestionario
sin_aa = z["sin_aa"]
J_AA = int(z["j_aa"]) - 1  # índice 0-based de AA en voto_lv

# --sin-mezcla: fuera las filas sin provincia. Solo cambian las filas de voto
# (y sus índices); la postestratificación queda igual
SIN_MEZCLA = args.sin_mezcla
if SIN_MEZCLA:
    keep = np.ones(N, dtype=bool)
    keep[z["fila2"] - 1] = False
    y, mujer, off = y[keep], mujer[keep], off[keep]
    sin_aa = sin_aa[keep]
    for k in ["casa", "mes", "edad", "edu", "rec", "ccaa"]:
        z[k] = z[k][keep]
    pos = np.full(N, -1, dtype=np.int64)
    pos[keep] = np.arange(int(keep.sum()))
    z["fila1"] = pos[z["fila1"] - 1] + 1  # prov1 va en el mismo orden que fila1
    for k in ["fila2", "ini", "len", "ent_prov"]:
        z[k] = np.zeros(0, dtype=z[k].dtype)
    z["ent_w"] = np.zeros(0)
    N = int(keep.sum())

idx_np = {
    k: z[k] - 1 for k in ["casa", "mes", "edad", "edu", "rec", "ccaa", "fila2", "ini"]
}
ini, len_fila = idx_np["ini"].astype(np.int64), z["len"].astype(np.int64)
ccaa_rec = (z["ccaa"] - 1) * n_rec + z["rec"] - 1
casa_rec = (z["casa"] - 1) * n_rec + z["rec"] - 1
mix_rows = idx_np["fila2"]
prov_fila = np.zeros(N, dtype=np.int32)
prov_fila[z["fila1"] - 1] = z["prov1"] - 1
prov_fila[mix_rows] = 0  # se sobreescribe con la mezcla

# entradas de la mezcla, en una matriz densa (N2 x L) con máscara
N2 = len(mix_rows)
E = int(z["ent_prov"].shape[0])
L = int(len_fila.max()) if len(len_fila) else 1
row_of_entry = np.repeat(np.arange(N2), len_fila)
pos_of_entry = np.arange(E) - np.repeat(ini, len_fila)
ent_pad_prov = np.zeros((N2, L), dtype=np.int32)
ent_pad_w = np.zeros((N2, L))
ent_mask = np.zeros((N2, L), dtype=bool)
ent_pad_prov[row_of_entry, pos_of_entry] = z["ent_prov"] - 1
ent_pad_w[row_of_entry, pos_of_entry] = z["ent_w"]
ent_mask[row_of_entry, pos_of_entry] = True

# postestratificación
P_N, P_mujer = z["P_N"], z["P_mujer"]
P_edad, P_edu, P_rec = z["P_edad"] - 1, z["P_edu"] - 1, z["P_rec"] - 1
P_ccaa, P_prov = z["P_ccaa"] - 1, z["P_prov"] - 1
P_mascara = z["P_mascara"]
i_cr = (z["P_ccaa"] - 1) * n_rec + z["P_rec"] - 1


# ------------------------------------------------------------ constantes ----
def q_zero_sum(n):
    # base ortonormal del subespacio de suma cero: QR reducida de la matriz
    # (n, n-1) con columnas e_i - e_n (su span es exactamente {x: sum(x) = 0});
    # el prior inducido es el normal estándar del subespacio, como el
    # sum_to_zero_vector de Stan
    M = np.zeros((n, n - 1))
    M[np.arange(n - 1), np.arange(n - 1)] = 1.0
    M[-1, :] = -1.0
    Q, _ = np.linalg.qr(M)  # reduced: Q es (n, n-1)
    return Q


Q_alpha = jnp.asarray(q_zero_sum(J))
Q_casa = jnp.asarray(q_zero_sum(n_casa))
# --suma0: los niveles de esos efectos suman 0 (además de centra() entre
# partidos), como casa. Quita la cresta entre alpha y la media de los niveles
# (correlación -0,60 con rec en el ajuste de Stan; ver PLAN.md)
n_niv = {
    "mes": n_mes,
    "edad": n_edad,
    "edu": n_edu,
    "rec": n_rec,
    "ccaa": n_ccaa,
    "prov": n_prov,
}
SUMA0 = [k for k in args.suma0.split(",") if k]
assert all(k in n_niv for k in SUMA0), SUMA0
Q_niv = {k: jnp.asarray(q_zero_sum(n_niv[k])) for k in SUMA0}
suf = "_s0_" + "_".join(SUMA0) if SUMA0 else ""
if SIN_MEZCLA:
    suf += "_sinmezcla"
if SD_FIJA:
    suf += "_sf" + "_".join(f"{k}{v:g}" for k, v in sorted(SD_FIJA.items()))


# ---------------------------------------------------------------- modelo ----
def centra(a):
    return a - a.mean(axis=-1, keepdims=True)


def z_efecto(nombre, n):
    # z (n x J) ~ N(0, 1); con --suma0, cada columna suma 0 entre niveles
    # (escala sqrt(n / (n - 1)): la sd marginal de cada nivel sigue siendo 1)
    if nombre not in SUMA0:
        return numpyro.sample(
            f"z_{nombre}", dist.Normal(0.0, 1.0).expand([n, J]).to_event(2)
        )
    u = numpyro.sample(
        f"u_{nombre}", dist.Normal(0.0, 1.0).expand([J, n - 1]).to_event(2)
    )
    return numpyro.deterministic(
        f"z_{nombre}", float(np.sqrt(n / (n - 1.0))) * (u @ Q_niv[nombre].T).T
    )


def model(
    Y,
    MUJER,
    OFF,
    CASA,
    MES,
    EDAD,
    EDU,
    REC,
    CCAA,
    CCAA_REC,
    CASA_REC,
    MIX,
    PROV_FILA,
    ENT_PROV,
    ENT_W,
    MASK,
    SIN_AA,
):
    za = numpyro.sample(
        "z_alpha", dist.Normal(0.0, s_alpha).expand([J - 1]).to_event(1)
    )
    zb = numpyro.sample(
        "z_bmujer", dist.Normal(0.0, s_b_mujer).expand([J - 1]).to_event(1)
    )
    alpha = Q_alpha @ za
    b_mujer = Q_alpha @ zb

    def sd_bloque(nombre, s):
        # con --sd-fija, la sd sale del modelo: solo queda su valor constante
        # (deterministic la deja en get_samples para la postestratificación)
        if nombre in SD_FIJA:
            return numpyro.deterministic(f"sd_{nombre}", jnp.full(J, SD_FIJA[nombre]))
        return numpyro.sample(
            f"sd_{nombre}", dist.HalfNormal(s).expand([J]).to_event(1)
        )

    sd_casa = sd_bloque("casa", s_sd_main)
    sd_mes = sd_bloque("mes", s_sd_main)
    sd_edad = sd_bloque("edad", s_sd_main)
    sd_edu = sd_bloque("edu", s_sd_main)
    sd_rec = sd_bloque("rec", s_sd_rec)
    sd_ccaa = sd_bloque("ccaa", s_sd_main)
    sd_prov = sd_bloque("prov", s_sd_main)
    sd_ccaa_rec = sd_bloque("ccaa_rec", s_sd_int)
    sd_casa_rec = sd_bloque("casa_rec", s_sd_int)

    z_casa = numpyro.sample(
        "z_casa", dist.Normal(0.0, 1.0).expand([J, n_casa - 1]).to_event(2)
    )
    # columna k = esc_casa * sd_casa[k] * (Q_casa z_k): suma 0 entre casas;
    # centra() suma 0 entre partidos, como en Stan
    a_casa = centra(esc_casa * (z_casa @ Q_casa.T).T * sd_casa[None, :])
    z_mes = z_efecto("mes", n_mes)
    a_mes = centra(z_mes * sd_mes[None, :])
    z_edad = z_efecto("edad", n_edad)
    a_edad = centra(z_edad * sd_edad[None, :])
    z_edu = z_efecto("edu", n_edu)
    a_edu = centra(z_edu * sd_edu[None, :])
    z_rec = z_efecto("rec", n_rec)
    a_rec = centra(z_rec * sd_rec[None, :])
    z_ccaa = z_efecto("ccaa", n_ccaa)
    a_ccaa = centra(z_ccaa * sd_ccaa[None, :])
    z_prov = z_efecto("prov", n_prov)
    a_prov = centra(z_prov * sd_prov[None, :])
    z_ccaa_rec = numpyro.sample(
        "z_ccaa_rec", dist.Normal(0.0, 1.0).expand([n_ccaa * n_rec, J]).to_event(2)
    )
    a_ccaa_rec = centra(z_ccaa_rec * sd_ccaa_rec[None, :])
    z_casa_rec = numpyro.sample(
        "z_casa_rec", dist.Normal(0.0, 1.0).expand([n_casa * n_rec, J]).to_event(2)
    )
    a_casa_rec = centra(z_casa_rec * sd_casa_rec[None, :])

    eta = (
        alpha[None, :]
        + MUJER[:, None] * b_mujer[None, :]
        + a_casa[CASA]
        + a_mes[MES]
        + a_edad[EDAD]
        + a_edu[EDU]
        + a_rec[REC]
        + a_ccaa[CCAA]
        + a_ccaa_rec[CCAA_REC]
        + a_casa_rec[CASA_REC]
        + OFF
    )
    lp = jax.nn.log_softmax(eta + a_prov[PROV_FILA])
    eta_mix = eta[MIX][:, None, :] + a_prov[ENT_PROV]
    lpe = jnp.where(
        MASK[..., None],
        jnp.log(ENT_W)[..., None] + jax.nn.log_softmax(eta_mix, axis=-1),
        -jnp.inf,
    )
    lp = lp.at[MIX].set(logsumexp(lpe, axis=1))
    # filas sin AA en el cuestionario: log(1 - p_AA) = log1p(-exp(lp_AA));
    # los y de AA son 0 en esas filas, así que solo falta renormalizar
    pen = jnp.where(
        SIN_AA == 1,
        Y.sum(-1) * jnp.log1p(-jnp.exp(lp[:, J_AA])),
        0.0,
    )
    numpyro.factor("ll", jnp.sum(Y * lp) - jnp.sum(pen))


# ------------------------------------------------------------------ bench ----
dtype_np = np.float32 if args.x32 else np.float64
fj = lambda a: jnp.asarray(a.astype(dtype_np))  # noqa: E731
datos_jax = dict(
    Y=fj(y),
    MUJER=fj(mujer),
    OFF=fj(off),
    CASA=jnp.asarray(idx_np["casa"], jnp.int32),
    MES=jnp.asarray(idx_np["mes"], jnp.int32),
    EDAD=jnp.asarray(idx_np["edad"], jnp.int32),
    EDU=jnp.asarray(idx_np["edu"], jnp.int32),
    REC=jnp.asarray(idx_np["rec"], jnp.int32),
    CCAA=jnp.asarray(idx_np["ccaa"], jnp.int32),
    CCAA_REC=jnp.asarray(ccaa_rec, jnp.int32),
    CASA_REC=jnp.asarray(casa_rec, jnp.int32),
    MIX=jnp.asarray(mix_rows, jnp.int32),
    PROV_FILA=jnp.asarray(prov_fila, jnp.int32),
    ENT_PROV=jnp.asarray(ent_pad_prov, jnp.int32),
    ENT_W=fj(ent_pad_w),
    MASK=jnp.asarray(ent_mask),
    SIN_AA=jnp.asarray(sin_aa, jnp.int32),
)

warmup = args.warmup or (150 if args.modo == "prueba" else 1000)
samples = args.samples or (150 if args.modo == "prueba" else 1000)
print(f"modo: {args.modo}  N: {N}  J: {J}  P: {P}  N2 (mezcla): {N2}  E: {E}")
print(f"niveles con suma 0: {', '.join(['casa'] + SUMA0)}")
if SIN_MEZCLA:
    print("sin filas mezcla: solo filas con provincia conocida")
if SD_FIJA:
    print("sd fijadas:", SD_FIJA)
print(
    f"iteraciones: {warmup} warmup + {samples} muestreo, {CHAINS} cadena(s) en paralelo, "
    f"{'float32' if args.x32 else 'float64'}"
)

mcmc = MCMC(
    # init_to_feasible: cero en la escala no restringida, como init = 0 de Stan
    NUTS(
        model,
        target_accept_prob=0.95,  # el adapt_delta de Stan
        max_tree_depth=8,
        init_strategy=init_to_feasible(),
    ),
    num_warmup=warmup,
    num_samples=samples,
    num_chains=CHAINS,
    progress_bar=False,
)
t0 = time.time()
mcmc.run(
    jax.random.PRNGKey(2026), **datos_jax, extra_fields=("num_steps", "accept_prob")
)
# run() despacha en asíncrono: sin esperar, t_mcmc mide solo compilación + despacho
jax.block_until_ready(mcmc.get_samples())
t_mcmc = time.time() - t0
extra = mcmc.get_extra_fields()
print(f"\nMCMC (compilación incluida): {t_mcmc:.1f} s = {t_mcmc / 60:.2f} min")
print(f"divergencias: {int(extra['diverging'].sum())} de {warmup + samples}")
print(f"pasos por árbol (mediana): {int(np.median(extra['num_steps']))}")
# extra solo trae el muestreo: se supone el mismo número medio de pasos en el
# warmup; las cadenas van en paralelo, así que el tiempo es el de una cadena
pasos_cadena = (warmup + samples) * float(np.mean(extra["num_steps"]))
print(
    f"ms por gradiente (aprox., compilación incluida): "
    f"{1000 * t_mcmc / pasos_cadena:.2f}"
)


# -------------------------------------------------- postestratificación ----
# v_casa: voto válido (%) en España según cada casa y la casa media, con los
# draws de la posterior. Las mismas fórmulas que stan/ajusta.R (sección 3).
def centra_np(a):
    return a - a.mean(axis=-1, keepdims=True)


s = mcmc.get_samples(group_by_chain=False)
nd = s["z_alpha"].shape[0]
alpha_d = s["z_alpha"] @ Q_alpha.T  # (nd, J)
b_mujer_d = s["z_bmujer"] @ Q_alpha.T
# columna k = esc_casa * sd_casa[k] * (Q_casa z_k); centra() por filas
a_casa_d = centra_np(
    esc_casa * (s["z_casa"] @ Q_casa.T).transpose(0, 2, 1) * s["sd_casa"][:, None, :]
)
A = {
    k: centra_np(s[f"z_{k}"] * s[f"sd_{k}"][:, None, :])
    for k in ["mes", "edad", "edu", "rec", "ccaa", "prov", "ccaa_rec", "casa_rec"]
}


def prob_valida(eta):
    # softmax con máscara de territorio, por filas
    e = eta - eta.max(axis=-1, keepdims=True)
    p = np.exp(e) * P_mascara
    return p / p.sum(axis=-1, keepdims=True)


# Participación (mrp_part_np.py), si está: la muestra d del voto se empareja
# con la d de la participación, como en stan/ajusta.R (v_part)
f_part = f"2026/11/mrp-multiencuesta/datos/part_numpyro_draws_{args.modo}.npz"
Tp = None
try:
    Tp = dict(np.load(f_part))
    if Tp["delta"].shape[0] < nd:
        print("(participación con menos muestras que el voto: se ignora)")
        Tp = None
    elif Tp["delta"].shape[0] > nd:
        Tp = {k: v[:nd] for k, v in Tp.items()}
        print(f"participación: {f_part} (recortada a {nd})")
    else:
        print(f"participación: {f_part}")
except FileNotFoundError:
    print(f"(sin {f_part}: solo v_casa)")

# por trozos de draws: con 4.000 muestras, los arrays intermedios (draws x P
# x J) no caben en memoria de golpe
t0 = time.time()
v_casa = np.zeros((nd, n_casa + 1, J))
v_part = np.zeros((nd, n_casa + 1, J))
# --celdas: guarda la casa media por celda, adelgazada a draws equiespaciados
# (como adelgaza() en el flujo de Stan: round(seq(1, nd, length.out = 1000)))
n_out = 1000 if args.modo == "nuts" else 50
pos_out = np.full(nd, -1, dtype=np.int64)
pos_out[np.round(np.linspace(0, nd - 1, n_out)).astype(int)] = np.arange(n_out)
p_celdas = np.zeros((n_out, P, J), dtype=np.float32) if args.celdas else None
pv_celdas = (
    np.zeros((n_out, P), dtype=np.float32) if args.celdas and Tp is not None else None
)
CH = 100
for a in range(0, nd, CH):
    sl = slice(a, min(a + CH, nd))
    base0 = (
        alpha_d[sl][:, None, :]
        + P_mujer[None, :, None] * b_mujer_d[sl][:, None, :]
        + A["mes"][sl][:, mes_post][:, None, :]
        + A["edad"][sl][:, P_edad, :]
        + A["edu"][sl][:, P_edu, :]
        + A["rec"][sl][:, P_rec, :]
        + A["ccaa"][sl][:, P_ccaa, :]
        + A["prov"][sl][:, P_prov, :]
        + A["ccaa_rec"][sl][:, i_cr, :]
    )  # (chunk, P, J)
    if Tp is not None:
        eta_t = (
            Tp["alpha"][sl, None]
            + P_mujer[None, :] * Tp["b"][sl, None]
            + Tp["mes"][sl, mes_post][:, None]
            + Tp["edad"][sl][:, P_edad]
            + Tp["edu"][sl][:, P_edu]
            + Tp["rec"][sl][:, P_rec]
            + Tp["ccaa"][sl][:, P_ccaa]
            + Tp["prov"][sl][:, P_prov]
        )
        w_part = P_N[None, :] / (1 + np.exp(-(eta_t + Tp["delta"][sl, None])))
    ef_media = np.zeros((sl.stop - sl.start, P, J))
    for h in range(n_casa + 1):
        if h < n_casa:
            ef = (
                a_casa_d[sl][:, h, :][:, None, :]
                + A["casa_rec"][sl][:, h * n_rec + P_rec, :]
            )
            ef_media += ef / n_casa
        else:
            ef = ef_media
        p = prob_valida(base0 + ef)
        if p_celdas is not None and h == n_casa:
            sel = pos_out[sl]
            m = sel >= 0
            p_celdas[sel[m]] = p[m].astype(np.float32)
            if pv_celdas is not None:
                # pv sin el N de la celda (como pv de 07_post_celdas.R): el
                # producto P_N * pv lo hace 08_escanos.R
                pv_celdas[sel[m]] = (w_part[m] / P_N).astype(np.float32)
        v_casa[sl, h, :] = (P_N[None, :, None] * p).sum(axis=1) / P_N.sum() * 100
        if Tp is not None:
            v_part[sl, h, :] = (
                (w_part[:, :, None] * p).sum(axis=1) / w_part.sum(axis=1)[:, None] * 100
            )
t_post = time.time() - t0
print(f"\npostestratificación: {nd} muestras, {t_post:.1f} s")

if CHAINS > 1:
    try:
        import arviz as az

        idata = az.from_numpyro(mcmc)
        summ = az.summary(
            idata,
            var_names=[
                "sd_casa",
                "sd_mes",
                "sd_edad",
                "sd_edu",
                "sd_rec",
                "sd_ccaa",
                "sd_prov",
                "sd_ccaa_rec",
                "sd_casa_rec",
            ],
        )
        print(
            f"Rhat máximo de las sd: {summ['r_hat'].max():.3f}"
            f"  ESS bulk mínimo: {summ['ess_bulk'].min():.0f}"
        )
    except Exception as e:
        print("(arviz no disponible:", e, ")")

if "casa_lv" in z:
    nombres = [str(c) for c in z["casa_lv"]]
else:
    nombres = [f"casa {h + 1}" for h in range(n_casa)] + ["casa media"]
partidos = [str(c) for c in z["voto_lv"]] if "voto_lv" in z else None
print("\n== MRP por casa (% voto válido, media de los draws) ==")
print("partidos:", partidos)
for h, nom in enumerate(nombres):
    fila = ", ".join(f"{v:.1f}" for v in v_casa[:, h, :].mean(axis=0))
    print(f"{nom:>16}: {fila}")
if Tp is not None:
    print("\n== MRP por casa, con participación calibrada (% voto válido) ==")
    for h, nom in enumerate(nombres):
        fila = ", ".join(f"{v:.1f}" for v in v_part[:, h, :].mean(axis=0))
        print(f"{nom:>16}: {fila}")
f_salida = (
    args.salida
    or f"2026/11/mrp-multiencuesta/datos/v_casa_numpyro_{args.modo}{suf}.npz"
)
np.savez_compressed(
    f_salida,
    v_casa=v_casa,
    **({"v_part": v_part} if Tp is not None else {}),
)
print(f"\nguardado {f_salida}")

if args.celdas:
    kw = {"p": p_celdas, "voto_lv": np.asarray(z["voto_lv"])}
    if pv_celdas is not None:
        kw["pv"] = pv_celdas
    np.savez_compressed(args.celdas, **kw)
    print(f"guardado {args.celdas}")
