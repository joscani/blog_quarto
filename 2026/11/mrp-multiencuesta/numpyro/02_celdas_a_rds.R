# Convierte el npz de celdas de numpyro (mrp_voto_np.py --celdas) al formato
# de post_celdas_casa_media*.rds que lee 08_escanos.R:
#   p     draws x celda x partido (casa media; la máscara de territorio ya va
#         aplicada y renormalizada, como prob_valida() en 05b)
#   post  tibble de celdas (prov, ccaa, sexo, edad, edu, rec, N, pv_media)
#   pv    draws x celda (probabilidad de participar, SIN el N de la celda)
#   voto_lv, mes, fuente, ndraws, seed
#
# Las etiquetas de edad/edu/sexo salen de los dimnames de tabla_post_prov.rds
# (el orden de esos niveles no cambia) y las de recuerdo de la versión anterior
# al alta de UPN (13 niveles: los 14 actuales menos UPN). Solo prov y N entran
# en el cálculo de escaños, así que una etiqueta equivocada no movería nada.
#
# Uso: Rscript 2026/11/mrp-multiencuesta/numpyro/02_celdas_a_rds.R <npz>

suppressMessages({
  library(reticulate)
  library(dplyr)
  library(tibble)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) %in% 1:2)
use_virtualenv("~/.virtualenvs/r-mrp-numpyro", required = TRUE)
np <- import("numpy")

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

npz <- np$load(args[1])
zd <- np$load(file.path(dir_datos, "voto_numpyro_nuts.npz"))
f_vc <- if (length(args) == 2) {
  args[2]
} else {
  file.path(dir_datos, "v_casa_numpyro_nuts.npz")
}
vc <- np$load(f_vc)

# py_to_r conserva las dimensiones de numpy tal cual: p es (draws, P, J)
p <- npz[["p"]]
pv <- npz[["pv"]]
voto_lv <- npz[["voto_lv"]]
nd <- dim(p)[1]

P_N <- as.numeric(py_to_r(zd[["P_N"]]))
P_mujer <- as.numeric(py_to_r(zd[["P_mujer"]]))
P_edad <- as.integer(py_to_r(zd[["P_edad"]]))
P_edu <- as.integer(py_to_r(zd[["P_edu"]]))
P_rec <- as.integer(py_to_r(zd[["P_rec"]]))
P_ccaa <- as.integer(py_to_r(zd[["P_ccaa"]]))
P_prov <- as.integer(py_to_r(zd[["P_prov"]]))
stopifnot(length(P_N) == dim(p)[2], length(voto_lv) == dim(p)[3])

# etiquetas
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds")) |>
  mutate(prov = as.character(as.integer(as.character(prov))))
tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
dn <- dimnames(tabla)
rec_lv <- setdiff(dn$rec, "UPN") # 13 niveles de antes del alta de UPN
stopifnot(length(rec_lv) == max(P_rec))

ccaa_lbl <- prov_ccaa$ccaa[match(as.character(P_prov), prov_ccaa$prov)]
# comprobación: el índice de ccaa del npz y el de prov_ccaa tienen que cuadrar
stopifnot(identical(
  as.integer(P_ccaa),
  as.integer(match(ccaa_lbl, levels(prov_ccaa$ccaa)))
))

post <- tibble(
  prov = as.character(P_prov),
  ccaa = as.character(ccaa_lbl),
  sexo = ifelse(P_mujer == 1, "M", "H"),
  edad = dn$edad[P_edad],
  edu = dn$edu[P_edu],
  rec = rec_lv[P_rec],
  N = P_N,
  pv_media = colMeans(pv)
)

# comprobación end-to-end: el agregado nacional de las celdas tiene que
# reproducir el v_part (casa media) que guardó mrp_voto_np.py, draw a draw
n_casa <- as.integer(py_to_r(zd[["n_casa"]]))
vp_arr <- py_to_r(vc[["v_part"]]) # (draws, n_casa + 1, J)
vp_ref <- vp_arr[, n_casa + 1, ] # casa media: (draws, J)
nd_ref <- dim(vp_arr)[1]
i_out <- round(seq(1, nd_ref, length.out = nd)) # los mismos draws que p
# sweep y no PN * pv: el reciclaje de R va por columnas
w <- sweep(pv, 2, post$N, `*`)
v_np <- sapply(seq_along(voto_lv), function(j) {
  rowSums(p[,, j] * w) / rowSums(w) * 100
})
delta <- max(abs(v_np - vp_ref[i_out, ]))
stopifnot(delta < 0.02)
cat(
  "comprobación v_part: diferencia máxima",
  sprintf("%.4f", delta),
  "puntos\n"
)

mes <- readRDS(file.path(dir_datos, "estim_escanos_mrp.rds"))$mes
dimnames(p) <- list(draw = seq_len(nd), celda = NULL, partido = voto_lv)
salida <- file.path(dir_datos, "post_celdas_casa_media_numpyro_part.rds")
saveRDS(
  list(
    p = p,
    post = post,
    voto_lv = voto_lv,
    pv = pv,
    mes = mes,
    fuente = "numpyro",
    ndraws = nd,
    seed = 2026
  ),
  salida
)
cat(
  "guardado",
  salida,
  "(",
  nd,
  "draws x",
  nrow(post),
  "celdas x",
  length(voto_lv),
  "partidos )\n"
)
