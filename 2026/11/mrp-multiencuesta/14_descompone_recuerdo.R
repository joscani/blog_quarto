# Descomposición del efecto del recuerdo de voto (material para el post III).
#
# Parte del ajuste de producción (datos/fit_separado_nuts_voto.rds) y repite
# la postestratificación nacional (casa media, electorado, con participación)
# quitando por partes los términos de recuerdo:
#   completo       a_rec + a_ccaa_rec + a_casa_rec        (la producción)
#   sin_casa_rec   sin la interacción casa x rec
#   sin_ccaa_rec   sin la interacción ccaa x rec
#   solo_rec       solo el efecto principal del recuerdo
#   sin_rec        sin nada de recuerdo (descomposición sobre el fit de
#                  producción; el reajuste sintrec además reestima el resto
#                  del modelo: parecido, pero no idéntico)
#
# OJO: la separación entre los tres términos depende del centra() (columnas
# centradas): lo único canónico es completo frente a sin_rec. Las variantes
# intermedias describen el modelo tal como está ajustado.
#
# Además calcula la matriz de transición recuerdo -> voto (casa media) y la
# fidelidad por casa, P(vota j | recordó votar a j).
#
# Salidas: datos/descompone_recuerdo.rds y las tablas impresas.

library(dplyr)
library(tidyr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

x <- readRDS(file.path(dir_datos, "datos_stan.rds"))
datos <- x$datos
voto_lv <- x$voto_lv
rec_lv <- x$lv$rec

av <- readRDS(file.path(dir_datos, "fit_separado_nuts_voto.rds"))
ap <- readRDS(file.path(dir_datos, "fit_separado_nuts_part.rds"))

matriz <- function(dm) matrix(dm, nrow(dm), dimnames = list(NULL, colnames(dm)))
adelgaza <- function(dm, n) {
  dm[round(seq(1, nrow(dm), length.out = n)), , drop = FALSE]
}
ndraws <- 1000L
dv <- adelgaza(matriz(av$fit$draws(format = "draws_matrix")), ndraws)
dp <- adelgaza(matriz(ap$fit$draws(format = "draws_matrix")), ndraws)
nd <- min(nrow(dv), nrow(dp))

J <- datos$J
P <- length(datos$P_N)
n_casa <- datos$n_casa
n_rec <- datos$n_rec
P_N <- datos$P_N
P_mujer <- datos$P_mujer
P_edad <- datos$P_edad
P_edu <- datos$P_edu
P_rec <- datos$P_rec
P_ccaa <- datos$P_ccaa
P_prov <- datos$P_prov
mascara <- matrix(datos$P_mascara, P, J)
i_cr <- (P_ccaa - 1) * n_rec + P_rec
mes_post <- datos$mes_post

arr <- function(dm, v, n) {
  m <- dm[, paste0(v, "[", rep(1:n, J), ",", rep(1:J, each = n), "]")]
  array(m, c(nrow(dm), n, J))
}
vecs <- function(dm, v, n) dm[, paste0(v, "[", 1:n, "]"), drop = FALSE]

A <- list(
  alpha = vecs(dv, "alpha", J),
  b_mujer = vecs(dv, "b_mujer", J),
  casa = arr(dv, "a_casa", n_casa),
  mes = arr(dv, "a_mes", datos$n_mes),
  edad = arr(dv, "a_edad", datos$n_edad),
  edu = arr(dv, "a_edu", datos$n_edu),
  rec = arr(dv, "a_rec", n_rec),
  ccaa = arr(dv, "a_ccaa", datos$n_ccaa),
  prov = arr(dv, "a_prov", datos$n_prov),
  ccaa_rec = arr(dv, "a_ccaa_rec", datos$n_ccaa * n_rec),
  casa_rec = arr(dv, "a_casa_rec", n_casa * n_rec)
)
Tp <- list(
  alpha = dp[, "alpha_t"],
  b = dp[, "b_mujer_t"],
  mes = vecs(dp, "a_mes_t", datos$n_mes),
  edad = vecs(dp, "a_edad_t", datos$n_edad),
  edu = vecs(dp, "a_edu_t", datos$n_edu),
  rec = vecs(dp, "a_rec_t", n_rec),
  ccaa = vecs(dp, "a_ccaa_t", datos$n_ccaa),
  prov = vecs(dp, "a_prov_t", datos$n_prov),
  delta = dp[, "delta"]
)

prob <- function(eta) {
  p <- exp(eta - apply(eta, 1, max)) * mascara
  p / rowSums(p)
}

# ---- 1. variantes de la postestratificación -----------------------------------

variantes <- list(
  completo = c(rec = TRUE, ccaa = TRUE, casa = TRUE),
  sin_casa_rec = c(rec = TRUE, ccaa = TRUE, casa = FALSE),
  sin_ccaa_rec = c(rec = TRUE, ccaa = FALSE, casa = TRUE),
  solo_rec = c(rec = TRUE, ccaa = FALSE, casa = FALSE),
  sin_rec = c(rec = FALSE, ccaa = FALSE, casa = FALSE)
)
nombres_v <- names(variantes)
v_var <- array(
  NA,
  c(nd, length(nombres_v), J),
  list(draw = NULL, variante = nombres_v, partido = voto_lv)
)

t0 <- Sys.time()
for (d in seq_len(nd)) {
  base0 <- outer(rep(1, P), A$alpha[d, ]) +
    outer(P_mujer, A$b_mujer[d, ]) +
    matrix(A$mes[d, mes_post, ], P, J, byrow = TRUE) +
    A$edad[d, P_edad, ] +
    A$edu[d, P_edu, ] +
    A$ccaa[d, P_ccaa, ] +
    A$prov[d, P_prov, ]
  eta_t <- Tp$alpha[d] +
    P_mujer * Tp$b[d] +
    Tp$mes[d, mes_post] +
    Tp$edad[d, P_edad] +
    Tp$edu[d, P_edu] +
    Tp$rec[d, P_rec] +
    Tp$ccaa[d, P_ccaa] +
    Tp$prov[d, P_prov]
  w <- plogis(eta_t + Tp$delta[d]) * P_N
  S <- sum(w)
  t_rec <- A$rec[d, P_rec, ]
  t_ccaa <- A$ccaa_rec[d, i_cr, ]
  ef_con <- 0
  ef_sin <- 0
  for (h in seq_len(n_casa)) {
    ef_h <- matrix(A$casa[d, h, ], P, J, byrow = TRUE)
    ef_sin <- ef_sin + ef_h / n_casa
    ef_con <- ef_con +
      (ef_h + A$casa_rec[d, (h - 1) * n_rec + P_rec, ]) / n_casa
  }
  for (nv in nombres_v) {
    v <- variantes[[nv]]
    eta <- base0 +
      (if (v[["rec"]]) t_rec else 0) +
      (if (v[["ccaa"]]) t_ccaa else 0) +
      (if (v[["casa"]]) ef_con else ef_sin)
    v_var[d, nv, ] <- colSums(w * prob(eta)) / S
  }
}
cat(
  "postestratificación por variante:",
  round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
  "min\n"
)

tab_variantes <- as.data.frame(apply(v_var, c(2, 3), mean) * 100) |>
  mutate(
    variante = rownames(apply(v_var, c(2, 3), mean)),
    hueco_PSOE_PP = PSOE - PP
  ) |>
  select(variante, PP, PSOE, VOX, SUMAR, OTROS, hueco_PSOE_PP)

# referencia: el reajuste sintrec (posterior completo sin recuerdo)
sint <- readRDS(file.path(dir_proy, "sintrec", "estim_separado_nuts.rds"))
ref_sint <- sint |>
  filter(casa == "casa media, con participación") |>
  select(partido, mrp) |>
  filter(partido %in% c("PP", "PSOE", "VOX", "SUMAR", "OTROS")) |>
  pivot_wider(names_from = partido, values_from = mrp) |>
  mutate(
    variante = "sintrec (reajuste)",
    hueco_PSOE_PP = PSOE - PP
  ) |>
  select(variante, PP, PSOE, VOX, SUMAR, OTROS, hueco_PSOE_PP)

cat("\n== % voto, casa media y electorado, por variante ==\n")
print(
  as.data.frame(
    bind_rows(tab_variantes, ref_sint) |>
      mutate(across(where(is.numeric), ~ round(.x, 1)))
  ),
  row.names = FALSE
)

# ---- 2. transición recuerdo -> voto -------------------------------------------
# p(vota j | recordó votar r), ponderando las celdas de población con ese
# recuerdo por su peso censal rakeado.

trans_media <- array(
  NA,
  c(nd, n_rec, J),
  list(draw = NULL, rec = rec_lv, partido = voto_lv)
)
filas_casa <- intersect(
  c("PP", "PSOE", "VOX", "SUMAR", "ABST", "NO_PODIA"),
  rec_lv
)
fidelidad_rec <- intersect(c("PP", "PSOE", "VOX", "SUMAR"), rec_lv)
trans_casa <- array(
  NA,
  c(nd, length(filas_casa), n_casa, J),
  list(draw = NULL, rec = filas_casa, casa = NULL, partido = voto_lv)
)

t0 <- Sys.time()
for (d in seq_len(nd)) {
  base0 <- outer(rep(1, P), A$alpha[d, ]) +
    outer(P_mujer, A$b_mujer[d, ]) +
    matrix(A$mes[d, mes_post, ], P, J, byrow = TRUE) +
    A$edad[d, P_edad, ] +
    A$edu[d, P_edu, ] +
    A$ccaa[d, P_ccaa, ] +
    A$prov[d, P_prov, ]
  ef_media <- 0
  ef_h <- vector("list", n_casa)
  for (h in seq_len(n_casa)) {
    ef_h[[h]] <- matrix(A$casa[d, h, ], P, J, byrow = TRUE)
    ef_media <- ef_media + ef_h[[h]] / n_casa
  }
  for (r in seq_len(n_rec)) {
    w_r <- P_N * (P_rec == r)
    if (sum(w_r) == 0) {
      next
    }
    S_r <- sum(w_r)
    i_r <- (P_ccaa - 1) * n_rec + r
    extra_media <- matrix(A$rec[d, r, ], P, J, byrow = TRUE) +
      A$ccaa_rec[d, i_r, ]
    p <- prob(base0 + ef_media + extra_media)
    trans_media[d, r, ] <- colSums(w_r * p) / S_r
    for (h in seq_len(n_casa)) {
      if (!rec_lv[r] %in% filas_casa) {
        next
      }
      extra_h <- extra_media + A$casa_rec[d, (h - 1) * n_rec + r, ]
      p_h <- prob(base0 + ef_h[[h]] + extra_h)
      trans_casa[d, match(rec_lv[r], filas_casa), h, ] <- colSums(w_r * p_h) /
        S_r
    }
  }
}
cat(
  "transiciones:",
  round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
  "min\n"
)

cols <- intersect(c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "OTROS"), voto_lv)
tab_trans <- 100 * apply(trans_media, c(2, 3), mean)
cat("\n== transición recuerdo -> voto, casa media (% en fila) ==\n")
print(round(tab_trans[, cols], 1))

cat("\n== fidelidad por casa: P(vota j | recordó votar a j) ==\n")
fid <- sapply(seq_len(n_casa), function(h) {
  sapply(fidelidad_rec, function(pr) {
    r <- match(pr, rec_lv)
    100 * mean(trans_casa[, match(pr, filas_casa), h, match(pr, voto_lv)])
  })
})
colnames(fid) <- x$lv$casa
print(round(t(fid), 1))

cat("\n== por casa: transición de los que recordaron votar PP y PSOE ==\n")
for (pr in c("PP", "PSOE")) {
  cat("\n-- recordó votar", pr, "--\n")
  m <- sapply(seq_len(n_casa), function(h) {
    100 * colMeans(trans_casa[, match(pr, filas_casa), h, ])[cols]
  })
  dimnames(m) <- list(partido = cols, casa = x$lv$casa)
  print(round(t(m), 1))
}

saveRDS(
  list(
    variantes = bind_rows(tab_variantes, ref_sint),
    trans_media = tab_trans,
    trans_casa = apply(trans_casa, c(2, 3, 4), mean),
    fidelidad = fid,
    draws = list(
      v_var = v_var,
      trans_media = trans_media,
      trans_casa = trans_casa
    )
  ),
  file.path(dir_datos, "descompone_recuerdo.rds")
)
cat("\nguardado datos/descompone_recuerdo.rds\n")
