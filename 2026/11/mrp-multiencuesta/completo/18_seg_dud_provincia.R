# Seguros y dudosos por provincia, para los mapas del post (III).
#
# Mismo cálculo que completo/12_escenarios_voto_seg.R y 17_votos_grupos.R
# (método seg, casa media, octubre de 2026), pero separa en cada provincia los
# votos de las dos mitades de cada celda, en el escenario base:
#   seguros: N * pi * p_seg
#   dudosos: N * (pi_cal - pi) * p_dud   (los dudosos que votan)
# La suma de las dos es votos["seg", base, , , ] de escenarios.rds (se comprueba).
#
# Draws: los mismos 200 que 17 (draw d = draw 5 * (d - 1) + 1 de escenarios.rds).
# También guarda la geometría de las provincias (mapSpain, Canarias movida).
#
# Salidas: completo/datos/seg_dud_prov.rds, completo/datos/provincias_sf.rds
#
#   Rscript 2026/11/mrp-multiencuesta/completo/18_seg_dud_provincia.R

suppressPackageStartupMessages({
  library(dplyr)
  library(posterior)
  library(Matrix)
})

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_completo <- file.path(dir_proy, "completo", "datos")
source(file.path(dir_proy, "00_partidos.R")) # territorio

# ---- 1. Celdas y niveles (como 12) ---------------------------------------------

entradas <- readRDS(file.path(dir_completo, "entradas_escenarios.rds"))
post <- entradas$post
lv <- entradas$lv
mes_post <- entradas$mes_post
part_base <- entradas$part_obj
x_voto <- readRDS(file.path(dir_completo, "datos_voto_seg.rds"))
for (k in names(lv)) stopifnot(identical(as.character(lv[[k]]), as.character(x_voto$lv[[k]])))
voto_lv <- as.character(x_voto$lv$voto)
J <- length(voto_lv)
P <- nrow(post)
N <- post$N
mujer <- as.numeric(post$sexo == "M")
ix <- function(x, niveles) {
  i <- match(as.character(x), niveles)
  stopifnot(!anyNA(i))
  i
}
i_edad <- ix(post$edad, lv$edad)
i_edu <- ix(post$edu, lv$edu)
i_rec <- ix(post$rec, lv$rec)
i_ccaa <- ix(post$ccaa, lv$ccaa)
i_prov <- ix(post$prov, lv$prov)

# ---- 2. Draws: los de 12, uno de cada 5 ----------------------------------------

sel_draws <- function(n_tot) round(seq(1, n_tot, length.out = 1000L))[seq(1, 1000, by = 5)]
nd <- 200L

vars_voto <- c("alpha", "b_mujer", "b_dud", "a_casa", "a_mes", "a_edad", "a_edu", "a_rec",
               "a_ccaa", "a_prov", "a_casa_rec", "a_dud_edad", "a_dud_rec")
dv <- as_draws_rvars(readRDS(file.path(dir_completo, "fit_voto_seg.rds"))$fit$draws(vars_voto))
g <- function(k) unname(draws_of(dv[[k]]))
sel <- sel_draws(dim(g("alpha"))[1])
V <- lapply(vars_voto, function(k) {
  a <- g(k)
  if (length(dim(a)) == 2) a[sel, , drop = FALSE] else a[sel, , , drop = FALSE]
})
names(V) <- vars_voto
n_casa <- dim(V$a_casa)[2]
n_rec <- length(lv$rec)

territorio_mod <- territorio[names(territorio) != "AC"]
log_mask <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) ifelse(post$ccaa %in% territorio_mod[[p]], 0, -Inf) else rep(0, P)
})
softmax_filas <- function(e) {
  e <- e - apply(e, 1, max)
  p <- exp(e)
  p / rowSums(p)
}

ap <- readRDS(file.path(dir_datos, "fit_separado_nuts_part.rds"))$fit
dp <- as_draws_matrix(ap$draws(c("alpha_t", "b_mujer_t", "a_mes_t", "a_edad_t",
                                 "a_edu_t", "a_rec_t", "a_ccaa_t", "a_prov_t")))
dp <- dp[sel_draws(nrow(dp)), ]
dp <- matrix(as.numeric(dp), nrow(dp), dimnames = list(NULL, colnames(dp)))
col_t <- function(v, i) unname(dp[, paste0(v, "[", i, "]"), drop = FALSE])

# ---- 3. Seguros y dudosos por provincia, en cada draw ---------------------------

esc <- readRDS(file.path(dir_completo, "escenarios_seg", "escenarios.rds"))
prov_ref <- esc$prov_ref
prov_i <- match(as.integer(as.character(post$prov)), as.integer(prov_ref$codigo_provincia))
stopifnot(!anyNA(prov_i))
A <- sparseMatrix(i = seq_len(P), j = prov_i, x = 1, dims = c(P, nrow(prov_ref)))

arr <- function() {
  array(0, c(nd, nrow(prov_ref), J),
        list(draw = NULL, provincia = prov_ref$provincia, partido = voto_lv))
}
seguros <- arr()
dudosos <- arr()

t0 <- Sys.time()
for (d in seq_len(nd)) {
  casa_media <- Reduce(`+`, lapply(seq_len(n_casa), function(h) {
    matrix(V$a_casa[d, h, ], n_rec, J, byrow = TRUE) +
      V$a_casa_rec[d, (h - 1) * n_rec + seq_len(n_rec), ]
  })) / n_casa
  fila <- function(a, i) a[d, i, , drop = TRUE]
  eta <- matrix(V$alpha[d, ], P, J, byrow = TRUE) +
    outer(mujer, V$b_mujer[d, ]) +
    matrix(V$a_mes[d, mes_post, ], P, J, byrow = TRUE) +
    fila(V$a_edad, i_edad) + fila(V$a_edu, i_edu) + fila(V$a_rec, i_rec) +
    fila(V$a_ccaa, i_ccaa) + fila(V$a_prov, i_prov) + casa_media[i_rec, ] + log_mask
  dd <- matrix(V$b_dud[d, ], P, J, byrow = TRUE) + fila(V$a_dud_edad, i_edad) +
    fila(V$a_dud_rec, i_rec)
  p_s <- softmax_filas(eta)
  p_d <- softmax_filas(eta + dd)

  eta_t <- as.numeric(dp[d, "alpha_t"] + mujer * dp[d, "b_mujer_t"] +
    dp[d, paste0("a_mes_t[", mes_post, "]")] +
    col_t("a_edad_t", i_edad)[d, ] + col_t("a_edu_t", i_edu)[d, ] +
    col_t("a_rec_t", i_rec)[d, ] + col_t("a_ccaa_t", i_ccaa)[d, ] +
    col_t("a_prov_t", i_prov)[d, ])
  pi <- plogis(eta_t)
  dl <- uniroot(function(x) sum(N * plogis(eta_t + x)) / sum(N) - part_base,
                c(-10, 10), tol = 1e-10)$root
  pcal <- plogis(eta_t + dl)

  seguros[d, , ] <- as.matrix(crossprod(A, N * pmin(pi, pcal) * p_s))
  dudosos[d, , ] <- as.matrix(crossprod(A, N * pmax(pcal - pi, 0) * p_d))
  if (d %% 50 == 0) cat("draw", d, "de", nd, "-", format(Sys.time(), "%H:%M:%S"), "\n")
}
cat("minutos:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "\n")

# comprobación: seguros + dudosos = escenarios.rds en los draws emparejados
i_base <- which(esc$escenarios == part_base)
ve <- esc$votos["seg", i_base, seq(1, 1000, by = 5), , ]
dif <- max(abs((seguros + dudosos) / ve - 1), na.rm = TRUE)
cat("diferencia relativa máx con escenarios.rds:", signif(dif, 3), "\n")
stopifnot(dif < 1e-8)

saveRDS(list(seguros = seguros, dudosos = dudosos, prov_ref = prov_ref,
             draws_escenarios = seq(1, 1000, by = 5)),
        file.path(dir_completo, "seg_dud_prov.rds"))

# ---- 4. Geometría de las provincias -------------------------------------------

provincias_sf <- mapSpain::esp_get_prov(moveCAN = TRUE, epsg = 4326, resolution = "10") |>
  transmute(codigo_provincia = as.integer(cpro))
stopifnot(all(as.integer(prov_ref$codigo_provincia) %in% provincias_sf$codigo_provincia))
saveRDS(provincias_sf, file.path(dir_completo, "provincias_sf.rds"))
cat("guardado:", file.path(dir_completo, "seg_dud_prov.rds"), "y provincias_sf.rds\n")
