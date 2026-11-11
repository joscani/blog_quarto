# Voto por grupos para el post (III), sección "Lo que solo da el MRP".
#
# Mismo cálculo que completo/12_escenarios_voto_seg.R (método seg, escenario
# base, casa media, octubre de 2026), pero en vez de sumar por provincia guarda
# los votos de cada draw agregados por provincia x sexo x edad x estudios (y por
# provincia x recuerdo). Guarda dos cosas por grupo:
#   votos:   lo que vota el grupo en el escenario base
#            (N * pi con p_seg + N * (pi_cal - pi) con p_dud);
#   reserva: lo que votarían los que no votan, N * (1 - pi_cal), si votaran
#            como los dudosos de su celda (p_dud).
#
# Draws: los 1.000 de 12 (misma selección y misma semilla de adelgazado), y de
# ellos uno de cada 5 (200). El draw d de aquí es el draw 5 * (d - 1) + 1 de
# escenarios.rds.
#
# Entradas: las de 12. Salida: completo/datos/votos_grupos.rds (~4 min).
#
#   Rscript 2026/11/mrp-multiencuesta/completo/17_votos_grupos.R

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

# ---- 3. Grupos: matrices indicadoras celda -> grupo ---------------------------

grupo_psee <- post |>
  transmute(prov = as.character(prov), sexo = as.character(sexo),
            edad = as.character(edad), edu = as.character(edu))
clave_psee <- do.call(paste, c(grupo_psee, sep = "|"))
niveles_psee <- unique(clave_psee)
G_psee <- sparseMatrix(i = seq_len(P), j = match(clave_psee, niveles_psee), x = 1,
                       dims = c(P, length(niveles_psee)))
clave_pr <- paste(post$prov, post$rec, sep = "|")
niveles_pr <- unique(clave_pr)
G_pr <- sparseMatrix(i = seq_len(P), j = match(clave_pr, niveles_pr), x = 1,
                     dims = c(P, length(niveles_pr)))

arr <- function(n_grupos, nombres) {
  array(0, c(nd, n_grupos, J), list(draw = NULL, grupo = nombres, partido = voto_lv))
}
votos_psee <- arr(length(niveles_psee), niveles_psee)
reserva_psee <- arr(length(niveles_psee), niveles_psee)
votos_pr <- arr(length(niveles_pr), niveles_pr)
reserva_pr <- arr(length(niveles_pr), niveles_pr)

# ---- 4. Votos por draw (mismo cálculo que 12, método seg, escenario base) -----

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

  v <- N * pmin(pi, pcal) * p_s + N * pmax(pcal - pi, 0) * p_d
  r <- N * (1 - pcal) * p_d
  votos_psee[d, , ] <- as.matrix(crossprod(G_psee, v))
  reserva_psee[d, , ] <- as.matrix(crossprod(G_psee, r))
  votos_pr[d, , ] <- as.matrix(crossprod(G_pr, v))
  reserva_pr[d, , ] <- as.matrix(crossprod(G_pr, r))
  if (d %% 50 == 0) cat("draw", d, "de", nd, "-", format(Sys.time(), "%H:%M:%S"), "\n")
}
cat("minutos:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "\n")

# comprobación: la provincia suma lo mismo que escenarios.rds en los draws emparejados
esc <- readRDS(file.path(dir_completo, "escenarios_seg", "escenarios.rds"))
i_base <- which(esc$escenarios == part_base)
prov_psee <- sub("\\|.*", "", niveles_psee)
vp <- sapply(esc$prov_ref$codigo_provincia, function(cp) {
  rowSums(votos_psee[, prov_psee == as.character(as.integer(cp)), , drop = FALSE])
})
ve <- apply(esc$votos["seg", i_base, seq(1, 1000, by = 5), , ], c(1, 2), sum)
cat("diferencia relativa máx con escenarios.rds:", signif(max(abs(vp / ve - 1)), 3), "\n")
stopifnot(max(abs(vp / ve - 1)) < 1e-8)

grupos_psee <- tibble(grupo = niveles_psee) |>
  tidyr::separate(grupo, c("prov", "sexo", "edad", "edu"), sep = "\\|", remove = FALSE)
grupos_pr <- tibble(grupo = niveles_pr) |>
  tidyr::separate(grupo, c("prov", "rec"), sep = "\\|", remove = FALSE)

saveRDS(list(votos_psee = votos_psee, reserva_psee = reserva_psee, grupos_psee = grupos_psee,
             votos_pr = votos_pr, reserva_pr = reserva_pr, grupos_pr = grupos_pr,
             prov_ref = esc$prov_ref, draws_escenarios = seq(1, 1000, by = 5)),
        file.path(dir_completo, "votos_grupos.rds"))
cat("guardado:", file.path(dir_completo, "votos_grupos.rds"), "\n")
