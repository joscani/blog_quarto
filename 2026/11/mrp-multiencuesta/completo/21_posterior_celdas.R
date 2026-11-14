# La posterior por celda, en formato ordenado, para enseñar a usarla.
#
# Guarda dos tablas que se unen por `celda`:
#   tabla_post.rds:        la tabla de postestratificación. Una fila por celda
#                          (provincia x sexo x edad x estudios x recuerdo del
#                          23-J) con N, las personas de 18 y más de la celda.
#   posterior_celdas_80.rds: la posterior. Una fila por draw y celda con
#                          part:  probabilidad de votar (calibrada al 80 %);
#                          PP, PSOE, ..., BLANCO: reparto del voto entre los
#                          que votan en esa celda (suman 1).
#
# Con eso, los votos de una celda a un partido son N * part * p_partido, y
# cualquier pregunta es sumar celdas.
#
# Mismo cálculo que 12_escenarios_voto_seg.R y 17_votos_grupos.R (método seg,
# escenario de participación del 80 %, casa media, octubre de 2026). Los mismos
# 1.000 draws que 12_escenarios_voto_seg.R (el draw d de aquí es el d de
# escenarios.rds).
# Aquí el reparto del voto de la celda ya mezcla seguros y dudosos:
#   p = (min(pi, pi_cal) * p_seg + max(pi_cal - pi, 0) * p_dud) / pi_cal
#
# Entradas: las de 12. Salidas en completo/datos/ (~1 min).
#
#   Rscript 2026/11/mrp-multiencuesta/completo/21_posterior_celdas.R

suppressPackageStartupMessages({
  library(dplyr)
  library(posterior)
})

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_completo <- file.path(dir_proy, "completo", "datos")
source(file.path(dir_proy, "00_partidos.R")) # territorio

# ---- 1. Celdas y niveles (como 12 y 17) ---------------------------------------

entradas <- readRDS(file.path(dir_completo, "entradas_escenarios.rds"))
post <- entradas$post
lv <- entradas$lv
mes_post <- entradas$mes_post
participacion <- 0.80 # uno de los escenarios de 12: 0.65, 0.704 (base), 0.75, 0.80
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

# ---- 2. Draws: 1.000 -----------------------------------------------------------

sel_draws <- function(n_tot) round(seq(1, n_tot, length.out = 1000L))
nd <- 1000L

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

# ---- 3. Participación y reparto del voto por celda y draw ----------------------
# Bucle sobre draws como en 17: la calibración de la participación (uniroot)
# es una ecuación distinta en cada draw.

part <- matrix(0, nd, P)
reparto <- array(0, c(nd, P, J), list(NULL, NULL, voto_lv))

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
  dl <- uniroot(function(x) sum(N * plogis(eta_t + x)) / sum(N) - participacion,
                c(-10, 10), tol = 1e-10)$root
  pcal <- plogis(eta_t + dl)

  part[d, ] <- pcal
  reparto[d, , ] <- (pmin(pi, pcal) * p_s + pmax(pcal - pi, 0) * p_d) / pcal
}

# ---- 4. Tablas ordenadas -------------------------------------------------------

prov_ref <- readRDS(file.path(dir_completo, "votos_grupos.rds"))$prov_ref

tabla_post <- post |>
  as_tibble() |>
  mutate(celda = row_number(), cod = as.integer(as.character(prov))) |>
  left_join(prov_ref |> transmute(cod = as.integer(codigo_provincia), provincia,
                                  escanos = diputados), by = "cod") |>
  transmute(
    celda, cod, provincia, ccaa = as.character(ccaa), escanos,
    sexo = factor(sexo, c("H", "M"), c("Hombres", "Mujeres")),
    edad = factor(edad, c("18-20", "21-24", "25-34", "35-44", "45-54", "55-64", "65+")),
    edu = factor(edu, c("baja", "media", "alta")),
    rec = factor(rec, lv$rec),
    N
  )
stopifnot(!anyNA(tabla_post))

posterior_celdas <- tibble(
  draw = rep(seq_len(nd), times = P), # draw varía más rápido, como en part y reparto
  celda = rep(seq_len(P), each = nd),
  part = as.vector(part)
) |>
  bind_cols(as_tibble(matrix(reparto, nd * P, J, dimnames = list(NULL, voto_lv))))

# comprobación 1: el reparto suma 1 en cada celda
stopifnot(max(abs(rowSums(posterior_celdas[voto_lv]) - 1)) < 1e-10)

# comprobación 2: los votos por provincia y partido son los de escenarios.rds
# (método seg, mismo escenario) en los 1.000 draws
esc <- readRDS(file.path(dir_completo, "escenarios_seg", "escenarios.rds"))
i_esc <- which(abs(esc$escenarios - participacion) < 1e-9)
stopifnot(length(i_esc) == 1)
mio <- posterior_celdas |>
  left_join(tabla_post |> select(celda, cod, N), by = "celda") |>
  summarise(across(all_of(voto_lv), \(p) sum(N * part * p)), .by = c(draw, cod)) |>
  arrange(cod, draw)
ref <- esc$votos["seg", i_esc, , , ] # draw x provincia x partido
stopifnot(identical(as.integer(esc$prov_ref$codigo_provincia), sort(unique(mio$cod))))
ref <- matrix(ref, ncol = J) # filas: draw rápido, luego provincia
dif <- max(abs(as.matrix(mio[voto_lv]) - ref) / pmax(ref, 1)) # ref = 0 fuera de territorio
cat("diferencia relativa máx con escenarios.rds:", signif(dif, 3), "\n")
stopifnot(dif < 1e-8)

saveRDS(tabla_post, file.path(dir_completo, "tabla_post.rds"))
fichero <- sprintf("posterior_celdas_%02d.rds", round(100 * participacion))
saveRDS(posterior_celdas, file.path(dir_completo, fichero))
cat("guardado: tabla_post.rds (", nrow(tabla_post), "celdas ) y", fichero, "(",
    nrow(posterior_celdas), "filas )\n")
