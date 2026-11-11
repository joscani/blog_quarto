# Tablas del anexo del post (III): "Resultados del modelo para explorar y descargar".
#
# Dos escenarios de participación nacional: la base (part_obj, 70,4 %) y el 80 %.
# Método seg, casa media, mes 2026-10: los mismos 1.000 draws de
# completo/12_escenarios_voto_seg.R. No se ajusta ningún modelo.
#
#   1. Nacional: % de voto (votos de las provincias sumados en cada draw) y escaños
#      (escaños provinciales sumados en cada draw).
#   2. Voto por provincia y 3. escaños por provincia: de escenarios.rds; los
#      escaños, D'Hondt draw a draw (resultados_provinciales() de iii-funciones.R).
#   4. Voto por provincia y edad: escenarios.rds no lo guarda, así que se repite el
#      cálculo por celda de 12 (mismos draws, misma fórmula del método seg) y se
#      suma por provincia x edad. Comprobación: sumando las edades sale, draw a
#      draw, el voto provincial de escenarios.rds.
# Los resúmenes (mediana, p5, p95, media) se calculan siempre al final, sobre los
# totales de cada draw. La fila "Diferencia" es la resta draw a draw.
#
# Entradas: las de 12 y completo/datos/escenarios_seg/escenarios.rds.
# Salida: completo/datos/anexo_tablas.rds (lo lee el post).
#
#   Rscript 2026/11/mrp-multiencuesta/completo/19_anexo_tablas.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(posterior)
  library(Matrix)
})

dir_proy <- here::here("2026/11/mrp-multiencuesta")
source(file.path(dir_proy, "iii-funciones.R")) # resultados_provinciales(), num(), dirs
source(file.path(dir_proy, "00_partidos.R")) # territorio

# ---- 1. Celdas, niveles y draws (como 12, método seg) ---------------------------

nd <- 1000L
entradas <- readRDS(file.path(dir_completo, "entradas_escenarios.rds"))
post <- entradas$post
lv <- entradas$lv
mes_post <- entradas$mes_post
part_base <- entradas$part_obj
escenarios <- c(part_base, 0.80)
x_voto <- readRDS(file.path(dir_completo, "datos_voto_seg.rds"))
for (k in names(lv)) stopifnot(identical(as.character(lv[[k]]), as.character(x_voto$lv[[k]])))
voto_lv <- as.character(x_voto$lv$voto)
mes_post_voto <- match(as.character(lv$mes[mes_post]), as.character(x_voto$lv$mes))
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

vars_voto <- c("alpha", "b_mujer", "b_dud", "a_casa", "a_mes", "a_edad", "a_edu", "a_rec",
               "a_ccaa", "a_prov", "a_casa_rec", "a_dud_edad", "a_dud_rec")
dv <- as_draws_rvars(readRDS(file.path(dir_completo, "fit_voto_seg.rds"))$fit$draws(vars_voto))
g <- function(k) unname(draws_of(dv[[k]]))
sel <- round(seq(1, dim(g("alpha"))[1], length.out = nd)) # la selección de 12
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
dp <- dp[round(seq(1, nrow(dp), length.out = nd)), ]
dp <- matrix(as.numeric(dp), nrow(dp), dimnames = list(NULL, colnames(dp)))
col_t <- function(v, i) unname(dp[, paste0(v, "[", i, "]"), drop = FALSE])

# ---- 2. Provincia x edad: matriz indicadora celda -> grupo -----------------------

prov_ref <- readRDS(file.path(dir_esc, "escenarios.rds"))$prov_ref
i_pr <- match(as.integer(as.character(post$prov)), as.integer(prov_ref$codigo_provincia))
stopifnot(!anyNA(i_pr))
grupos <- expand_grid(i_prov = seq_len(nrow(prov_ref)), i_edad = seq_along(lv$edad))
G <- sparseMatrix(i = seq_len(P), j = (i_pr - 1) * length(lv$edad) + i_edad, x = 1,
                  dims = c(P, nrow(grupos)))

votos_pe <- array(0, c(length(escenarios), nd, nrow(grupos), J),
                  list(NULL, NULL, NULL, voto_lv))

# ---- 3. Votos por draw (mismo cálculo que 12, método seg) ------------------------

t0 <- Sys.time()
for (d in seq_len(nd)) {
  casa_media <- Reduce(`+`, lapply(seq_len(n_casa), function(h) {
    matrix(V$a_casa[d, h, ], n_rec, J, byrow = TRUE) +
      V$a_casa_rec[d, (h - 1) * n_rec + seq_len(n_rec), ]
  })) / n_casa
  fila <- function(a, i) a[d, i, , drop = TRUE]
  eta <- matrix(V$alpha[d, ], P, J, byrow = TRUE) +
    outer(mujer, V$b_mujer[d, ]) +
    matrix(V$a_mes[d, mes_post_voto, ], P, J, byrow = TRUE) +
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
  for (e in seq_along(escenarios)) {
    dl <- uniroot(function(x) sum(N * plogis(eta_t + x)) / sum(N) - escenarios[e],
                  c(-10, 10), tol = 1e-10)$root
    pcal <- plogis(eta_t + dl)
    v <- N * pmin(pi, pcal) * p_s + N * pmax(pcal - pi, 0) * p_d
    votos_pe[e, d, , ] <- as.matrix(crossprod(G, v))
  }
  if (d %% 100 == 0) cat("draw", d, "de", nd, "-", format(Sys.time(), "%H:%M:%S"), "\n")
}
cat("minutos:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "\n")

# ---- 4. Provincia (voto y escaños) y comprobaciones -----------------------------

pr <- resultados_provinciales(escenarios) # comprueba escaños = totales de 12
# provincia x edad sumado por edades = votos provinciales de escenarios.rds
idx <- array(votos_pe, c(length(escenarios), nd, length(lv$edad), nrow(prov_ref), J))
vp <- apply(idx, c(1, 2, 4, 5), sum)
dif_rel <- max(abs(vp / pr$v - 1), na.rm = TRUE) # NaN donde el partido no se presenta
cat("provincia x edad vs escenarios.rds, diferencia relativa máx:", signif(dif_rel, 3), "\n")
stopifnot(dif_rel < 1e-8, all((vp == 0) == (pr$v == 0)))

etq <- pr$etq
etq_dif <- paste0("Diferencia (", etq[2], " − ", etq[1], ")")
niv_esc <- c(etq, etq_dif)
cuantiles <- function(x) c(mediana = median(x), p5 = unname(quantile(x, 0.05)),
                           p95 = unname(quantile(x, 0.95)))

# a: escenario x draw x ... ; devuelve una fila por celda de las dimensiones extra
# y por escenario (más la diferencia draw a draw)
resume_arr <- function(a, nombres, media = FALSE) {
  a3 <- array(a, c(dim(a)[1:2], prod(dim(a)[-(1:2)])))
  piezas <- list(a3[1, , , drop = TRUE], a3[2, , , drop = TRUE], a3[2, , ] - a3[1, , ])
  bind_rows(lapply(seq_along(piezas), function(k) {
    m <- piezas[[k]]
    q <- apply(m, 2, cuantiles)
    out <- mutate(nombres, escenario = niv_esc[k],
                  mediana = q["mediana", ], p5 = q["p5", ], p95 = q["p95", ])
    if (media) out$media <- colMeans(m)
    out
  }))
}

# 1. nacional
v_nac <- apply(pr$v, c(1, 2, 4), sum) # escenario x draw x partido
pct_nac <- 100 * v_nac / c(apply(v_nac, 1:2, sum))
esc_nac <- apply(pr$esc, c(1, 2, 4), sum) # escaños provinciales sumados por draw
partidos_dh <- dimnames(pr$esc)[[4]]
t_nac <- resume_arr(pct_nac, tibble(partido = voto_lv)) |>
  left_join(
    resume_arr(esc_nac, tibble(partido = partidos_dh), media = TRUE) |>
      rename(esc_media = media, esc_mediana = mediana, esc_p5 = p5, esc_p95 = p95),
    by = c("partido", "escenario")
  ) |>
  mutate(partido = factor(partido, voto_lv), escenario = factor(escenario, niv_esc)) |>
  arrange(partido, escenario) |>
  select(partido, escenario, mediana, p5, p95, esc_media, esc_mediana, esc_p5, esc_p95)

# comprobación con los CSV de 12 (los que usa el post)
csv_v <- read.csv(file.path(dir_esc, "escenarios_voto.csv")) |>
  filter(metodo == "seg", participacion %in% etq)
csv_e <- read.csv(file.path(dir_esc, "escenarios_escanos.csv")) |>
  filter(metodo == "seg", participacion %in% etq)
chk_v <- inner_join(csv_v, t_nac |> mutate(across(c(partido, escenario), as.character)),
                    by = c("partido", participacion = "escenario"))
chk_e <- inner_join(csv_e, t_nac |> mutate(across(c(partido, escenario), as.character)),
                    by = c("partido", participacion = "escenario"))
stopifnot(nrow(chk_v) == 2 * J, nrow(chk_e) == 2 * length(partidos_dh))
cat("nacional vs CSV de 12, dif. máx: voto", signif(max(abs(chk_v$mediana.x - chk_v$mediana.y)), 3),
    "| escaños media", signif(max(abs(chk_e$media - chk_e$esc_media)), 3),
    "| escaños p5/p95", max(abs(chk_e$q05 - chk_e$esc_p5)), max(abs(chk_e$q95 - chk_e$esc_p95)), "\n")
stopifnot(max(abs(chk_v$mediana.x - chk_v$mediana.y)) < 1e-9,
          max(abs(chk_e$media - chk_e$esc_media)) < 1e-9)

# 4. provincia x edad
pct_pe <- 100 * votos_pe / c(apply(votos_pe, 1:3, sum))
nombres_pe <- expand_grid(partido = voto_lv,
                          grupo = seq_len(nrow(grupos))) |>
  transmute(provincia = prov_ref$provincia[grupos$i_prov[grupo]],
            edad = as.character(lv$edad)[grupos$i_edad[grupo]], partido)
presente_pe <- c(apply(votos_pe > 0, 3:4, any))
t_pe <- resume_arr(pct_pe, nombres_pe) |>
  filter(rep(presente_pe, 3)) |>
  mutate(provincia = factor(provincia, prov_ref$provincia),
         edad = factor(edad, as.character(lv$edad)),
         partido = factor(partido, voto_lv),
         escenario = factor(escenario, niv_esc)) |>
  arrange(provincia, edad, partido, escenario) |>
  select(provincia, edad, partido, escenario, mediana, p5, p95)

anexo <- list(nacional = t_nac, voto_prov = pr$voto, escanos_prov = pr$escanos,
              voto_prov_edad = t_pe, escenarios = escenarios, etiquetas = niv_esc,
              nd = nd)
saveRDS(anexo, file.path(dir_completo, "anexo_tablas.rds"))
cat("filas:", sapply(anexo[1:4], nrow), "\n")
cat("guardado:", file.path(dir_completo, "anexo_tablas.rds"), "\n")
