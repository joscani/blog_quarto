# ============================================================================
# FULL BAYESIAN 29-N · 17: sensibilidad de la Omega del error
#
# La Omega del error es común a generales y autonómicas, así que el patrón
# de correlaciones lo traen sobre todo las 39 autonómicas. En las cinco
# generales, el error medido de PP y Vox correlacionaba al revés (+0,92,
# encogido a +0,5 en el pipeline de octubre). Con cinco generales no hay
# Omega propia que estimar (el prior se comería el posterior): aquí se
# comprueba a posteriori cuánto pesaría. Mismos draws de voto latente, mu
# y sigma; solo cambia la correlación con la que se sortea el error.
#
#   base       Omega del posterior tal cual
#   vox_pp_05  Omega del posterior con la correlación PP-Vox forzada a +0,5
#   r_error    la Omega entera del pipeline de octubre (R_error)
#
# Salidas (en 2026/10/full-bayesian/):
#   sensibilidad_omega_mayorias.csv   variante, p_pp_vox_176
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/17_sensibilidad_omega.R
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(posterior)
})
source(here::here("2026/10/full-bayesian/11_datos_conjunto.R"))
source(here::here("2026/10/simular_escanos.R"))
source(here::here("2026/10/full-bayesian/funciones_simular.R"))

fit <- readRDS(file.path(dir_fb, "fit_conjunto.rds"))
dc <- datos_conjunto()

# voto latente del día E (igual que en 16_ablacion_error.R)
Kc <- dc$K[dc$C]
Tc <- dc$T[dc$C]
v <- unclass(fit$draws(
  sprintf("v_obj[%d,%d]", seq_len(Kc), Tc),
  format = "draws_matrix"
))[, seq_len(Kc)]
colnames(v) <- sub("EH_Bildu", "EH Bildu", fams_campana(dc$objetivo))
stopifnot(setequal(colnames(v), fams14), abs(max(rowSums(v) - 1)) < 1e-6)

# mu y sigma del error por draw, y Omega mediana del posterior. La salida
# de Stan es column-major: Omega[i,j] está en la columna (j-1)*G+i
G <- length(grupos_error)
mg <- unclass(fit$draws("mu_gen", format = "draws_matrix"))
sg <- unclass(fit$draws("sigma_gen", format = "draws_matrix"))
om <- unclass(fit$draws("Omega", format = "draws_matrix"))
Omega_med <- matrix(0, G, G)
for (i in 1:G) {
  for (j in 1:G) {
    Omega_med[i, j] <- median(om[, (j - 1) * G + i])
  }
}
dimnames(Omega_med) <- list(grupos_error, grupos_error)

# variantes de correlación
Omega_A <- Omega_med
Omega_A["PP", "VOX"] <- Omega_A["VOX", "PP"] <- 0.5
L_A <- unclass(as.matrix(t(chol(Matrix::nearPD(Omega_A)$mat))))
R4 <- R_error[familias_error[1:4], familias_error[1:4]]
L_B <- unclass(as.matrix(t(chol(Matrix::nearPD(R4)$mat))))

# cholesky de la Omega de cada draw, y unos mismos z para las tres variantes
L_arr <- array(0, c(nrow(om), G, G))
for (s in seq_len(nrow(om))) {
  L_arr[s, , ] <- t(chol(matrix(om[s, ], G, G)))
}
set.seed(777)
z <- matrix(rnorm(nrow(om) * G), nrow = nrow(om))

gen_err <- function(L, per_draw = FALSE) {
  out <- vapply(
    seq_len(nrow(om)),
    \(s) {
      Ls <- if (per_draw) L_arr[s, , ] else L
      mg[s, ] + sg[s, ] * (Ls %*% z[s, ])
    },
    numeric(G)
  )
  out <- t(out)
  colnames(out) <- grupos_error
  out
}
err_base <- gen_err(NULL, per_draw = TRUE)
err_A <- gen_err(L_A)
err_B <- gen_err(L_B)

run <- function(err) {
  set.seed(2029)
  sim <- simular(
    n_draws = nrow(v),
    listas = "mapa",
    provincial = TRUE,
    shares_ext = combinar_conjunto(v, err)
  )
  mean(bloques(sim$draws)$derecha >= 176)
}

tibble(
  variante = c("base", "vox_pp_05", "r_error"),
  p_pp_vox_176 = c(run(err_base), run(err_A), run(err_B))
) |>
  write_csv(file.path(dir_fb, "sensibilidad_omega_mayorias.csv"))

cat("\nP(PP + VOX >= 176):\n")
print(read_csv(
  file.path(dir_fb, "sensibilidad_omega_mayorias.csv"),
  show_col_types = FALSE
))
