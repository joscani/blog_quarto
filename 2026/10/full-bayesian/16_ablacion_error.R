# ============================================================================
# FULL BAYESIAN 29-N · 16: la ablación del error, desde el ajuste conjunto
#
# El modelo conjunto guarda, en cada draw, el voto latente del día E
# (v_obj) y un sorteo del error del consenso (err_new). Este script lleva
# a escaños solo la primera capa (sin error). La horquilla con las dos
# capas es la de conjunto_escanos.csv (12_ajustar_conjunto.R) y la del
# pipeline de octubre, la versión "actual" de comparar_escanos.csv
# (05_comparar.R); los tres con el escenario mapa y la semilla 2029.
#
# Salidas (en 2026/10/full-bayesian/):
#   conjunto_sin_error.csv   escaños por partido (mediana, IC90)
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/16_ablacion_error.R
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

Kc <- dc$K[dc$C]
Tc <- dc$T[dc$C]
v <- unclass(fit$draws(
  sprintf("v_obj[%d,%d]", seq_len(Kc), Tc),
  format = "draws_matrix"
))[, seq_len(Kc)]
colnames(v) <- sub("EH_Bildu", "EH Bildu", fams_campana(dc$objetivo))
stopifnot(setequal(colnames(v), fams14), abs(max(rowSums(v) - 1)) < 1e-6)

# solo el voto latente: el error del consenso no entra
shares <- 100 * v[, fams14]
set.seed(2029)
sim <- simular(
  n_draws = nrow(shares),
  listas = "mapa",
  provincial = TRUE,
  shares_ext = shares
)

resumen_escanos(sim, width = 0.9) |>
  select(partido, mediana = escanos, q05 = .lower, q95 = .upper) |>
  write_csv(file.path(dir_fb, "conjunto_sin_error.csv"))

cat("\nEscaños con solo el voto latente (mediana [IC90]):\n")
print(
  read_csv(
    file.path(dir_fb, "conjunto_sin_error.csv"),
    show_col_types = FALSE
  ) |>
    mutate(txt = sprintf("%g [%g-%g]", mediana, round(q05), round(q95))) |>
    select(partido, txt),
  n = Inf
)
