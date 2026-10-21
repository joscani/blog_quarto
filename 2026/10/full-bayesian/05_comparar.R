# ============================================================================
# FULL BAYESIAN 29-N · 05: comparación con el pipeline actual (escenario mapa)
#
# Simula el pipeline de octubre (brms con tendencia lineal + error plug-in:
# mu PSOE encogido, PP-VOX 0,5, sigma_bloque 1,5) con el escenario mapa:
#
#   actual      simular(listas = "mapa")
#
# Es la referencia "Pipeline de octubre" de la figura de resultados y la
# tabla de ablación del post full-bayesian. La comparación con el modelo
# conjunto la hace el propio post leyendo conjunto_escanos.csv y
# conjunto_mayorias.csv (12_ajustar_conjunto.R). Las versiones fb y
# fb_sin_err de la vieja arquitectura de dos módulos se retiraron con ella
# (los scripts, archivados en _archivo_modulos_ab/).
#
# Salidas (en 2026/10/full-bayesian/):
#   comparar_escanos.csv   partido: mediana, IC90 y ancho
#   comparar_mayorias.csv  P(PP+VOX >= 176), P(izquierda >= 176)
#
# Uso (desde la raíz), con mod_meta_grande_sep.rds fresco (lo reajusta el
# driver del mapa, escenarios_mapa_29N.R, o ajustar_grande_sep(refit = TRUE)):
#   Rscript 2026/10/full-bayesian/05_comparar.R
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(posterior)
})
source(here::here("2026/10/simular_escanos.R"))

dir_fb <- here::here("2026/10/full-bayesian")
seed_comb <- 2029
n_draws <- 6000

set.seed(seed_comb)
actual <- simular(n_draws = n_draws, listas = "mapa", provincial = TRUE)

grandes <- c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "AA", "ERC")

esc <- resumen_escanos(actual, width = 0.9) |>
  filter(partido %in% grandes) |>
  transmute(
    version = "actual",
    partido,
    mediana = escanos,
    q05 = .lower,
    q95 = .upper,
    ancho90 = .upper - .lower
  )
may <- bloques(actual$draws) |>
  summarise(
    version = "actual",
    derecha_mediana = median(derecha),
    derecha_q05 = quantile(derecha, 0.05),
    derecha_q95 = quantile(derecha, 0.95),
    p_pp_vox_176 = mean(derecha >= 176),
    p_izquierda_176 = mean(izquierda >= 176)
  )
write_csv(esc, file.path(dir_fb, "comparar_escanos.csv"))
write_csv(may, file.path(dir_fb, "comparar_mayorias.csv"))

cat("\nEscaños (mediana [IC90], ancho):\n")
esc |>
  mutate(txt = sprintf("%g [%g-%g] %g", mediana, q05, q95, ancho90)) |>
  select(partido, version, txt) |>
  print()
cat("\nMayorías:\n")
print(may |> mutate(across(where(is.numeric), \(x) round(x, 3))))
