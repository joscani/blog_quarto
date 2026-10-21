# ============================================================================
# FULL BAYESIAN 29-N · 04: combinación draw a draw y escaños (escenario mapa)
#
# Para cada draw s:
#   1. Voto de la media de la industria el día E: v_T del módulo A (7
#      categorías, proporciones).
#   2. Error del consenso: err_new_casa del módulo B (logit, 4 grupos): mu +
#      delta_new + sesgo medio de las casas de 2026. Se aplica
#      al logit de PP, PSOE, VOX y SUMAR+PODEMOS (la historia solo conoce a
#      Sumar/Unidas Podemos agregado). El resto (SALF + BLOQUE) no lleva
#      error propio; al renormalizar a 1 absorbe el ajuste en proporción a
#      su tamaño.
#   3. SUMAR+PODEMOS se reparte con el peso de cada lista en el draw (antes
#      del error); SALF + BLOQUE, igual.
#   4. Composición del BLOQUE: draws del modelo 2 de brms (ajustar_bloque()).
#   5. simular(listas = "mapa", shares_ext = ...): shock provincial,
#      reglas del mapa (Podemos fuera de Cataluña, lista conjunta en
#      Canarias y C. Valenciana, AA en Andalucía), umbral del 3 % y D'Hondt
#      con el reparto del RD 806/2026.
#
# Los dos módulos son independientes: el draw s de A se empareja con el draw
# s de B (los dos con 6.000 draws).
#
# Salidas (en 2026/10/full-bayesian/):
#   fb_voto.csv        voto nacional por partido (mediana, IC90), en %
#   fb_escanos.csv     escaños por partido (mediana, media, etiqueta, IC90)
#   fb_mayorias.csv    P(mayorías)
#   fb_bloques.csv     escaños por bloque
#   fb_sim.rds         la simulación completa (no va a git)
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/04_simular.R
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(cmdstanr)
  library(posterior)
  library(tidybayes)
})
source(here::here("2026/10/full-bayesian/01_datos.R"))
source(here::here("2026/10/simular_escanos.R"))
source(here::here("2026/10/full-bayesian/funciones_simular.R"))

seed_comb <- 2029

fit_a <- readRDS(file.path(dir_fb, "fit_estado.rds"))
fit_b <- readRDS(file.path(dir_fb, "fit_error.rds"))
sim <- simular_fb(fit_a, fit_b, seed = seed_comb)
v_err <- sim$v_err
saveRDS(sim, file.path(dir_fb, "fb_sim.rds"))

# ---- comprobaciones del mapa -----------------------------------------------------

ep <- sim$prov_escanos
pv <- sim$prov_voto
provs <- dimnames(ep)[[2]]
ccaa <- sim$prov_ref$ccaa[match(provs, sim$prov_ref$provincia)]
sin_pod <- grepl("Cataluña|Canarias|Valenciana", ccaa)
stopifnot(
  all(apply(ep, 1, sum) == 350),
  all(apply(ep[, "Madrid", ], 1, sum) == 38),
  all(apply(ep[, "Cádiz", ], 1, sum) == 8),
  all(pv[, sin_pod, "PODEMOS"] == 0),
  all(pv[, !grepl("Andalucía", ccaa), "AA"] == 0)
)
cat("Comprobaciones del mapa: OK\n")

# ---- resúmenes -------------------------------------------------------------------

presentes <- names(which(colSums(sim$draws[, -1]) > 0))

voto <- as_tibble(100 * v_err) |>
  pivot_longer(everything(), names_to = "partido", values_to = "voto") |>
  group_by(partido) |>
  median_qi(voto, .width = 0.9) |>
  select(partido, mediana = voto, q05 = .lower, q95 = .upper) |>
  arrange(desc(mediana))

etiq <- escanos_resto_mayor(sim)
escanos <- resumen_escanos(sim, width = 0.9) |>
  filter(partido %in% presentes) |>
  select(partido, mediana = escanos, q05 = .lower, q95 = .upper) |>
  left_join(etiq |> rename(etiqueta = escanos), by = "partido") |>
  relocate(media, etiqueta, .after = mediana)
mayorias <- prob_mayorias(sim)
bl <- bloques(sim$draws) |>
  select(derecha, izquierda) |>
  pivot_longer(everything(), names_to = "bloque", values_to = "escanos") |>
  group_by(bloque) |>
  summarise(
    mediana = median(escanos),
    q05 = quantile(escanos, 0.05),
    q95 = quantile(escanos, 0.95),
    p_176 = mean(escanos >= 176)
  )
write_csv(bl, file.path(dir_fb, "fb_bloques.csv"))

write_csv(voto, file.path(dir_fb, "fb_voto.csv"))
write_csv(escanos, file.path(dir_fb, "fb_escanos.csv"))
write_csv(mayorias, file.path(dir_fb, "fb_mayorias.csv"))

cat("\nVoto nacional con error (%):\n")
print(voto |> mutate(across(where(is.numeric), \(x) round(x, 1))))
cat("\nEscaños:\n")
print(escanos |> mutate(across(where(is.numeric), \(x) round(x, 1))), n = Inf)
cat("\nSuma de etiquetas:", sum(escanos$etiqueta), "\n")
print(mayorias)
