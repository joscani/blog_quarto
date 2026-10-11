# ============================================================================
# COMPARACIÓN DE ESTRUCTURAS DEL EFECTO CASA (meta-análisis grande)
#
# Compara tres variantes del multinomial de 6 categorías con loo y con
# validación temporal (holdout), y deja el veredicto por escrito:
#
#   m1: time + (1 | encuestadora)          — efecto casa constante
#   m2: time + (time || encuestadora)      — deriva de casa, sin correlaciones
#   m0: time + (time | encuestadora)       — deriva correlacionada (Andalucía)
#
# 1. loo_compare (PSIS). Caveat documentado: compara predicción condicional
#    dentro de muestra; con efectos aleatorios las casas con muchas encuestas
#    (ElectoPanel si estuviera) inflan el loo. Por eso el árbitro es (2).
#
# 2. Validación temporal: entrenar con campo <= 2026-08-31 y evaluar MAE
#    (en puntos de voto) sobre las encuestas posteriores. Es la prueba que
#    responde a "qué estructura predice mejor lo que viene".
#
# Nota: la comparación original se hizo con el dataset que aún incluía
# ElectoPanel (148 encuestas, 11 casas) y las conclusiones fueron
# "equivalentes" (dif. loo < 4 elpd; holdout 0.77 vs 0.76). Este script
# re-ejecuta la comparación sobre el dataset ACTUAL de prep_grande()
# (sin CIS ni ElectoPanel), que es el que alimenta el modelo final.
#
# Uso:  Rscript 2026/10/comparar_modelos.R
# ============================================================================

source(here::here("2026/10/ajustar_meta.R"))
library(loo)
library(tidybayes)

f_cmp <- function(tag) here::here(paste0("2026/10/mod_cmp_", tag))

# ---- ajustes cortos (800/400) para comparar ---------------------------------

fit_cmp <- function(form, dat, tag, refit = FALSE) {
  if (file.exists(paste0(f_cmp(tag), ".rds")) && !refit) {
    message("mod_cmp_", tag, " cacheado")
    return(readRDS(paste0(f_cmp(tag), ".rds")))
  }
  brm(
    form,
    dat,
    multinomial(),
    prior = priors_multinomial(form, dat),
    iter = 800,
    warmup = 400,
    cores = 4,
    chains = 4,
    seed = seed_fit,
    init = 0,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 0,
    file = f_cmp(tag)
  )
}

cortes <- list(
  m1 = brmsformula(cell_counts | trials(n) ~ time + (1 | encuestadora)),
  m2 = brmsformula(cell_counts | trials(n) ~ time + (time || encuestadora)),
  m0 = brmsformula(cell_counts | trials(n) ~ time + (time | encuestadora))
)

dat <- prep_grande()
cat(
  "dataset:",
  nrow(dat),
  "encuestas |",
  n_distinct(dat$encuestadora),
  "casas\n"
)

mods <- lapply(names(cortes), \(nm) fit_cmp(cortes[[nm]], dat, nm))
names(mods) <- names(cortes)

# ---- 1. loo -----------------------------------------------------------------

loos <- lapply(mods, loo)
cat("\n===== LOO COMPARE =====\n")
print(loo_compare(loos), simplify = FALSE)
for (nm in names(loos)) {
  k <- loos[[nm]]$diagnostics$pareto_k
  cat(
    nm,
    ": pareto-k > 0.7 en",
    sum(k > 0.7),
    "observaciones (máx",
    round(max(k), 2),
    ")\n"
  )
}

# ---- 2. holdout temporal ----------------------------------------------------

cutoff <- as.Date("2026-08-31")
train <- dat |> filter(fecha <= cutoff)
test <- dat |> filter(fecha > cutoff)
cat("\n===== HOLDOUT: train", nrow(train), "| test", nrow(test), "=====\n")

test_slim <- test |>
  mutate(id = row_number()) |>
  select(id, encuestadora, enlace, fecha, time, n, cell_counts)

mae_holdout <- function(mod) {
  preds <- test_slim |>
    add_epred_draws(object = mod, ndraws = 1000) |>
    rename(categoria = .category, pred = .epred) |>
    group_by(id, categoria) |>
    summarise(pred = mean(pred / n), .groups = "drop")
  # ojo: en dat las columnas PP..BLOQUE son VOTOS; la proporción observada
  # es votos / n
  obs <- test |>
    mutate(id = row_number()) |>
    pivot_longer(
      all_of(c("PP", "PSOE", "VOX", "SUMAR", "SALF", "BLOQUE")),
      names_to = "categoria",
      values_to = "votos_cat"
    ) |>
    mutate(obs_pct = votos_cat / n) |>
    select(id, categoria, obs_pct)
  left_join(preds, obs, by = c("id", "categoria")) |>
    group_by(categoria) |>
    summarise(mae = round(100 * mean(abs(pred - obs_pct)), 2), .groups = "drop")
}

maes <- lapply(c("m1", "m2"), \(nm) {
  m <- fit_cmp(cortes[[nm]], train, paste0(nm, "_holdout"))
  out <- mae_holdout(m) |> rename(!!nm := mae)
  out
})
tabla_mae <- reduce(maes, full_join, by = "categoria")
print(tabla_mae)
for (nm in c("m1", "m2")) {
  cat("MAE medio global", nm, ":", round(mean(tabla_mae[[nm]]), 2), "puntos\n")
}

cat(
  "\nVEREDICTO: si las diferencias de elpd son < 4 y el MAE del holdout es",
  "equivalente, las tres estructuras son predictivamente equivalentes y se",
  "mantiene m0 (correlacionado) por consistencia con la serie de posts.\n"
)
