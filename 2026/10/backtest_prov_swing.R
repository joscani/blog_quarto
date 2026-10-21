# ============================================================================
# CAMBIO 2 — SWING PROPORCIONAL VS LOGIT: backtest provincial pareado
#
# La misma capa nacional con dos mecanismos de reparto provincial:
#   - proporcional: share_prov = share_nac x ratio_ref (el baseline)
#   - logit: la share nacional del draw desplaza la odds de la share
#     provincial de referencia, partido a partido; el shock jerárquico
#     multiplica encima sin recalibrar (tau sigue en log-ratio)
#
# Pareado: por elección, la misma semilla y los mismos draws de tau y de
# shock, de modo que la única diferencia entre variantes es el mecanismo.
#
# Salidas:
#   2026/10/backtest_prov_swing_celdas.csv  — celda a celda, con swing
#   2026/10/backtest_prov_swing_resumen.csv — cobertura y error por familia
# ============================================================================

source(here::here("2026/10/backtest_provincial.R"))
source(here::here("2026/10/shock_jerarquico.R"))
library(posterior) # as_draws_df en extraer_tau_draws

b <- 2000
mapa_bt <- mapa_ccaa_congreso()
fit <- readRDS(here::here("2026/10/mod_tau_shock.rds"))
tau <- extraer_tau_draws(fit, b, seed = 9157)
preps <- preparar_preps()

seeds_bt <- c(
  `26j` = 4127,
  `28a` = 9083,
  `10n` = 1564,
  `23j` = 6301,
  `20d` = 2458
)

run <- function(id, swing) {
  e <- elecciones[[which(map_chr(elecciones, "id") == id)]]
  set.seed(seeds_bt[[id]])
  backtest_prov(
    e,
    preps,
    b = b,
    modo = "shock",
    swing = swing,
    tau = tau,
    mapa = mapa_bt
  ) |>
    mutate(swing = swing, .before = 1)
}

res <- bind_rows(lapply(c("proporcional", "logit"), \(sw) {
  cat("swing:", sw, "\n")
  bind_rows(lapply(elecciones, \(e) {
    cat("  ", e$id, "\n")
    run(e$id, sw)
  }))
}))

write_csv(res, here::here("2026/10/backtest_prov_swing_celdas.csv"))

resumen <- res |>
  group_by(swing, familia) |>
  summarise(
    celdas = n(),
    cobertura = mean(dentro),
    cobertura_sin_ref0 = mean(dentro[!ref_ratio_cero]),
    fallos_estructura = sum(ref_ratio_cero & real > q95),
    error_med = mean(abs(real - mediana)),
    sesgo_med = mean(real - mediana),
    ancho_med_ic90 = mean(q95 - q05),
    .groups = "drop"
  )
write_csv(resumen, here::here("2026/10/backtest_prov_swing_resumen.csv"))
cat("\n== resumen ==\n")
print(resumen, n = Inf)
