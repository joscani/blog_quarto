# ============================================================================
# BACKTEST PAREADO: PROPORCIONAL Y LOGIT CON LOS DOS JUEGOS DE TAUS
#
# Tres variantes con la misma capa nacional, los mismos draws de tau/shock
# y la misma semilla por elección; solo cambian el mecanismo de reparto y
# el juego de taus:
#   1. proporcional + taus en log-ratio   (lo de producción)
#   2. logit + taus en log-ratio          (el swing con taus ajenos)
#   3. logit + taus en logit              (el swing con sus propios taus)
#
# Requiere antes: calibrar_tau_logit.R (mod_tau_shock_logit.rds).
#
# Salidas:
#   2026/10/backtest_prov_swing_tau_celdas.csv
#   2026/10/backtest_prov_swing_tau_resumen.csv
# ============================================================================

source(here::here("2026/10/backtest_provincial.R"))
source(here::here("2026/10/shock_jerarquico.R"))
library(posterior)

b <- 2000
mapa_bt <- mapa_ccaa_congreso()
tau_ratio <- extraer_tau_draws(
  readRDS(here::here("2026/10/mod_tau_shock.rds")),
  b,
  seed = 9157
)
tau_logit <- extraer_tau_draws(
  readRDS(here::here("2026/10/mod_tau_shock_logit.rds")),
  b,
  seed = 3171
)
preps <- preparar_preps()

seeds_bt <- c(
  `26j` = 4127,
  `28a` = 9083,
  `10n` = 1564,
  `23j` = 6301,
  `20d` = 2458
)

configs <- list(
  list(
    variante = "proporcional + tau ratio",
    swing = "proporcional",
    tau = tau_ratio
  ),
  list(variante = "logit + tau ratio", swing = "logit", tau = tau_ratio),
  list(variante = "logit + tau logit", swing = "logit", tau = tau_logit)
)

res <- bind_rows(lapply(configs, \(cfg) {
  cat("\nvariante:", cfg$variante, "\n")
  bind_rows(lapply(elecciones, \(e) {
    cat("  ", e$id, "\n")
    set.seed(seeds_bt[[e$id]])
    backtest_prov(
      e,
      preps,
      b = b,
      modo = "shock",
      swing = cfg$swing,
      tau = cfg$tau,
      mapa = mapa_bt
    ) |>
      mutate(variante = cfg$variante, .before = 1)
  }))
}))

write_csv(res, here::here("2026/10/backtest_prov_swing_tau_celdas.csv"))

resumen <- res |>
  group_by(variante, familia) |>
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
write_csv(resumen, here::here("2026/10/backtest_prov_swing_tau_resumen.csv"))
cat("\n== resumen ==\n")
print(resumen, n = Inf)
