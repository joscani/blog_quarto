# ============================================================================
# TAUS EN ESCALA LOGIT PARA EL SWING EN ODDS
#
# El swing en odds (simular(swing = "logit"), backtest_prov_swing.R) mete el
# shock jerárquico como desplazamiento del log-odds provincial:
#
#   logit p_fp^(s) = logit p_fp + delta_f + eta_fc + eps_fp
#   delta_f = logit p_nac^(s)_f - logit p_nac(ref)_f
#
# Los tau de producción (mod_tau_shock.rds) están calibrados en log-ratio:
# dlr = Δ log(p_prov/p_nac) entre elecciones. La escala análoga para el
# swing logit es el cambio del log-odds RATIO provincia/nación:
#
#   dlo = [logit p_prov(a) - logit p_nac(a)] -
#         [logit p_prov(de) - logit p_nac(de)]
#
# Mismo modelo brms que mod_tau_shock.rds (familia + repetición fijos, ccaa
# aleatorio, sigma por familia×repetición), mismos priors, semilla nueva.
#
# Salidas:
#   2026/10/mod_tau_shock_logit.rds   — fit brms en escala logit
#   2026/10/tau_shock_draws_logit.csv — draws (interfaz de backtest_prov)
# ============================================================================

library(tidyverse)
library(brms)

source(here::here("2026/10/backtest_provincial.R"))
source(here::here("2026/10/shock_jerarquico.R"))

preps <- preparar_preps()

# higiene: el rebuild en escala ratio debe reproducir el dlr del fit de
# producción (723 observaciones)
d_tau_ratio <- construir_dlr(preps, escala = "ratio")
fams_tau <- sort(unique(d_tau_ratio$familia))
stopifnot(identical(fams_tau, c("PP", "PSOE", "SUMAR", "VOX")))
add_celdas <- \(d) {
  d |>
    mutate(
      repeticion = str_detect(etiqueta, "repetición"),
      celda_sigma = factor(
        paste(familia, repeticion, sep = "_"),
        levels = as.vector(outer(
          fams_tau,
          c("FALSE", "TRUE"),
          paste,
          sep = "_"
        ))
      )
    )
}
d_tau_ratio <- add_celdas(d_tau_ratio)
f_fit_full <- here::here("2026/10/mod_tau_shock.rds")
if (file.exists(f_fit_full)) {
  fit_full <- readRDS(f_fit_full)
  stopifnot(nrow(d_tau_ratio) == nrow(fit_full$data))
}

# el dataset en escala logit, misma estructura
d_tau_logit <- construir_dlr(preps, escala = "logit") |> add_celdas()
stopifnot(nrow(d_tau_logit) == nrow(d_tau_ratio))

ajustar_tau_brm <- function(d, seed) {
  brm(
    bf(
      dlr ~ 0 + familia + repeticion + (0 + familia || codigo_ccaa),
      sigma ~ 0 + celda_sigma
    ),
    data = d,
    prior = c(
      prior(normal(0, 1), class = b),
      prior(normal(0, 0.2), class = sd),
      prior(normal(-2.1, 1), class = b, dpar = "sigma")
    ),
    chains = 4,
    iter = 2000,
    warmup = 1000,
    cores = 4,
    backend = "cmdstanr",
    adapt_delta = 0.95,
    seed = seed,
    silent = 2,
    refresh = 0
  )
}

f_fit <- here::here("2026/10/mod_tau_shock_logit.rds")
fit_logit <- if (file.exists(f_fit)) {
  message("fit logit en caché")
  readRDS(f_fit)
} else {
  message("ajustando tau en escala logit (seed 7331)")
  f <- ajustar_tau_brm(d_tau_logit, 7331)
  saveRDS(f, f_fit)
  f
}

# draws con la misma interfaz (extraer_tau_draws vale: mismos nombres de
# parámetros); semilla distinta de la del fit ratio para independencia
tau_logit <- extraer_tau_draws(fit_logit, b = 4000, seed = 3171)
write_csv(tau_logit, here::here("2026/10/tau_shock_draws_logit.csv"))

# ---- comparativa de taus -----------------------------------------------------
# ratio: del fit de producción; logit: del nuevo. En puntos aproximados,
# tau x p(1-p) con la share nacional de referencia del 23J
tau_ratio <- extraer_tau_draws(fit_full, b = 4000, seed = 9157)

p_ref <- preps[["23j"]]$real_nac |>
  filter(familia %in% fams_tau) |>
  mutate(p = pct / 100) |>
  select(familia, p)

comp <- bind_rows(
  tau_ratio |>
    group_by(familia) |>
    summarise(across(c(tau_c, tau_p), mean)) |>
    mutate(escala = "ratio"),
  tau_logit |>
    group_by(familia) |>
    summarise(across(c(tau_c, tau_p), mean)) |>
    mutate(escala = "logit")
) |>
  left_join(p_ref, by = "familia") |>
  mutate(
    tau_c_puntos = tau_c * p * (1 - p) * 100,
    tau_p_puntos = tau_p * p * (1 - p) * 100
  ) |>
  select(familia, escala, tau_c, tau_p, tau_c_puntos, tau_p_puntos)

cat("\n== taus: log-ratio (producción) vs logit (nuevo) ==\n")
print(comp |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)
write_csv(comp, here::here("2026/10/tau_logit_comparativa.csv"))
