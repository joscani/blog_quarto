suppressMessages({library(tidyverse); library(brms)})
suppressMessages({source(here::here("2026/10/backtest_provincial.R")); source(here::here("2026/10/shock_jerarquico.R"))})
preps <- suppressMessages(preparar_preps())
dlr <- construir_dlr(preps, voto_nacional = "presentes")
fit <- readRDS(here::here("2026/10/mod_tau_shock.rds"))
cat("dlr igual al del fit:", isTRUE(all.equal(dlr$dlr, fit$data$dlr, tolerance = 1e-12)), "\n")
# LOTO
for (tr in unique(dlr$transition)) {
  fl <- readRDS(here::here("2026/10", sprintf("mod_tau_loto_%s.rds", tr)))
  cat("LOTO", tr, isTRUE(all.equal(dlr$dlr[dlr$transition != tr], fl$data$dlr, tolerance = 1e-12)), "\n")
}
cmp <- function(a, f) { b <- read_csv(here::here("2026/10", f), show_col_types = FALSE); isTRUE(all.equal(as.data.frame(a), as.data.frame(b), tolerance = 1e-8, check.attributes = FALSE)) }
dr <- extraer_tau_draws(fit, b = 2000)
cat("draws b=2000 seed 9157:", cmp(dr, "tau_shock_draws.csv"), "\n")
resum <- function(d) d |> group_by(familia) |> summarise(tau_c_med = median(tau_c), tau_c_ic05 = quantile(tau_c, .05), tau_c_ic95 = quantile(tau_c, .95), tau_p_med = median(tau_p), tau_p_ic05 = quantile(tau_p, .05), tau_p_ic95 = quantile(tau_p, .95))
cat("brms resumen de draws b=2000:", cmp(resum(dr), "tau_shock_brms.csv"), "\n")
a <- as_draws_df(fit); full <- map_dfr(c("PP","PSOE","SUMAR","VOX"), \(f) tibble(familia = f, tau_c = a[[paste0("sd_codigo_ccaa__familia", f)]], tau_p = exp(a[[paste0("b_sigma_celda_sigma", f, "_FALSE")]])))
cat("brms resumen de los 4000 draws:", cmp(resum(full), "tau_shock_brms.csv"), "\n")
print(resum(full)); 
cat("lme4 presentes:", cmp(ajustar_tau(dlr), "tau_shock.csv"), "  lme4 oficial:", cmp(ajustar_tau(construir_dlr(preps)), "tau_shock.csv"), "\n")
reg <- ajustar_tau_regionales(construir_dlr_regionales(preps))
cat("regionales:", cmp(reg, "tau_regionales.csv"), "\n"); print(reg); print(read_csv(here::here("2026/10/tau_regionales.csv"), show_col_types = FALSE))
