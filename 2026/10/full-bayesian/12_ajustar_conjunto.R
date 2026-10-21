# ============================================================================
# FULL BAYESIAN 29-N · 12: ajuste del modelo conjunto (modelo_conjunto.stan)
#
# 1. Prior predictive: voto de 2026 el día E con prior_only = 1.
# 2. Ajuste y diagnósticos.
# 3. Parámetros clave: sigma_rw y kappa (volatilidad, y cuánto más en
#    campaña), sigma_gen, mu_gen, gamma y Omega (error del consenso).
# 4. Escaños con el escenario mapa (simular_conjunto()).
#
# Salidas (en 2026/10/full-bayesian/):
#   fit_conjunto.rds             objeto CmdStanMCMC (no va a git)
#   conjunto_diagnostico.csv     divergencias, Rhat, ESS
#   conjunto_parametros.csv      parámetros clave (mediana, IC90)
#   conjunto_tendencia.csv       voto semanal de 2026 (media de las casas)
#   conjunto_escanos.csv         escaños por partido (mediana, media, IC90)
#   conjunto_mayorias.csv        P(mayorías)
#   conjunto_prior_predictive.csv  voto del día E solo con los priors (%)
#   conjunto_sim.rds             la simulación completa (no va a git)
#
# Uso (desde la raíz):
#   Rscript 2026/10/full-bayesian/12_ajustar_conjunto.R [principal | v2]
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(cmdstanr)
  library(posterior)
  library(tidybayes)
})
source(here::here("2026/10/full-bayesian/11_datos_conjunto.R"))
source(here::here("2026/10/simular_escanos.R"))
source(here::here("2026/10/full-bayesian/funciones_simular.R"))

seed_cj <- 1811
# variante (primer argumento): "principal" (volatilidad propia de cada
# campaña, sin sufijo) o "v2" (volatilidad común, sufijo _v2)
args <- commandArgs(trailingOnly = TRUE)
variante <- if (length(args) > 0) args[1] else "principal"
pr <- priors_conjunto
pr$sigma_rw_campana <- as.integer(variante != "v2")
sufijo <- if (variante == "principal") "" else paste0("_", variante)
f_sal <- \(nombre, ext = "csv") file.path(dir_fb, paste0(nombre, sufijo, ".", ext))
mod <- cmdstan_model(file.path(dir_fb, "modelo_conjunto.stan"))
dc <- datos_conjunto()
Kc <- dc$K[dc$C]
Tc <- dc$T[dc$C]
fams_obj <- fams_campana(dc$objetivo)

muestrear <- function(prior_only = 0L, iter = 1500) {
  mod$sample(
    data = stan_conjunto(dc, prior_only, priors = pr),
    seed = seed_cj,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = iter,
    adapt_delta = 0.95,
    refresh = 250
  )
}

# ---- 1. prior predictive -------------------------------------------------------

fp <- muestrear(prior_only = 1L, iter = 500)
vp <- unclass(fp$draws(sprintf("v_obj[%d,%d]", seq_len(Kc), Tc), format = "draws_matrix"))
cat("\nPrior predictive, voto el día E (%, q05-q50-q95):\n")
pp_tab <- round(100 * t(apply(vp, 2, quantile, c(0.05, 0.5, 0.95))), 1) |> `rownames<-`(fams_obj)
print(pp_tab)
write_csv(
  as_tibble(pp_tab, rownames = "partido") |> rename(q05 = `5%`, mediana = `50%`, q95 = `95%`),
  f_sal("conjunto_prior_predictive")
)

# ---- 2. ajuste ----------------------------------------------------------------

t0 <- Sys.time()
fit <- muestrear()
cat("\nTiempo de ajuste:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
fit$save_object(f_sal("fit_conjunto", "rds"))

claves <- c("sigma_rw", "sigma_rw_c", "tau_rw", "kappa", "sigma_casa", "log_phi", "mu_gen", "mu_aut", "sigma_gen",
            "sigma_aut", "gamma", "Omega", "sigma_b", "err_new")
diag <- fit$diagnostic_summary(quiet = TRUE)
res <- fit$summary(c(claves, "theta1"), "rhat", "ess_bulk", "ess_tail")
diag_tab <- tibble(
  divergencias = sum(diag$num_divergent),
  max_treedepth = sum(diag$num_max_treedepth),
  rhat_max = max(res$rhat, na.rm = TRUE),
  ess_bulk_min = min(res$ess_bulk, na.rm = TRUE),
  ess_tail_min = min(res$ess_tail, na.rm = TRUE)
)
cat("\nDiagnósticos:\n")
print(diag_tab)
write_csv(diag_tab, f_sal("conjunto_diagnostico"))

# ---- 3. parámetros clave -------------------------------------------------------

grupos_rw <- c("PP", "PSOE", "VOX", "SUMAR", "Podemos+SALF", "regionales", "RESTO")
par <- fit$summary(claves, "median", ~ quantile(.x, c(0.05, 0.95))) |>
  mutate(
    i = suppressWarnings(as.integer(str_match(variable, "\\[(\\d+)")[, 2])),
    j = suppressWarnings(as.integer(str_match(variable, ",(\\d+)\\]")[, 2])),
    base = sub("\\[.*", "", variable)
  ) |>
  filter(base != "Omega" | i > j) |>
  filter(base != "sigma_rw_c" | variante != "v2") |>
  mutate(
    partido = case_when(
      base %in% c("sigma_rw", "sigma_casa") ~ grupos_rw[i],
      base == "sigma_rw_c" ~ paste(dc$campanas[i], grupos_rw[j]),
      base == "log_phi" ~ dc$campanas[i],
      base == "Omega" ~ paste(grupos_error[i], grupos_error[j], sep = "-"),
      !is.na(i) ~ grupos_error[i],
      TRUE ~ ""
    )
  ) |>
  select(parametro = base, partido, mediana = median, q05 = `5%`, q95 = `95%`)
cat("\nParámetros clave (logit):\n")
print(par |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)
write_csv(par, f_sal("conjunto_parametros"))

# tendencia semanal de 2026
vt <- fit$draws("v_obj", format = "draws_matrix")
tend <- tibble(var = colnames(vt)) |>
  mutate(
    k = as.integer(str_match(var, "\\[(\\d+),")[, 2]),
    semana = as.integer(str_match(var, ",(\\d+)\\]")[, 2]),
    partido = fams_obj[k],
    fecha = inicio_2026 + 7 * (semana - 1),
    mediana = 100 * apply(vt, 2, median),
    q05 = 100 * apply(vt, 2, quantile, 0.05),
    q95 = 100 * apply(vt, 2, quantile, 0.95)
  ) |>
  select(partido, semana, fecha, mediana, q05, q95)
write_csv(tend, f_sal("conjunto_tendencia"))
cat("\nVoto de la media de las casas el día E (%):\n")
print(tend |> filter(semana == Tc) |> mutate(across(where(is.numeric), \(x) round(x, 1))), n = Inf)

# ---- 4. escaños --------------------------------------------------------------------

sim <- simular_conjunto(fit, dc)
saveRDS(sim, f_sal("conjunto_sim", "rds"))
presentes <- names(which(colSums(sim$draws[, -1]) > 0))
esc <- resumen_escanos(sim, width = 0.9) |>
  filter(partido %in% presentes) |>
  select(partido, mediana = escanos, q05 = .lower, q95 = .upper) |>
  left_join(escanos_resto_mayor(sim) |> rename(etiqueta = escanos), by = "partido")
may <- prob_mayorias(sim)
write_csv(esc, f_sal("conjunto_escanos"))
write_csv(may, f_sal("conjunto_mayorias"))
cat("\nEscaños (mapa):\n")
print(esc |> mutate(across(where(is.numeric), \(x) round(x, 1))), n = Inf)
print(may)
