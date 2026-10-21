# ============================================================================
# FULL BAYESIAN 29-N · 08: sensibilidad a los priors
#
# ¿Cuánto dependen las conclusiones de las escalas de los priors? Sin
# priorsense (no está en el renv), la alternativa de Bayesian Workflow
# (§8.5): multiplicar cada escala por 0,5 y por 2, una a una, reajustar el
# módulo que la usa y repetir la simulación de escaños. El otro módulo se
# queda en su ajuste base.
#
# Escalas (priors_estado y priors_error en 01_datos.R):
#   módulo A: sd_rw, sd_casa, sd_enc
#   módulo B: sd_mu, sd_log_s0, sd_tau_s, sd_gamma, sd_enc, sd_b, eta_lkj
# 10 escalas x 2 = 20 reajustes + el base.
#
# Cantidades de interés: escaños del PP (mediana e IC90), mediana de PP+VOX,
# P(PP+VOX >= 176) y voto del PP (mediana e IC90).
#
# Salida (en 2026/10/full-bayesian/):
#   sensibilidad.csv   una fila por variante; se escribe tras cada reajuste
#                      (si se corta, se retoma donde iba)
#
# Uso (desde la raíz), después de 02 y 03:
#   Rscript 2026/10/full-bayesian/08_sensibilidad.R      (unos 35 minutos)
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

f_out <- file.path(dir_fb, "sensibilidad.csv")
seed_estado <- 2911
seed_error <- 1129
seed_comb <- 2029

mod_a <- cmdstan_model(file.path(dir_fb, "modelo_estado.stan"))
mod_b <- cmdstan_model(file.path(dir_fb, "modelo_error.stan"))
da <- datos_estado()
db <- datos_error()
fit_a0 <- readRDS(file.path(dir_fb, "fit_estado.rds"))
fit_b0 <- readRDS(file.path(dir_fb, "fit_error.rds"))

ajustar_a <- function(pr) {
  mod_a$sample(
    data = stan_estado(da, priors = pr),
    seed = seed_estado,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1500,
    adapt_delta = 0.99,
    refresh = 0,
    show_messages = FALSE
  )
}
ajustar_b <- function(pr) {
  mod_b$sample(
    data = stan_error(db, priors = pr),
    seed = seed_error,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1500,
    adapt_delta = 0.99,
    refresh = 0,
    show_messages = FALSE
  )
}

resumir <- function(sim, modulo, escala, factor, fit) {
  d <- bloques(sim$draws)
  diag <- fit$diagnostic_summary(quiet = TRUE)
  pp_voto <- 100 * sim$v_err[, "PP"]
  tibble(
    modulo,
    escala,
    factor,
    pp_mediana = median(d$PP),
    pp_q05 = quantile(d$PP, 0.05),
    pp_q95 = quantile(d$PP, 0.95),
    derecha_mediana = median(d$derecha),
    p_pp_vox_176 = mean(d$derecha >= 176),
    pp_voto_mediana = median(pp_voto),
    pp_voto_q05 = quantile(pp_voto, 0.05),
    pp_voto_q95 = quantile(pp_voto, 0.95),
    divergencias = sum(diag$num_divergent)
  )
}

hecho <- if (file.exists(f_out)) {
  read_csv(f_out, show_col_types = FALSE)
} else {
  tibble()
}
ya <- function(m, e, f) {
  nrow(hecho) > 0 &&
    any(hecho$modulo == m & hecho$escala == e & abs(hecho$factor - f) < 1e-9)
}
guardar <- function(fila) {
  hecho <<- bind_rows(hecho, fila)
  write_csv(hecho, f_out)
  cat(sprintf(
    "%s %-10s x%-4s PP %3.0f [%3.0f-%3.0f]  P(PP+VOX>=176) %.3f  div %d  (%s)\n",
    fila$modulo, fila$escala, fila$factor, fila$pp_mediana, fila$pp_q05,
    fila$pp_q95, fila$p_pp_vox_176, fila$divergencias, format(Sys.time(), "%H:%M")
  ))
}

# ---- base ------------------------------------------------------------------------

if (!ya("base", "-", 1)) {
  guardar(resumir(simular_fb(fit_a0, fit_b0, seed_comb), "base", "-", 1, fit_b0))
}

# ---- módulo A --------------------------------------------------------------------

for (e in names(priors_estado)) {
  for (f in c(0.5, 2)) {
    if (ya("A", e, f)) next
    pr <- priors_estado
    pr[[e]] <- pr[[e]] * f
    fit_a <- ajustar_a(pr)
    guardar(resumir(simular_fb(fit_a, fit_b0, seed_comb), "A", e, f, fit_a))
  }
}

# ---- módulo B --------------------------------------------------------------------

for (e in names(priors_error)) {
  for (f in c(0.5, 2)) {
    if (ya("B", e, f)) next
    pr <- priors_error
    pr[[e]] <- pr[[e]] * f
    fit_b <- ajustar_b(pr)
    guardar(resumir(simular_fb(fit_a0, fit_b, seed_comb), "B", e, f, fit_b))
  }
}

cat("\nSensibilidad (escaños del PP y P(PP+VOX >= 176)):\n")
print(
  hecho |>
    mutate(across(where(is.numeric), \(x) round(x, 3))) |>
    select(modulo, escala, factor, pp_mediana, pp_q05, pp_q95, p_pp_vox_176, divergencias),
  n = Inf
)
