# ============================================================================
# FULL BAYESIAN 29-N · 14: ¿qué aprende el modelo conjunto y qué pone el prior?
#
# Contracción = 1 - var(posterior) / var(prior), parámetro a parámetro:
#   ~1: lo deciden los datos; ~0: lo decide el prior.
# El prior sale de un ajuste con prior_only = 1 (sin verosimilitud); el
# posterior, de fit_conjunto.rds. Mismos datos de diseño y mismas escalas de
# los priors (priors_conjunto).
#
# Complementa 09_recuperacion.R, que se hizo sobre la versión modular.
#
# Salida (en 2026/10/full-bayesian/):
#   contraccion_conjunto.csv   parámetro: sd prior, sd posterior, contracción
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/14_contraccion.R
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(cmdstanr)
  library(posterior)
})
source(here::here("2026/10/full-bayesian/11_datos_conjunto.R"))

mod <- cmdstan_model(file.path(dir_fb, "modelo_conjunto.stan"))
dc <- datos_conjunto()
fit_post <- readRDS(file.path(dir_fb, "fit_conjunto.rds"))
fit_prior <- mod$sample(
  data = stan_conjunto(dc, prior_only = 1L),
  seed = 1811,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  refresh = 0,
  show_messages = FALSE
)

claves <- c("sigma_rw", "tau_rw", "kappa", "sigma_casa", "log_phi", "mu_gen",
            "mu_aut", "sigma_gen", "sigma_aut", "gamma", "Omega", "sigma_b")
grupos_rw <- c("PP", "PSOE", "VOX", "SUMAR", "Podemos+SALF", "regionales", "RESTO")
sd_de <- \(fit) {
  d <- as_draws_df(fit$draws(claves))
  tibble(variable = setdiff(names(d), c(".chain", ".iteration", ".draw")),
         sd = map_dbl(variable, \(v) sd(d[[v]])))
}
tab <- inner_join(sd_de(fit_prior), sd_de(fit_post), by = "variable",
                  suffix = c("_prior", "_post")) |>
  mutate(
    i = suppressWarnings(as.integer(str_match(variable, "\\[(\\d+)")[, 2])),
    j = suppressWarnings(as.integer(str_match(variable, ",(\\d+)\\]")[, 2])),
    parametro = sub("\\[.*", "", variable)
  ) |>
  filter(parametro != "Omega" | i > j) |>
  mutate(
    partido = case_when(
      parametro %in% c("sigma_rw", "sigma_casa") ~ grupos_rw[i],
      parametro == "log_phi" ~ dc$campanas[i],
      parametro == "Omega" ~ paste(grupos_error[i], grupos_error[j], sep = "-"),
      !is.na(i) ~ grupos_error[i],
      TRUE ~ ""
    ),
    contraccion = 1 - sd_post^2 / sd_prior^2
  ) |>
  select(parametro, partido, sd_prior, sd_post, contraccion) |>
  arrange(contraccion)
write_csv(tab, file.path(dir_fb, "contraccion_conjunto.csv"))
print(tab |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)
