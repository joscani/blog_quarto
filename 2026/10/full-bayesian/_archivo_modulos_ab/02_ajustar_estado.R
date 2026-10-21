# ============================================================================
# FULL BAYESIAN 29-N · 02: ajuste del módulo A (estado latente)
#
# 1. Prior predictive: con prior_only = 1, ¿qué voto del día E admite el
#    prior? Debe ser amplio pero no absurdo.
# 2. Ajuste con cmdstanr y diagnósticos (divergencias, Rhat, ESS).
# 3. Posterior predictive: % de encuestas observadas dentro de su IC90.
#
# Salidas (en 2026/10/full-bayesian/):
#   fit_estado.rds             objeto CmdStanMCMC (no va a git: release)
#   estado_tendencia.csv       voto semanal de la media de la industria
#   estado_diagnostico.csv     Rhat, ESS y divergencias
#   estado_rebano.csv          dispersión de las encuestas frente al muestreo
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/02_ajustar_estado.R
# ============================================================================

source(here::here("2026/10/full-bayesian/01_datos.R"))
suppressMessages({
  library(cmdstanr)
  library(posterior)
})

seed_estado <- 2911

mod_a <- cmdstan_model(file.path(dir_fb, "modelo_estado.stan"))
da <- datos_estado()

# ---- 1. prior predictive -------------------------------------------------------

fit_prior <- mod_a$sample(
  data = stan_estado(da, prior_only = 1L),
  seed = seed_estado,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 500,
  refresh = 0,
  show_messages = FALSE
)
v_prior <- fit_prior$draws(
  paste0("v[", seq_len(da$K), ",", da$T, "]"),
  format = "draws_matrix"
)
cat("\nPrior predictive del voto el día E (%, q05-q50-q95):\n")
print(round(
  100 * t(apply(v_prior, 2, quantile, c(0.05, 0.5, 0.95))) |>
    `rownames<-`(da$cats),
  1
))

# ---- 2. ajuste -----------------------------------------------------------------

fit <- mod_a$sample(
  data = stan_estado(da),
  seed = seed_estado,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1500,
  adapt_delta = 0.99,
  refresh = 0,
  show_messages = FALSE
)
fit$save_object(file.path(dir_fb, "fit_estado.rds"))

diag <- fit$diagnostic_summary(quiet = TRUE)
resumen <- fit$summary(
  c("eta", "sigma_rw", "sigma_casa", "sigma_enc", "beta"),
  "rhat",
  "ess_bulk",
  "ess_tail"
)
diag_tab <- tibble(
  divergencias = sum(diag$num_divergent),
  max_treedepth = sum(diag$num_max_treedepth),
  rhat_max = max(resumen$rhat, na.rm = TRUE),
  ess_bulk_min = min(resumen$ess_bulk, na.rm = TRUE),
  ess_tail_min = min(resumen$ess_tail, na.rm = TRUE)
)
cat("\nDiagnósticos:\n")
print(diag_tab)
write_csv(diag_tab, file.path(dir_fb, "estado_diagnostico.csv"))

cat("\nsigma del paseo (logit por semana), de las casas y de cada encuesta:\n")
print(
  fit$summary(
    c("sigma_rw", "sigma_casa", "sigma_enc"),
    "median",
    ~ quantile(.x, c(0.05, 0.95))
  ) |>
    mutate(across(where(is.numeric), \(x) round(x, 3)))
)

# ---- tendencia semanal ---------------------------------------------------------

v <- fit$draws("v", format = "draws_matrix")
tendencia <- tibble(var = colnames(v)) |>
  mutate(
    k = as.integer(sub("v\\[(\\d+),.*", "\\1", var)),
    semana = as.integer(sub("v\\[\\d+,(\\d+)\\]", "\\1", var)),
    partido = da$cats[k],
    fecha = inicio_semanas + 7 * (semana - 1),
    mediana = 100 * apply(v, 2, median),
    q05 = 100 * apply(v, 2, quantile, 0.05),
    q95 = 100 * apply(v, 2, quantile, 0.95)
  ) |>
  select(partido, semana, fecha, mediana, q05, q95)
write_csv(tendencia, file.path(dir_fb, "estado_tendencia.csv"))

cat("\nVoto el día E (semana", da$T, ", %):\n")
print(
  tendencia |>
    filter(semana == da$T) |>
    mutate(ancho90 = q95 - q05, across(where(is.numeric), \(x) round(x, 1)))
)

# ---- 3. posterior predictive ---------------------------------------------------

y_rep <- fit$draws("y_rep", format = "draws_matrix")
dentro <- sapply(seq_len(da$N), \(i) {
  sapply(seq_len(da$K), \(k) {
    col <- sprintf("y_rep[%d,%d]", i, k)
    q <- quantile(y_rep[, col], c(0.05, 0.95))
    da$y[i, k] >= q[1] && da$y[i, k] <= q[2]
  })
})
cat(
  "\nPosterior predictive: celdas encuesta x partido dentro de su IC90:",
  round(100 * mean(dentro), 1),
  "% (n =",
  length(dentro),
  ")\n"
)
print(round(100 * rowMeans(dentro), 1) |> setNames(da$cats))

# ---- 4. efecto rebaño ------------------------------------------------------------

# Si cada encuesta fuera una muestra aleatoria independiente, su desviación
# respecto a la media de la industria de su semana tendría, como mínimo, la
# varianza del muestreo p(1 - p) / n (más el efecto casa). Cociente =
# varianza observada / varianza de muestreo. Por debajo de 1: las encuestas
# están más juntas de lo que permite el azar (rebaño, ponderaciones
# parecidas, "no irse del consenso").
v_med <- matrix(
  apply(v, 2, median),
  nrow = da$K,
  dimnames = list(da$cats, NULL)
)
p_obs <- da$y / rowSums(da$y)
rebano <- tibble(partido = da$cats) |>
  mutate(
    cociente = sapply(seq_len(da$K), \(k) {
      p_tend <- v_med[k, da$semana]
      n <- rowSums(da$y)
      mean((p_obs[, k] - p_tend)^2) / mean(p_tend * (1 - p_tend) / n)
    })
  )
cat("\nRebaño: varianza observada / varianza de muestreo (1 = azar puro):\n")
print(rebano |> mutate(cociente = round(cociente, 2)))
write_csv(rebano, file.path(dir_fb, "estado_rebano.csv"))
