# ============================================================================
# FULL BAYESIAN 29-N · 09: ¿recupera el módulo B parámetros conocidos?
#
# Bayesian Workflow (§6.3): ajustar a datos simulados con parámetros
# conocidos. Si el modelo no los recupera, el posterior de los datos reales
# lo está fijando el prior (o hay un error de código).
#
# Con ~290 parámetros y solo 5 generales, la duda es si se separan gamma
# (nivel del error en generales), la correlación y sigma_b (sesgo de las
# casas). La primera versión (Omega de las generales mezclada con la de las
# autonómicas con un peso w) mostró que ni w ni esa correlación se
# aprendían: se simplificó a una Omega común.
#
# Procedimiento:
#   1. S = 20 draws del posterior real (fit_error.rds) hacen de "verdad":
#      mu, sigma_enc, delta de cada elección y b de cada casa.
#   2. Con el mismo diseño (encuestas, partidos observados, elección, casa,
#      tipo) se simula z = mu + delta + b + N(0, sigma_enc).
#   3. Se reajusta y se mira si la verdad cae en el IC90 del posterior.
#   4. Contracción = 1 - var(posterior) / var(prior): cuánto aprende el
#      modelo de los datos (0: nada, el prior manda; 1: todo).
# Con draws del posterior como verdad (no del prior), esto comprueba la
# recuperación en la zona que importa; no es una SBC completa.
#
# Salidas (en 2026/10/full-bayesian/):
#   recuperacion_celdas.csv    dataset x parámetro: verdad, mediana, IC90
#   recuperacion_resumen.csv   parámetro: cobertura del IC90, contracción
#
# Uso (desde la raíz), después de 03:
#   Rscript 2026/10/full-bayesian/09_recuperacion.R      (unos 20 minutos)
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(cmdstanr)
  library(posterior)
})
source(here::here("2026/10/full-bayesian/01_datos.R"))

S <- 20
seed_rec <- 6061
mod_b <- cmdstan_model(file.path(dir_fb, "modelo_error.stan"))
db <- datos_error()
fit0 <- readRDS(file.path(dir_fb, "fit_error.rds"))
D <- db$D

pars <- c(
  sprintf("mu[%d,1]", 1:D),
  sprintf("sigma_gen[%d]", 1:D),
  sprintf("sigma_aut[%d]", 1:D),
  sprintf("sigma_b[%d]", 1:D),
  "gamma",
  "Omega[3,1]",
  sprintf("b26[%d]", 1:D)
)
etiqueta <- function(v) {
  k <- suppressWarnings(as.integer(str_match(v, "\\[(\\d+)")[, 2]))
  case_when(
    v == "Omega[3,1]" ~ "Omega VOX-PP",
    str_starts(v, "mu") ~ paste("mu_gen", db$cats[k]),
    !is.na(k) ~ paste(sub("\\[.*", "", v), db$cats[k]),
    TRUE ~ v
  )
}

ajustar <- function(d, prior_only = 0L) {
  mod_b$sample(
    data = stan_error(d, prior_only),
    seed = seed_rec,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 500,
    iter_sampling = 500,
    adapt_delta = 0.99,
    refresh = 0,
    show_messages = FALSE
  )
}

# varianza del prior de cada parámetro (para la contracción)
fit_prior <- ajustar(db, prior_only = 1L)
var_prior <- apply(unclass(fit_prior$draws(pars, format = "draws_matrix")), 2, var)

# ---- datasets simulados ------------------------------------------------------------

set.seed(seed_rec)
dr0 <- as_draws_df(fit0$draws(c("mu", "sigma_enc", "delta", "b", pars)))
elegidos <- sample(nrow(dr0), S)
obs <- db$obs == 1
g <- db$tipo[db$eleccion]

celdas <- map_dfr(seq_len(S), \(s) {
  v <- dr0[elegidos[s], ]
  val <- \(nm) as.numeric(v[[nm]])
  z <- matrix(0, db$J, D)
  for (k in 1:D) {
    mu_k <- sapply(g, \(gg) val(sprintf("mu[%d,%d]", k, gg)))
    de_k <- sapply(db$eleccion, \(e) val(sprintf("delta[%d,%d]", k, e)))
    b_k <- sapply(db$casa, \(h) val(sprintf("b[%d,%d]", k, h)))
    se_k <- sapply(g, \(gg) val(sprintf("sigma_enc[%d,%d]", k, gg)))
    z[, k] <- mu_k + de_k + b_k + rnorm(db$J, 0, se_k)
  }
  z[!obs] <- 0
  d_s <- db
  d_s$z <- z
  fit_s <- ajustar(d_s)
  post <- unclass(fit_s$draws(pars, format = "draws_matrix"))
  diag <- fit_s$diagnostic_summary(quiet = TRUE)
  cat(sprintf("dataset %2d de %d (%s), divergencias %d\n", s, S, format(Sys.time(), "%H:%M"), sum(diag$num_divergent)))
  tibble(
    dataset = s,
    variable = pars,
    parametro = etiqueta(pars),
    verdad = sapply(pars, val),
    mediana = apply(post, 2, median),
    q05 = apply(post, 2, quantile, 0.05),
    q95 = apply(post, 2, quantile, 0.95),
    var_post = apply(post, 2, var),
    divergencias = sum(diag$num_divergent)
  )
})

resumen <- celdas |>
  mutate(
    dentro = verdad >= q05 & verdad <= q95,
    contraccion = 1 - var_post / var_prior[variable]
  ) |>
  group_by(parametro) |>
  summarise(
    cobertura_ic90 = mean(dentro),
    contraccion = median(contraccion),
    # sesgo de la mediana respecto a la verdad, en sd posteriores
    sesgo_sd = median((mediana - verdad) / sqrt(var_post)),
    .groups = "drop"
  ) |>
  arrange(contraccion)

write_csv(celdas, file.path(dir_fb, "recuperacion_celdas.csv"))
write_csv(resumen, file.path(dir_fb, "recuperacion_resumen.csv"))
cat("\nRecuperación (", S, "datasets; cobertura nominal 0,90):\n")
print(resumen |> mutate(across(where(is.numeric), \(x) round(x, 2))), n = Inf)
