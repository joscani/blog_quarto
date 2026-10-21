# ============================================================================
# FULL BAYESIAN 29-N · 10: backtest de la cadena completa en escaños
#
# Pregunta: si hubiéramos usado este modelo en el 10N-2019 y el 23J-2023,
# ¿habría caído el resultado real dentro de su IC90 de escaños?
#
# Para cada elección, con SOLO la información disponible antes de ella:
#   1. Módulo A con las encuestas de esa campaña (errores_encuestas_ventana.csv,
#      hasta 75 días antes). Solo encuestas publicables: fin de campo al
#      menos 6 días antes (la ley prohíbe publicar en los 5 últimos). El CSV
#      no trae la n: se supone 1.000 por encuesta, como en
#      backtest_pipeline.R. Categorías: PP, PSOE, VOX, SUMAR y BLOQUE (el
#      resto, con Cs en 2019). Prior de la semana 1: log-ratios de la general
#      anterior.
#   2. Módulo B solo con generales y autonómicas ANTERIORES a la elección;
#      el sesgo medio b se calcula con las casas de esa campaña.
#   3. Combinación draw a draw (logit) y escaños con backtest_eleccion() de
#      backtest_pipeline.R (shock provincial con referencia geográfica de la
#      elección anterior, competencia real de los partidos no seguidos,
#      umbral y D'Hondt), con tracked_ext en lugar del agregado y el error
#      plug-in.
# Se compara con el backtest del pipeline actual (backtest_escanos.csv,
# kappa = 0,5, geografía previa).
#
# Salidas (en 2026/10/full-bayesian/):
#   backtest_fb_escanos.csv   elección x partido: real, mediana, IC90, dentro
#   backtest_fb_votos.csv     lo mismo en voto
#   backtest_fb_resumen.csv   cobertura y anchura frente al pipeline actual
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/10_backtest.R
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(cmdstanr)
  library(posterior)
})
source(here::here("2026/10/full-bayesian/01_datos.R"))
source(here::here("2026/10/backtest_pipeline.R"))

b_bt <- 2000
seed_bt <- 4177
cats_bt <- c("PP", "PSOE", "VOX", "SUMAR", "BLOQUE")
previa <- c(`10n` = "28a", `23j` = "10n")
dias_min <- 6

mod_a <- cmdstan_model(file.path(dir_fb, "modelo_estado.stan"))
mod_b <- cmdstan_model(file.path(dir_fb, "modelo_error.stan"))

# ---- módulo A con las encuestas de la campaña ---------------------------------------

datos_estado_bt <- function(id) {
  dia_e <- fechas_hist[[id]]
  enc <- read_csv(
    here::here("2026/10/errores_encuestas_ventana.csv"),
    show_col_types = FALSE
  ) |>
    filter(eleccion == id, dias >= dias_min) |>
    filter(if_all(c(PP, PSOE, VOX, SUMAR), \(x) !is.na(x))) |>
    mutate(BLOQUE = 100 - PP - PSOE - VOX - SUMAR, casa = norm_casa(casa)) |>
    filter(BLOQUE > 0)
  inicio <- min(enc$fecha)
  sem <- \(f) as.integer(floor(as.numeric(as.Date(f) - inicio) / 7)) + 1L
  y <- round(10 * as.matrix(enc[, cats_bt]))
  storage.mode(y) <- "integer"
  casas <- sort(unique(enc$casa))
  real_prev <- errores_hist()$real |> filter(eleccion == previa[[id]])
  r_prev <- 100 - real_prev$PP - real_prev$PSOE - real_prev$VOX - real_prev$SUMAR
  list(
    N = nrow(enc),
    K = length(cats_bt),
    H = length(casas),
    T = sem(dia_e),
    y = y,
    semana = sem(enc$fecha),
    casa = match(enc$casa, casas),
    eta1_mu = log(unlist(real_prev[cats_bt[1:4]]) / r_prev) |> unname(),
    eta1_sd = eta1_sd,
    casas = casas,
    n_encuestas = nrow(enc)
  )
}

# ---- una elección ----------------------------------------------------------------

backtest_fb <- function(id, preps, tau, mapa) {
  dia_e <- fechas_hist[[id]]
  da <- datos_estado_bt(id)
  fit_a <- mod_a$sample(
    data = stan_estado(da),
    seed = 2911,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1500,
    adapt_delta = 0.99,
    refresh = 0,
    show_messages = FALSE
  )
  db <- datos_error(antes_de = dia_e, casas_obj = da$casas)
  fit_b <- mod_b$sample(
    data = stan_error(db),
    seed = 1129,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1500,
    adapt_delta = 0.99,
    refresh = 0,
    show_messages = FALSE
  )
  div <- c(
    A = sum(fit_a$diagnostic_summary(quiet = TRUE)$num_divergent),
    B = sum(fit_b$diagnostic_summary(quiet = TRUE)$num_divergent)
  )

  v <- unclass(fit_a$draws(
    sprintf("v[%d,%d]", seq_len(da$K), da$T),
    format = "draws_matrix"
  ))[, seq_len(da$K)]
  colnames(v) <- cats_bt
  e <- unclass(fit_b$draws("err_new_casa", format = "draws_matrix"))[, 1:4]
  colnames(e) <- cats_error

  set.seed(seed_bt)
  s <- sample(nrow(v), b_bt)
  v <- v[s, ]
  e <- e[s, ]
  g <- plogis(qlogis(v[, 1:4]) + e[, colnames(v)[1:4]])
  tot <- rowSums(g) + v[, "BLOQUE"]
  tracked <- 100 * cbind(g / tot, BLOQUE = v[, "BLOQUE"] / tot)

  el <- elecciones[[which(map_chr(elecciones, "id") == id)]]
  set.seed(seed_bt)
  res <- backtest_eleccion(
    el,
    preps,
    b = b_bt,
    d_ref = preps[[prev_de[[id]]]],
    modo = "shock",
    tau = tau,
    mapa = mapa,
    tracked_ext = tracked
  )
  cat(sprintf(
    "%s: %d encuestas, %d casas, %d generales y %d autonómicas previas en B, divergencias A %d / B %d\n",
    id, da$n_encuestas, da$H, sum(db$tipo == 1), sum(db$tipo == 2), div["A"], div["B"]
  ))
  res
}

# ---- ejecución -------------------------------------------------------------------

preps <- preparar_preps()
fit_full <- readRDS(here::here("2026/10/mod_tau_shock.rds"))
tau <- extraer_tau_draws(fit_full, b = b_bt)
mapa <- mapa_ccaa_congreso()

res <- map(names(previa), \(id) backtest_fb(id, preps, tau, mapa))
esc <- map_dfr(res, "escanos")
vot <- map_dfr(res, "votos")
write_csv(esc, file.path(dir_fb, "backtest_fb_escanos.csv"))
write_csv(vot, file.path(dir_fb, "backtest_fb_votos.csv"))

# pipeline actual, mismas elecciones (backtest_pipeline.R ya guardado)
act_esc <- read_csv(here::here("2026/10/backtest_escanos.csv"), show_col_types = FALSE) |>
  filter(eleccion %in% names(previa))
act_vot <- read_csv(here::here("2026/10/backtest_votos.csv"), show_col_types = FALSE) |>
  filter(eleccion %in% names(previa))

resumen <- bind_rows(
  esc |> mutate(version = "full bayesian", tipo = "escaños"),
  vot |> mutate(version = "full bayesian", tipo = "voto"),
  act_esc |> mutate(version = "actual", tipo = "escaños"),
  act_vot |> mutate(version = "actual", tipo = "voto")
) |>
  group_by(version, tipo) |>
  summarise(
    celdas = n(),
    cobertura_ic90 = mean(dentro),
    ancho_medio_ic90 = mean(q95 - q05),
    error_mediano = median(abs(mediana - real)),
    .groups = "drop"
  )
write_csv(resumen, file.path(dir_fb, "backtest_fb_resumen.csv"))

cat("\nEscaños (real, mediana [IC90]):\n")
bind_rows(
  esc |> mutate(version = "fb"),
  act_esc |> mutate(version = "actual")
) |>
  mutate(txt = sprintf("%g [%g-%g]%s", mediana, q05, q95, ifelse(dentro, "", " FUERA"))) |>
  select(eleccion, familia, real, version, txt) |>
  pivot_wider(names_from = version, values_from = txt) |>
  print(n = Inf)
cat("\nResumen:\n")
print(resumen |> mutate(across(where(is.numeric), \(x) round(x, 2))))
cat("\nP(PP+VOX >= 176):", paste(map_chr(res, "id"), round(map_dbl(res, "p_derecha"), 3), collapse = " | "), "\n")
