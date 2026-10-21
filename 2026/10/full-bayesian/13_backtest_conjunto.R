# ============================================================================
# FULL BAYESIAN 29-N · 13: backtest del modelo conjunto en escaños
#
# Igual que 10_backtest.R, pero con el modelo conjunto: para el 10N-2019 y el
# 23J-2023, datos_conjunto(fuera = id) deja solo lo anterior a la elección
# (campañas pasadas con su resultado, la campaña objetivo SIN resultado y
# las autonómicas previas). El voto el día E (v_obj) más el error predicho
# (err_new) pasan a backtest_eleccion() como tracked_ext (geografía de la
# elección anterior, shock provincial, competencia real de los no seguidos,
# umbral y D'Hondt).
#
# La pregunta: ¿son las horquillas del modelo conjunto demasiado anchas? Si
# el resultado real cae siempre muy dentro, sí.
#
# Salidas (en 2026/10/full-bayesian/):
#   backtest_conjunto_escanos.csv   elección x partido: real, mediana, IC90
#   backtest_conjunto_votos.csv     lo mismo en voto
#   backtest_conjunto_resumen.csv   cobertura y anchura frente al pipeline
#                                   actual (y a lo que haya de versiones
#                                   anteriores en el directorio)
#
# Uso (desde la raíz):
#   Rscript 2026/10/full-bayesian/13_backtest_conjunto.R [v2 | principal] [ids] [corte_dias]
#   (2 ajustes de ~5 y ~13 minutos)
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(cmdstanr)
  library(posterior)
})
source(here::here("2026/10/full-bayesian/11_datos_conjunto.R"))
source(here::here("2026/10/backtest_pipeline.R"))

# variante (primer argumento): "principal" (volatilidad propia de cada
# campaña) o "v2" (volatilidad común, la primera versión).
args <- commandArgs(trailingOnly = TRUE)
variante <- if (length(args) > 0) args[1] else "v2"
# elecciones a probar (segundo argumento, separadas por comas)
ids <- if (length(args) > 1) strsplit(args[2], ",")[[1]] else c("10n", "23j")
# horizonte (tercer argumento): días antes del día E en que se cortan las
# encuestas de la campaña objetivo; 0 = la víspera
corte <- if (length(args) > 2) as.integer(args[3]) else 0L
pr <- priors_conjunto
# ("sigma_gen", la escala propia de las generales, se descartó: ver
# modelo_conjunto_con_modo2.stan.bak)
# "v2" y "sigma_gen" se probaron con la volatilidad común; "principal", con
# la propia de cada campaña
pr$sigma_rw_campana <- as.integer(variante == "principal")
sufijo <- paste0(
  if (variante == "v2") "" else paste0("_", variante), # "v2" sin sufijo: fue la primera
  if (identical(ids, c("10n", "23j"))) {
    ""
  } else {
    paste0("_", paste(ids, collapse = "_"))
  },
  if (corte > 0) paste0("_c", corte) else ""
)

b_bt <- 2000
seed_bt <- 4177
mod <- cmdstan_model(file.path(dir_fb, "modelo_conjunto.stan"))

backtest_conjunto <- function(id, preps, tau, mapa) {
  dc <- datos_conjunto(fuera = id, corte_dias = corte)
  t0 <- Sys.time()
  fit <- mod$sample(
    data = stan_conjunto(dc, priors = pr),
    seed = 1811,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1500,
    adapt_delta = 0.95,
    refresh = 250
  )
  fit$save_object(file.path(
    dir_fb,
    paste0("fit_conjunto_bt_", id, sufijo, ".rds")
  ))
  d <- fit$diagnostic_summary(quiet = TRUE)
  rh <- fit$summary(
    c("sigma_rw", "kappa", "sigma_gen", "Omega", "theta1"),
    "rhat"
  )
  cat(sprintf(
    "\n%s: %.1f min, divergencias %d, rhat max %.3f\n",
    id,
    as.numeric(difftime(Sys.time(), t0, units = "mins")),
    sum(d$num_divergent),
    max(rh$rhat, na.rm = TRUE)
  ))

  Kc <- dc$K[dc$C]
  v <- unclass(fit$draws(
    sprintf("v_obj[%d,%d]", seq_len(Kc), dc$T[dc$C]),
    format = "draws_matrix"
  ))[, seq_len(Kc)]
  colnames(v) <- fams_campana(id)
  e <- unclass(fit$draws("err_new", format = "draws_matrix"))[, 1:4]
  colnames(e) <- grupos_error
  # grupos de la campaña (sin VOX en 2015-2016)
  gr <- intersect(grupos_error, colnames(v))

  set.seed(seed_bt)
  s <- sample(nrow(v), b_bt)
  v <- v[s, ]
  e <- e[s, ]
  g <- plogis(qlogis(v[, gr, drop = FALSE]) + e[, gr, drop = FALSE])
  tot <- rowSums(g) + v[, "RESTO"]
  tracked <- 100 * cbind(g / tot, BLOQUE = v[, "RESTO"] / tot)

  el <- elecciones[[which(map_chr(elecciones, "id") == id)]]
  set.seed(seed_bt)
  backtest_eleccion(
    el,
    preps,
    b = b_bt,
    d_ref = preps[[prev_de[[id]]]],
    modo = "shock",
    tau = tau,
    mapa = mapa,
    tracked_ext = tracked
  )
}

preps <- preparar_preps()
tau <- extraer_tau_draws(
  readRDS(here::here("2026/10/mod_tau_shock.rds")),
  b = b_bt
)
mapa <- mapa_ccaa_congreso()

res <- map(ids, \(id) backtest_conjunto(id, preps, tau, mapa))
esc <- map_dfr(res, "escanos")
vot <- map_dfr(res, "votos")
write_csv(
  esc,
  file.path(dir_fb, paste0("backtest_conjunto_escanos", sufijo, ".csv"))
)
write_csv(
  vot,
  file.path(dir_fb, paste0("backtest_conjunto_votos", sufijo, ".csv"))
)

leer <- \(f, v) {
  read_csv(f, show_col_types = FALSE) |>
    filter(eleccion %in% ids) |>
    mutate(version = v)
}
# la comparación con la primera corrida del conjunto (sin sufijo) y con la
# versión modular (backtest_fb_*.csv, archivada en _archivo_modulos_ab/)
# solo si esos ficheros siguen existiendo
todas_esc <- bind_rows(
  esc |> mutate(version = paste0("conjunto", sufijo)),
  if (
    sufijo != "" &&
      file.exists(file.path(dir_fb, "backtest_conjunto_escanos.csv"))
  ) {
    leer(file.path(dir_fb, "backtest_conjunto_escanos.csv"), "conjunto")
  },
  if (file.exists(file.path(dir_fb, "backtest_fb_escanos.csv"))) {
    leer(file.path(dir_fb, "backtest_fb_escanos.csv"), "modular")
  },
  leer(here::here("2026/10/backtest_escanos.csv"), "actual")
)
todas_vot <- bind_rows(
  vot |> mutate(version = paste0("conjunto", sufijo)),
  if (
    sufijo != "" &&
      file.exists(file.path(dir_fb, "backtest_conjunto_votos.csv"))
  ) {
    leer(file.path(dir_fb, "backtest_conjunto_votos.csv"), "conjunto")
  },
  if (file.exists(file.path(dir_fb, "backtest_fb_votos.csv"))) {
    leer(file.path(dir_fb, "backtest_fb_votos.csv"), "modular")
  },
  leer(here::here("2026/10/backtest_votos.csv"), "actual")
)
resumen <- bind_rows(
  todas_esc |> mutate(tipo = "escaños"),
  todas_vot |> mutate(tipo = "voto")
) |>
  group_by(version, tipo) |>
  summarise(
    celdas = n(),
    cobertura_ic90 = mean(dentro),
    ancho_medio_ic90 = mean(q95 - q05),
    error_mediano = median(abs(mediana - real)),
    # posición del real dentro del IC90: 0 = en el centro, 1 = en el borde
    posicion_media = mean(abs(real - mediana) / ((q95 - q05) / 2)),
    .groups = "drop"
  )
write_csv(
  resumen,
  file.path(dir_fb, paste0("backtest_conjunto_resumen", sufijo, ".csv"))
)

cat("\nEscaños (real, mediana [IC90]):\n")
todas_esc |>
  mutate(
    txt = sprintf(
      "%g [%g-%g]%s",
      mediana,
      q05,
      q95,
      ifelse(dentro, "", " FUERA")
    )
  ) |>
  select(eleccion, familia, real, version, txt) |>
  pivot_wider(names_from = version, values_from = txt) |>
  print(n = Inf)
cat("\nResumen:\n")
print(resumen |> mutate(across(where(is.numeric), \(x) round(x, 2))))
write_csv(
  tibble(
    eleccion = map_chr(res, "id"),
    p_pp_vox_176 = map_dbl(res, "p_derecha"),
    real_pp_vox = map_dbl(res, "real_derecha")
  ),
  file.path(dir_fb, paste0("backtest_conjunto_mayorias", sufijo, ".csv"))
)
cat(
  "\nP(PP+VOX >= 176):",
  paste(
    map_chr(res, "id"),
    round(map_dbl(res, "p_derecha"), 3),
    collapse = " | "
  ),
  "\n"
)
