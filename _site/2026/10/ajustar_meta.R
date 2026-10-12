# ============================================================================
# AJUSTE DE LOS DOS MODELOS DEL META-ANÁLISIS (arquitectura en cascada)
#
# Modelo 1 (grande) — multinomial de 6 categorías:
#   PP | PSOE | VOX | SUMAR (= Sumar + Podemos) | SALF | BLOQUE
#   cell_counts | trials(n) ~ time + (1 | encuestadora)
#
#   - Con una sola covariable, brms::multinomial() ajusta un intercepto y
#     una pendiente de `time` por cada categoría no-referencia (la primera,
#     PP, es la referencia).
#   - (1 | encuestadora): efecto casa constante (antes era (time |
#     encuestadora), la deriva correlacionada del post de Andalucía; con la
#     ventana corta de jul-oct 2026 y pocas casas por nivel queda dominada
#     por el prior, y el holdout ya daba ambas estructuras equivalentes).
#   - BLOQUE = 100 - (PP+PSOE+VOX+SUMAR+SALF): toda la masa que no es de
#     los 5 partidos, incluidos los regionales. No se interpreta por sí
#     mismo: su composición la estima el modelo 2.
#
# Modelo 2 (composición del BLOQUE) — multinomial condicional de 8 subceldas:
#   Junts | ERC | PNV | EH Bildu | BNG | CC | UPN | RESTO
#   cell_counts | trials(bloque) ~ time
#
#   - Solo encuestas que reportan los 7 regionales (97 de 10 casas tras el
#     relleno con objetivo176; el CIS queda fuera, igual que en el modelo 1).
#     Por eso NO lleva efecto casa: ElectoPanel concentra el 63% del peso y
#     un efecto aleatorio tan desbalanceado sería ruido; la composición del
#     bloque se modela como tendencia temporal común.
#   - trials = votos totales del bloque de esa encuesta (no la muestra
#     completa): es un multinomial condicional, la masa del bloque se
#     conserva.
#
# `time` = días desde el origen, ahora 2026-07-01 (inicio de la ventana del
# modelo 1; antes era 2025-06-01 y con la ventana corta dejaba los interceptos
# muy lejos de los datos). Los interceptos corresponden a esa fecha; para
# predecir "hoy" (o el día de las elecciones) se evalúa el modelo con el
# `time` de esa fecha.
#
# Priors (los mismos que en Andalucía):
#   b / Intercept -> Normal(0, 2) en escala log-odds
#   sd            -> Exponential(1)
#   cor           -> lkj(1)
#
# Detalles de sampling del modelo grande (aprendidos a la leña):
#   - init = 0: con valores iniciales aleatorios las cadenas no llegan a
#     arrancar (gradiente no finito en cientos de reintentos).
#   - max_treedepth = 12: con el efecto aleatorio correlacionado de 22
#     casas, con treedepth 10 el 100% de transiciones pega el límite.
#
# Salidas: 2026/10/mod_meta_grande.rds y 2026/10/mod_meta_bloque.rds
# (fitted brmsfit con caché: si existen, no se re-ajustan salvo refit = TRUE)
# ============================================================================

library(brms)
library(cmdstanr)
library(dplyr)
library(tidyr)
library(readr)

options(brms.backend = "cmdstanr")

f_grande <- here::here("2026/10/encuestas_meta_grande.csv")
f_bloque <- here::here("2026/10/encuestas_meta_bloque.csv")
f_mod_g <- here::here("2026/10/mod_meta_grande")
f_mod_b <- here::here("2026/10/mod_meta_bloque")
ref_time <- "2026-07-01" # origen de `time` (inicio de la ventana del modelo 1)
fecha_elecciones <- "2026-11-29"
# time al que habrá que evaluar el modelo para predecir el día E:
# as.numeric(as.Date(fecha_elecciones) - as.Date(ref_time)) = 546 días
time_eleccion <- as.numeric(as.Date(fecha_elecciones) - as.Date(ref_time))

min_encuestas_casa <- 1
# casas con menos de min_encuestas_casa encuestas quedan fuera del modelo
# grande. Con la ventana corta (jul-oct 2026) había solo 3 casas con >=3
# encuestas: se baja a 1 para incluir los singletons (IMOP, Data10, NC
# Report...), que el shrinkage jerárquico absorbe como efecto casa.

seed_fit <- 7314

# ---- preparación de datos ---------------------------------------------------

prep_grande <- function() {
  g <- read_csv(f_grande, show_col_types = FALSE) |>
    mutate(time = as.numeric(fecha - as.Date(ref_time)))
  # encuestas sin tamaño muestral publicado: fuera, como en Andalucía
  # (si no, brms las descarta en silencio)
  sin_n <- g |> filter(is.na(muestra)) |> distinct(encuestadora, campo)
  if (nrow(sin_n) > 0) {
    cat(
      "Fuera (sin n publicado):",
      nrow(sin_n),
      "encuestas —",
      paste(sin_n$encuestadora, collapse = ", "),
      "\n"
    )
    g <- g |> filter(!is.na(muestra))
  }
  # si ciento76 trae la misma encuesta dos veces (mismo enlace y campo),
  # pivot_wider fallaría: dedup defensivo por (encuesta x categoría),
  # conservando la primera fila de cada par
  g_d <- distinct(g, encuestadora, campo, enlace, categoria, .keep_all = TRUE)
  if (nrow(g_d) < nrow(g)) {
    warning(
      nrow(g) - nrow(g_d),
      " filas duplicadas de encuesta; conservo la primera"
    )
    g <- g_d
  }
  # casas con muy pocas encuestas: fuera del modelo grande
  # (ojo: en formato largo hay 6 filas por encuesta, se cuentan encuestas)
  por_casa <- g |>
    distinct(encuestadora, campo, enlace) |>
    count(encuestadora)
  pocas <- por_casa |> filter(n < min_encuestas_casa) |> pull(encuestadora)
  if (length(pocas) > 0) {
    cat(
      "Fuera (menos de",
      min_encuestas_casa,
      "encuestas):",
      length(pocas),
      "casas —",
      paste(pocas, collapse = ", "),
      "\n"
    )
    g <- g |> filter(!encuestadora %in% pocas)
  }
  # CIS fuera: su estimación de voto es metodológicamente distinta (y sus
  # barómetros muy grandes); se estimará aparte con sus microdatos
  # corrigiendo el recuerdo de voto. Ver posts de MRP/raking.
  #
  # ElectoPanel fuera del modelo grande: es un panel continuo de electomanía
  # que suponía el 46% del dataset, no encuestas puntuales independientes.
  # SIGUE dentro del modelo de composición del bloque (prep_bloque): es la
  # casa con más cobertura regional, aunque desde el relleno con objetivo176
  # ya no es la única (57 de 97 encuestas, 63% del peso).
  casas_fuera <- c("CIS", "ElectoPanel")
  fuera_cis <- g |>
    filter(encuestadora %in% casas_fuera) |>
    distinct(encuestadora, campo, enlace) |>
    nrow()
  if (fuera_cis > 0) {
    cat(
      "Fuera (casas_fuera:",
      paste(casas_fuera, collapse = ", "),
      "):",
      fuera_cis,
      "encuestas\n"
    )
    g <- g |> filter(!encuestadora %in% casas_fuera)
  }
  wide <- g |>
    pivot_wider(
      id_cols = c(encuestadora, enlace, fecha, time, muestra),
      names_from = categoria,
      values_from = votos
    )
  # trials = suma de conteos (el redondeo puede diferir de `muestra` en ±1)
  wide$n <- rowSums(wide |> select(PP, PSOE, VOX, SUMAR, SALF, BLOQUE))
  wide$cell_counts <- with(wide, cbind(PP, PSOE, VOX, SUMAR, SALF, BLOQUE))
  wide
}

prep_bloque <- function() {
  b <- read_csv(f_bloque, show_col_types = FALSE) |>
    mutate(time = as.numeric(fecha - as.Date(ref_time)))
  b_d <- distinct(b, encuestadora, campo, enlace, subcela, .keep_all = TRUE)
  if (nrow(b_d) < nrow(b)) {
    warning(
      nrow(b) - nrow(b_d),
      " filas duplicadas de encuesta; conservo la primera"
    )
    b <- b_d
  }
  wide <- b |>
    pivot_wider(
      id_cols = c(
        encuestadora,
        enlace,
        fecha,
        time,
        muestra,
        bloque_pct,
        bloque_votos
      ),
      names_from = subcela,
      values_from = votos
    ) |>
    # nombre sin espacio para que brms genere parámetros limpios
    rename(EH_Bildu = `EH Bildu`)
  wide$bloque <- rowSums(
    wide |> select(Junts, ERC, PNV, EH_Bildu, BNG, CC, UPN, RESTO)
  )
  wide$cell_counts <- with(
    wide,
    cbind(Junts, ERC, PNV, EH_Bildu, BNG, CC, UPN, RESTO)
  )
  wide
}

# ---- priors -----------------------------------------------------------------

priors_multinomial <- function(formula, data) {
  p <- get_prior(formula, data, family = multinomial())
  p |>
    mutate(
      prior = case_when(
        class == "b" ~ "normal(0, 2)",
        class == "Intercept" ~ "normal(0, 2)",
        class == "sd" ~ "exponential(1)",
        class == "cor" ~ "lkj(1)",
        TRUE ~ prior
      )
    ) |>
    # las filas globales de b y sd (coef vacío) sobran: cada coeficiente ya
    # tiene su prior individual y brms avisa de que ignora las globales.
    # Ojo: no toca las filas de Intercept (coef vacío ahí es su fila única)
    filter(!(class %in% c("b", "sd") & (is.na(coef) | coef == "")))
}

# ---- ajuste -----------------------------------------------------------------

ajustar_grande <- function(refit = FALSE, iter = 4000, warmup = 1000) {
  dat <- prep_grande()
  # con pocas casas y ventana corta, la pendiente aleatoria correlacionada
  # de Andalucía queda dominada por el prior (3-13 niveles); se usa efecto
  # casa constante. El holdout de comparar_modelos.R ya las daba equivalentes.
  form <- brmsformula(cell_counts | trials(n) ~ time + (1 | encuestadora))
  if (file.exists(paste0(f_mod_g, ".rds")) && !refit) {
    message("mod_meta_grande ya existe (refit = TRUE para reajustar)")
    return(invisible(readRDS(paste0(f_mod_g, ".rds"))))
  }
  brm(
    form,
    dat,
    multinomial(),
    prior = priors_multinomial(form, dat),
    iter = iter,
    warmup = warmup,
    cores = 4,
    chains = 4,
    seed = seed_fit,
    init = 0,
    file = f_mod_g,
    file_refit = if (refit) "always" else "never",
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 0
  )
}

ajustar_bloque <- function(refit = FALSE, iter = 4000, warmup = 1000) {
  dat <- prep_bloque()
  form <- brmsformula(cell_counts | trials(bloque) ~ time)
  if (file.exists(paste0(f_mod_b, ".rds")) && !refit) {
    message("mod_meta_bloque ya existe (refit = TRUE para reajustar)")
    return(invisible(readRDS(paste0(f_mod_b, ".rds"))))
  }
  brm(
    form,
    dat,
    multinomial(),
    prior = priors_multinomial(form, dat),
    iter = iter,
    warmup = warmup,
    cores = 4,
    chains = 4,
    seed = seed_fit,
    file = f_mod_b,
    file_refit = if (refit) "always" else "never",
    control = list(adapt_delta = 0.95),
    refresh = 0
  )
}

# ---- diagnóstico ------------------------------------------------------------

diagnostico <- function(modelo) {
  s <- summary(modelo)
  fx <- as.data.frame(s$fixed) |>
    tibble::rownames_to_column("param") |>
    as_tibble() |>
    mutate(across(where(is.numeric), \(x) round(x, 3)))
  div <- nuts_params(modelo) |>
    filter(Parameter == "divergent__") |>
    summarise(n_divergent = sum(Value)) |>
    pull(n_divergent)
  rhat_max <- max(s$fixed[, "Rhat"])
  list(fixed = fx, n_divergent = div, rhat_max = rhat_max)
}

if (sys.nframe() == 0) {
  mod_g <- ajustar_grande()
  mod_b <- ajustar_bloque()
  d_g <- diagnostico(mod_g)
  d_b <- diagnostico(mod_b)
  cat("\n===== MODELO GRANDE: efectos fijos =====\n")
  print(d_g$fixed)
  cat(
    "\ndivergentes:",
    d_g$n_divergent,
    "| rhat máx:",
    round(d_g$rhat_max, 3),
    "\n"
  )
  cat("\n===== MODELO BLOQUE: efectos fijos =====\n")
  print(d_b$fixed)
  cat(
    "\ndivergentes:",
    d_b$n_divergent,
    "| rhat máx:",
    round(d_b$rhat_max, 3),
    "\n"
  )
}

# ----------------------------------------------------------------------------
# RUTINA DE ACTUALIZACIÓN CUANDO ENTREN ENCUESTAS NUEVAS (ola oct-nov 2026):
#
#   source("2026/10/preparar_meta.R")
#   preparar_todo(actualizar = TRUE)   # scrapea ciento76 y regenera los CSV
#
#   source("2026/10/ajustar_meta.R")
#   ajustar_grande(refit = TRUE)       # re-ajusta el modelo 1
#   ajustar_bloque(refit = TRUE)       # re-ajusta el modelo 2 (composición)
#
#   source("2026/10/simular_escanos.R")
#   sim <- simular(n_draws = 2000)     # escaños día E (29 nov 2026)
#   resumen_escanos(sim); prob_mayorias(sim)
#
# Notas:
#  - la ventana del modelo 1 es fecha FIJA (fecha_inicio_grande = 2026-07-01
#    en preparar_meta.R); si se quiere rodante, cambiarla ahí (p. ej. a
#    max(fecha) - 120 días dentro de leer_encuestas_meta()).
#  - ajustar_bloque() sigue con el dataset largo hasta decidir su ventana.
#  - ojo con brm(file=...): sin refit = TRUE devuelve el modelo cacheado.
#  - la ref_provincial_23J_13.csv es congelada (no se regenera en la rutina);
#    si algún día hay que rehacerla, ver la construcción de simular_escanos.R
#    (11 familias del 23J + SALF y su alpha desde las Europeas 2024).
#  - línea base PREVIA a la ola (día E, mediana de escaños): PP 133, PSOE 105,
#    VOX 67, SUMAR 17, ERC 9, EH Bildu 7, Junts 4, PNV 4, BNG 2, CC 1,
#    UPN 0-1, SALF 0; P(PP+VOX >= 176) = 1.0. En % de voto: PP 32.6,
#    PSOE 27.1, VOX 19.2, SUMAR 8.0, SALF 1.8, BLOQUE 11.3.
# ----------------------------------------------------------------------------
