# ============================================================================
# FULL BAYESIAN 29-N · 01: preparación de datos
#
# La sourcea 11_datos_conjunto.R. Da acceso a:
#
#   - prep_grande() y f_grande_sep (via ajustar_meta.R): las encuestas de
#     2026 del escenario separadas, con los mismos filtros que el modelo 1
#     de brms (sin CIS, sin ElectoPanel, sin encuestas sin n publicado).
#   - errores_hist(): 124 encuestas de la última semana de campaña en las
#     generales de 2015-2023 (errores_encuestas_historicas.csv).
#   - errores_autonomicas(): las de la última semana en 39 autonómicas
#     (2018-2026, de 06_scrapear_autonomicas.R).
#   - datos_error(): une las dos fuentes en el formato del modelo conjunto.
#   - norm_casa(): nombre de casa comparable entre fuentes.
#
# El error se mide en logit de cada partido frente a todos los demás:
#   z_jk = logit(real_k) - logit(encuesta_jk)
# con k en PP, PSOE, VOX, SUMAR. No se usa el log-ratio frente al resto R
# (el BLOQUE histórico): R incluía a Cs en 2015-2019 (30 % en el 28A, 10,5 %
# en el 23J) y su propio error contaminaba las cuatro componentes (sd 0,14-
# 0,20 en log-ratio frente a 0,09-0,15 en logit). En el 20D y el 26J VOX no
# existía como partido relevante: su componente de z queda sin observar.
#
# (Los scripts de la vieja arquitectura de dos módulos, archivados en
# _archivo_modulos_ab/, usaban además de esto datos_estado(), stan_estado(),
# stan_error() y los priors por defecto de los dos modelos viejos; se
# quitaron al archivarlos.)
# ============================================================================

suppressMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
})

dir_fb <- here::here("2026/10/full-bayesian")

source(here::here("2026/10/ajustar_meta.R"))

cats_estado <- c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "SALF", "BLOQUE")
cats_error <- c("PP", "PSOE", "VOX", "SUMAR")

elecciones_hist <- c("20d", "26j", "28a", "10n", "23j")
fechas_hist <- as.Date(c(
  `20d` = "2015-12-20",
  `26j` = "2016-06-26",
  `28a` = "2019-04-28",
  `10n` = "2019-11-10",
  `23j` = "2023-07-23"
))

# encuestadoras de las encuestas de 2026 que entran en el modelo (era el
# `casas` de datos_estado(), del viejo módulo A)
casas_2026 <- function() {
  sort(unique(prep_grande(f_grande_sep, cats_estado)$encuestadora))
}

# nombre de casa comparable entre fuentes: sin notas [1], sin el tamaño y el
# margen que añade algún anexo, sin el medio que la encarga ("NC
# Report/La Razón", "40dB - El País"), sin tildes ni mayúsculas
norm_casa <- function(x) {
  x |>
    str_remove_all("\\[.*?\\]") |>
    # el anexo del 20D añade "(n)\nMargen de error: ..." al nombre
    str_remove("(?s)\\s*[(\\n].*$") |>
    str_remove("/.*$") |>
    str_remove("\\s+-\\s+.*$") |>
    str_squish() |>
    tolower() |>
    stringi::stri_trans_general("Latin-ASCII") |>
    str_replace("^demoscopia y servicios$", "demoscopia servicios")
}

# ---- generales 2015-2023 -------------------------------------------------------

errores_hist <- function() {
  d <- read_csv(
    here::here("2026/10/errores_encuestas_historicas.csv"),
    show_col_types = FALSE
  ) |>
    mutate(eleccion = factor(eleccion, levels = elecciones_hist))
  sin_vox <- c("20d", "26j")
  enc <- d |>
    mutate(
      VOX = if_else(eleccion %in% sin_vox, NA_real_, VOX),
      R = 100 - PP - PSOE - SUMAR - coalesce(VOX, 0)
    )
  # resultado real de cada elección = encuesta + error (constante dentro de
  # la elección; se comprueba)
  real <- d |>
    transmute(
      eleccion,
      PP = PP + e_PP,
      PSOE = PSOE + e_PSOE,
      VOX = if_else(eleccion %in% sin_vox, NA_real_, VOX + e_VOX),
      SUMAR = SUMAR + e_SUMAR
    )
  real_u <- real |>
    group_by(eleccion) |>
    summarise(across(all_of(cats_error), \(x) mean(x)), .groups = "drop")
  disp <- real |>
    group_by(eleccion) |>
    summarise(across(all_of(cats_error), \(x) diff(range(x))), .groups = "drop")
  stopifnot(all(as.matrix(disp[, -1]) < 0.05, na.rm = TRUE))
  real_u <- real_u |>
    mutate(R = 100 - PP - PSOE - SUMAR - coalesce(VOX, 0))
  rr <- real_u[match(enc$eleccion, real_u$eleccion), ]
  z <- sapply(cats_error, \(k) qlogis(rr[[k]] / 100) - qlogis(enc[[k]] / 100))
  list(encuestas = enc, real = real_u, z = z, casa = norm_casa(enc$casa))
}

# ---- autonómicas 2018-2026 ------------------------------------------------------

# (06_scrapear_autonomicas.R): solo PP, PSOE y VOX; SUMAR queda sin observar
# (sus equivalentes regionales no son comparables). VOX por debajo del 3 %
# real: sin observar (el logit amplifica mucho el error de un partido tan
# pequeño; en 2026 VOX ronda el 18 %)
errores_autonomicas <- function() {
  a <- read_csv(
    file.path(dir_fb, "errores_autonomicas.csv"),
    show_col_types = FALSE
  ) |>
    mutate(z_VOX = if_else(real_VOX < 3, NA_real_, z_VOX))
  z <- cbind(PP = a$z_PP, PSOE = a$z_PSOE, VOX = a$z_VOX, SUMAR = NA_real_)
  list(
    eleccion = a$eleccion,
    dia_e = a$dia_e,
    z = z,
    casa = norm_casa(a$fuente)
  )
}

# ---- las dos fuentes juntas -------------------------------------------------------

# fuera: una general que se deja fuera (validación dejando fuera una
# elección); las autonómicas siempre entran.
# antes_de: solo la información disponible antes de esa fecha (generales y
# autonómicas anteriores), para el backtest temporal de 13_backtest_conjunto.R.
# casas_obj: casas cuyo sesgo medio se calcula (b26); por defecto, las de 2026.
datos_error <- function(
    fuera = NULL,
    con_autonomicas = TRUE,
    antes_de = NULL,
    casas_obj = NULL
) {
  h <- errores_hist()
  keep <- if (is.null(fuera)) {
    rep(TRUE, nrow(h$z))
  } else {
    h$encuestas$eleccion != fuera
  }
  if (!is.null(antes_de)) {
    keep <- keep &
      fechas_hist[as.character(h$encuestas$eleccion)] < as.Date(antes_de)
  }
  z <- h$z[keep, , drop = FALSE]
  el <- as.character(droplevels(h$encuestas$eleccion[keep]))
  casa <- h$casa[keep]
  if (con_autonomicas) {
    a <- errores_autonomicas()
    ka <- if (is.null(antes_de)) {
      rep(TRUE, length(a$eleccion))
    } else {
      a$dia_e < as.Date(antes_de)
    }
    z <- rbind(z, a$z[ka, cats_error])
    el <- c(el, a$eleccion[ka])
    casa <- c(casa, a$casa[ka])
  }
  casas <- sort(unique(casa))
  # casas objetivo (por defecto las de 2026) en el índice de las históricas;
  # 0 = sin historia
  casas26 <- norm_casa(casas_obj %||% casas_2026())
  # las generales primero: tipo 1; autonómicas: tipo 2
  niveles <- unique(el)
  tipo <- if_else(niveles %in% elecciones_hist, 1L, 2L)
  obs <- !is.na(z)
  # elecciones sin ningún partido observado: fuera
  z[!obs] <- 0
  list(
    J = nrow(z),
    D = ncol(z),
    E = length(niveles),
    z = z,
    obs = matrix(as.integer(obs), nrow(z)),
    eleccion = match(el, niveles),
    tipo = tipo,
    H = length(casas),
    casa = match(casa, casas),
    H26 = length(casas26),
    idx26 = coalesce(match(casas26, casas), 0L),
    elecciones = niveles,
    casas = casas,
    casas26 = casas26,
    cats = cats_error
  )
}
