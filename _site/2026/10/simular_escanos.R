# ============================================================================
# SIMULACIÓN DE ESCAÑOS DEL CONGRESO (desde los draws del meta-análisis)
#
# Replica la maquinaria de escanos-andalucia-2026.qmd adaptada al Congreso:
#
#   1. Draws del posterior del modelo 1 (6 categorías) y del modelo 2
#      (composición del BLOQUE, 8 subceldas) en la fecha objetivo. Ambos
#      con ref_time = 2026-07-01. La predicción usa una casa nueva
#      (allow_new_levels) para que el efecto casa entre como incertidumbre.
#   2. Combinación a 13 categorías: 5 grandes + SALF + BLOQUE repartido con
#      la composición del modelo 2 (masa conservada, suman 100).
#   3. Distribución provincial: share_prov = share_nacional x
#      (Dirichlet_23J / share_23J_provincial). El patrón geográfico de
#      referencia es el 23J (ref_provincial_23J_13.csv). SALF no existía en
#      el 23J: su patrón viene de las Europeas 2024. La Dirichlet se centra
#      en los shares provinciales de referencia (concentración = votos
#      totales de la provincia / 1000), así que E[multiplicador] = ratio.
#      En provincias donde un partido no tuvo votos en el 23J (share_23J = 0)
#      el multiplicador es 1: su ratio 0 la anula de todos modos.
#   4. Umbral legal del 3% provincial (LOREG art. 163) antes del D'Hondt.
#      Los votos en blanco cuentan para el umbral legal y aquí no se simulan
#      (~0.5%): el umbral aplicado sobre shares de candidaturas es
#      marginalmente más estricto (3%/1.005). Ceuta y Melilla (1 escaño) no
#      necesitan tratamiento especial: D'Hondt con 1 escaño = mayoría simple.
#   5. D'Hondt provincia a provincia (dhondt), acumulando escaños.
#   6. CAPA DE ERROR DE ENCUESTAS (aquí empírica; en Andalucía era una matriz
#      especificada a mano): cada muestra del posterior recibe un
#      error de encuesta e ~ N(mu, Sigma) sobre los 5 grandes (SALF no tiene
#      historia, sigma = 0) y el BLOQUE absorbe el resto: shares = 100 - suma
#      de los otros. Calibrada con las generales 20D-2015, 26J-2016, 28A-2019,
#      10N-2019 y 23J-2023:
#        - Encuestas individuales: Anexos de Wikipedia de sondeos, campo en la
#          última semana de campaña (14 días para el 26J, campaña corta);
#          ~70 encuestas. Error de cada encuesta = real - encuesta.
#        - Agregado: real - media de la última semana (5 elecciones). mu es esa
#          media (sesgo sistemático del agregado) y sigma su sd muestral.
#        - R (correlaciones): empírica sobre las ~70 encuestas individuales.
#        - Encuentros y saldos: las encuestas infraestiman sistemáticamente al
#          PSOE (+1.8 pp de media) y sobreestiman al PP (+1.0); la correlación
#          PP-VOX es prácticamente nula (-0.10), a diferencia de la -0.5 a mano
#          del post andaluz. El error del BLOQUE = -(suma de los otros), así
#          que la masa se conserva y no hay que modelarlo aparte.
#        - El detalle encuesta a encuesta está congelado en
#          2026/10/errores_encuestas_historicas.csv (par encuesta y errores).
#
# Salidas:
#   simular(n_draws): lista con draws (tibble .draw x 13 partidos), etc.
#   resumen_escanos(sim): mediana + IC por partido
#   prob_mayorias(sim): P(PP>=176), P(PP+VOX>=176), P(izquierda>=176)
#   sim_validacion_23j(): con shares del 23J y ratios de la ref sin ruido
#     debe reproducir los escaños oficiales de los partidos mapeados. Ojo:
#     RESTO (agregado) puede arrastrar 1-2 escaños "fantasma" que en la
#     realidad fueron de PP/PSOE (así era en la calibración del swing).
#
# Uso:
#   source("2026/10/simular_escanos.R")
#   sim <- simular(n_draws = 2000)        # día E (29 nov 2026) por defecto
#   resumen_escanos(sim); prob_mayorias(sim)
#   sim_validacion_23j()
# ============================================================================

library(brms)
library(dplyr)
library(tidyr)
library(readr)
library(tidybayes)
library(gtools)

source(here::here("2026/10/ajustar_meta.R"))

f_ref13 <- here::here("2026/10/ref_provincial_23J_13.csv")
ref13 <- read_csv(f_ref13, show_col_types = FALSE)

fams_bloque <- c("Junts", "ERC", "PNV", "EH Bildu", "BNG", "CC", "UPN", "RESTO")
# en el fit del modelo 2 la subcelda va sin espacio (prep_bloque la renombra)
fams_bloque_fit <- c(
  "Junts",
  "ERC",
  "PNV",
  "EH_Bildu",
  "BNG",
  "CC",
  "UPN",
  "RESTO"
)
fams13 <- c("PP", "PSOE", "VOX", "SUMAR", "SALF", fams_bloque)
stopifnot(setequal(fams13, unique(ref13$partido)))

# ---- capa de error de encuestas (20D-2015 a 23J-2023) ------------------------
# error en puntos porcentuales sobre el share nacional; el BLOQUE absorbe.
# Calibración con 124 encuestas de la última campaña de 5 elecciones
# (errores = real - encuesta; detalle en errores_encuestas_historicas.csv).
# Quedan fuera los sondeos internos de partidos (PSOE, Podemos...) y la
# ventana usa el fin del TRABAJO DE CAMPO (los rangos con en dash de los
# anexos se parsean al último día).
# Solo se conserva lo robusto:
#   - sigma por partido: 1.3-2.6 pp, estable en todas las elecciones;
#   - mu: solo el PSOE (+1.3; positivo en 4 de 5 elecciones, estable sin el
#     26J). PP, VOX y Sumar: mu ~ 0 (cambia de signo según qué elecciones
#     entren);
#   - PP-VOX: errores con el MISMO signo en las 3 elecciones con VOX
#     (correlación +0.92 a nivel de elección). Con n = 3 se encoge a +0.5;
#     el resto de correlaciones es inestable y va a 0.
familias_error <- c("PP", "PSOE", "VOX", "SUMAR", "SALF")
error_mu <- c(PP = 0, PSOE = 1.32, VOX = 0, SUMAR = 0, SALF = 0)
error_sigma <- c(PP = 2.57, PSOE = 1.32, VOX = 1.29, SUMAR = 1.89, SALF = 0)
R_error <- matrix(
  c(
    #          PP    PSOE   VOX   SUMAR  SALF
    1.00,
    0,
    0.50,
    0,
    0, # PP
    0,
    1.00,
    0,
    0,
    0, # PSOE
    0.50,
    0,
    1.00,
    0,
    0, # VOX
    0,
    0,
    0,
    1.00,
    0, # SUMAR
    0,
    0,
    0,
    0,
    1 # SALF (sin error)
  ),
  nrow = 5,
  byrow = TRUE,
  dimnames = list(familias_error, familias_error)
)
Sigma_error <- outer(error_sigma, error_sigma) * R_error
stopifnot(all(eigen(Sigma_error)$values > -1e-8))

prov_ref <- ref13 |>
  distinct(codigo_provincia, provincia, ccaa, diputados) |>
  arrange(codigo_provincia)

# datos por provincia en el ORDEN de fams13 (crítico: shares y Dirichlet
# van en el mismo orden). Nombres con sufijo para no pisar columnas de
# ref13 dentro de summarise (si partido = list(partido[...]) va primero,
# las expresiones siguientes ven la lista, no la columna).
prov_datos <- ref13 |>
  group_by(codigo_provincia) |>
  summarise(
    partido13 = list(partido[match(fams13, partido)]),
    # shares de referencia renormalizados: pct_prov no suma 100 donde SALF
    # (Europeas) se añadió encima del 23J. La Dirichlet se centra en ellos y
    # el multiplicador divide por ellos, así E[mult] = ratio exactamente.
    # (Antes alpha = votos / 1000 mezclaba votos de Europeas con el total del
    # 23J y centraba a SALF en ~0.68 de su ratio.)
    pct23 = list(pct_prov[match(fams13, partido)] / sum(pct_prov)),
    ratio13 = list(ratio[match(fams13, partido)]),
    # suelo de 0.5 solo para partidos con votos: los que no se presentan
    # (ratio 0) van con alpha 0 y no le quitan masa al resto
    alpha13 = list({
      p <- pct_prov[match(fams13, partido)] / sum(pct_prov)
      ifelse(p > 0, pmax(p * sum(votos) / 1000, 0.5), 0)
    }),
    escanos = diputados[1],
    .groups = "drop"
  )
stopifnot(nrow(prov_datos) == 52, sum(prov_datos$escanos) == 350)

# ---- draws de los dos modelos ------------------------------------------------

extraer_shares13 <- function(
  n_draws = 2000,
  time_obj = time_eleccion,
  con_error = TRUE
) {
  mod_g <- ajustar_grande()
  mod_b <- ajustar_bloque()
  d1 <- tibble(time = time_obj, encuestadora = "prediccion_nueva", n = 1) |>
    add_epred_draws(mod_g, ndraws = n_draws, allow_new_levels = TRUE) |>
    ungroup() |>
    select(.draw, .category, .epred) |>
    pivot_wider(names_from = .category, values_from = .epred)
  d2 <- tibble(time = time_obj, bloque = 1) |>
    add_epred_draws(mod_b, ndraws = n_draws) |>
    ungroup() |>
    select(.draw, .category, .epred) |>
    pivot_wider(names_from = .category, values_from = .epred)
  stopifnot(nrow(d1) == nrow(d2))
  # epred del modelo 2: proporciones condicionales dentro del BLOQUE
  bloque_comp <- as.matrix(d2[, fams_bloque_fit])
  shares <- as.matrix(d1[, c("PP", "PSOE", "VOX", "SUMAR", "SALF", "BLOQUE")])
  # epred puede venir en proporciones (suman 1) o en conteos con n = 1:
  # normalizamos a porcentaje
  sc <- if (max(rowSums(shares)) < 1.5) 100 else 1
  shares6 <- shares * sc
  if (con_error) {
    # capa de error: e ~ N(mu, Sigma) sobre los 5 grandes; BLOQUE = resto
    e <- MASS::mvrnorm(n_draws, error_mu, Sigma_error)
    shares6[, 1:5] <- pmax(shares6[, 1:5] + e, 0)
    shares6[, 6] <- 100 - rowSums(shares6[, 1:5])
  }
  out <- cbind(shares6[, 1:5, drop = FALSE], shares6[, 6] * bloque_comp)
  colnames(out) <- fams13
  stopifnot(all(abs(rowSums(out) - 100) < 0.5))
  out
}

# ---- D'Hondt -----------------------------------------------------------------

dhondt <- function(votos, escanos) {
  result <- setNames(numeric(length(votos)), names(votos))
  for (i in seq_len(escanos)) {
    ganador <- which.max(votos / (result + 1))
    result[ganador] <- result[ganador] + 1
  }
  result
}

# ---- simulación --------------------------------------------------------------

simular <- function(
  n_draws = 2000,
  time_obj = time_eleccion,
  con_ruido = TRUE,
  con_error = TRUE,
  umbral = 0.03,
  resto_compite = FALSE
) {
  shares_m <- extraer_shares13(n_draws, time_obj, con_error)
  # Dirichlet pre-generados por provincia (n_draws x 13, orden fams13)
  dir_muestras <- lapply(prov_datos$alpha13, \(a) {
    gtools::rdirichlet(n_draws, a)
  })

  una <- function(k) {
    vapply(
      seq_len(nrow(prov_datos)),
      \(i) {
        pd <- prov_datos[i, ]
        if (con_ruido) {
          mult <- ifelse(
            pd$pct23[[1]] > 0,
            pd$ratio13[[1]] * dir_muestras[[i]][k, ] / pd$pct23[[1]],
            pd$ratio13[[1]]
          )
        } else {
          mult <- ifelse(pd$pct23[[1]] > 0, pd$ratio13[[1]], 1)
        }
        v <- shares_m[k, ] * mult
        v <- v / sum(v)
        v[v < umbral] <- 0
        # RESTO agregado gana escaños fantasma (partidos pequeños que en la
        # realidad compiten por separado y casi siempre quedan bajo el umbral);
        # sus votos se tratan como no representados
        if (!resto_compite) {
          v["RESTO"] <- 0
        }
        dhondt(v, pd$escanos)
      },
      numeric(13)
    ) |>
      t() |>
      colSums()
  }

  esc <- purrr::map_dfr(seq_len(n_draws), \(k) {
    una(k) |>
      t() |>
      as_tibble(.name_repair = "minimal") |>
      setNames(fams13)
  }) |>
    mutate(.draw = seq_len(n_draws), .before = 1)

  list(
    draws = esc,
    time = time_obj,
    n_draws = n_draws,
    con_ruido = con_ruido,
    con_error = con_error
  )
}

# ---- resúmenes ---------------------------------------------------------------

resumen_escanos <- function(sim, width = 0.8) {
  sim$draws |>
    pivot_longer(-.draw, names_to = "partido", values_to = "escanos") |>
    group_by(partido) |>
    tidybayes::median_qi(escanos, .width = width) |>
    arrange(desc(escanos))
}

bloques <- function(d) {
  d |>
    mutate(
      derecha = PP + VOX,
      # sin SALF: partido a caballo, no lo asigno a ningún bloque
      izquierda = PSOE +
        SUMAR +
        ERC +
        Junts +
        `EH Bildu` +
        PNV +
        BNG +
        CC +
        UPN
    )
}

prob_mayorias <- function(sim) {
  d <- bloques(sim$draws)
  tibble::tibble(
    escenario = c("PP solo >= 176", "PP + VOX >= 176", "izquierda >= 176"),
    prob = c(
      mean(d$PP >= 176),
      mean(d$derecha >= 176),
      mean(d$izquierda >= 176)
    )
  )
}

# ---- validación con el 23J ---------------------------------------------------

sim_validacion_23j <- function() {
  nac <- ref13 |>
    group_by(partido) |>
    summarise(votos_nac = sum(votos), .groups = "drop") |>
    arrange(match(partido, fams13))
  shares <- setNames(100 * nac$votos_nac / sum(nac$votos_nac), nac$partido)
  shares["SALF"] <- 0

  esc <- vapply(
    seq_len(nrow(prov_datos)),
    \(i) {
      pd <- prov_datos[i, ]
      v <- shares[fams13] * pd$ratio13[[1]]
      v <- v / sum(v)
      v[v < 0.03] <- 0
      v["RESTO"] <- 0
      dhondt(v, pd$escanos)
    },
    numeric(13)
  ) |>
    t() |>
    colSums()
  setNames(esc, fams13)
}
