# ============================================================================
# PREPARAR DATOS DEL META-ANÁLISIS DE ENCUESTAS GENERALES
#
# Arquitectura en cascada ("bloque conservado"):
#
#   Modelo 1 (grande):  multinomial de 6 categorías
#     PP | PSOE | VOX | SUMAR (= Sumar + Podemos) | SALF | BLOQUE
#     con TODAS las encuestas que reportan los 6 principales.
#     BLOQUE = 100 - (los 5 anteriores).
#
#   Modelo 2 (composición del BLOQUE): multinomial condicional de 8 subceldas
#     Junts | ERC | PNV | EH Bildu | BNG | CC | UPN | RESTO
#     solo con las encuestas completas (las que reportan regionales).
#     RESTO = Otros_residuo (CUP, PACMA, ... partidos sin escaños provinciales;
#     SALF va como categoría grande, no dentro del bloque).
#
#   En la simulación, el draw del BLOQUE del modelo 1 se reparte con la
#   composición del modelo 2 -> 13 valores que suman 100 y masa conservada.
#
# Solo se usan encuestas con trabajo de campo desde fecha_inicio (jun 2025).
#
# Artefactos:
#   1. ref_provincial_23J.csv — 23J por provincia y partido con el ratio
#      pct_prov/pct_nac para distribuir draws entre provincias.
#   2. encuestas_meta_grande.csv — formato largo (encuesta x 6 categorías).
#   3. encuestas_meta_bloque.csv — formato largo (encuesta x 8 subceldas),
#      con bloque_pct y bloque_votos como trials del multinomial condicional.
#
# Uso:
#   source("2026/10/preparar_meta.R")
#   preparar_todo()                  # rehace los 3 CSV
#   preparar_todo(actualizar = TRUE) # 1) actualiza encuestas 2) rehace todo
# ============================================================================

library(readr)
library(dplyr)
library(stringr)
library(tidyr)

f_ref <- here::here("2026/10/ref_provincial_23J.csv")
f_grande <- here::here("2026/10/encuestas_meta_grande.csv")
f_bloque <- here::here("2026/10/encuestas_meta_bloque.csv")
f_enc <- here::here("encuestas_con_partidos_regionales.csv")
f_script <- "descargar_encuestas_ciento76.R"

# Ventanas temporales por modelo:
#  - Modelo 1 (nowcast): solo desde julio 2026. Se espera un cambio de régimen
#    en oct-nov 2026; la data antigua informaría de una carrera que ya no existe.
#  - Modelo 2 (composición del bloque): ventana aún por decidir; de momento
#    toda la que había (jun 2025 en adelante). 84% de las completas son de
#    ElectoPanel, ojo al decidir.
fecha_inicio_bloque <- "2025-06-01"
fecha_inicio_grande <- "2026-07-01"

principales <- c("PP", "PSOE", "VOX", "Sumar", "Podemos", "SALF")
regionales <- c("Junts", "ERC", "PNV", "EH Bildu", "BNG", "CC", "UPN")

# ---- fechas de campo --------------------------------------------------------
# Campo viene como "28 feb–4 mar 26": nos quedamos con el final del trabajo
# de campo, que es la referencia temporal estándar.

parse_fecha_campo <- function(campo) {
  fin <- str_match(campo, "^[^–]*–\\s*(.+)$")[, 2]
  fin[is.na(fin)] <- campo[is.na(fin)]
  meses <- c(
    ene = 1,
    feb = 2,
    mar = 3,
    abr = 4,
    may = 5,
    jun = 6,
    jul = 7,
    ago = 8,
    sep = 9,
    oct = 10,
    nov = 11,
    dic = 12
  )
  m <- str_match(fin, "^([0-9]+)\\s+([a-z]{3})\\s+([0-9]{2})$")
  fecha <- as.Date(rep(NA, length(fin)))
  ok <- !is.na(m[, 1])
  fecha[ok] <- as.Date(
    sprintf("20%s-%02d-%02d", m[ok, 4], meses[m[ok, 3]], as.integer(m[ok, 2]))
  )
  fecha
}

# ---- tabla de referencia provincial del 23J ---------------------------------

construir_ref_provincial <- function(f_salida = f_ref, force = FALSE) {
  if (file.exists(f_salida) && !force) {
    message(
      "ref_provincial ya existe, no la recalculo (force = TRUE para rehacerla)"
    )
    return(invisible(read_csv(f_salida, show_col_types = FALSE)))
  }
  library(infoelectoral)
  # ojo: el mes va con cero ("07"), sin él el paquete construye una URL 404
  res23 <- provincias(tipo_eleccion = "congreso", anno = 2023, mes = "07")
  prov <- res23 |> filter(codigo_provincia != "99") # 99 = totales CCAA/España

  map_partido <- c(
    "000005" = "PP",
    "000002" = "PSOE",
    "000006" = "VOX",
    "000010" = "SUMAR",
    "000050" = "ERC",
    "000057" = "Junts",
    "000071" = "EH Bildu",
    "000075" = "PNV",
    "000065" = "BNG",
    "000031" = "CC",
    "000074" = "UPN"
  )

  prov_base <- prov |>
    distinct(codigo_provincia, provincia = ambito_territorial, n_diputados)
  votos_cand <- prov |> distinct(codigo_provincia, votos_candidaturas)
  votos_fam <- prov |>
    mutate(partido = coalesce(map_partido[codigo_partido_nacional], "OTROS")) |>
    group_by(codigo_provincia, partido) |>
    summarise(votos = sum(votos), .groups = "drop")

  nacional <- votos_fam |>
    group_by(partido) |>
    summarise(votos_nac = sum(votos), .groups = "drop") |>
    mutate(pct_nac = 100 * votos_nac / sum(votos_nac))

  ref <- expand_grid(prov_base, partido = unique(votos_fam$partido)) |>
    left_join(votos_fam, by = c("codigo_provincia", "partido")) |>
    mutate(votos = replace_na(votos, 0)) |>
    left_join(nacional, by = "partido") |>
    left_join(votos_cand, by = "codigo_provincia") |>
    left_join(
      infoelectoral::codigos_provincias |> select(codigo_provincia, ccaa),
      by = "codigo_provincia"
    ) |>
    mutate(
      pct_prov = 100 * votos / votos_candidaturas,
      ratio = pct_prov / pct_nac
    ) |>
    select(
      codigo_provincia,
      provincia,
      ccaa,
      partido,
      votos,
      votos_candidaturas,
      pct_prov,
      pct_nac,
      ratio,
      diputados = n_diputados
    )

  # verificación: los escaños por partido deben reproducir el 23J oficial
  esc <- prov |>
    mutate(partido = coalesce(map_partido[codigo_partido_nacional], "OTROS")) |>
    group_by(partido) |>
    summarise(diputados = sum(diputados), .groups = "drop")
  stopifnot(
    esc$diputados[esc$partido == "PP"] == 137,
    esc$diputados[esc$partido == "PSOE"] == 121,
    esc$diputados[esc$partido == "SUMAR"] == 31,
    sum(esc$diputados) == 350,
    esc$diputados[esc$partido == "OTROS"] == 0
  )

  write_csv(ref, f_salida)
  message("Escrito ", f_salida, " (", nrow(ref), " filas)")
  invisible(ref)
}

# ---- datasets del meta-análisis ---------------------------------------------

leer_encuestas_meta <- function(fecha_min = fecha_inicio_bloque) {
  enc <- read_csv(f_enc, show_col_types = FALSE) |>
    mutate(fecha = parse_fecha_campo(Campo)) |>
    filter(!is.na(fecha), fecha >= fecha_min)
  enc
}

# Modelo 1: todas las encuestas con los 6 principales, 6 categorías
construir_meta_grande <- function(f_salida = f_grande) {
  dat <- leer_encuestas_meta(fecha_inicio_grande) |>
    filter(if_all(all_of(principales), ~ !is.na(.x))) |>
    mutate(
      SUMAR = Sumar + Podemos,
      BLOQUE = 100 - (PP + PSOE + VOX + SUMAR + SALF)
    ) |>
    select(
      encuestadora = Encuestadora,
      campo = Campo,
      fecha,
      muestra = Muestra,
      enlace,
      PP,
      PSOE,
      VOX,
      SUMAR,
      SALF,
      BLOQUE
    )

  stopifnot(all(dat$BLOQUE >= -1e-8))
  meta <- dat |>
    pivot_longer(
      c(PP, PSOE, VOX, SUMAR, SALF, BLOQUE),
      names_to = "categoria",
      values_to = "pct"
    ) |>
    mutate(votos = round(muestra * pct / 100)) |>
    arrange(fecha, encuestadora, categoria)

  write_csv(meta, f_salida)
  cat(
    "Escrito",
    f_salida,
    ":",
    nrow(dat),
    "encuestas x 6 categorías =",
    nrow(meta),
    "filas\n"
  )
  invisible(meta)
}

# Modelo 2: composición del BLOQUE, solo encuestas completas (8 subceldas)
construir_meta_bloque <- function(f_salida = f_bloque) {
  dat <- leer_encuestas_meta() |>
    filter(!is.na(Otros_residuo)) |>
    mutate(
      SUMAR = Sumar + Podemos,
      bloque_pct = 100 - (PP + PSOE + VOX + SUMAR + SALF),
      bloque_votos = round(Muestra * bloque_pct / 100),
      RESTO = Otros_residuo
    ) |>
    select(
      encuestadora = Encuestadora,
      campo = Campo,
      fecha,
      muestra = Muestra,
      enlace,
      bloque_pct,
      bloque_votos,
      all_of(regionales),
      RESTO
    )

  stopifnot(all(dat$bloque_pct >= -1e-8), all(dat$RESTO >= -1e-8))
  meta <- dat |>
    pivot_longer(
      all_of(c(regionales, "RESTO")),
      names_to = "subcela",
      values_to = "pct"
    ) |>
    mutate(votos = round(bloque_votos * pct / 100)) |>
    arrange(fecha, encuestadora, subcela)

  write_csv(meta, f_salida)
  cat(
    "Escrito",
    f_salida,
    ":",
    nrow(dat),
    "encuestas x 8 subceldas =",
    nrow(meta),
    "filas\n"
  )
  invisible(meta)
}

# ---- orquestación -----------------------------------------------------------

cargar_pipeline_encuestas <- function() {
  # evalúa solo la parte del pipeline de descargar_encuestas_ciento76.R
  # (saltándose el bloque exploratorio que re-scrapea la portada)
  lineas <- readLines(f_script)
  ini <- grep("PIPELINE REPRODUCIBLE", lineas)
  eval(
    parse(text = paste(lineas[ini:length(lineas)], collapse = "\n")),
    envir = globalenv()
  )
}

preparar_todo <- function(actualizar = FALSE) {
  if (actualizar) {
    cargar_pipeline_encuestas()
    actualizar_encuestas(solo_nuevas = TRUE)
  }
  construir_ref_provincial()
  construir_meta_grande()
  construir_meta_bloque()
}

if (sys.nframe() == 0) {
  preparar_todo(actualizar = FALSE)
}
