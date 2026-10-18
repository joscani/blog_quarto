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
#   4. descargas/objetivo176/encuestas.json — snapshot de
#      https://objetivo176.es/data/encuestas.js (datadicto, @datadicto_es):
#      histórico de encuestas de generales desde ago-2023 con los 7
#      regionales cuando la casa los reporta. Es la fuente de relleno de
#      Junts..UPN (y SALF) donde el scrape y las tablas de electomanía no
#      llegan; sustituye a la extracción LLM de prensa, que queda retirada
#      por coste y fiabilidad. En las encuestas que ambos datasets traen
#      completas los valores coinciden dígito a dígito.
#   5. 2026/10/sobrescribe_objetivo176.csv — overrides curados a mano
#      (enlace, partido, valor): donde hay conflicto con un valor congelado,
#      manda objetivo176. Nació de los 5 conflictos detectados en el primer
#      relleno (algunos venían de la extracción LLM de prensa, retirada).
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
library(jsonlite)

f_ref <- here::here("2026/10/ref_provincial_23J.csv")
f_grande <- here::here("2026/10/encuestas_meta_grande.csv")
# variante del modelo 1 con SUMAR y PODEMOS como categorías separadas
# (7 categorías; escenario "listas" del simulador)
f_grande_sep <- here::here("2026/10/encuestas_meta_grande_sep.csv")
f_bloque <- here::here("2026/10/encuestas_meta_bloque.csv")
f_enc <- here::here("encuestas_con_partidos_regionales.csv")
f_script <- "descargar_encuestas_ciento76.R"
f_o176 <- here::here("descargas/objetivo176/encuestas.json")

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

# Modelo 1: todas las encuestas con los 6 principales, 6 categorías.
# separadas = TRUE: SUMAR y PODEMOS como categorías propias (7 categorías,
# escenario "listas" del simulador). El filtro if_all(principales) ya exige
# las dos columnas (414 de 415 encuestas las traen), así que la muestra no
# cambia.
construir_meta_grande <- function(f_salida = f_grande, separadas = FALSE) {
  dat <- leer_encuestas_meta(fecha_inicio_grande) |>
    filter(if_all(all_of(principales), ~ !is.na(.x)))
  cats <- c("PP", "PSOE", "VOX", "SUMAR", "SALF", "BLOQUE")
  if (separadas) {
    f_salida <- f_grande_sep
    cats <- c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "SALF", "BLOQUE")
    dat <- dat |>
      mutate(
        SUMAR = Sumar,
        PODEMOS = Podemos,
        BLOQUE = 100 - (PP + PSOE + VOX + SUMAR + PODEMOS + SALF)
      )
  } else {
    dat <- dat |>
      mutate(
        SUMAR = Sumar + Podemos,
        BLOQUE = 100 - (PP + PSOE + VOX + SUMAR + SALF)
      )
  }
  dat <- dat |>
    select(
      encuestadora = Encuestadora,
      campo = Campo,
      fecha,
      muestra = Muestra,
      enlace,
      all_of(cats)
    )

  stopifnot(all(dat$BLOQUE >= -1e-8))
  meta <- dat |>
    pivot_longer(
      all_of(cats),
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
    "encuestas x",
    length(cats),
    "categorías =",
    nrow(meta),
    "filas\n"
  )
  invisible(meta)
}

# Modelo 2: composición del BLOQUE, solo encuestas completas (8 subceldas)
construir_meta_bloque <- function(f_salida = f_bloque) {
  dat <- leer_encuestas_meta() |>
    # el CIS queda fuera también aquí (misma razón que en el modelo 1:
    # metodología de recuerdo de voto). Sin el filtro, sus muestras
    # enormes se llevarían ~32% del peso de los trials del multinomial.
    # ElectoPanel se queda: es la casa con más cobertura regional.
    filter(!is.na(Otros_residuo), Encuestadora != "CIS") |>
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

# ---- relleno de regionales con objetivo176 ----------------------------------

actualizar_objetivo176 <- function(f_salida = f_o176) {
  # descarga data/encuestas.js y lo deja como JSON parseable por R.
  # El array va de la primera "[" tras el marcador al último "]" del
  # fichero (después del array solo hay "];").
  f_tmp <- tempfile(fileext = ".js")
  download.file(
    "https://objetivo176.es/data/encuestas.js",
    f_tmp,
    mode = "wb",
    quiet = TRUE
  )
  src <- read_file(f_tmp)
  m <- str_locate(src, "ENCUESTAS_HISTORICO\\s*=\\s*\\[")
  if (is.na(m[, "start"][1])) {
    stop("No encuentro el array ENCUESTAS_HISTORICO en encuestas.js")
  }
  arr <- str_sub(src, m[, "end"][1])
  fin <- max(str_locate_all(arr, "]")[[1]][, "start"])
  arr <- str_sub(arr, 1, fin)
  # El site cambió de formato en oct-2026 y jsonlite (estricto) dejó de
  # tragarlo. Tres arreglos:
  # 1. claves desnudas (PP:32.6; antes iban entrecomilladas) -> entrecomilladas
  arr <- gsub("([{,]\\s*)([A-Za-z][A-Za-z0-9_. ]*?)\\s*:", '\\1"\\2":', arr)
  # 2. comillas simples -> dobles (el fichero no usa apóstrofos dentro de strings)
  arr <- gsub("'([^']*)'", '"\\1"', arr)
  # 3. banners de comentario dentro del array (// ══ 2026 ══) y coma final
  #    antes del ]: solo líneas que empiezan por // (un // dentro de un
  #    string nunca va a principio de línea)
  lineas_arr <- str_split(arr, "\n")[[1]]
  arr <- paste(
    lineas_arr[!str_detect(str_trim(lineas_arr), "^//")],
    collapse = "\n"
  )
  arr <- gsub(",\\s*\\]", "]", arr)
  n <- length(fromJSON(arr, simplifyVector = FALSE))
  stopifnot(n > 100)
  write_lines(arr, f_salida)
  message("objetivo176 actualizado: ", n, " encuestas → ", f_salida)
  invisible(f_salida)
}

rellenar_regionales_o176 <- function(
  f_in = f_enc,
  f_json = f_o176,
  tol_dias = 2
) {
  # rellena Junts..UPN (y SALF vía SALF_manual) donde falten, emparejando
  # por casa y fin de campo (±tol_días). Nunca sobrescribe valores
  # congelados: coalesce sobre los huecos, igual que el merge de prensa.
  # Los valores escritos aquí quedan congelados en las siguientes pasadas
  # incrementales de actualizar_encuestas() (vían columnas_prev).
  # Excepción: sobrescribe_objetivo176.csv — overrides curados a mano que
  # SÍ sobrescriben (política: en conflicto, manda objetivo176).
  o <- fromJSON(f_json, simplifyVector = FALSE)
  o_df <- bind_rows(lapply(o, \(e) {
    p <- e$partidos %||% list()
    tibble::tibble(
      casa = str_remove(e$firma, "/.*"),
      fecha = as.Date(e$fecha_hasta, format = "%d/%m/%Y"),
      Junts = p$JUNTS %||% NA_real_,
      ERC = p$ERC %||% NA_real_,
      PNV = p$PNV %||% NA_real_,
      `EH Bildu` = p$`EH BILDU` %||% NA_real_,
      BNG = p$BNG %||% NA_real_,
      CC = p$CC %||% NA_real_,
      UPN = p$UPN %||% NA_real_,
      SALF = p$SALF %||% NA_real_
    )
  }))

  enc <- read_csv(f_in, show_col_types = FALSE) |>
    mutate(fecha_fin = parse_fecha_campo(Campo))
  if (!"SALF_manual" %in% names(enc)) {
    enc$SALF_manual <- NA_real_
  }

  n_celdas <- 0
  n_salf <- 0
  rellenos <- c()
  discrepancias <- c()
  for (i in seq_len(nrow(enc))) {
    if (is.na(enc$fecha_fin[i])) {
      next
    }
    hueco_reg <- anyNA(unlist(enc[i, regionales]))
    hueco_salf <- is.na(enc$SALF[i]) & is.na(enc$SALF_manual[i])
    if (!hueco_reg && !hueco_salf) {
      next
    }

    nom <- tolower(enc$Encuestadora[i])
    cand <- o_df |>
      filter(
        abs(as.numeric(fecha - enc$fecha_fin[i])) <= tol_dias,
        str_detect(nom, fixed(tolower(casa))) |
          str_detect(tolower(casa), fixed(nom))
      )
    if (nrow(cand) == 0) {
      next
    }
    if (nrow(cand) > 1) {
      cat(
        "AVISO: casa ambigua para",
        enc$Encuestadora[i],
        format(enc$fecha_fin[i]),
        "- no relleno\n"
      )
      next
    }
    cand <- cand[1, ]

    # control: donde ya hay valor congelado, o176 debería coincidir
    for (cl in c(regionales, "SALF")) {
      v <- enc[[cl]][i]
      vo <- cand[[cl]][1]
      if (!is.na(v) && !is.na(vo) && abs(v - vo) > 0.05) {
        discrepancias <- c(
          discrepancias,
          sprintf(
            "%s %s · %s: nuestro %.1f vs objetivo176 %.1f",
            enc$Encuestadora[i],
            enc$Campo[i],
            cl,
            v,
            vo
          )
        )
      }
    }

    tocadas <- 0
    for (cl in regionales) {
      if (is.na(enc[[cl]][i]) && !is.na(cand[[cl]][1])) {
        enc[[cl]][i] <- cand[[cl]][1]
        tocadas <- tocadas + 1
      }
    }
    if (tocadas > 0) {
      n_celdas <- n_celdas + tocadas
    }
    if (hueco_salf && !is.na(cand$SALF[1])) {
      enc$SALF_manual[i] <- cand$SALF[1]
      n_salf <- n_salf + 1
    }
    if (tocadas > 0 || (hueco_salf && !is.na(cand$SALF[1]))) {
      enc$fecha_extraccion[i] <- format(Sys.Date(), "%Y-%m-%d")
      rellenos <- c(
        rellenos,
        sprintf(
          "%s (%s): %d regionales%s",
          enc$Encuestadora[i],
          enc$Campo[i],
          tocadas,
          ifelse(hueco_salf && !is.na(cand$SALF[1]), " + SALF", "")
        )
      )
    }
  }

  # overrides curados a mano (sobrescribe_objetivo176.csv): mandan sobre
  # los valores congelados. Nota: SALF no es congelada (el scrape la trae
  # fresca en cada pasada de actualizar_encuestas), así que el override de
  # SALF se reaplica aquí en cada pasada, siempre antes de construir los
  # metas.
  f_over <- here::here("2026/10/sobrescribe_objetivo176.csv")
  if (file.exists(f_over)) {
    ov <- read_csv(f_over, show_col_types = FALSE)
    stopifnot(all(ov$partido %in% c(regionales, "SALF")))
    n_ok <- 0
    for (k in seq_len(nrow(ov))) {
      i <- match(ov$enlace[k], enc$enlace)
      if (is.na(i)) {
        cat("AVISO: override sin fila en el CSV:", ov$enlace[k], "\n")
        next
      }
      enc[[ov$partido[k]]][i] <- ov$valor[k]
      n_ok <- n_ok + 1
    }
    cat("overrides aplicados:", n_ok, "de", nrow(ov), "celdas\n")
  }

  # recomputar el residuo y el flag de completos con los mismos criterios
  # que actualizar_encuestas()
  enc <- enc |>
    mutate(
      SALF = coalesce(SALF, SALF_manual),
      suma_conocida = PP +
        PSOE +
        VOX +
        Sumar +
        Podemos +
        SALF +
        rowSums(across(all_of(regionales))),
      Otros_residuo = ifelse(
        !is.na(suma_conocida),
        round(100 - suma_conocida, 2),
        NA_real_
      ),
      regionales_completos = !is.na(suma_conocida)
    ) |>
    select(-suma_conocida, -fecha_fin)

  write_csv(enc, f_in)
  cat(
    "objetivo176:",
    length(rellenos),
    "encuestas tocadas (",
    n_celdas,
    "celdas regionales,",
    n_salf,
    "SALF imputada a SALF_manual)\n"
  )
  if (length(rellenos)) {
    cat(paste0("  - ", rellenos, "\n"), sep = "")
  }
  if (length(discrepancias)) {
    cat(
      "DISCREPANCIAS con valores congelados (no sobrescritos; las de",
      "sobrescribe_objetivo176.csv se corrigen abajo):\n"
    )
    cat(paste0("  - ", discrepancias, "\n"), sep = "")
  }
  invisible(enc)
}

# ---- orquestación -----------------------------------------------------------

cargar_pipeline_encuestas <- function() {
  # evalúa solo la parte del pipeline de descargar_encuestas_ciento76.R
  # (saltándose el bloque exploratorio que re-scrapea la portada)
  lineas <- readLines(f_script)
  ini <- grep("PIPELINE REPRODUCIBLE", lineas)
  # cabecera (library() y sin_guion()) hasta el bloque exploratorio, que
  # empieza en `url <- ...`: sin ella, en una sesión limpia (Rscript) faltan
  # read_html_live y sin_guion
  fin_cab <- grep("^url <- ", lineas)[1] - 1
  stopifnot(!is.na(fin_cab), fin_cab < ini)
  eval(
    parse(
      text = paste(c(lineas[1:fin_cab], lineas[ini:length(lineas)]), collapse = "\n")
    ),
    envir = globalenv()
  )
}

preparar_todo <- function(actualizar = FALSE) {
  if (actualizar) {
    cargar_pipeline_encuestas()
    actualizar_encuestas(solo_nuevas = TRUE)
    actualizar_objetivo176()
  }
  rellenar_regionales_o176()
  construir_ref_provincial()
  construir_meta_grande()
  # 7 categorías (SUMAR y PODEMOS separadas): entrada de ajustar_grande_sep()
  # para los escenarios listas = "separadas" y "mapa" de simular()
  construir_meta_grande(separadas = TRUE)
  construir_meta_bloque()
}

if (sys.nframe() == 0) {
  preparar_todo(actualizar = FALSE)
}
