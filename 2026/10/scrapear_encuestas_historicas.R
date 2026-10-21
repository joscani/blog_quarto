# ============================================================================
# SCRAPEO DE ENCUESTAS PRE-CAMPAÑA (Wikipedia) Y FACTOR DE ANTELACIÓN
#
# errores_encuestas_historicas.csv solo conserva la última semana de campo de
# cada elección (es la ventana con la que está calibrada la capa de error de
# simular_escanos.R). Para calibrar cuánto crece el error con la distancia al
# antes del día E (el factor de antelación que ciento76 estima en 1,4 a 49 días) hacen
# falta encuestas más antiguas: este script scrapea los Anexos de sondeos de
# Wikipedia (es.wikipedia para 20D/28A/10N; en.wikipedia para 26J y 23J: el
# anexo español de 26J no trae la fila de resultados de 2016 y el del 23J no
# trae fecha de trabajo de campo) desde ~75 días antes del día E.
#
# Por elección se scrapea UNA tabla (la principal de intención de voto, a
# pelo: los anexos tienen estructura distinta entre elecciones, con las
# cabeceras de partido a veces como texto y a veces como logo, que se
# identifica por el filename del fichero del logo en el src).
#
# El error se calcula contra la fila de RESULTADOS de la propia tabla: así el
# % de la encuesta y el real comparten base (votos válidos con blancos, las
# diferencias con la base de errores_encuestas_historicas.csv están por
# debajo de ~0,4 pp y no afectan al cociente de sigmas). La familia SUMAR se
# scrapea como la COALICIÓN principal de la época (Podemos+IU-UP en 20D, el
# bloque Podemos en 26J, Unidas Podemos en 28A/10N, Sumar en 23J): NO coincide
# exactamente con el mapeo de backtest_pipeline.R (que suma Más País,
# Compromís... por código nacional), así que la familia SUMAR queda FUERA de
# la estimación del factor, que se apoya en PP, PSOE y VOX (mapeo limpio).
#
# Se excluyen los sondeos internos de partidos (PSOE, PP, Podemos...) y
# electoPanel (panel continuo, igual que en el modelo de producción).
#
# Salidas:
#   2026/10/errores_encuestas_ventana.csv  — encuesta a encuesta, con dias =
#     días hasta el día E y el error e_* = real - encuesta
#   2026/10/factor_antelacion.csv          — sigma por ventana de antelación
#     y el factor f = sigma(bin 35-56 d) / sigma(bin 0-7 d), por familia y
#     agregado (media de varianzas entre elecciones, sd de la mezcla)
# ============================================================================

library(rvest)
library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(stringr)
library(here)

# ---- configuración por elección ----------------------------------------------

# columnas (1-indexadas) de la tabla principal de cada anexo; SUMAR es la
# coalición principal (ver cabecera del script)
config_elec <- tribble(
  ~id   , ~fecha_eleccion , ~url                                                                                                                                      , ~wiki , ~col_fuente , ~col_fecha , ~col_n , ~col_pp , ~col_psoe , ~col_vox , ~cols_sumar ,
  "20d" , "2015-12-20"    , "https://es.wikipedia.org/wiki/Anexo:Sondeos_de_intenci%C3%B3n_de_voto_para_las_elecciones_generales_de_Espa%C3%B1a_de_2015"              , "es"  ,           1 ,          2 , NA     ,       3 ,         4 , NA       , c(5, 16)    ,
  "26j" , "2016-06-26"    , "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2016_Spanish_general_election"                                                     , "en"  ,           1 ,          2 ,      3 ,       5 ,         6 , NA       ,          16 ,
  "28a" , "2019-04-28"    , "https://es.wikipedia.org/wiki/Anexo:Sondeos_de_intenci%C3%B3n_de_voto_para_las_elecciones_generales_de_Espa%C3%B1a_de_abril_de_2019"     , "es"  ,           1 ,          2 ,      3 ,       5 ,         6 ,       15 ,           7 ,
  "10n" , "2019-11-10"    , "https://es.wikipedia.org/wiki/Anexo:Sondeos_de_intenci%C3%B3n_de_voto_para_las_elecciones_generales_de_Espa%C3%B1a_de_noviembre_de_2019" , "es"  ,           1 ,          2 ,      3 ,       6 ,         5 ,        7 ,           8 ,
  "23j" , "2023-07-23"    , "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2023_Spanish_general_election"                                                     , "en"  ,           1 ,          2 ,      3 ,       6 ,         5 ,        7 ,           8
)
ventana_max_dias <- 75

# sondeos internos de partidos y panel continuo: fuera, igual que en la
# calibración de producción
fuente_partido <- regex(
  "^(PSOE|PP|Podemos|Vox|VOX|Ciudadanos|C's|UP|Más País|Unidas Podemos|Sumar)($|[/\\[( ]) ",
  ignore_case = TRUE
)

# ---- helpers -----------------------------------------------------------------

# pct en la celda: primer número de la celda, haya % o no: "26,6%\n123" (es) /
# "28.5\n117/121" (en); "–", "-", "", "?", "[a]", "N/a" -> NA. El número tras
# el salto son escaños (o rango).
parse_pct <- function(x) {
  num <- str_match(str_trim(x), "^([0-9]+[.,][0-9]+|[0-9]+)")[, 2]
  suppressWarnings(as.numeric(str_replace(num, ",", ".")))
}

# fecha de trabajo de campo: "19 Dic 2015", "25–27 Abr 2019",
# "22 Feb–25 Abr 2019", "13–22 Jul 2023" (en.wiki). En rangos vale el ÚLTIMO
# día con el mes/año del último token de mes.
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
parse_fecha <- function(x, anno) {
  # pares día-mes (el mes puede faltar si el rango comparte mes: "25–27 Abr").
  # El (?![0-9]) evita que "2016" aporte tokens ("20" y "16"): el último
  # número debe ser el día, no un trozo del año.
  if (length(x) != 1 || is.na(x)) {
    return(rep(as.Date(NA), max(length(x), 1L)))
  }
  # regmatches() espera el resultado de gregexpr sin des-listar
  tokens <- regmatches(
    x,
    gregexpr(
      "(?<![0-9])[0-9]{1,2}\\s*[A-Za-zÁ-ú]{0,3}(?![0-9])",
      x,
      perl = TRUE
    )
  )[[1]]
  if (length(tokens) == 0) {
    return(as.Date(NA))
  }
  # el último token con letras trae el mes; los días en solitario comparten
  # ese mes solo si van antes... regla simple: último día = último número,
  # último mes = último token con letras
  dia <- as.numeric(str_extract(str_trim(tokens[length(tokens)]), "^\\d{1,2}"))
  con_letras <- tokens[str_detect(tokens, "[A-Za-z]")]
  mes_token <- tolower(str_extract(
    con_letras[length(con_letras)],
    "[A-Za-zÁ-ú]+"
  ))
  mes <- meses[mes_token][1]
  out <- rep(as.Date(NA), length(x))
  ok <- !is.na(dia) & !is.na(mes)
  out[ok] <- as.Date(sprintf("%04d-%02d-%02d", anno, mes[ok], dia[ok]))
  out
}

# ---- scrapeo de una elección -------------------------------------------------

scrapear_eleccion <- function(cfg) {
  anno <- as.integer(substr(cfg$fecha_eleccion, 1, 4))
  h <- read_html(cfg$url)
  t <- html_elements(h, "table.wikitable")[[1]]
  filas <- html_elements(t, "tr")
  celdas <- map(filas, \(r) html_text2(html_elements(r, "th, td")) |> trimws())

  # cabecera: primera fila con >= 3 celdas no vacías
  n_celdas <- map_int(celdas, length)
  i_hdr <- which(n_celdas >= 10)[1]
  stopifnot(!is.na(i_hdr))

  # fila de resultados: fuente con "Resultado", "Elecciones generales" o
  # "general election"
  es_resultado <- map_lgl(
    celdas,
    \(x) {
      length(x) > 0 &&
        str_detect(
          x[1],
          regex(
            "resultado|elecciones generales|general election",
            ignore_case = TRUE
          )
        )
    }
  )
  i_res <- which(es_resultado & seq_along(celdas) > i_hdr)[1]
  stopifnot(!is.na(i_res))
  res <- celdas[[i_res]]

  get_val <- function(fila, col) {
    if (is.na(col) || length(fila) < col) {
      return(NA_real_)
    }
    parse_pct(fila[col])
  }
  # cols_sumar llega como list-column de tribble(); se desenvuelve
  cols_s <- unlist(cfg$cols_sumar)

  real <- c(
    PP = get_val(res, cfg$col_pp),
    PSOE = get_val(res, cfg$col_psoe),
    VOX = get_val(res, cfg$col_vox),
    SUMAR = sum(sapply(cols_s, \(c) get_val(res, c)), na.rm = TRUE)
  )
  # VOX no existe en 20D ni 26J (col_vox = NA): el NA de su real es correcto
  stopifnot(
    all(!is.na(real[c("PP", "PSOE", "SUMAR")])),
    is.na(cfg$col_vox) || !is.na(real[["VOX"]]),
    sum(real, na.rm = TRUE) < 100
  )

  filas_dat <- seq_along(celdas)
  filas_dat <- filas_dat[
    filas_dat > i_hdr &
      !es_resultado &
      n_celdas >= max(c(cols_s, cfg$col_psoe, cfg$col_pp), na.rm = TRUE)
  ]

  datos <- map_dfr(filas_dat, \(i) {
    f <- celdas[[i]]
    casa <- f[cfg$col_fuente]
    fecha <- parse_fecha(f[cfg$col_fecha], anno)
    tibble(
      eleccion = cfg$id,
      casa = casa,
      fecha = fecha[1],
      n = if (is.na(cfg$col_n)) {
        NA_real_
      } else {
        suppressWarnings(as.numeric(str_remove_all(f[cfg$col_n], "[.,]")))
      },
      PP = get_val(f, cfg$col_pp),
      PSOE = get_val(f, cfg$col_psoe),
      VOX = get_val(f, cfg$col_vox),
      SUMAR = sum(sapply(cols_s, \(c) get_val(f, c)), na.rm = TRUE)
    )
  })

  # limpieza: sin fecha o sin ningún % -> fuera; sondeos internos y
  # electoPanel -> fuera; ventana de 75 días -> fuera
  datos <- datos |>
    filter(!is.na(fecha)) |>
    filter(!is.na(PP) | !is.na(PSOE) | !is.na(VOX) | !is.na(SUMAR)) |>
    filter(!str_detect(casa, fuente_partido)) |>
    filter(!str_detect(casa, regex("electopanel", ignore_case = TRUE))) |>
    mutate(
      dia_e = as.Date(cfg$fecha_eleccion),
      dias = as.numeric(dia_e - fecha)
    ) |>
    filter(dias >= 0, dias <= ventana_max_dias)

  for (f in c("PP", "PSOE", "VOX", "SUMAR")) {
    datos[[paste0("e_", f)]] <- real[[f]] - datos[[f]]
  }
  attr(datos, "real") <- real
  datos
}

# ---- ejecución ---------------------------------------------------------------

calibrar_antelacion <- function() {
  enc_ventana <- setNames(
    lapply(seq_len(nrow(config_elec)), \(i) {
      scrapear_eleccion(config_elec[i, ])
    }),
    config_elec$id
  )

  # validación: encuestas por elección y ventana, y comprobación de que los
  # reales scrapeados cuadran con los de errores_encuestas_historicas.csv
  err_h <- read_csv(
    here::here("2026/10/errores_encuestas_historicas.csv"),
    show_col_types = FALSE
  )
  mapa_id <- c(
    `20D-2015` = "20d",
    `26J-2016` = "26j",
    `28A-2019` = "28a",
    `10N-2019` = "10n",
    `23J-2023` = "23j"
  )
  cat("\n== encuestas scrapeadas por elección ==\n")
  print(map_int(enc_ventana, nrow))

  cat("\n== reales scrapeados vs reales del CSV de la última semana ==\n")
  for (id in names(enc_ventana)) {
    real_w <- attr(enc_ventana[[id]], "real")
    ev <- names(mapa_id)[mapa_id == id]
    fila_h <- err_h |> filter(eleccion == ev) |> slice(1)
    for (f in c("PP", "PSOE", "VOX", "SUMAR")) {
      real_h <- fila_h[[f]] + fila_h[[paste0("e_", f)]]
      cat(sprintf(
        "%s %-6s scrapeado %6.2f | histórico %6.2f | dif %+.2f\n",
        id,
        f,
        real_w[[f]],
        real_h,
        real_w[[f]] - real_h
      ))
    }
  }

  cat("\n== distribución de dias ==\n")
  bind_rows(enc_ventana) |>
    mutate(
      bin = cut(
        dias,
        c(0, 7, 21, 35, 56, 75),
        labels = c("0-7", "8-21", "22-35", "36-56", "57-75"),
        right = TRUE,
        include.lowest = TRUE
      )
    ) |>
    count(eleccion, bin) |>
    pivot_wider(names_from = bin, values_from = n)

  out <- bind_rows(enc_ventana) |> arrange(eleccion, desc(dias))
  write_csv(out, here::here("2026/10/errores_encuestas_ventana.csv"))
  cat("\nGuardado:", here::here("2026/10/errores_encuestas_ventana.csv"), "\n")

  # ---- factor de antelación ----------------------------------------------------
  # sigma por ventana y familia: en cada elección, sd de los errores de la
  # ventana (encuestas individuales); entre elecciones, media de las
  # varianzas (cada elección pesa igual, al contrario que en el sd global,
  # donde el 23J con ~60 encuestas dominaría). Factor = sd(36-56) / sd(0-7).
  df <- out |>
    pivot_longer(
      c(e_PP, e_PSOE, e_VOX, e_SUMAR),
      names_to = "familia",
      values_to = "e"
    ) |>
    mutate(familia = str_remove(familia, "e_")) |>
    filter(!is.na(e)) |>
    mutate(
      bin = cut(
        dias,
        c(0, 7, 21, 35, 56, 75),
        labels = c("0-7", "8-21", "22-35", "36-56", "57-75"),
        right = TRUE,
        include.lowest = TRUE
      )
    )

  # SUMAR fuera del factor (mapeo de coalición, ver cabecera)
  fac <- df |>
    filter(familia != "SUMAR") |>
    group_by(eleccion, familia, bin) |>
    summarise(v = var(e), n = n(), .groups = "drop")

  sigma_bin <- fac |>
    group_by(familia, bin) |>
    summarise(
      sigma = sqrt(mean(v[!is.na(v)])),
      n_ele = sum(!is.na(v)),
      .groups = "drop"
    ) |>
    filter(n_ele >= 2)

  factor_fam <- sigma_bin |>
    select(familia, bin, sigma) |>
    pivot_wider(names_from = bin, values_from = sigma) |>
    mutate(factor = `36-56` / `0-7`)

  # agregado: factor de la mezcla (todas las familias y elecciones en la
  # ventana, con la misma regla de media de varianzas)
  fac_mix <- fac |>
    group_by(bin) |>
    summarise(
      sigma = sqrt(mean(v[!is.na(v)])),
      n_obs = sum(!is.na(v)),
      .groups = "drop"
    ) |>
    filter(n_obs >= 2)
  factor_mix <- fac_mix |>
    select(bin, sigma) |>
    pivot_wider(names_from = bin, values_from = sigma) |>
    mutate(familia = "MIX", factor = `36-56` / `0-7`) |>
    select(familia, `0-7`, `8-21`, `22-35`, `36-56`, `57-75`, factor)

  cat("\n== sigma por ventana (fam x bin) ==\n")
  print(factor_fam, n = Inf)
  cat("\n== factor agregado ==\n")
  print(factor_mix)

  write_csv(
    bind_rows(factor_fam, factor_mix),
    here::here("2026/10/factor_antelacion.csv")
  )
  cat("\nGuardado:", here::here("2026/10/factor_antelacion.csv"), "\n")
  invisible(list(encuestas = out, factor = bind_rows(factor_fam, factor_mix)))
}
