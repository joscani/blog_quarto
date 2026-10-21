# ============================================================================
# FULL BAYESIAN 29-N · 06: errores de las encuestas en autonómicas (2018-2026)
#
# Para aprender la correlación del error entre PP, PSOE y VOX hacen falta más
# elecciones con VOX que las 3 generales. Este script scrapea las encuestas
# de las autonómicas desde Andalucía 2018 (en.wikipedia: la página
# "Opinion polling for the ..." si existe; si no, la sección "Opinion polls"
# del artículo de la elección).
#
# Columnas: se detectan por el title del enlace de la cabecera (PP, PSOE,
# Vox y sus federaciones regionales). Si el partido fue en coalición (PP en
# Navarra Suma 2019, PP+Cs en el País Vasco 2020), su columna no se
# encuentra y queda sin observar.
#
# Resultado real: la fila "YYYY regional election" de la propia tabla (misma
# base que las encuestas). Ventana: fin de campo en los 7 días previos; si
# hay menos de 3 encuestas, 14 días. Fuera: electoPanel (panel continuo),
# sondeos de partidos y encuestas a pie de urna (fin de campo el día E).
#
# Error en logit, como en el módulo B: z = logit(real) - logit(encuesta).
#
# Salidas (en 2026/10/full-bayesian/):
#   errores_autonomicas.csv   encuesta x (PP, PSOE, VOX): % y error en logit
#   autonomicas_resumen.csv   elección: fecha, n encuestas, reales, error medio
#   cache_wiki/               HTML descargado (no va a git)
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/06_scrapear_autonomicas.R
# ============================================================================

suppressMessages({
  library(tidyverse)
  library(rvest)
  library(xml2)
})

dir_fb <- here::here("2026/10/full-bayesian")
dir_cache <- file.path(dir_fb, "cache_wiki")
dir.create(dir_cache, showWarnings = FALSE)
ua <- "Mozilla/5.0 (blog muestrear-no-es-pecado; canadasreche@gmail.com)"

autonomicas <- tribble(
  ~anno , ~gentilicio          ,
   2018 , "Andalusian"         ,
   2019 , "Valencian"          ,
   2019 , "Aragonese"          ,
   2019 , "Asturian"           ,
   2019 , "Balearic"           ,
   2019 , "Canarian"           ,
   2019 , "Cantabrian"         ,
   2019 , "Castilian-Leonese"  ,
   2019 , "Castilian-Manchegan",
   2019 , "Extremaduran"       ,
   2019 , "Madrilenian"        ,
   2019 , "Murcian"            ,
   2019 , "Navarrese"          ,
   2019 , "Riojan"             ,
   2020 , "Basque"             ,
   2020 , "Galician"           ,
   2021 , "Catalan"            ,
   2021 , "Madrilenian"        ,
   2022 , "Castilian-Leonese"  ,
   2022 , "Andalusian"         ,
   2023 , "Valencian"          ,
   2023 , "Aragonese"          ,
   2023 , "Asturian"           ,
   2023 , "Balearic"           ,
   2023 , "Canarian"           ,
   2023 , "Cantabrian"         ,
   2023 , "Castilian-Manchegan",
   2023 , "Extremaduran"       ,
   2023 , "Madrilenian"        ,
   2023 , "Murcian"            ,
   2023 , "Navarrese"          ,
   2023 , "Riojan"             ,
   2024 , "Galician"           ,
   2024 , "Basque"             ,
   2024 , "Catalan"            ,
   2025 , "Extremaduran"       ,
   2026 , "Aragonese"          ,
   2026 , "Castilian-Leonese"  ,
   2026 , "Andalusian"
) |>
  mutate(id = paste(anno, gentilicio))

# patrones del title del enlace de cada columna de partido
patron_partido <- c(
  # el PSOE vasco es "PSE–EE (PSOE)", con raya: se aceptan guion y raya
  PP = "^(PP|People's Party( of [A-Za-z ]+)?|PPdeG|PP[-–]A|PPCV|PPC)$",
  PSOE = "^(PSOE|PSC|PSdeG|PSE[-–]EE|PSN|PSPV|PSIB|PSOE[-–]A|Socialist|Socialists' Party of [A-Za-z ]+|Spanish Socialist Workers' Party)",
  VOX = "^Vox"
)
fuente_fuera <- regex(
  "electopanel|^(PSOE|PP|Vox|Podemos|Cs|Sumar)\\b",
  ignore_case = TRUE
)

meses_en <- c(
  jan = 1,
  feb = 2,
  mar = 3,
  apr = 4,
  may = 5,
  jun = 6,
  jul = 7,
  aug = 8,
  sep = 9,
  oct = 10,
  nov = 11,
  dec = 12
)

# "19–24 May 2023", "28 Apr–3 May 2023", "28 May 2023": último día, último
# mes y año de la celda
parse_fecha_en <- function(x) {
  x <- str_squish(x)
  anno <- str_extract(x, "\\d{4}(?!.*\\d{4})")
  mes <- str_extract(tolower(x), "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?!.*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))")
  mes <- meses_en[substr(mes, 1, 3)]
  dias <- str_extract_all(str_remove(x, "\\d{4}"), "(?<!\\d)\\d{1,2}(?!\\d)")[[1]]
  if (is.na(anno) || is.na(mes) || length(dias) == 0) {
    return(as.Date(NA))
  }
  as.Date(sprintf("%s-%02d-%02d", anno, mes, as.integer(dias[length(dias)])))
}

parse_pct <- function(x) {
  num <- str_match(str_squish(x), "^([0-9]+\\.[0-9]+|[0-9]+)")[, 2]
  suppressWarnings(as.numeric(num))
}

descargar <- function(pagina) {
  f <- file.path(dir_cache, paste0(pagina, ".html"))
  if (!file.exists(f)) {
    ok <- tryCatch(
      {
        download.file(
          paste0("https://en.wikipedia.org/wiki/", pagina),
          f,
          quiet = TRUE,
          method = "libcurl",
          headers = c(`User-Agent` = ua)
        )
        TRUE
      },
      error = \(e) FALSE,
      warning = \(w) FALSE
    )
    Sys.sleep(1)
    if (!ok) {
      unlink(f)
      return(NULL)
    }
  }
  read_html(f)
}

# tabla de encuestas: la primera wikitable con columnas de PP o PSOE y Vox,
# en la página de sondeos o tras la sección Opinion polls del artículo
tabla_encuestas <- function(anno, gentilicio) {
  base <- paste0(anno, "_", gentilicio, "_regional_election")
  h <- descargar(paste0("Opinion_polling_for_the_", base))
  tablas <- if (!is.null(h)) {
    html_elements(h, "table.wikitable")
  } else {
    h <- descargar(base)
    if (is.null(h)) {
      return(NULL)
    }
    xml_find_all(
      h,
      "//*[@id='Opinion_polls']/following::table[contains(@class,'wikitable')]"
    )
  }
  for (t in tablas) {
    cab <- html_elements(html_elements(t, "tr")[[1]], "th, td")
    titulos <- map_chr(cab, \(x) {
      a <- html_element(x, "a")
      tt <- if (inherits(a, "xml_missing")) NA else html_attr(a, "title")
      coalesce(tt, str_squish(html_text2(x)))
    })
    cols <- map_int(patron_partido, \(p) {
      i <- which(str_detect(titulos, p))
      if (length(i) == 0) NA_integer_ else i[1]
    })
    if (!is.na(cols[["VOX"]]) && any(!is.na(cols[c("PP", "PSOE")]))) {
      return(list(tabla = t, cols = cols, n_cab = length(cab)))
    }
  }
  NULL
}

scrapear_autonomica <- function(anno, gentilicio) {
  tb <- tabla_encuestas(anno, gentilicio)
  if (is.null(tb)) {
    return(NULL)
  }
  filas <- html_elements(tb$tabla, "tr")
  celdas <- map(filas, \(r) str_squish(html_text2(html_elements(r, "th, td"))))
  celdas <- celdas[map_int(celdas, length) == tb$n_cab]
  d <- map_dfr(celdas, \(f) {
    tibble(
      fuente = f[1],
      fecha = parse_fecha_en(f[2]),
      PP = if (is.na(tb$cols[["PP"]])) NA_real_ else parse_pct(f[tb$cols[["PP"]]]),
      PSOE = if (is.na(tb$cols[["PSOE"]])) NA_real_ else parse_pct(f[tb$cols[["PSOE"]]]),
      VOX = parse_pct(f[tb$cols[["VOX"]]])
    )
  })
  # resultado: la fila "YYYY regional election" del año de la elección
  res <- d |>
    filter(str_detect(fuente, regex(paste0("^", anno, " .*election"), ignore_case = TRUE)))
  if (nrow(res) == 0 || is.na(res$fecha[1])) {
    return(NULL)
  }
  res <- res[1, ]
  enc <- d |>
    filter(
      !str_detect(fuente, regex("election", ignore_case = TRUE)),
      !str_detect(fuente, fuente_fuera),
      !is.na(fecha)
    ) |>
    mutate(dias = as.numeric(res$fecha - fecha)) |>
    # día 0 = encuestas a pie de urna (GfK/FORTA, Sigma Dos/Telemadrid...):
    # no son encuestas preelectorales; con ellas había 35 de 252
    filter(dias >= 1)
  ventana <- if (sum(enc$dias <= 7) >= 3) 7 else 14
  enc <- enc |> filter(dias <= ventana)
  if (nrow(enc) == 0) {
    return(NULL)
  }
  enc |>
    mutate(
      eleccion = paste(anno, gentilicio),
      dia_e = res$fecha,
      ventana = ventana,
      real_PP = res$PP,
      real_PSOE = res$PSOE,
      real_VOX = res$VOX,
      z_PP = qlogis(real_PP / 100) - qlogis(PP / 100),
      z_PSOE = qlogis(real_PSOE / 100) - qlogis(PSOE / 100),
      z_VOX = qlogis(real_VOX / 100) - qlogis(VOX / 100)
    ) |>
    relocate(eleccion, dia_e, ventana)
}

errores <- map2_dfr(autonomicas$anno, autonomicas$gentilicio, \(a, g) {
  out <- tryCatch(scrapear_autonomica(a, g), error = \(e) {
    message(a, " ", g, ": ", conditionMessage(e))
    NULL
  })
  cat(sprintf("%-28s %3d encuestas\n", paste(a, g), if (is.null(out)) 0L else nrow(out)))
  out
})

# encuestas con algún % imposible o con el real sin columna: el z queda NA
errores <- errores |>
  mutate(across(starts_with("z_"), \(z) if_else(is.finite(z), z, NA_real_)))

resumen <- errores |>
  group_by(eleccion, dia_e, ventana) |>
  summarise(
    n_encuestas = n(),
    across(c(real_PP, real_PSOE, real_VOX), first),
    across(c(z_PP, z_PSOE, z_VOX), \(z) mean(z, na.rm = TRUE)),
    .groups = "drop"
  ) |>
  arrange(dia_e)

write_csv(errores, file.path(dir_fb, "errores_autonomicas.csv"))
write_csv(resumen, file.path(dir_fb, "autonomicas_resumen.csv"))
cat("\n", nrow(resumen), "elecciones,", nrow(errores), "encuestas\n")
print(resumen |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)
