library(rvest)
library(dplyr)
library(stringr)
library(readr)
library(tidyr)
library(purrr)

# ciento76 marca los datos ausentes con un guion (–, — o -); convertir esas
# celdas a NA antes de parsear evita los warnings de parse_number (el NA es
# justo lo que toca para "no hay dato")
sin_guion <- function(x) {
  x <- as.character(x)
  x[grepl("^\\s*[–—-]+\\s*$", x)] <- NA_character_
  x
}

# url <- "https://www.ciento76.es/encuestas"

# html <- read_html(url)

# # Ver qué tablas encuentra
# tables <- html_table(html, fill = TRUE)

# length(tables)

# # Normalmente será la tabla electoral que buscamos
# encuestas <- tables[[1]]

# encuestas

# library(rvest)
# library(dplyr)
# library(janitor)

url <- "https://www.ciento76.es/encuestas"

x <- read_html_live(url)

# Para ver la página mientras depuras
# x$view()

# Ver los botones disponibles
x |>
  html_elements("button") |>
  html_text2()


x$session$Runtime$evaluate(
  expression = "
    [...document.querySelectorAll('button')]
      .find(el => el.innerText.trim() === 'Ver las 476 encuestas')
      .click()
  "
)

Sys.sleep(1)

tables <- x |>
  html_elements("table") |>
  html_table(fill = TRUE)

lapply(tables, dim)
encuestas <- tables[[1]]

# El enlace a la fuente de cada encuesta está en la <a> de la celda de la casa.
# Las filas de la tabla y las que tienen enlace van en el mismo orden (476 == 476).
enlaces <- x |>
  html_elements("table") |>
  (\(t) t[[1]])() |>
  html_elements("tr") |>
  lapply(\(tr) html_elements(tr, "a") |> html_attr("href")) |>
  (\(l) unlist(l[lengths(l) == 1]))()

stopifnot(length(enlaces) == nrow(encuestas))

encuestas <- encuestas |>
  mutate(enlace = enlaces) |>
  mutate(
    across(
      c(Muestra, PP, PSOE, VOX, Sumar, Podemos, SALF),
      ~ parse_number(
        sin_guion(.x),
        locale = locale(decimal_mark = ",")
      )
    )
  )
write_csv(encuestas, "encuestas_electorales.csv")


# ============================================================================
# PIPELINE REPRODUCIBLE
#
# actualizar_encuestas() reconstruye encuestas_con_partidos_regionales.csv
# desde cero: scrapea ciento76, extrae Junts/ERC/PNV/EH Bildu/BNG/CC/UPN y
# Otros de las tablas de electomanía (estáticas o Datawrapper renderizado en
# vivo), y fusiona la extracción LLM de la prensa ya existente.
#
# Por defecto es INCREMENTAL (solo_nuevas = TRUE): conserva los valores ya
# congelados del CSV anterior y solo procesa las encuestas nuevas. Para
# re-scrapear todo (p. ej. el día de publicación):
#
#   actualizar_encuestas(solo_nuevas = FALSE)
#
# La parte de prensa NUEVA (artículos que no estén en prensa_partidos_llm.csv)
# requiere pasada LLM a mano (los subagentes), no es scriptable. Si en algún
# momento se consiguen los regionales de algún artículo (a mano o con LLM),
# basta con editar/rellenar las columnas junts..upn de prensa_partidos_llm.csv
# y volver a ejecutar: el merge rellena huecos sin sobrescribir valores ya
# congelados.
#
# Coste: ~15-20 min en modo completo (160+ renders de Datawrapper);
# segundos en modo incremental sin encuestas nuevas.
# ============================================================================

UA <- paste(
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

partidos_regionales <- c("Junts", "ERC", "PNV", "EH Bildu", "BNG", "CC", "UPN")
patrones_regionales <- c(
  Junts = "^junts$",
  ERC = "^erc$",
  PNV = "^pnv$",
  `EH Bildu` = "^eh[- ]bildu$",
  BNG = "^bng$",
  CC = "^cc[a]?$",
  UPN = "^upn$"
)

scrape_ciento76 <- function() {
  x <- read_html_live("https://www.ciento76.es/encuestas")
  # el botón incluye el número de encuestas, que puede crecer
  expr <- paste0(
    "[...document.querySelectorAll('button')]",
    ".find(el => /Ver las \\d+ encuestas/.test(el.innerText.trim()))",
    ".click()"
  )
  x$session$Runtime$evaluate(expression = expr)
  Sys.sleep(2)
  tabs <- x |> html_elements("table") |> html_table(fill = TRUE)
  enc <- tabs[[1]]
  trs <- x |> html_elements("table") |> (\(t) t[[1]])() |> html_elements("tr")
  hrefs <- lapply(trs, \(tr) html_elements(tr, "a") |> html_attr("href"))
  enlaces <- unlist(hrefs[lengths(hrefs) == 1])
  stopifnot(length(enlaces) == nrow(enc))
  enc <- enc |> mutate(enlace = enlaces)
  enc |>
    # desde el 17 oct 2026 ciento76 llama "Frente Amplio" a la columna de
    # Sumar; el pipeline sigue usando Sumar como nombre interno
    rename(any_of(c(Sumar = "Frente Amplio"))) |>
    mutate(
      across(
        c(Muestra, PP, PSOE, VOX, Sumar, Podemos, SALF),
        ~ parse_number(sin_guion(.x), locale = locale(decimal_mark = ","))
      )
    )
}

dw_tabla_id <- function(url_pagina) {
  # id del gráfico Datawrapper de la tabla de intención de voto
  # (descarta el de "evolución", que es un gráfico de líneas)
  r <- tryCatch(
    request(url_pagina) |>
      req_user_agent(UA) |>
      req_retry(max_tries = 2) |>
      req_perform() |>
      resp_body_string(),
    error = \(e) NULL
  )
  if (is.null(r)) {
    return(NA_character_)
  }
  ifs <- unique(c(
    str_extract_all(r, "<iframe[^>]*>")[[1]],
    str_extract_all(r, "<iframe[^>]*lazy-src[^>]*>")[[1]]
  ))
  x <- str_extract(
    ifs[
      grepl("intención de voto", ifs, ignore.case = TRUE) &
        !grepl("evolución", ifs, ignore.case = TRUE) &
        grepl("dwcdn\\.net", ifs)
    ],
    "dwcdn\\.net/([a-zA-Z0-9]+)"
  )
  if (length(x)) {
    str_match(x[1], "dwcdn\\.net/([a-zA-Z0-9]+)")[, 2]
  } else {
    NA_character_
  }
}

tabla_dw <- function(id, wait = 5) {
  # tabla completa (partido, pct) renderizada del embed de Datawrapper
  tryCatch(
    {
      xl <- read_html_live(sprintf("https://datawrapper.dwcdn.net/%s/1/", id))
      Sys.sleep(wait)
      tabs <- xl |> html_elements("table") |> html_table(fill = TRUE)
      if (length(tabs) == 0) {
        return(NULL)
      }
      tab <- tabs[[which.max(sapply(tabs, nrow))]]
      col_p <- which(names(tab) %in% c("Partido", "partido"))
      col_p <- if (length(col_p)) col_p[1] else 1L
      pct_cols <- sapply(tab, \(c) {
        sum(grepl("^\\s*[0-9]+[.,][0-9]\\s*%", as.character(c)))
      })
      if (max(pct_cols) == 0) {
        return(NULL)
      }
      tibble(
        partido = trimws(as.character(tab[[col_p]])),
        pct = parse_number(
          sin_guion(tab[[which.max(pct_cols)]]),
          locale = locale(decimal_mark = ",")
        )
      )
    },
    error = \(e) NULL
  )
}

tabla_estatica <- function(url_pagina) {
  tryCatch(
    {
      h <- request(url_pagina) |>
        req_user_agent(UA) |>
        req_retry(max_tries = 2) |>
        req_perform() |>
        resp_body_html()
      tabs <- h |> html_elements("table") |> html_table(fill = TRUE)
      if (length(tabs) == 0) {
        return(NULL)
      }
      tab <- tabs[[which.max(sapply(tabs, nrow))]]
      col_p <- which(names(tab) %in% c("Partido", "partido"))
      col_p <- if (length(col_p)) col_p[1] else 1L
      pct_cols <- sapply(tab, \(c) {
        sum(grepl("^\\s*[0-9]+[.,][0-9]\\s*%", as.character(c)))
      })
      if (max(pct_cols) == 0) {
        return(NULL)
      }
      tibble(
        partido = trimws(as.character(tab[[col_p]])),
        pct = parse_number(
          sin_guion(tab[[which.max(pct_cols)]]),
          locale = locale(decimal_mark = ",")
        )
      )
    },
    error = \(e) NULL
  )
}

partidos_de_tabla <- function(tab) {
  # vector con los 7 partidos regionales + Otros (suma de partidos menores)
  # y el flag de página "adelanto" (las que electomanía actualiza en vivo)
  v <- sapply(patrones_regionales, \(p) {
    i <- which(grepl(p, tolower(tab$partido)))
    if (length(i) == 0) {
      return(NA_real_)
    }
    tab$pct[i[1]]
  })
  # principales = los 6 de la tabla de ciento76 + los 7 regionales
  principales <- c(
    "^pp$",
    "^psoe$",
    "^vox$",
    "^sumar",
    "^podemos",
    "^salf",
    unname(patrones_regionales)
  )
  list(
    regionales = setNames(as.numeric(v), names(patrones_regionales)),
    otros = sum(
      tab$pct[
        !grepl(paste(principales, collapse = "|"), tolower(tab$partido)) &
          !is.na(tab$pct)
      ],
      na.rm = TRUE
    )
  )
}

actualizar_encuestas <- function(
  f_salida = "encuestas_con_partidos_regionales.csv",
  f_prensa_llm = "prensa_partidos_llm.csv",
  wait_dw = 5,
  solo_nuevas = TRUE
) {
  enc <- scrape_ciento76()
  dom <- str_remove(
    str_match(enc$enlace, "https?://(?:www\\.)?([^/]+)")[, 2],
    "^www\\."
  )

  # --- valores ya congelados de la ejecución anterior ---
  columnas_prev <- c(
    partidos_regionales,
    "Otros_fuente",
    "Otros_residuo",
    "fecha_extraccion",
    "pagina_volatil",
    # parche manual para principales que la fuente no reporta (p. ej. SALF
    # imputado); el valor fresco del scrape manda salvo que sea NA
    "SALF_manual"
  )
  prev <- NULL
  if (file.exists(f_salida)) {
    prev <- read_csv(f_salida, show_col_types = FALSE)
    prev <- prev[,
      intersect(names(prev), c("enlace", columnas_prev)),
      drop = FALSE
    ]
    # puede haber enlaces repetidos en la tabla de ciento76: sin dedupe,
    # el many-to-one de abajo fallaría en la siguiente pasada incremental
    prev <- prev |>
      group_by(enlace) |>
      summarise(
        across(everything(), \(x) x[which(!is.na(x))[1]]),
        .groups = "drop"
      )
  }
  incremental <- solo_nuevas && !is.null(prev)

  if (!is.null(prev)) {
    enc <- enc |>
      left_join(prev, by = "enlace", relationship = "many-to-one")
    if (!"fecha_extraccion" %in% names(enc)) {
      enc$fecha_extraccion <- NA_character_
    }
  } else {
    for (cl in columnas_prev) {
      enc[[cl]] <- NA
    }
  }
  if (!"SALF_manual" %in% names(enc)) {
    enc$SALF_manual <- NA_real_
  }
  enc$SALF <- coalesce(enc$SALF, enc$SALF_manual)

  # ¿qué enlaces de electomanía hay que procesar?
  idx_el <- which(dom == "electomania.es")
  if (incremental) {
    a_procesar <- idx_el[is.na(enc$fecha_extraccion[idx_el])]
    cat(
      "Modo incremental:",
      length(a_procesar),
      "enlaces de electomanía nuevos,",
      length(idx_el) - length(a_procesar),
      "ya congelados\n"
    )
  } else {
    a_procesar <- idx_el
    cat("Modo completo:", length(a_procesar), "enlaces de electomanía\n")
  }

  # --- electomanía: tabla estática primero, Datawrapper como alternativa ---
  volatile <- setNames(logical(length(idx_el)), enc$enlace[idx_el])
  res <- list()
  for (k in seq_along(a_procesar)) {
    u <- enc$enlace[a_procesar[k]]
    tab <- tabla_estatica(u)
    if (!is.null(tab)) {
      volatile[[u]] <- TRUE
    } else {
      id <- dw_tabla_id(u)
      tab <- if (!is.na(id)) tabla_dw(id, wait = wait_dw) else NULL
    }
    if (!is.null(tab)) {
      res[[u]] <- partidos_de_tabla(tab)
    }
    if (k %% 20 == 0) cat("electomanía:", k, "de", length(a_procesar), "\n")
  }

  hoy <- format(Sys.Date(), "%Y-%m-%d")
  for (u in names(res)) {
    i <- which(enc$enlace == u)
    for (cl in partidos_regionales) {
      enc[[cl]][i] <- res[[u]]$regionales[[cl]]
    }
    enc$Otros_fuente[i] <- round(res[[u]]$otros, 2)
    enc$pagina_volatil[i] <- volatile[[u]]
    enc$fecha_extraccion[i] <- hoy
  }

  # --- prensa: fusionar la extracción LLM previa por enlace ---
  # (ojo: puede haber enlaces repetidos en la tabla de ciento76)
  if (file.exists(f_prensa_llm)) {
    prensa <- read_csv(f_prensa_llm, show_col_types = FALSE) |>
      rename(
        Junts = junts,
        ERC = erc,
        PNV = pnv,
        `EH Bildu` = eh_bildu,
        BNG = bng,
        CC = cc,
        UPN = upn
      ) |>
      select(enlace, all_of(partidos_regionales)) |>
      group_by(enlace) |>
      summarise(
        across(all_of(partidos_regionales), \(x) x[which(!is.na(x))[1]]),
        .groups = "drop"
      )
    idx_pr <- which(!dom %in% c("electomania.es", "cis.es"))
    # coalesce sobre TODAS las filas de prensa: rellena huecos con valores
    # nuevos (LLM o manuales añadidos a f_prensa_llm) sin sobrescribir nunca
    # los congelados. Así, si se consiguen a mano los regionales de un
    # artículo, basta con editarlos ahí y volver a ejecutar.
    for (cl in partidos_regionales) {
      vals <- prensa[[cl]][match(enc$enlace[idx_pr], prensa$enlace)]
      huecos <- is.na(enc[[cl]][idx_pr]) & !is.na(vals)
      enc[[cl]][idx_pr[huecos]] <- vals[huecos]
      # sella la fecha solo donde el merge ha aportado algo nuevo
      enc$fecha_extraccion[idx_pr[huecos]] <- hoy
    }
    incompletos <- idx_pr[rowSums(is.na(enc[idx_pr, partidos_regionales])) > 0]
    if (length(incompletos)) {
      cat(
        "\nATENCIÓN:",
        length(incompletos),
        "encuestas de prensa con regionales incompletos.\n"
      )
      cat(
        "Si consigues los datos a mano, edita las columnas junts..upn en",
        f_prensa_llm,
        "y vuelve a ejecutar.\n"
      )
    }
  } else {
    warning(
      "No encuentro ",
      f_prensa_llm,
      ": la prensa quedará sin datos regionales"
    )
  }

  # --- residuo: solo donde se conoce todo ---
  enc <- enc |>
    mutate(
      suma_conocida = PP +
        PSOE +
        VOX +
        Sumar +
        Podemos +
        SALF +
        rowSums(across(all_of(partidos_regionales))),
      Otros_residuo = ifelse(
        !is.na(suma_conocida),
        round(100 - suma_conocida, 2),
        NA_real_
      ),
      # TRUE si la encuesta trae los 7 regionales (=> tiene dato regional
      # completo y Otros_residuo es calculable; es la que entra al modelo 2)
      regionales_completos = !is.na(suma_conocida)
    ) |>
    select(-suma_conocida)

  write_csv(enc, f_salida)
  cat(
    "\nEscrito",
    f_salida,
    "-",
    nrow(enc),
    "filas |",
    sum(!is.na(enc$Otros_fuente)),
    "con Otros de fuente\n"
  )
  invisible(enc)
}
