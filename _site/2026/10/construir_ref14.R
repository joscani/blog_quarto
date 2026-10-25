# ============================================================================
# CAMBIO 3 — SUMAR Y PODEMOS SEPARADAS: peso provincial y referencia a 14 fams
#
#   1. Europeas 2024 provinciales (infoelectoral): votos de SUMAR (000032,
#      todas las variantes territoriales) y PODEMOS (000025, coalición con
#      Alianza Verde) por provincia. Salida: europeas24_provincias.csv con
#      s_pod = votos_PODEMOS / (SUMAR + PODEMOS), el peso con el que el voto
#      de SUMAR del 23J se reparte entre las dos familias.
#   2. ref_provincial_23J_14.csv: la _13 (11 familias del 23J + SALF de
#      Europeas + RESTO) con la fila de SUMAR partida en SUMAR y PODEMOS
#      según s_pod. La base del % sigue siendo la del 23J (votos_candidaturas
#      del 23J), igual que en la _13: pct_prov para las dos familias nuevas,
#      pct_nac y ratio recalculados; el resto de familias idéntico.
#
# La _13 se conserva intacta. Validaciones:
#   - masa conservada: SUMAR_14 + PODEMOS = SUMAR_13 provincia a provincia;
#   - escaños y códigos de provincia idénticos a la _13;
#   - PODEMOS nacional estimado en un rango razonable (~2,5-4% del 23J).
# ============================================================================

library(infoelectoral)
library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(stringr)

f_europeas <- here::here("2026/10/europeas24_provincias.csv")
f_ref13 <- here::here("2026/10/ref_provincial_23J_13.csv")
f_ref_base <- here::here("2026/10/ref_provincial_23J.csv")
f_ref14 <- here::here("2026/10/ref_provincial_23J_14.csv")

cod_sumar <- "000032"
cod_podemos <- "000025"

# ---- 1. s_pod provincial ------------------------------------------------------

construir_s_pod <- function(force = FALSE) {
  if (file.exists(f_europeas) && !force) {
    message("europeas24_provincias.csv ya existe (force = TRUE para rehacerla)")
    return(invisible(read_csv(f_europeas, show_col_types = FALSE)))
  }
  res <- provincias(tipo_eleccion = "europeas", anno = 2024, mes = "06") |>
    filter(codigo_provincia != "99")
  # infoelectoral::codigos_partidos solo llega a 2019 y los códigos
  # nacionales se reciclan entre elecciones: la validación se hace contra
  # las siglas de la propia descarga. SUMAR (000032) agrupa todas sus
  # variantes territoriales; PODEMOS (000025) es la coalición con
  # Alianza Verde de las europeas 2024
  siglas_s <- res |>
    filter(codigo_partido_nacional == cod_sumar) |>
    distinct(siglas) |>
    pull()
  siglas_p <- res |>
    filter(codigo_partido_nacional == cod_podemos) |>
    distinct(siglas) |>
    pull()
  cat("SUMAR:", paste(siglas_s, collapse = ", "), "\n")
  cat("PODEMOS:", paste(siglas_p, collapse = ", "), "\n")
  stopifnot(
    length(siglas_s) > 0,
    length(siglas_p) > 0,
    all(str_detect(siglas_s, regex("sumar", ignore_case = TRUE))),
    all(str_detect(siglas_p, regex("podemos", ignore_case = TRUE)))
  )

  out <- res |>
    filter(codigo_partido_nacional %in% c(cod_sumar, cod_podemos)) |>
    mutate(
      fam = if_else(codigo_partido_nacional == cod_sumar, "sumar", "podemos")
    ) |>
    group_by(codigo_provincia, provincia = ambito_territorial, fam) |>
    summarise(votos = sum(votos), .groups = "drop") |>
    pivot_wider(names_from = fam, values_from = votos, values_fill = 0) |>
    mutate(
      s_pod = podemos / (sumar + podemos),
      s_pod = replace_na(s_pod, NA)
    )
  # provincias sin suma ni podemos en las europeas (no debería haber: las dos
  # coaliciones compiten en las 52): peso nacional de reserva
  s_pod_nac <- sum(out$podemos) / sum(out$sumar + out$podemos)
  cat(sprintf("s_pod nacional: %.3f\n", s_pod_nac))
  out <- out |> mutate(s_pod = coalesce(s_pod, s_pod_nac))
  stopifnot(nrow(out) == 52, all(!is.na(out$s_pod)))
  stopifnot(sum(out$sumar) > 700000, sum(out$sumar) < 900000) # Sumar ~803 mil

  write_csv(out, f_europeas)
  message("Escrito ", f_europeas, " (", nrow(out), " filas)")
  invisible(out)
}

# ---- 2. referencia a 14 familias ---------------------------------------------

fams14 <- c(
  "PP",
  "PSOE",
  "VOX",
  "SUMAR",
  "PODEMOS",
  "SALF",
  "Junts",
  "ERC",
  "PNV",
  "EH Bildu",
  "BNG",
  "CC",
  "UPN",
  "RESTO"
)

construir_ref14 <- function(force = FALSE) {
  if (file.exists(f_ref14) && !force) {
    message("ref_provincial_23J_14.csv ya existe (force = TRUE para rehacerla)")
    return(invisible(read_csv(f_ref14, show_col_types = FALSE)))
  }
  ref13 <- read_csv(f_ref13, show_col_types = FALSE)
  base <- read_csv(f_ref_base, show_col_types = FALSE) |>
    distinct(codigo_provincia, votos_candidaturas)
  s_pod <- construir_s_pod() |> select(codigo_provincia, s_pod)

  # nacional 23J: para el pct_nac de las dos familias nuevas (la suma
  # de votos_candidaturas de todas las provincias = votos válidos 23J)
  votos_cand_nac <- sum(base$votos_candidaturas)

  ref14 <- ref13 |>
    left_join(base, by = "codigo_provincia") |>
    left_join(s_pod, by = "codigo_provincia") |>
    # partir el SUMAR del 23J: PODEMOS recibe la fracción s_pod
    mutate(
      partido = if_else(partido == "SUMAR", "SUMAR_SPLIT", partido)
    ) |>
    bind_rows(
      ref13 |>
        filter(partido == "SUMAR") |>
        left_join(s_pod, by = "codigo_provincia") |>
        left_join(base, by = "codigo_provincia") |>
        mutate(partido = "PODEMOS", votos = round(votos * s_pod))
    ) |>
    mutate(partido = if_else(partido == "SUMAR_SPLIT", "SUMAR", partido))

  # votos de SUMAR = originales menos lo que se lleva PODEMOS
  ref14 <- ref14 |>
    group_by(codigo_provincia) |>
    mutate(
      v_pod = votos[partido == "PODEMOS"],
      votos = if_else(partido == "SUMAR", votos - v_pod, votos)
    ) |>
    ungroup() |>
    select(-v_pod)

  # pct_nac de las dos familias nuevas: 100 x votos NACIONALES / candidaturas
  # del 23J (misma base que el resto de familias de la _13); el resto conserva
  # su pct_nac de la _13 (idéntico por construcción)
  nac <- ref14 |>
    group_by(partido) |>
    summarise(votos_nac = sum(votos), .groups = "drop")
  ref14 <- ref14 |>
    left_join(nac, by = "partido") |>
    mutate(
      pct_prov = if_else(
        partido %in% c("SUMAR", "PODEMOS"),
        100 * votos / votos_candidaturas,
        pct_prov
      ),
      pct_nac = if_else(
        partido %in% c("SUMAR", "PODEMOS"),
        100 * votos_nac / votos_cand_nac,
        pct_nac
      ),
      ratio = pct_prov / pct_nac
    ) |>
    select(-votos_candidaturas, -s_pod, -votos_nac) |>
    mutate(partido = factor(partido, fams14)) |>
    arrange(codigo_provincia, partido) |>
    mutate(partido = as.character(partido))

  # ---- validaciones ----
  stopifnot(setequal(unique(ref14$partido), fams14))
  stopifnot(nrow(ref14) == nrow(ref13) + 52)
  # masa conservada provincia a provincia (con redondeo de PODEMOS)
  chk <- ref14 |>
    filter(partido %in% c("SUMAR", "PODEMOS")) |>
    group_by(codigo_provincia) |>
    summarise(v14 = sum(votos), .groups = "drop") |>
    left_join(
      ref13 |>
        filter(partido == "SUMAR") |>
        select(codigo_provincia, v13 = votos),
      by = "codigo_provincia"
    )
  stopifnot(all(abs(chk$v14 - chk$v13) <= 1))
  # los escaños y demás columnas intactos
  stopifnot(identical(
    ref14 |> distinct(codigo_provincia, diputados) |> arrange(codigo_provincia),
    ref13 |> distinct(codigo_provincia, diputados) |> arrange(codigo_provincia)
  ))
  # PODEMOS nacional estimado dentro de un rango razonable: las europeas
  # de 2024 le dan ~41% del voto Sumar+Podemos (578 mil frente a 818 mil),
  # aplicado al 12,4% de SUMAR en el 23J da ~5,2%
  pct_pod <- 100 * sum(ref14$votos[ref14$partido == "PODEMOS"]) / votos_cand_nac
  cat(sprintf("PODEMOS nacional estimado sobre 23J: %.2f%%\n", pct_pod))
  stopifnot(pct_pod > 2, pct_pod < 7)
  # el total nacional de la _14 debe igualar al de la _13
  stopifnot(
    abs(
      sum(ref14$votos) - sum(ref13$votos)
    ) <=
      26
  ) # redondeo de 52 provincias

  write_csv(ref14, f_ref14)
  message("Escrito ", f_ref14, " (", nrow(ref14), " filas)")
  invisible(ref14)
}

if (sys.nframe() == 0) {
  construir_s_pod()
  construir_ref14()
}
