# Modelo completo (PARTICIPACION.md, "La idea general"), paso 1: datos.
#
# Separado del flujo principal: lee datos/encuestas.rds (01_armoniza.R) y
# escribe completo/datos/encuestas_completo.rds. No toca 00_partidos.R ni
# 01_armoniza.R.
#
# Añade a encuestas.rds:
#   1. El voto en blanco como categoría propia, BLANCO, en voto y voto_sim. En
#      la armonización va a OTROS en las cuatro casas (no está en las listas
#      de no válidos). Se recupera de la intención directa del fichero
#      original de cada estudio: mismas filas en el mismo orden (01_armoniza.R
#      no filtra al leer), comprobado con el número de filas, la edad o el
#      sexo y con que todo blanco esté en OTROS. La simpatía no se toca: a
#      quien vota en blanco no se le pregunta.
#      "Escaños en Blanco" (CIS) es un partido, no voto en blanco.
#   2. La ideología en 5 tramos (ideol5). Las escalas no son iguales: el CIS
#      pregunta de 1 a 10 y el resto de 0 a 10.
#        CIS 1-10:  1-2 izq, 3-4 cizq, 5-6 centro, 7-8 cder, 9-10 der
#        0-10:      0-2 izq, 3-4 cizq, 5 centro,   6-7 cder, 8-10 der
#      La casa en el modelo de ideología absorbe lo que quede de diferencia.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/01_extra.R

library(dplyr)
library(haven)
library(readr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_salida <- file.path(dir_proy, "completo", "datos")
dir.create(dir_salida, showWarnings = FALSE, recursive = TRUE)

enc <- readRDS(file.path(dir_proy, "datos", "encuestas.rds"))

# ficheros y variable de intención directa de cada estudio (como 01_armoniza.R)
ficheros <- tribble(
  ~estudio         , ~fichero                                                                               , ~var                ,
  "CIS 3567"       , "descargas/MD3567/3567.sav"                                                            , "INTENCIONG"        ,
  "CIS 3571"       , "2026/10/mrp-raking/datos/MD3571/3571.sav"                                             , "INTENCIONG"        ,
  "CIS 3577"       , "descargas/MD3577/3577.sav"                                                            , "INTENCIONG"        ,
  "CIS 3582"       , "descargas/MD3582/3582.sav"                                                            , "INTENCIONG"        ,
  "40dB jul"       , "descargas/92_Julio_2026/03_Datos_julio_2026.sav"                                      , "p3"                ,
  "40dB sep"       , "2026/10/mrp-raking/datos/2026_09_barometro_descargables/03_Datos_septiembre_2026.sav" , "p3"                ,
  "40dB oct"       , "2026/10/mrp-raking/datos/2026_10_barometro_descargables/03_Datos_octubre_2026.sav"    , "p3"                ,
  "40dB flash oct" , "descargas/2026_10_08_barometro_flash_descargables/03_Datos_octubre_2026_flash.sav"    , "p2"                ,
  "MIC jun"        , "descargas/microdatos-2026-07/microdatos-julio-2026.sav"                               , "intencion_voto_raw",
  "MIC sep"        , "descargas/microdatos-2026-09/microdatos-septiembre-2026.sav"                          , "intencion_voto_raw",
  "Ateneo jun"     , "descargas/atdt-dat-expresidentes-corrupcion.csv"                                      , "intencion"
)
stopifnot(setequal(ficheros$estudio, unique(enc$estudio)))
etiquetas_blanco <- c("En blanco", "Votaría en blanco", "Blanco")

# ---- 1. Voto en blanco -------------------------------------------------------

enc$blanco <- NA
for (i in seq_len(nrow(ficheros))) {
  s <- ficheros$estudio[i]
  f <- here::here(ficheros$fichero[i])
  crudo <- if (grepl("\\.csv$", f)) {
    read_csv(f, col_types = cols(.default = "c"))
  } else {
    read_sav(f)
  }
  txt <- as.character(as_factor(crudo[[ficheros$var[i]]]))
  filas <- which(enc$estudio == s)
  stopifnot(length(filas) == nrow(crudo))
  # misma persona en la misma fila: la edad en años (CIS y 40dB) o el sexo
  if ("EDAD" %in% names(crudo)) {
    stopifnot(all(as.numeric(crudo$EDAD) == enc$edad_n[filas], na.rm = TRUE))
  }
  b <- txt %in% etiquetas_blanco
  # todo blanco tiene que haber ido a OTROS (o a NA si no hay voto)
  stopifnot(all(enc$voto[filas][b] == "OTROS"))
  enc$blanco[filas] <- b
  cat(sprintf("%-15s filas %5d  blanco %4d\n", s, length(filas), sum(b)))
}

partidos_completo <- c(levels(enc$voto), "BLANCO")
enc <- enc |>
  mutate(
    voto = factor(
      if_else(blanco, "BLANCO", as.character(voto)),
      levels = partidos_completo
    ),
    voto_sim = factor(
      if_else(blanco, "BLANCO", as.character(voto_sim)),
      levels = partidos_completo
    )
  )

# ---- 2. Ideología en 5 tramos ----------------------------------------------------

tramos <- c("izq", "cizq", "centro", "cder", "der")
enc <- enc |>
  mutate(
    escala_1_10 = casa == "CIS",
    x = if_else(escala_1_10, round(9 * ideol01 + 1), round(10 * ideol01)),
    ideol5 = case_when(
      is.na(x) ~ NA_character_,
      escala_1_10 & x <= 2 ~ "izq",
      escala_1_10 & x <= 4 ~ "cizq",
      escala_1_10 & x <= 6 ~ "centro",
      escala_1_10 & x <= 8 ~ "cder",
      escala_1_10 ~ "der",
      x <= 2 ~ "izq",
      x <= 4 ~ "cizq",
      x == 5 ~ "centro",
      x <= 7 ~ "cder",
      TRUE ~ "der"
    ),
    ideol5 = factor(ideol5, levels = tramos)
  ) |>
  select(-escala_1_10, -x)

# ---- comprobaciones ---------------------------------------------------------------

cat("\n== % por tramo de ideología y casa ==\n")
print(round(100 * prop.table(table(enc$casa, enc$ideol5, useNA = "ifany"), 1), 1))

cat("\n== % voto en blanco sobre quien da partido (voto_sim), por casa ==\n")
print(
  enc |>
    filter(!is.na(voto_sim)) |>
    group_by(casa) |>
    summarise(
      n = n(),
      blanco = round(100 * mean(voto_sim == "BLANCO"), 1),
      otros = round(100 * mean(voto_sim == "OTROS"), 1)
    ) |>
    as.data.frame(),
  row.names = FALSE
)

saveRDS(enc, file.path(dir_salida, "encuestas_completo.rds"))
cat("\nEscrito", file.path(dir_salida, "encuestas_completo.rds"), "\n")
