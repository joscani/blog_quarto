# Exporta la tabla de postestratificación (datos/tabla_post_prov.rds) a CSV en
# formato largo: una fila por celda, con los ceros explícitos. Para terceros.
#
#   Rscript 2026/11/mrp-multiencuesta/12_exporta_tabla.R

library(dplyr)
library(readr)

tabla <- readRDS(here::here(
  "2026/11/mrp-multiencuesta/datos/tabla_post_prov.rds"
))

prov_ccaa <- readRDS(here::here(
  "2026/11/mrp-multiencuesta/datos/prov_ccaa.rds"
)) |>
  transmute(prov = as.integer(as.character(prov)), ccaa = as.character(ccaa))

# Nombres de provincia del 29-N (códigos del INE)
nombres_prov <- read_csv(
  here::here("2026/10/coaliciones_izquierda_29N_provincias.csv"),
  show_col_types = FALSE
) |>
  transmute(prov = as.integer(codigo_provincia), provincia)

codigos_ccaa <- read_csv(
  here::here("2026/10/mrp-raking/datos/ccaa_codigos.csv"),
  col_types = cols(.default = "c"),
  show_col_types = FALSE
) |>
  transmute(ccaa, cod_ine_ccaa = cod_ine)

salida <- as.data.frame.table(
  tabla,
  responseName = "personas",
  stringsAsFactors = FALSE
) |>
  mutate(prov = as.integer(prov)) |>
  left_join(prov_ccaa, by = "prov") |>
  left_join(nombres_prov, by = "prov") |>
  left_join(codigos_ccaa, by = "ccaa") |>
  transmute(
    prov,
    provincia,
    ccaa,
    cod_ine_ccaa,
    sexo = recode(sexo, H = "Hombres", M = "Mujeres"),
    edad,
    edu,
    recuerdo = recode(
      rec,
      ABST = "Abstención",
      NO_PODIA = "No podía votar (18-20)"
    ),
    personas = round(personas, 2)
  )

stopifnot(
  nrow(salida) == length(tabla),
  !anyNA(salida),
  # el redondeo a 2 decimales puede desviar como mucho 0,005 por celda
  abs(sum(salida$personas) - sum(tabla)) < nrow(salida) * 0.005
)

write_csv(
  salida,
  here::here(
    "2026/11/mrp-multiencuesta/datos/tabla_postestratificacion_provincia.csv"
  ),
  na = ""
)
