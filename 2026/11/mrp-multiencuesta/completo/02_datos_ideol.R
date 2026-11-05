# Modelo completo, parte 1: datos del modelo de ideología (mrp_ideol.stan).
#
# Versión sencilla: una fila por entrevistado con ideología (ideol5 de
# 01_extra.R), sexo, edad, recuerdo y CCAA. Sin estudios, mes ni provincia
# (ver la cabecera de stan/mrp_ideol.stan); como todas las casas traen la
# CCAA, no hace falta mezclar provincias. Quien no da ideología no entra
# (entre el 2,8 % del CIS y el 8,0 % de 40dB).
#
# Lee datos/ (solo lectura) y completo/datos/encuestas_completo.rds; escribe
# completo/datos/datos_ideol.rds.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/02_datos_ideol.R

library(dplyr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_completo <- file.path(dir_proy, "completo", "datos")
source(file.path(dir_proy, "00_partidos.R")) # rec_lv

enc <- readRDS(file.path(dir_completo, "encuestas_completo.rds"))
tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))

dat <- enc |>
  filter(!is.na(ideol5), !is.na(sexo), !is.na(edad), !is.na(rec), !is.na(ccaa))
cat("entrevistados:", nrow(dat), "de", nrow(enc), "\n")
print(table(dat$casa, dat$ideol5))

lv <- list(
  casa = c("CIS", "40dB", "More in Common", "Ateneo del Dato"),
  edad = levels(enc$edad),
  rec = rec_lv,
  ccaa = levels(prov_ccaa$ccaa),
  tramo = levels(enc$ideol5)
)
idx <- function(x, niveles) {
  i <- match(as.character(x), niveles)
  stopifnot(!anyNA(i))
  i
}

# población agregada a sexo x edad x recuerdo x CCAA (lo que usa el modelo)
post <- as.data.frame.table(tabla, responseName = "N") |>
  left_join(prov_ccaa, by = "prov") |>
  group_by(sexo, edad, rec, ccaa) |>
  summarise(N = sum(N), .groups = "drop") |>
  filter(N > 1e-6)

datos <- list(
  s_sd_rec = 2,
  s_sd_main = 0.5,
  s_c = 3,
  s_b_mujer = 1,
  N = nrow(dat),
  T = length(lv$tramo),
  y = idx(dat$ideol5, lv$tramo),
  mujer = as.numeric(dat$sexo == "M"),
  n_casa = length(lv$casa),
  n_edad = length(lv$edad),
  n_rec = length(lv$rec),
  n_ccaa = length(lv$ccaa),
  casa = idx(dat$casa, lv$casa),
  edad = idx(dat$edad, lv$edad),
  rec = idx(dat$rec, lv$rec),
  ccaa = idx(dat$ccaa, lv$ccaa),
  P = nrow(post),
  P_N = post$N,
  P_mujer = as.numeric(post$sexo == "M"),
  P_edad = idx(post$edad, lv$edad),
  P_rec = idx(post$rec, lv$rec),
  P_ccaa = idx(post$ccaa, lv$ccaa)
)

saveRDS(
  list(datos = datos, lv = lv),
  file.path(dir_completo, "datos_ideol.rds")
)
cat("Escrito", file.path(dir_completo, "datos_ideol.rds"), "\n")
