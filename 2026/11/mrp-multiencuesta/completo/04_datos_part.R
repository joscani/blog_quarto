# Modelo completo, parte 2: datos del modelo de participación (mrp_part3.stan).
#
# Una fila por entrevistado con probabilidad de votar (CIS, 40dB, More in
# Common; Ateneo no la pregunta) e ideología (ideol5 de 01_extra.R). Nivel:
#   1 = improbable (0-4), 2 = probable (5-9), 3 = seguro (10).
# Quien no da ideología no entra: el modelo de ideología tampoco lo cuenta, y
# así la cadena ideología -> participación -> voto usa la misma población.
#
# Lee completo/datos/encuestas_completo.rds y datos_ideol.rds (los niveles);
# escribe completo/datos/datos_part.rds.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/04_datos_part.R

library(dplyr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_completo <- file.path(dir_proy, "completo", "datos")

enc <- readRDS(file.path(dir_completo, "encuestas_completo.rds"))
lv_ideol <- readRDS(file.path(dir_completo, "datos_ideol.rds"))$lv

dat <- enc |>
  filter(
    !is.na(prob_voto),
    !is.na(ideol5),
    !is.na(sexo),
    !is.na(edad),
    !is.na(edu),
    !is.na(rec),
    !is.na(ccaa)
  ) |>
  mutate(
    nivel = case_when(
      prob_voto <= 4 ~ 1L,
      prob_voto <= 9 ~ 2L,
      TRUE ~ 3L
    )
  )
cat("entrevistados:", nrow(dat), "\n")
cat("\n% por nivel (1 improbable, 2 probable, 3 seguro) y casa:\n")
print(round(100 * prop.table(table(dat$casa, dat$nivel), 1), 1))
cat("\n% seguros por tramo de ideología:\n")
print(round(100 * prop.table(table(dat$ideol5, dat$nivel), 1), 1))

lv <- list(
  casa = c("CIS", "40dB", "More in Common"),
  mes = sort(unique(as.character(dat$mes))),
  edad = lv_ideol$edad,
  edu = levels(enc$edu),
  rec = lv_ideol$rec,
  ccaa = lv_ideol$ccaa,
  ideol = lv_ideol$tramo,
  nivel = c("improbable", "probable", "seguro")
)
idx <- function(x, niveles) {
  i <- match(as.character(x), niveles)
  stopifnot(!anyNA(i))
  i
}

datos <- list(
  s_sd_rec = 2,
  s_sd_main = 0.5,
  s_c = 3,
  s_b = 1,
  N = nrow(dat),
  y = dat$nivel,
  mujer = as.numeric(dat$sexo == "M"),
  n_casa = length(lv$casa),
  n_mes = length(lv$mes),
  n_edad = length(lv$edad),
  n_edu = length(lv$edu),
  n_rec = length(lv$rec),
  n_ccaa = length(lv$ccaa),
  n_ideol = length(lv$ideol),
  casa = idx(dat$casa, lv$casa),
  mes = idx(dat$mes, lv$mes),
  edad = idx(dat$edad, lv$edad),
  edu = idx(dat$edu, lv$edu),
  rec = idx(dat$rec, lv$rec),
  ccaa = idx(dat$ccaa, lv$ccaa),
  ideol = idx(dat$ideol5, lv$ideol)
)

saveRDS(
  list(datos = datos, lv = lv),
  file.path(dir_completo, "datos_part.rds")
)
cat("\nEscrito", file.path(dir_completo, "datos_part.rds"), "\n")
