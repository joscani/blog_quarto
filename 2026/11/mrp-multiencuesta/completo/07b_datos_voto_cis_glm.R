# Modelo completo, parte 3, PROTOTIPO solo con el CIS, versión glm: datos de
# completo/stan/mrp_voto_cis_glm.stan.
#
# Las mismas entrevistas que 07_datos_voto_cis.R, pero una fila por
# entrevistado: y es el partido (1..J) y x la matriz de diseño, con una
# columna por nivel de cada efecto (sexo = mujer, mes, edad, estudios,
# recuerdo, CCAA, provincia, ideología, nivel). off_ccaa lleva el -20 de los
# partidos regionales fuera de su territorio, por CCAA.
#
# Escribe completo/datos/datos_voto_cis_glm.rds.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/07b_datos_voto_cis_glm.R

library(dplyr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_completo <- file.path(dir_proy, "completo", "datos")
source(file.path(dir_proy, "00_partidos.R")) # partidos_lv, territorio

enc <- readRDS(file.path(dir_completo, "encuestas_completo.rds"))
lv <- readRDS(file.path(dir_completo, "datos_voto_cis.rds"))$lv
territorio_mod <- territorio[names(territorio) != "AC"]

dat <- enc |>
  filter(
    casa == "CIS",
    !is.na(voto_sim),
    !is.na(ideol5),
    !is.na(prob_voto),
    !is.na(sexo),
    !is.na(edad),
    !is.na(edu),
    !is.na(rec),
    !is.na(prov)
  ) |>
  mutate(
    voto = if_else(voto_sim == "AC", "OTROS", as.character(voto_sim)),
    nivel = case_when(
      prob_voto <= 4 ~ "improbable",
      prob_voto <= 9 ~ "probable",
      TRUE ~ "seguro"
    )
  )
cat("entrevistas del CIS:", nrow(dat), "\n")

# matriz de diseño: una columna por nivel (sin nivel de referencia: el modelo
# identifica con la suma cero)
una_col <- function(v, niveles) {
  m <- outer(match(as.character(v), niveles), seq_along(niveles), "==") * 1
  stopifnot(all(rowSums(m) == 1))
  m
}
x <- cbind(
  mujer = as.numeric(dat$sexo == "M"),
  una_col(dat$mes, lv$mes),
  una_col(dat$edad, lv$edad),
  una_col(dat$edu, lv$edu),
  una_col(dat$rec, lv$rec),
  una_col(dat$ccaa, lv$ccaa),
  una_col(dat$prov, lv$prov),
  una_col(dat$ideol5, lv$ideol),
  una_col(dat$nivel, lv$nivel)
)
cat("columnas de x:", ncol(x), "\n")

off_ccaa <- sapply(lv$voto, function(p) {
  if (p %in% names(territorio_mod)) {
    ifelse(lv$ccaa %in% territorio_mod[[p]], 0, -20)
  } else {
    rep(0, length(lv$ccaa))
  }
})
y <- match(dat$voto, lv$voto)
stopifnot(!anyNA(y))
# ningún voto fuera del territorio del partido
stopifnot(all(off_ccaa[cbind(match(as.character(dat$ccaa), lv$ccaa), y)] == 0))

datos <- list(
  s_alpha = 1,
  s_b_mujer = 1,
  s_sd_rec = 2,
  s_sd_main = 0.5,
  N = nrow(dat),
  J = length(lv$voto),
  y = y,
  n_mes = length(lv$mes),
  n_edad = length(lv$edad),
  n_edu = length(lv$edu),
  n_rec = length(lv$rec),
  n_ccaa = length(lv$ccaa),
  n_prov = length(lv$prov),
  n_ideol = length(lv$ideol),
  n_nivel = length(lv$nivel),
  K = ncol(x),
  x = unname(x),
  off_ccaa = unname(off_ccaa)
)

saveRDS(
  list(datos = datos, lv = lv),
  file.path(dir_completo, "datos_voto_cis_glm.rds")
)
cat("Escrito", file.path(dir_completo, "datos_voto_cis_glm.rds"), "\n")
