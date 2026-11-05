# Modelo completo, parte 3, PROTOTIPO solo con el CIS: datos de
# completo/stan/mrp_voto_cis.stan.
#
# Celdas mes x ccaa x prov x sexo x edad x edu x rec x ideología x nivel con
# el recuento de cada partido (voto + simpatía, BLANCO aparte). Solo
# entrevistas del CIS con partido, ideología y probabilidad de votar.
# Aliança Catalana va con OTROS, como en 03_celdas.R. Los partidos regionales
# llevan un offset de -20 fuera de su territorio (03_celdas.R).
#
# Lee completo/datos/encuestas_completo.rds, datos_part.rds (niveles) y
# 00_partidos.R; escribe completo/datos/datos_voto_cis.rds.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/07_datos_voto_cis.R

library(dplyr)
library(tidyr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_completo <- file.path(dir_proy, "completo", "datos")
source(file.path(dir_proy, "00_partidos.R")) # partidos_lv, territorio

enc <- readRDS(file.path(dir_completo, "encuestas_completo.rds"))
lv_part <- readRDS(file.path(dir_completo, "datos_part.rds"))$lv
prov_ccaa <- readRDS(file.path(dir_proy, "datos", "prov_ccaa.rds"))

voto_lv <- c(setdiff(partidos_lv, "AC"), "BLANCO")
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
    voto = factor(
      if_else(voto_sim == "AC", "OTROS", as.character(voto_sim)),
      levels = voto_lv
    ),
    nivel = case_when(
      prob_voto <= 4 ~ "improbable",
      prob_voto <= 9 ~ "probable",
      TRUE ~ "seguro"
    )
  )
stopifnot(!anyNA(dat$voto))
cat("entrevistas del CIS:", nrow(dat), "\n")
print(round(100 * prop.table(table(dat$voto)), 1))

celdas <- dat |>
  count(mes, ccaa, prov, sexo, edad, edu, rec, ideol5, nivel, voto) |>
  pivot_wider(names_from = voto, values_from = n, values_fill = 0)
for (p in setdiff(voto_lv, names(celdas))) {
  celdas[[p]] <- 0L
}
celdas$y <- as.matrix(celdas[, voto_lv])
cat("filas:", nrow(celdas), "\n")

off <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) {
    ifelse(celdas$ccaa %in% territorio_mod[[p]], 0, -20)
  } else {
    rep(0, nrow(celdas))
  }
})
# ningún voto fuera del territorio del partido
stopifnot(all(celdas$y[off < 0] == 0))

lv <- list(
  mes = lv_part$mes,
  edad = lv_part$edad,
  edu = lv_part$edu,
  rec = lv_part$rec,
  ccaa = lv_part$ccaa,
  prov = levels(prov_ccaa$prov),
  ideol = lv_part$ideol,
  nivel = lv_part$nivel,
  voto = voto_lv
)
idx <- function(x, niveles) {
  i <- match(as.character(x), niveles)
  stopifnot(!anyNA(i))
  i
}

datos <- list(
  s_alpha = 1,
  s_b_mujer = 1,
  s_sd_rec = 2,
  s_sd_main = 0.5,
  N = nrow(celdas),
  J = length(voto_lv),
  y = unname(celdas$y),
  mujer = as.numeric(celdas$sexo == "M"),
  n_mes = length(lv$mes),
  n_edad = length(lv$edad),
  n_edu = length(lv$edu),
  n_rec = length(lv$rec),
  n_ccaa = length(lv$ccaa),
  n_prov = length(lv$prov),
  n_ideol = length(lv$ideol),
  n_nivel = length(lv$nivel),
  mes = idx(celdas$mes, lv$mes),
  edad = idx(celdas$edad, lv$edad),
  edu = idx(celdas$edu, lv$edu),
  rec = idx(celdas$rec, lv$rec),
  ccaa = idx(celdas$ccaa, lv$ccaa),
  prov = idx(celdas$prov, lv$prov),
  ideol = idx(celdas$ideol5, lv$ideol),
  nivel = idx(celdas$nivel, lv$nivel),
  off = unname(off)
)

saveRDS(
  list(datos = datos, lv = lv),
  file.path(dir_completo, "datos_voto_cis.rds")
)
cat("Escrito", file.path(dir_completo, "datos_voto_cis.rds"), "\n")
