# Datos del ejemplo de postestratificación del post (II), por recuerdo del 23-J:
#   - % de cada recuerdo en la muestra (las nueve encuestas, sin pesos);
#   - % en la población (tabla de postestratificación);
#   - P(voto PSOE | recuerdo) y P(voto PP | recuerdo) del modelo (media de la
#     posterior, casa media, si votara todo el mundo), agregando las celdas de
#     cada recuerdo con su población.
# Salida: datos/datos_postestrat.rds. Necesita el post_celdas (07_post_celdas.R).
#
#   Rscript 2026/11/mrp-multiencuesta/07c_postestrat_datos.R

library(dplyr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

grupo <- function(r) {
  r <- as.character(r)
  ifelse(r %in% c("PP", "PSOE", "VOX", "SUMAR", "ABST", "NO_PODIA"), r, "otros")
}

post <- readRDS(file.path(dir_datos, "post_celdas_casa_media_part.rds"))

poblacion <- post$post |>
  mutate(
    p_psoe = colMeans(post$p[, , "PSOE"]),
    p_pp = colMeans(post$p[, , "PP"]),
    rec = grupo(rec)
  ) |>
  summarise(
    p_psoe = weighted.mean(p_psoe, N),
    p_pp = weighted.mean(p_pp, N),
    N = sum(N),
    .by = rec
  ) |>
  mutate(poblacion = N / sum(N))

muestra <- readRDS(file.path(dir_datos, "encuestas.rds")) |>
  filter(!is.na(rec)) |>
  count(rec = grupo(rec)) |>
  mutate(muestra = n / sum(n))

res <- poblacion |>
  left_join(muestra, by = "rec") |>
  select(rec, muestra, poblacion, p_psoe, p_pp)

saveRDS(res, file.path(dir_datos, "datos_postestrat.rds"))
print(res)
