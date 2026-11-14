# VOX entre los de 18-20 por comunidad: bruto frente a MRP.
# Los de 18-20 no votaron en 2023 (todos NO_PODIA): el recuerdo no recoge lo
# propio de cada comunidad y el modelo, sin término CCAA x edad, les aplica la
# curva nacional de edad desplazada. ¿Dónde se separa más del bruto?
#
# z = (bruto - MRP) / error estándar binomial del bruto con p = MRP. Es
# orientativo: el bruto es la muestra (sin ponderar) y el MRP la población.
#
# Salida: completo/datos/vox_18_20_ccaa.rds
#
#   Rscript 2026/11/mrp-multiencuesta/completo/34_vox_18_20_ccaa.R   (~1 min)

library(dplyr)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo/datos")
tramo <- function(edad) {
  case_when(edad == "18-20" ~ "18-20",
            edad %in% c("21-24", "25-34") ~ "21-34",
            .default = "35+")
}

encuestas <- readRDS(file.path(dir_completo, "encuestas_completo.rds")) |>
  filter(!is.na(voto_sim), !is.na(ccaa)) |>
  mutate(ccaa = as.character(ccaa), tramo = tramo(edad),
         vox = as.character(voto_sim) == "VOX")
bruto <- encuestas |>
  summarise(n = n(), votos_vox = sum(vox), bruto = 100 * mean(vox), .by = c(ccaa, tramo))

celdas <- readRDS(file.path(dir_completo, "tabla_post.rds")) |>
  transmute(celda, N, ccaa, tramo = tramo(as.character(edad)))
mrp_draws <- readRDS(file.path(dir_completo, "posterior_celdas_80.rds")) |>
  select(draw, celda, part, VOX) |>
  left_join(celdas, by = "celda") |>
  summarise(vox = 100 * sum(N * part * VOX) / sum(N * part), .by = c(draw, ccaa, tramo))
mrp <- mrp_draws |>
  summarise(mrp = mean(vox), q05 = quantile(vox, 0.05), q95 = quantile(vox, 0.95),
            .by = c(ccaa, tramo))

tabla <- inner_join(bruto, mrp, by = c("ccaa", "tramo")) |>
  mutate(z = (bruto - mrp) / (100 * sqrt(mrp / 100 * (1 - mrp / 100) / n)))
stopifnot(nrow(tabla) == nrow(bruto))

# Intervalo de credibilidad del bruto, solo con los datos del grupo:
# Beta(1 + votos VOX, 1 + resto), prior uniforme
tabla <- tabla |>
  mutate(bruto_q05 = 100 * qbeta(0.05, 1 + votos_vox, 1 + n - votos_vox),
         bruto_q95 = 100 * qbeta(0.95, 1 + votos_vox, 1 + n - votos_vox))

print(tabla |>
        filter(tramo == "18-20", n >= 20) |>
        arrange(z) |>
        select(ccaa, n, bruto, bruto_q05, bruto_q95, mrp, q05, q95, z) |>
        mutate(across(where(is.double), \(x) round(x, 1))))
cat("\nz medio por tramo:\n")
print(tabla |> summarise(z_medio = mean(z), .by = tramo))

saveRDS(tabla, file.path(dir_completo, "vox_18_20_ccaa.rds"))
