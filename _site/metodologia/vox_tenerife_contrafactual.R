# Efecto de corregir el ratio de VOX en Santa Cruz de Tenerife.
#
# El 23J, VOX no tuvo lista al Congreso allí, y su ratio de referencia era 0.
# simular_escanos.R lo corrige con el ratio del 10N (0,765). Este script vuelve a
# poner el ratio a 0 solo en memoria y compara las dos simulaciones, con el mismo
# voto nacional y la misma semilla. Los pesos del recentrado siguen siendo los
# corregidos, así que "sin corregir" es una aproximación muy cercana al cálculo
# anterior a la corrección.
#
# Salida: metodologia/vox_tenerife_contrafactual.csv
#   Rscript metodologia/vox_tenerife_contrafactual.R

library(tidyverse)
source(here::here("2026/10/simular_escanos.R"))

guardada <- readRDS(here::here("2026/10/full-bayesian/vivo/conjunto_sim.rds"))
shares <- guardada$shares

ejecutar <- function() {
  set.seed(2029)
  simular(
    n_draws = nrow(shares),
    listas = "mapa",
    provincial = TRUE,
    shares_ext = shares
  )
}

corregido <- ejecutar()
stopifnot(identical(corregido$draws$VOX, guardada$draws$VOX))

i_tenerife <- which(prov_datos14$codigo_provincia == "38")
j_vox <- match("VOX", fams14)
stopifnot(prov_datos14$ratio_f[[i_tenerife]][j_vox] > 0)

prov_datos14$ratio_f[[i_tenerife]][j_vox] <- 0
sin_corregir <- ejecutar()

tenerife <- "Santa Cruz de Tenerife"

fila <- function(medida, a, b) {
  tibble(medida = medida, sin_corregir = a, corregido = b)
}

resultado <- bind_rows(
  fila(
    "VOX en Tenerife: escaños medios",
    mean(sin_corregir$prov_escanos[, tenerife, "VOX"]),
    mean(corregido$prov_escanos[, tenerife, "VOX"])
  ),
  fila(
    "VOX en Tenerife: voto mediano (%)",
    100 * median(sin_corregir$prov_voto[, tenerife, "VOX"]),
    100 * median(corregido$prov_voto[, tenerife, "VOX"])
  ),
  fila(
    "PP en Tenerife: escaños medios",
    mean(sin_corregir$prov_escanos[, tenerife, "PP"]),
    mean(corregido$prov_escanos[, tenerife, "PP"])
  ),
  fila(
    "PSOE en Tenerife: escaños medios",
    mean(sin_corregir$prov_escanos[, tenerife, "PSOE"]),
    mean(corregido$prov_escanos[, tenerife, "PSOE"])
  ),
  fila(
    "VOX en España: escaños medios",
    mean(sin_corregir$draws$VOX),
    mean(corregido$draws$VOX)
  ),
  fila(
    "PP en España: escaños medios",
    mean(sin_corregir$draws$PP),
    mean(corregido$draws$PP)
  ),
  fila(
    "PSOE en España: escaños medios",
    mean(sin_corregir$draws$PSOE),
    mean(corregido$draws$PSOE)
  ),
  fila(
    "PP + VOX en España: escaños medios",
    mean(sin_corregir$draws$PP + sin_corregir$draws$VOX),
    mean(corregido$draws$PP + corregido$draws$VOX)
  ),
  fila(
    "P(PP + VOX >= 176) (%)",
    100 * mean(sin_corregir$draws$PP + sin_corregir$draws$VOX >= 176),
    100 * mean(corregido$draws$PP + corregido$draws$VOX >= 176)
  )
)

print(resultado)
write_csv(
  resultado,
  here::here("metodologia/vox_tenerife_contrafactual.csv")
)
