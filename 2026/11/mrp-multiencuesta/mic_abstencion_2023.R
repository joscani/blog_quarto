#' ---
#' title: "More in Common: la abstención de 2023 y el PSOE"
#' author: "José Luis Cañadas Reche"
#' date: "2026-10-26"
#' format:
#'   html:
#'     embed-resources: true
#' execute:
#'   message: false
#'   warning: false
#' ---
#'
#' Microdatos de More in Common de junio y septiembre de 2026, juntos.
#' Variables brutas: `participacion_2023_raw`, `voto_2023_raw`,
#' `intencion_voto_raw` y `peso`.
#'
#' - Abstención 2023 = "No pudo" + "No suele votar". Es la misma definición
#'   que vuestra variable `voto_2023_final`: todos los "No pudo" y "No suele
#'   votar" están ahí como "Abstención".
#' - "Declara partido" = `intencion_voto_raw` distinto de Blanco, Nulo, NS, NC
#'   y No votaría.
#'
#' Se ejecuta con `Rscript mic_abstencion_2023.R` o se convierte en HTML con
#' `quarto render mic_abstencion_2023.R`.

#+ datos
library(haven)
library(dplyr)
library(knitr)

options(OutDec = ",")

# Rutas desde la raíz del proyecto; cámbialas si los .sav están en otro sitio
ficheros <- here::here(c(
  "descargas/microdatos-2026-07/microdatos-julio-2026.sav",
  "descargas/microdatos-2026-09/microdatos-septiembre-2026.sav"
))

# Las etiquetas se convierten fichero a fichero, antes de juntarlos
lee <- function(f) {
  read_sav(f) |>
    transmute(
      participacion = as.character(as_factor(participacion_2023_raw)),
      recuerdo = as.character(as_factor(voto_2023_raw)),
      intencion = as.character(as_factor(intencion_voto_raw)),
      peso = as.numeric(peso)
    )
}

mic <- lapply(ficheros, lee) |>
  bind_rows() |>
  mutate(
    abst_2023 = participacion %in% c("No pudo", "No suele votar"),
    declara_partido = !intencion %in% c("Blanco", "Nulo", "NS", "NC", "No votaría")
  )

#' ## 1. Peso de la abstención de 2023
#'
#' Entre quienes tienen recuerdo de 2023 (votó y dice a quién, o abstención).
#' Con su peso, en el CIS queda en el 13 % y en 40dB en el 10 %.

#+ peso-abstencion
abst <- mic |>
  filter(abst_2023 | (participacion == "Votó" & !is.na(recuerdo))) |>
  summarise(
    n = n(),
    pct_abst_sin_peso = 100 * mean(abst_2023),
    pct_abst_con_peso = 100 * weighted.mean(abst_2023, peso)
  )

kable(abst, digits = 1)

#' ## 2. Abstencionistas de 2023 que declaran PSOE
#'
#' Entre los que declaran partido. Con su peso, en el CIS es el 24 % y en
#' 40dB el 15 %.

#+ abst-psoe
abst_psoe <- mic |>
  filter(abst_2023, declara_partido) |>
  summarise(
    n = n(),
    pct_psoe_sin_peso = 100 * mean(intencion == "PSOE"),
    pct_psoe_con_peso = 100 * weighted.mean(intencion == "PSOE", peso)
  )

kable(abst_psoe, digits = 1)

#' Con el peso, la abstención de 2023 pasa del
#' `r round(abst$pct_abst_sin_peso, 1)` % al
#' `r round(abst$pct_abst_con_peso, 1)` % de la muestra, y de esos
#' abstencionistas el `r round(abst_psoe$pct_psoe_con_peso, 1)` % declara
#' PSOE.
