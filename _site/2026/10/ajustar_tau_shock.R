# ============================================================================
# TAU DEL SHOCK GEOGRÁFICO: AJUSTE Y FICHEROS DE PRODUCCIÓN
#
# Produce los cinco ficheros que consume simular() y los backtests:
#   2026/10/mod_tau_shock.rds    — fit brms de los tau de PP, PSOE, SUMAR, VOX
#   2026/10/tau_shock_draws.csv  — 2.000 draws por familia (simular() los lee)
#   2026/10/tau_shock_brms.csv   — mediana e IC90 de los tau (4.000 draws)
#   2026/10/tau_shock.csv        — tau puntuales con lme4 (comparación)
#   2026/10/tau_regionales.csv   — tau de los regionales (dos etapas, plug-in)
#
# Hasta el 7 nov 2026 ningún script escribía estos ficheros: el fit del
# 12 oct se hizo con un dlr construido a mano en la sesión. Este script lo
# reproduce: con voto_nacional = "presentes", construir_dlr() da EXACTAMENTE
# los datos de mod_tau_shock.rds y de los cuatro mod_tau_loto_*.rds, y los
# cuatro CSV coinciden con los publicados (comprobado el 7 nov 2026).
#
# "presentes" calcula el voto nacional de cada familia solo con las
# provincias donde tiene votos (ver construir_dlr() en shock_jerarquico.R).
# El resultado oficial ("oficial") cambia el dlr en PP 10N→23J (−0,0139) y
# VOX 10N→23J (+0,0202). Pasar a "oficial" exige reajustar con
# sobrescribir = TRUE y regenerar todo lo que depende de los tau.
#
# Uso, desde la raíz de blog_quarto:
#   Rscript 2026/10/ajustar_tau_shock.R
# Sin cambios en los datos, no reajusta: carga el fit, comprueba que sus
# datos coinciden con el dlr y que cada CSV coincide con lo que se calcula.
# Si algo difiere, se para sin sobrescribir. Para reajustar y sobrescribir,
# poner sobrescribir <- TRUE.
# ============================================================================

library(tidyverse)
library(brms)

source(here::here("2026/10/backtest_provincial.R"))
source(here::here("2026/10/shock_jerarquico.R"))

sobrescribir <- FALSE
voto_nacional <- "presentes"
seed_fit <- 5183

ruta <- function(f) here::here("2026/10", f)

# escribe el fichero si no existe (o si sobrescribir); si existe, comprueba
# que coincide con lo calculado y se para si no
escribir_o_comprobar <- function(d, f) {
  if (!file.exists(ruta(f)) || sobrescribir) {
    write_csv(d, ruta(f))
    message("escrito: ", f)
    return(invisible(d))
  }
  previo <- read_csv(ruta(f), show_col_types = FALSE)
  igual <- isTRUE(all.equal(
    as.data.frame(d),
    as.data.frame(previo),
    tolerance = 1e-8,
    check.attributes = FALSE
  ))
  if (!igual) {
    stop(f, " no coincide con lo calculado. Revisa o usa sobrescribir <- TRUE")
  }
  message("reproducido: ", f)
  invisible(d)
}

# ---- 1. datos ----------------------------------------------------------------
preps <- preparar_preps()
dlr <- construir_dlr(preps, voto_nacional = voto_nacional)
d_tau <- preparar_d_tau(dlr)

# ---- 2. fit brms -------------------------------------------------------------
f_fit <- ruta("mod_tau_shock.rds")
if (file.exists(f_fit) && !sobrescribir) {
  fit <- readRDS(f_fit)
  if (!isTRUE(all.equal(d_tau$dlr, fit$data$dlr, tolerance = 1e-12))) {
    stop("los datos de mod_tau_shock.rds no coinciden con el dlr actual")
  }
  message("reproducido: datos de mod_tau_shock.rds")
} else {
  fit <- ajustar_tau_brm(d_tau, seed_fit)
  saveRDS(fit, f_fit)
  message("escrito: mod_tau_shock.rds")
}

# ---- 3. ficheros derivados ---------------------------------------------------
escribir_o_comprobar(extraer_tau_draws(fit, b = 2000), "tau_shock_draws.csv")
escribir_o_comprobar(resumir_tau_brm(fit), "tau_shock_brms.csv")
escribir_o_comprobar(ajustar_tau(dlr), "tau_shock.csv")
escribir_o_comprobar(
  ajustar_tau_regionales(construir_dlr_regionales(preps)),
  "tau_regionales.csv"
)
