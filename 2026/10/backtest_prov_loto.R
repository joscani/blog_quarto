# ============================================================================
# LEAVE-ONE-TRANSITION-OUT (LOTO) DE LOS TAUS DEL SHOCK
#
# Los tau del shock se estiman con las 4 transiciones (20D→26J, 26J→28A,
# 28A→10N, 10N→23J) y el backtest provincial prueba esas mismas elecciones:
# la validación es in-sample. Aquí se repite el backtest con los tau
# reestimados SIN la transición que se prueba (análogo al leave-one-out del
# sesgo del PSOE en el backtest nacional).
#
# El 20D no entra en el LOTO: su geografía viene del 11D, que no está en la
# calibración, así que es el único objetivo REALMENTE fuera de muestra (se
# reporta con los tau completos).
#
# Configuración idéntica a backtest_provincial.R (geo previa, kappa = 0.5,
# b = 2000). Para cada elección objetivo se corre DOS veces con la misma
# semilla: tau in-sample (fit completo) y tau LOTO; la diferencia de
# cobertura es atribuible solo a la reestimación de los tau.
#
# Salidas:
#   2026/10/backtest_prov_loto.csv — celdas de las 4 elecciones objetivo ×
#   {in-sample, loto} + el 20D (oos). Reanudable: si el CSV existe, salta
#   las elecciones ya calculadas; los fits LOTO se cachean en RDS.
# ============================================================================

library(tidyverse)
library(brms)

source(here::here("2026/10/backtest_provincial.R"))
source(here::here("2026/10/shock_jerarquico.R"))

# ---- 1. datos de ajuste ------------------------------------------------------
#
# El dlr se construye igual que en ajustar_tau_shock.R, que produce el fit
# completo (mod_tau_shock.rds): voto_nacional = "presentes". Con esa regla,
# los datos de los cinco fits (completo y LOTO) coinciden exactamente con
# este dlr (comprobado el 7 nov 2026).
#
# Corrección del 7 nov 2026: esta cabecera decía que el fit se había
# ajustado con el dlr "actual" y que el anterior incluía UPN-PP en el PP. No
# era así. Este script usaba un objeto dlr de la sesión, construido a mano
# con el voto nacional de cada familia calculado solo en las provincias donde
# tiene votos. Ver construir_dlr() en shock_jerarquico.R.
# ------------------------------------------------------------------------------

if (!exists("preps")) {
  preps <- preparar_preps()
}
dlr <- construir_dlr(preps, voto_nacional = "presentes")
d_tau <- preparar_d_tau(dlr)

# ---- 2. refits LOTO (uno por transición excluida) ---------------------------

trans_loto <- tribble(
  ~eleccion , ~transition , ~tipo        ,
  "26j"     , "20d->26j"  , "repetición" ,
  "28a"     , "26j->28a"  , "fresca"     ,
  "10n"     , "28a->10n"  , "repetición" ,
  "23j"     , "10n->23j"  , "fresca"
)

# ajustar_tau_brm() está en shock_jerarquico.R (la misma que el fit completo)

# seeds fijos por fit (elección arbitraria, reproducible)
seeds_fit <- c(
  `20d->26j` = 8412,
  `26j->28a` = 3359,
  `28a->10n` = 6210,
  `10n->23j` = 7904
)

f_fit_loto <- function(tr) {
  here::here("2026/10", sprintf("mod_tau_loto_%s.rds", tr))
}

fits_loto <- setNames(
  map(trans_loto$transition, \(tr) {
    if (file.exists(f_fit_loto(tr))) {
      message("fit LOTO en caché: ", tr)
      f <- readRDS(f_fit_loto(tr))
      stopifnot(isTRUE(all.equal(
        d_tau$dlr[d_tau$transition != tr],
        f$data$dlr,
        tolerance = 1e-12
      )))
      f
    } else {
      message("refit LOTO: ", tr)
      f <- ajustar_tau_brm(d_tau |> filter(transition != tr), seeds_fit[[tr]])
      saveRDS(f, f_fit_loto(tr))
      f
    }
  }),
  trans_loto$transition
)

# fit completo (in-sample): mod_tau_shock.rds, que produce
# ajustar_tau_shock.R con el mismo dlr
f_fit_full <- here::here("2026/10/mod_tau_shock.rds")
if (!file.exists(f_fit_full)) {
  stop("falta mod_tau_shock.rds: ejecuta antes 2026/10/ajustar_tau_shock.R")
}
fit_full <- readRDS(f_fit_full)
stopifnot(isTRUE(all.equal(d_tau$dlr, fit_full$data$dlr, tolerance = 1e-12)))

# ---- 3. backtests pareados ---------------------------------------------------

b <- 2000
mapa_bt <- mapa_ccaa_congreso()

# una semilla por elección objetivo (la misma para las dos variantes, para
# que la diferencia sea solo los tau)
seeds_bt <- c(
  `26j` = 4127,
  `28a` = 9083,
  `10n` = 1564,
  `23j` = 6301,
  `20d` = 2458
)

run_bt <- function(id, tau) {
  e <- elecciones[[which(map_chr(elecciones, "id") == id)]]
  set.seed(seeds_bt[[id]])
  backtest_prov(e, preps, b = b, modo = "shock", tau = tau, mapa = mapa_bt)
}

f_salida <- here::here("2026/10/backtest_prov_loto.csv")
done <- if (file.exists(f_salida)) {
  read_csv(f_salida, show_col_types = FALSE)
} else {
  # con las columnas ya definidas: evita warnings de columnas ausentes
  # en los any(done$eleccion == ...) del bucle
  tibble(
    eleccion = character(),
    variante = character(),
    codigo_provincia = character(),
    familia = character(),
    real = double(),
    mediana = double(),
    q05 = double(),
    q95 = double(),
    ref_ratio_cero = logical(),
    dentro = logical(),
    validos_prov = double()
  )
}

# tau draws: mismos para todas las elecciones (extraer_tau_draws lleva su
# propia semilla); se muestrean una vez fuera del bucle
tau_in <- extraer_tau_draws(fit_full, b = b)

# VOX en el LOTO de 23J: al excluir 10n->23j no le queda ninguna transición
# fresca (28a->10n es repetición), así que su tau_p fresco sería prior-only
# (brms descarta la celda vacía y extraer_tau_draws no encuentra la columna).
# Se extrae lo que hay y VOX se rellena con el tau del fit completo: el LOTO
# de 23J testea PP/PSOE/SUMAR. El hallazgo (la calibración de VOX descansa
# por completo en 10N->23J) va al post.
tau_lo <- map(fits_loto, \(fit) {
  dr <- as_draws_df(fit)
  tiene_vox_fresco <- "b_sigma_celda_sigmaVOX_FALSE" %in% names(dr)
  if (tiene_vox_fresco) {
    return(extraer_tau_draws(fit, b = b))
  }
  tau <- extraer_tau_draws(fit, b = b, familias = c("PP", "PSOE", "SUMAR"))
  bind_rows(
    tau,
    tau_in |> filter(familia == "VOX")
  )
})

for (i in seq_len(nrow(trans_loto))) {
  id <- trans_loto$eleccion[i]
  tr <- trans_loto$transition[i]
  if (any(done$eleccion == id & done$variante == "loto")) {
    next
  }
  message("backtest LOTO: ", id, " (sin ", tr, ")")
  nuevo <- bind_rows(
    run_bt(id, tau_in) |> mutate(variante = "in-sample"),
    run_bt(id, tau_lo[[tr]]) |> mutate(variante = "loto")
  )
  done <- bind_rows(done, nuevo)
  write_csv(done, f_salida)
}

# el 20D: objetivo fuera de muestra por construcción (geografía del 11D);
# solo variante in-sample
if (!any(done$eleccion == "20d")) {
  message("backtest 20D (fuera de muestra por construcción)")
  done <- bind_rows(
    done,
    run_bt("20d", tau_in) |> mutate(variante = "in-sample (oos)")
  )
  write_csv(done, f_salida)
}

# las celdas in-sample SON las del shock con los draws brms de producción:
# se congelan como backtest_prov_celdas_shock_brms.csv (sin la columna de
# variante, como el fichero original)
done |>
  filter(variante %in% c("in-sample", "in-sample (oos)")) |>
  select(-variante) |>
  write_csv(here::here("2026/10/backtest_prov_celdas_shock_brms.csv"))

# ---- 4. resumen --------------------------------------------------------------

resumen_loto <- done |>
  filter(!ref_ratio_cero) |>
  group_by(eleccion, variante) |>
  summarise(
    celdas = n(),
    cobertura = mean(dentro),
    ancho_medio = mean(q95 - q05),
    .groups = "drop"
  ) |>
  mutate(
    eleccion = factor(
      eleccion,
      c("20d", "26j", "28a", "10n", "23j")
    ),
    variante = factor(
      variante,
      c("loto", "in-sample", "in-sample (oos)")
    )
  ) |>
  arrange(eleccion, variante)

write_csv(resumen_loto, here::here("2026/10/backtest_prov_loto_resumen.csv"))
resumen_loto |>
  mutate(across(where(is.numeric), \(x) round(x, 2))) |>
  print(n = 20, width = Inf)
