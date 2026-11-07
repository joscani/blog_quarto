# ============================================================================
# PRUEBA: efecto comunidad agrupado por comunidad × transición
#
# Modelo actual:  (0 + familia || codigo_ccaa)
# Modelo nuevo:   (0 + familia | gr(ccaa_tr, by = rep_f, cor = FALSE))
#   ccaa_tr = comunidad × transición; by = fresca / repetición, para que las
#   repeticiones no encojan el tau_c fresco. tau_c = sd de la celda fresca.
#
# Los dos modelos se ajustan con los MISMOS datos: los del fit de producción
# (mod_tau_shock.rds). Los dos se prueban con el MISMO backtest: preps de
# hoy, mismas semillas que backtest_prov_loto.R. La diferencia entre ambos es
# solo tau.
#
# Ejecutar con cwd = ~/proyectos/blog_quarto. Escribe solo en esta carpeta.
# ============================================================================

library(tidyverse)
library(brms)

out <- here::here("2026/10/prueba_ccaa_transicion")
hora <- function(...) message(format(Sys.time(), "%H:%M"), " ", ...)

source(here::here("2026/10/backtest_provincial.R"))
source(here::here("2026/10/shock_jerarquico.R"))

preps <- preparar_preps()
dlr <- construir_dlr(preps)
fit_prod <- readRDS(here::here("2026/10/mod_tau_shock.rds"))

# mismas filas y orden que el fit de producción; el dlr se toma del fit
stopifnot(
  nrow(dlr) == nrow(fit_prod$data),
  identical(dlr$familia, as.character(fit_prod$data$familia)),
  identical(dlr$codigo_ccaa, as.character(fit_prod$data$codigo_ccaa))
)
fams_tau <- c("PP", "PSOE", "SUMAR", "VOX")
d_tau <- dlr |>
  mutate(
    dlr = fit_prod$data$dlr,
    repeticion = str_detect(etiqueta, "repetición"),
    celda_sigma = factor(
      paste(familia, repeticion, sep = "_"),
      levels = as.vector(outer(fams_tau, c("FALSE", "TRUE"), paste, sep = "_"))
    ),
    ccaa_tr = paste(codigo_ccaa, transition, sep = "_"),
    rep_f = factor(if_else(repeticion, "rep", "fresca"))
  )

ajustar_ct <- function(d, seed) {
  brm(
    bf(
      dlr ~ 0 + familia + repeticion +
        (0 + familia | gr(ccaa_tr, by = rep_f, cor = FALSE)),
      sigma ~ 0 + celda_sigma
    ),
    data = d,
    prior = c(
      prior(normal(0, 1), class = b),
      prior(normal(0, 0.2), class = sd),
      prior(normal(-2.1, 1), class = b, dpar = "sigma")
    ),
    chains = 4, iter = 2000, warmup = 1000, cores = 4,
    backend = "cmdstanr", adapt_delta = 0.95,
    seed = seed, silent = 2, refresh = 0
  )
}

# tau_c = sd fresca del grupo ccaa_tr; tau_p = sigma fresca
extraer_ct <- function(fit, b, seed = 9157) {
  dr <- as_draws_df(fit)
  set.seed(seed)
  idx <- sample.int(nrow(dr), b, replace = TRUE)
  map_dfr(fams_tau, \(f) {
    col_c <- sprintf("sd_ccaa_tr__familia%s:rep_ffresca", f)
    col_p <- sprintf("b_sigma_celda_sigma%s_FALSE", f)
    if (!all(c(col_c, col_p) %in% names(dr))) {
      return(NULL)
    }
    tibble(
      draw = seq_len(b), familia = f,
      tau_c = dr[[col_c]][idx],
      tau_p = exp(dr[[col_p]][idx])
    )
  })
}

trans_loto <- tribble(
  ~eleccion, ~transition,
  "26j", "20d->26j",
  "28a", "26j->28a",
  "10n", "28a->10n",
  "23j", "10n->23j"
)
seeds_fit <- c(`20d->26j` = 8412, `26j->28a` = 3359, `28a->10n` = 6210, `10n->23j` = 7904)
seeds_bt <- c(`26j` = 4127, `28a` = 9083, `10n` = 1564, `23j` = 6301, `20d` = 2458)
b <- 2000

cache_fit <- function(nombre, d, seed) {
  f <- file.path(out, paste0(nombre, ".rds"))
  if (file.exists(f)) {
    return(readRDS(f))
  }
  hora("ajustando ", nombre)
  fit <- ajustar_ct(d, seed)
  saveRDS(fit, f)
  fit
}

# ---- ajustes del modelo nuevo -----------------------------------------------
fit_ct <- cache_fit("ct_full", d_tau, 5183)
fits_ct_loto <- setNames(
  map(trans_loto$transition, \(tr) {
    cache_fit(
      paste0("ct_loto_", gsub("->", "_", tr)),
      filter(d_tau, transition != tr),
      seeds_fit[[tr]]
    )
  }),
  trans_loto$transition
)

# diagnósticos
diag <- map_dfr(c(list(full = fit_ct), fits_ct_loto), \(f) {
  s <- posterior::summarise_draws(as_draws_df(f), "rhat", "ess_bulk")
  tibble(
    rhat_max = max(s$rhat, na.rm = TRUE),
    ess_min = min(s$ess_bulk, na.rm = TRUE),
    divergentes = sum(nuts_params(f) |> filter(Parameter == "divergent__") |> pull(Value))
  )
}, .id = "fit")
write_csv(diag, file.path(out, "diagnosticos_ct.csv"))
print(diag)

# ---- tau de los dos modelos (ajuste completo) --------------------------------
comp_tau <- bind_rows(
  extraer_tau_draws(fit_prod, b = 4000) |> mutate(modelo = "actual"),
  extraer_ct(fit_ct, b = 4000) |> mutate(modelo = "ccaa_tr")
) |>
  group_by(familia, modelo) |>
  summarise(
    tau_c_med = median(tau_c),
    tau_c_q05 = quantile(tau_c, .05),
    tau_c_q95 = quantile(tau_c, .95),
    tau_p_med = median(tau_p),
    .groups = "drop"
  )
write_csv(comp_tau, file.path(out, "comparacion_tau.csv"))
print(comp_tau, width = Inf)

# ---- tau para los backtests --------------------------------------------------
rellenar_vox <- function(t, t_full) {
  falta <- setdiff(fams_tau, unique(t$familia))
  bind_rows(t, filter(t_full, familia %in% falta))
}

tau_in <- list(
  actual = extraer_tau_draws(fit_prod, b = b),
  ccaa_tr = extraer_ct(fit_ct, b = b)
)
tau_lo <- list(
  actual = setNames(map(trans_loto$transition, \(tr) {
    fit <- readRDS(here::here("2026/10", sprintf("mod_tau_loto_%s.rds", tr)))
    dr <- as_draws_df(fit)
    fams <- if ("b_sigma_celda_sigmaVOX_FALSE" %in% names(dr)) NULL else c("PP", "PSOE", "SUMAR")
    rellenar_vox(extraer_tau_draws(fit, b = b, familias = fams), tau_in$actual)
  }), trans_loto$transition),
  ccaa_tr = map(fits_ct_loto, \(fit) rellenar_vox(extraer_ct(fit, b = b), tau_in$ccaa_tr))
)

# ---- backtests pareados ------------------------------------------------------
mapa_bt <- mapa_ccaa_congreso()
run_bt <- function(id, tau) {
  e <- elecciones[[which(map_chr(elecciones, "id") == id)]]
  set.seed(seeds_bt[[id]])
  backtest_prov(e, preps, b = b, modo = "shock", tau = tau, mapa = mapa_bt)
}

f_celdas <- file.path(out, "celdas.csv")
done <- if (file.exists(f_celdas)) {
  read_csv(f_celdas, show_col_types = FALSE, col_types = cols(codigo_provincia = col_character()))
} else {
  tibble(eleccion = character(), variante = character(), modelo = character())
}

for (m in c("actual", "ccaa_tr")) {
  for (i in seq_len(nrow(trans_loto))) {
    id <- trans_loto$eleccion[i]
    tr <- trans_loto$transition[i]
    if (any(done$eleccion == id & done$modelo == m)) next
    hora("backtest ", m, " ", id)
    done <- bind_rows(
      done,
      run_bt(id, tau_in[[m]]) |> mutate(variante = "in-sample", modelo = m),
      run_bt(id, tau_lo[[m]][[tr]]) |> mutate(variante = "loto", modelo = m)
    )
    write_csv(done, f_celdas)
  }
  if (!any(done$eleccion == "20d" & done$modelo == m)) {
    hora("backtest ", m, " 20d")
    done <- bind_rows(
      done,
      run_bt("20d", tau_in[[m]]) |> mutate(variante = "in-sample (oos)", modelo = m)
    )
    write_csv(done, f_celdas)
  }
}

# ---- resumen -----------------------------------------------------------------
resumen <- done |>
  filter(!ref_ratio_cero) |>
  group_by(eleccion, variante, modelo) |>
  summarise(
    celdas = n(),
    cobertura = mean(dentro),
    ancho_medio = mean(q95 - q05),
    .groups = "drop"
  ) |>
  mutate(eleccion = factor(eleccion, c("20d", "26j", "28a", "10n", "23j"))) |>
  arrange(eleccion, variante, modelo)
write_csv(resumen, file.path(out, "resumen.csv"))
print(resumen, n = 40, width = Inf)

# PSOE en Cataluña, 23J
cat_psoe <- done |>
  filter(eleccion == "23j", familia == "PSOE", codigo_provincia %in% c("08", "17", "25", "43")) |>
  select(modelo, variante, codigo_provincia, real, mediana, q05, q95, dentro) |>
  arrange(codigo_provincia, variante, modelo)
write_csv(cat_psoe, file.path(out, "psoe_cataluna_23j.csv"))
print(cat_psoe, n = 40, width = Inf)
hora("terminado")
