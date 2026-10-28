# MRP solo con los barómetros del CIS (mrp_cis.stan): sin efecto casa ni
# filas mezcla, porque el CIS siempre trae la provincia y una sola casa no
# separa su sesgo del intercepto. Es la referencia de "una sola encuesta"
# frente al modelo de las 4 casas (stan/ajusta.R).
#
# Los datos vienen de 06_stan_datos.R (la rama datos_cis de datos/datos_stan.rds).
#
# Salidas: datos/fit_stan_<modo>_nuevo_cis.rds y estim_stan_<modo>_nuevo_cis.rds.

library(dplyr)
library(tidyr)
library(cmdstanr)

modo <- "prueba" # prueba | nuts | prior (prior predictive check, sin verosimilitud)
priors <- "nuevo"

dir_proy <- "2026/11/mrp-multiencuesta"
dir_datos <- file.path(dir_proy, "datos")
stopifnot(modo %in% c("prueba", "nuts", "prior"))

x <- readRDS(file.path(dir_datos, "datos_stan.rds"))
stopifnot(x$modo == modo, x$priors == priors)
datos <- x$datos_cis
lv <- x$lv_cis
voto_lv <- x$voto_lv
datos$prior_only <- as.integer(modo == "prior")
cat("modo:", modo, "  N:", datos$N, "  filas de participación:", datos$NT, "\n")

# Se guarda el ajuste con una huella del modelo y de los datos: si ninguno de
# los dos cambia, se carga en vez de reajustar (como file_refit = "on_change").
f_stan <- file.path(dir_proy, "stan", "mrp_cis.stan")
huella <- digest::digest(list(readLines(f_stan), datos, modo))
f_fit <- file.path(
  dir_datos,
  paste0("fit_stan_", modo, "_", priors, "_cis.rds")
)

mod <- cmdstan_model(f_stan)

if (modo == "prior") {
  fit <- mod$sample(
    data = datos,
    seed = 2026,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 250,
    iter_sampling = 250,
    init = 0,
    refresh = 100
  )
  # Qué dice el prior en la escala de los datos, antes de ver ningún voto
  pm <- as.vector(fit$draws("p_max", format = "draws_matrix"))
  cat("\n== prior: probabilidad del partido más votado en una celda ==\n")
  print(round(quantile(pm, c(0.05, 0.25, 0.5, 0.75, 0.95)), 2))
  cat("celdas con un partido > 90 %:", round(100 * mean(pm > 0.9), 1), "%\n")
  cat("celdas con un partido > 99 %:", round(100 * mean(pm > 0.99), 1), "%\n")

  v <- fit$draws("v_casa", format = "draws_matrix")
  m <- v[, paste0("v_casa[1,", seq_along(voto_lv), "]"), drop = FALSE]
  cat("\n== prior: voto en España según el CIS (%) ==\n")
  print(
    data.frame(
      partido = voto_lv,
      q05 = round(apply(m, 2, quantile, 0.05), 1),
      mediana = round(apply(m, 2, median), 1),
      q95 = round(apply(m, 2, quantile, 0.95), 1)
    ),
    row.names = FALSE
  )
  saveRDS(
    list(p_max = pm, v_casa = v),
    file.path(dir_datos, paste0("prior_check_", priors, "_cis.rds"))
  )
  quit(save = "no")
}

if (file.exists(f_fit) && identical(readRDS(f_fit)$huella, huella)) {
  cat("modelo y datos sin cambios: cargo", f_fit, "\n")
  fit <- readRDS(f_fit)$fit
} else {
  t0 <- Sys.time()
  fit <- mod$sample(
    data = datos,
    seed = 2026,
    chains = if (modo == "prueba") 1 else 4,
    parallel_chains = 4,
    iter_warmup = if (modo == "prueba") 150 else 1000,
    iter_sampling = if (modo == "prueba") 150 else 1000,
    init = 0,
    adapt_delta = 0.95,
    max_treedepth = 8, # igual que 05_brms_mm.R
    refresh = if (modo == "prueba") 25 else 100
  )
  cat(
    "\ntiempo de ajuste:",
    round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
    "min\n"
  )
  fit$save_object(file.path(dir_datos, "tmp_fit.rds")) # lee todas las muestras en memoria
  saveRDS(list(fit = fit, huella = huella), f_fit)
  file.remove(file.path(dir_datos, "tmp_fit.rds"))
}

# ---------------------------------------------------------------------------
# Diagnóstico y resultados
# ---------------------------------------------------------------------------

cat("\n== diagnóstico del muestreador ==\n")
print(fit$diagnostic_summary())
td <- fit$sampler_diagnostics(format = "draws_df")
cat(
  "iteraciones en max_treedepth (8):",
  sum(td$treedepth__ >= 8),
  "de",
  nrow(td),
  "\n"
)
cat("pasos leapfrog por iteración (mediana):", median(td$n_leapfrog__), "\n")

sds <- fit$summary(
  variables = c(
    "sd_mes",
    "sd_edad",
    "sd_edu",
    "sd_rec",
    "sd_ccaa",
    "sd_prov",
    "sd_ccaa_rec"
  )
)
cat(
  "\nRhat máximo de las sd:",
  round(max(sds$rhat), 3),
  "  ESS bulk mínimo:",
  round(min(sds$ess_bulk)),
  "\n"
)

v <- fit$draws("v_casa", format = "draws_matrix")
res <- tibble(
  casa = "CIS solo",
  partido = voto_lv,
  mrp = colMeans(
    m <- v[, paste0("v_casa[1,", seq_along(voto_lv), "]"), drop = FALSE]
  ),
  q05 = apply(m, 2, quantile, 0.05),
  q95 = apply(m, 2, quantile, 0.95)
)

cat(
  "\n== MRP con el CIS,",
  max(lv$mes),
  ", si votara todo el mundo (% voto válido) ==\n"
)
print(
  as.data.frame(
    res %>% select(-q05, -q95) %>% mutate(across(-partido, ~ round(.x, 1)))
  ),
  row.names = FALSE
)

cat("\n== participación ==\n")
print(fit$summary(variables = c("part_bruta", "delta")), width = 120)
vp <- fit$draws("v_part", format = "draws_matrix")
m <- vp[, paste0("v_part[1,", seq_along(voto_lv), "]"), drop = FALSE]
res_part <- tibble(
  casa = "CIS solo, con participación",
  partido = voto_lv,
  mrp = colMeans(m),
  q05 = apply(m, 2, quantile, 0.05),
  q95 = apply(m, 2, quantile, 0.95)
)
cat("\n== MRP con participación calibrada al 70,4 % (% voto válido) ==\n")
print(
  as.data.frame(
    res_part %>% select(-q05, -q95) %>% mutate(across(-partido, ~ round(.x, 1)))
  ),
  row.names = FALSE
)
res <- bind_rows(res, res_part)

saveRDS(
  res,
  file.path(dir_datos, paste0("estim_stan_", modo, "_", priors, "_cis.rds"))
)
cat("\nguardado", paste0("estim_stan_", modo, "_", priors, "_cis.rds"), "\n")
