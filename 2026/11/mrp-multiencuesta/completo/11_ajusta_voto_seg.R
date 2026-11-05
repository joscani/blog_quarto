# Ajuste de completo/stan/mrp_voto_seg.stan (voto + simpatía, seguro /
# dudoso) con las 4 casas. Datos de completo/10_datos_voto_seg.R.
#
# 4 cadenas en paralelo x 3 hilos (reduce_sum) = 12 núcleos.
# Salidas:
#   completo/datos/csv_voto_seg/   CSV de CmdStan (se escriben mientras ajusta)
#   completo/datos/fit_voto_seg.rds  el ajuste, con sus draws en memoria
#
#   Rscript 2026/11/mrp-multiencuesta/completo/11_ajusta_voto_seg.R > \
#     2026/11/mrp-multiencuesta/completo/datos/log_voto_seg.txt 2>&1
#   MODO=prueba Rscript ...   # 1 cadena, 100 + 100 iteraciones
#   ADAPT_DELTA=0.8 Rscript ...  # por defecto 0.95. 2026-11-03: en NumPyro, 0.8 no
#     ganó (ESS mín por minuto 13,1 frente a 12,2; 255 pasos con los dos). Con otro valor que
#     0.95, el CSV y el fit llevan el sufijo _ad<valor>

library(cmdstanr)
library(posterior)

modo <- Sys.getenv("MODO", "final")
adapt_delta <- as.numeric(Sys.getenv("ADAPT_DELTA", "0.95"))
stopifnot(adapt_delta > 0, adapt_delta < 1)
suf <- paste0(if (modo == "prueba") "_prueba", if (adapt_delta != 0.95) paste0("_ad", adapt_delta))
stopifnot(modo %in% c("final", "prueba"))
hilos <- as.integer(Sys.getenv("HILOS", "3")) # hilos por cadena; HILOS=2 deja 4 núcleos libres
stopifnot(hilos >= 1L)
dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_completo <- file.path(dir_proy, "completo", "datos")
dir_csv <- file.path(dir_completo, paste0("csv_voto_seg", suf))
dir.create(dir_csv, showWarnings = FALSE, recursive = TRUE)

x <- readRDS(file.path(dir_completo, "datos_voto_seg.rds"))
datos <- x$datos
datos$grainsize <- max(1L, as.integer(ceiling(datos$N / (4 * hilos))))
cat("modo:", modo, "  adapt_delta:", adapt_delta, "  filas:", datos$N, "  entradas de mezcla:", datos$E,
    "  grainsize:", datos$grainsize, "\n")

mod <- cmdstan_model(
  file.path(dir_proy, "completo", "stan", "mrp_voto_seg.stan"),
  include_paths = file.path(dir_proy, "stan"),
  cpp_options = list(stan_threads = TRUE)
)

t0 <- Sys.time()
cat("inicio:", format(t0), "\n")
fit <- mod$sample(
  data = datos,
  seed = 2026,
  chains = if (modo == "prueba") 1 else 4,
  parallel_chains = 4,
  threads_per_chain = hilos,
  iter_warmup = if (modo == "prueba") 100 else 1000,
  iter_sampling = if (modo == "prueba") 100 else 1000,
  init = 0,
  adapt_delta = adapt_delta,
  max_treedepth = 8, # como el (II): con 10 iba a ~25 s por iteración (2026-11-03)
  refresh = 10,
  output_dir = dir_csv
)
minutos <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

fit$draws()
fit$sampler_diagnostics() # en memoria antes de guardar
f_fit <- file.path(dir_completo, paste0("fit_voto_seg", suf, ".rds"))
saveRDS(list(fit = fit, min = minutos, lv = x$lv), f_fit)
cat("guardado", f_fit, "\n")

td <- fit$sampler_diagnostics(format = "draws_df")
cat("\n== diagnóstico ==\n")
cat("minutos:", round(minutos, 1), "\n")
cat("divergencias:", sum(td$divergent__), " de", nrow(td), "\n")
cat("en max_treedepth (8):", sum(td$treedepth__ >= 8), "\n")
cat("pasos por iteración: mediana", median(td$n_leapfrog__), "\n")
s <- fit$summary(c("alpha", "b_dud", "sd_dud_edad", "sd_dud_rec", "sd_casa", "sd_rec"))
cat("Rhat máx:", round(max(s$rhat, na.rm = TRUE), 3),
    "  ESS bulk mín:", round(min(s$ess_bulk, na.rm = TRUE)), "\n\n")
s$partido <- x$lv$voto[as.integer(sub(".*\\[(\\d+)\\]$", "\\1", s$variable))]
s$variable <- sub("\\[(\\d+)\\]", "", s$variable)
cat("== b_dud (dudoso frente a seguro, log-odds) y sd de sus interacciones ==\n")
print(as.data.frame(s[s$variable %in% c("b_dud", "sd_dud_edad", "sd_dud_rec"),
                      c("variable", "partido", "mean", "q5", "q95", "rhat", "ess_bulk")]),
      digits = 3, row.names = FALSE)
