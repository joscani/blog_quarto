# Modelo completo, parte 2: ajuste de completo/stan/mrp_part3.stan.
#
# adapt_delta = 0.95: el modelo de ideología, con la misma estructura, dejó
# 9 divergencias de 4.000 con el valor por defecto (0.8).
#
# Entrada: completo/datos/datos_part.rds (04_datos_part.R).
# Salida:  completo/datos/fit_part.rds y los CSV en completo/datos/csv_part/.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/05_ajusta_part.R

library(cmdstanr)
library(posterior)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo")
dir_csv <- file.path(dir_completo, "datos", "csv_part")
dir.create(dir_csv, showWarnings = FALSE, recursive = TRUE)

d <- readRDS(file.path(dir_completo, "datos", "datos_part.rds"))
mod <- cmdstan_model(file.path(dir_completo, "stan", "mrp_part3.stan"))

inicio <- Sys.time()
fit <- mod$sample(
  data = d$datos,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  init = 0.5,
  seed = 2026,
  refresh = 100,
  output_dir = dir_csv
)
cat("\ntiempo:", format(Sys.time() - inicio), "\n")

print(fit$diagnostic_summary())
resumen <- fit$summary(
  c("c", "c_media", "b_mujer", "a_ideol", "a_edad", "a_edu", "a_rec",
    "a_mes", "sd_mes", "sd_edad", "sd_edu", "sd_rec", "sd_ccaa")
)
print(resumen, n = 80)
cat("\nRhat máximo:", round(max(fit$summary()$rhat, na.rm = TRUE), 3), "\n")
cat("ESS bulk mínimo:", round(min(fit$summary()$ess_bulk, na.rm = TRUE)), "\n")

fit$save_object(file.path(dir_completo, "datos", "fit_part.rds"))
cat("Escrito", file.path(dir_completo, "datos", "fit_part.rds"), "\n")
