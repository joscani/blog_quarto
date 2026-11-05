# Modelo completo, parte 1: ajuste de completo/stan/mrp_ideol.stan (versión
# sencilla, una fila por entrevistado y vectorizada: sin reduce_sum).
#
# init = 0.5: en las pruebas con mrp_ideol_prov.stan, una de las cuatro
# cadenas adaptaba un paso 5 veces menor con la inicialización por defecto
# (uniforme en -2..2).
#
# Entrada: completo/datos/datos_ideol.rds (02_datos_ideol.R).
# Salida:  completo/datos/fit_ideol.rds (el CmdStanMCMC, con save_object) y
#          los CSV de las cadenas en completo/datos/csv_ideol/.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/03_ajusta_ideol.R

library(cmdstanr)
library(posterior)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo")
dir_csv <- file.path(dir_completo, "datos", "csv_ideol")
dir.create(dir_csv, showWarnings = FALSE, recursive = TRUE)

d <- readRDS(file.path(dir_completo, "datos", "datos_ideol.rds"))
mod <- cmdstan_model(file.path(dir_completo, "stan", "mrp_ideol.stan"))

inicio <- Sys.time()
fit <- mod$sample(
  data = d$datos,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  init = 0.5,
  seed = 2026,
  refresh = 100,
  output_dir = dir_csv
)
cat("\ntiempo:", format(Sys.time() - inicio), "\n")

print(fit$diagnostic_summary())
resumen <- fit$summary(
  c("c", "b_mujer", "a_casa", "a_edad", "a_rec", "sd_casa", "sd_edad",
    "sd_rec", "sd_ccaa", "ideol_nac")
)
print(resumen, n = 60)
cat("\nRhat máximo:", round(max(fit$summary()$rhat, na.rm = TRUE), 3), "\n")
cat("ESS bulk mínimo:", round(min(fit$summary()$ess_bulk, na.rm = TRUE)), "\n")

fit$save_object(file.path(dir_completo, "datos", "fit_ideol.rds"))
cat("Escrito", file.path(dir_completo, "datos", "fit_ideol.rds"), "\n")
