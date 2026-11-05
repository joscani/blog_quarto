# Modelo completo, parte 3, PROTOTIPO solo con el CIS: ajuste de
# completo/stan/mrp_voto_cis.stan.
#
# Entrada: completo/datos/datos_voto_cis.rds (07_datos_voto_cis.R).
# Salida:  completo/datos/fit_voto_cis.rds y los CSV en
#          completo/datos/csv_voto_cis/.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/08_ajusta_voto_cis.R

library(cmdstanr)
library(posterior)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo")
dir_csv <- file.path(dir_completo, "datos", "csv_voto_cis")
dir.create(dir_csv, showWarnings = FALSE, recursive = TRUE)

d <- readRDS(file.path(dir_completo, "datos", "datos_voto_cis.rds"))
mod <- cmdstan_model(file.path(dir_completo, "stan", "mrp_voto_cis.stan"))

inicio <- Sys.time()
fit <- mod$sample(
  data = d$datos,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  init = 0,
  seed = 2026,
  refresh = 50,
  output_dir = dir_csv
)
cat("\ntiempo:", format(Sys.time() - inicio), "\n")

td <- fit$sampler_diagnostics(format = "draws_df")
cat("pasos por iteración: mediana", median(td$n_leapfrog__), "\n")
print(table(td$treedepth__))
print(fit$diagnostic_summary())
sds <- c("sd_mes", "sd_edad", "sd_edu", "sd_rec", "sd_ccaa", "sd_prov",
         "sd_ideol", "sd_nivel")
print(fit$summary(sds), n = 130)
cat("\nRhat máximo:", round(max(fit$summary()$rhat, na.rm = TRUE), 3), "\n")
cat("ESS bulk mínimo:", round(min(fit$summary()$ess_bulk, na.rm = TRUE)), "\n")

fit$save_object(file.path(dir_completo, "datos", "fit_voto_cis.rds"))
cat("Escrito", file.path(dir_completo, "datos", "fit_voto_cis.rds"), "\n")
