# Modelo completo, parte 3: PRUEBA de parametrización del modelo de voto.
#
# Compara completo/stan/mrp_voto2.stan (suma cero entre niveles; recuerdo y
# edad centrados) con el ajuste de stan/mrp_voto.stan, que llegó a
# max_treedepth = 8 (255 pasos) en las 4.000 iteraciones (149 min).
#
# Mismos datos (datos/datos_stan.rds, solo lectura) y mismos ajustes que
# stan/ajusta.R (adapt_delta 0.95, init 0, 3 hilos), salvo max_treedepth =
# 10, para ver cuántos pasos necesita de verdad. Ajuste corto: 1 cadena, 200
# iteraciones de warmup y 100 de sampling.
#
# Salida: CSV en completo/datos/csv_prueba_voto/ y el resumen en pantalla.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/06_prueba_voto.R
#   MODELO=mrp_voto3 Rscript ...   # sin CCAA x recuerdo

library(cmdstanr)
library(posterior)

# modelo a probar: mrp_voto2 (por defecto) o mrp_voto3 (sin CCAA x recuerdo)
modelo <- Sys.getenv("MODELO", "mrp_voto2")

hilos <- 3L
dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_csv <- file.path(dir_proy, "completo", "datos", paste0("csv_prueba_", modelo))
dir.create(dir_csv, showWarnings = FALSE, recursive = TRUE)

x <- readRDS(file.path(dir_proy, "datos", "datos_stan.rds"))
stopifnot(x$modo == "nuts")
datos <- c(
  x$datos,
  list(
    s_b_mujer = 1,
    grainsize = max(1L, as.integer(ceiling(x$datos$N / (4 * hilos))))
  )
)
cat("filas de voto:", datos$N, "  grainsize:", datos$grainsize, "\n")

mod <- cmdstan_model(
  file.path(dir_proy, "completo", "stan", paste0(modelo, ".stan")),
  include_paths = file.path(dir_proy, "stan"),
  cpp_options = list(stan_threads = TRUE)
)

t0 <- Sys.time()
fit <- mod$sample(
  data = datos,
  seed = 2026,
  chains = 1,
  threads_per_chain = hilos,
  iter_warmup = 200,
  iter_sampling = 100,
  init = 0,
  adapt_delta = 0.95,
  max_treedepth = 10,
  refresh = 25,
  output_dir = dir_csv
)
minutos <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

td <- fit$sampler_diagnostics(format = "draws_df")
cat("\n== sampling (100 iteraciones) ==\n")
cat("minutos en total:", round(minutos, 1), "\n")
cat("pasos por iteración: mediana", median(td$n_leapfrog__),
    " media", round(mean(td$n_leapfrog__)), " (mrp_voto.stan: 255, en el tope)\n")
cat("profundidad del árbol:\n")
print(table(td$treedepth__))
cat("divergencias:", sum(td$divergent__), "\n")
cat("tamaño de paso:", round(fit$metadata()$step_size_adaptation, 4), "\n")
