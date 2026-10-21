# ============================================================================
# FULL BAYESIAN 29-N · 07: ¿de dónde salen las divergencias del módulo A?
#
# 02_ajustar_estado.R las quitó subiendo adapt_delta a 0,99. Bayesian
# Workflow (§12.4, "folk theorem") pide mirar antes la geometría: una
# divergencia suele señalar una zona del posterior que el sampler no puede
# recorrer (típicamente un embudo: una sigma de grupo cerca de 0).
#
# 1. Reajuste con adapt_delta = 0,8 (el valor por defecto de Stan).
# 2. Para cada escala (sigma_rw, sigma_casa, sigma_enc), compara su
#    log-valor en las transiciones divergentes y en las demás. Si las
#    divergentes se concentran en valores pequeños, hay embudo en esa escala.
# 3. Gráfico de pares de la escala más señalada frente a uno de sus efectos.
#
# Salidas (en 2026/10/full-bayesian/):
#   divergencias_escalas.csv   escala: n divergencias, mediana del log en
#                              divergentes y en el resto, diferencia en sd
#   divergencias_pares.png     pares de la escala más señalada
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/07_divergencias.R [semilla]
# ============================================================================

source(here::here("2026/10/full-bayesian/01_datos.R"))
suppressMessages({
  library(cmdstanr)
  library(posterior)
  library(bayesplot)
  library(ggplot2)
})

# semilla: por defecto 333, la que más divergencias dio entre 1, 22, 333 y
# 4444 (3-17 de 6.000; con la 2911 de 02_ajustar_estado.R salieron 0)
args <- commandArgs(trailingOnly = TRUE)
seed_estado <- if (length(args) > 0) as.integer(args[1]) else 333L
mod_a <- cmdstan_model(file.path(dir_fb, "modelo_estado.stan"))
da <- datos_estado()

fit <- mod_a$sample(
  data = stan_estado(da),
  seed = seed_estado,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1500,
  adapt_delta = 0.8,
  refresh = 0,
  show_messages = FALSE
)
np <- nuts_params(fit)
div <- np$Value[np$Parameter == "divergent__"] == 1
cat("Divergencias con adapt_delta = 0,8:", sum(div), "de", length(div), "\n")

escalas <- c(
  sprintf("sigma_rw[%d]", 1:(da$K - 1)),
  sprintf("sigma_casa[%d]", 1:(da$K - 1)),
  sprintf("sigma_enc[%d]", 1:(da$K - 1))
)
dr <- unclass(fit$draws(escalas, format = "draws_matrix"))

tab <- tibble(escala = escalas) |>
  mutate(
    partido = da$cats[as.integer(sub(".*\\[(\\d+)\\]", "\\1", escala))],
    log_div = sapply(escala, \(v) median(log(dr[div, v]))),
    log_resto = sapply(escala, \(v) median(log(dr[!div, v]))),
    # cuántas sd del posterior separan a las divergentes del resto
    # (negativo: las divergentes están en valores pequeños de la escala)
    dif_sd = sapply(escala, \(v) {
      (median(log(dr[div, v])) - median(log(dr[!div, v]))) / sd(log(dr[, v]))
    })
  ) |>
  arrange(dif_sd)
cat("\nEscalas ordenadas por cuánto se separan las divergentes:\n")
print(tab |> mutate(across(where(is.numeric), \(x) round(x, 2))), n = Inf)
write_csv(tab, file.path(dir_fb, "divergencias_escalas.csv"))

# pares: la escala más señalada frente al log de los demás de su tipo
if (sum(div) > 0) {
  peor <- tab$escala[1]
  otras <- setdiff(escalas[startsWith(escalas, sub("\\[.*", "", peor))], peor)[1:2]
  g <- mcmc_pairs(
    fit$draws(c(peor, otras)),
    np = np,
    transformations = "log",
    off_diag_args = list(size = 0.6, alpha = 0.3)
  )
  ggsave(file.path(dir_fb, "divergencias_pares.png"), g, width = 7, height = 7, dpi = 150)
  cat("\nGráfico:", file.path(dir_fb, "divergencias_pares.png"), "\n")
}
