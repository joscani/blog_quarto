# Escenarios de participación (PARTICIPACION.md): voto de los no seguros con
# los indecisos imputados por su ideología.
#
# Con voto_sim, el 48,5 % de los abstencionistas no seguros sigue sin partido,
# y son más de izquierda por ideología. Aquí:
#   1. Multinomial del partido (los grupos de 15_delta_seguro.R) con voto_sim,
#      sobre los decididos, con la ideología en tramos (4 + "sin ideología")
#      que interactúa con el recuerdo. La casa × tramo absorbe que el CIS
#      pregunta de 1 a 10 (centro entre 0,444 y 0,556) y las otras de 0 a 10
#      (centro 0,5).
#   2. Voto predicho de todos los no seguros, decididos e indecisos.
#   3. Corrección de Δ por recuerdo, draw a draw:
#        corr = log P(grupo | todos los no seguros) - log P(grupo | no seguros
#               decididos)
#      y Δ_imp = Δ_sim (de 15_delta_seguro.R con voto_sim) + corr.
# Supuesto: un indeciso votaría como un decidido con la misma casa, edad,
# recuerdo, seguridad e ideología.
#
# Entrada: DIR_SALIDA/delta_seguro_sim.rds. Salidas en DIR_SALIDA:
#   delta_seguro_imp.rds   misma estructura que delta_seguro.rds
#   fit_delta_ideologia.rds
#
#   DIR_SALIDA=/ruta Rscript 2026/11/mrp-multiencuesta/15b_imputa_ideologia.R

library(dplyr)
library(tidyr)
library(brms)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_salida <- Sys.getenv("DIR_SALIDA")
stopifnot(nzchar(dir_salida), dir.exists(dir_salida))

ds <- readRDS(file.path(dir_salida, "delta_seguro_sim.rds"))
grupos <- ds$grupos
nd <- dim(ds$delta)[1]

enc <- readRDS(file.path(dir_datos, "encuestas.rds")) |>
  filter(!is.na(vota_seguro), !is.na(rec), !is.na(edad)) |>
  mutate(
    rec = factor(rec, levels = ds$rec_lv),
    edad = factor(edad, levels = ds$edad_lv),
    seguro = vota_seguro,
    tramo = cut(
      ideol01,
      c(-Inf, 0.3, 0.5, 0.7, Inf),
      labels = c("izq", "cizq", "cder", "der"),
      right = FALSE
    ),
    tramo = factor(
      if_else(is.na(tramo), "ns", as.character(tramo)),
      levels = c("izq", "cizq", "cder", "der", "ns")
    ),
    grupo = factor(
      if_else(voto_sim %in% grupos, as.character(voto_sim), "RESTO"),
      levels = grupos
    ),
    grupo = if_else(is.na(voto_sim), NA, grupo)
  )
cat("== no seguros por tramo de ideología: decididos e indecisos ==\n")
print(
  enc |>
    filter(seguro == 0) |>
    count(tramo, decidido = !is.na(grupo)) |>
    pivot_wider(names_from = decidido, values_from = n, names_prefix = "decidido_") |>
    as.data.frame(),
  row.names = FALSE
)

# ---- 1. Modelo con ideología ----------------------------------------------------

dat <- enc |>
  filter(!is.na(grupo)) |>
  count(casa, edad, rec, tramo, seguro, grupo) |>
  pivot_wider(names_from = grupo, values_from = n, values_fill = 0) |>
  mutate(n = rowSums(across(all_of(grupos))))
dat$y <- as.matrix(dat[, grupos])
cat("\nfilas agregadas:", nrow(dat), " entrevistas:", sum(dat$n), "\n")

dpars <- paste0("mu", grupos[-1])
priors <- do.call(
  c,
  lapply(dpars, function(dp) {
    c(
      set_prior("normal(0, 1.5)", class = "b", dpar = dp),
      set_prior("normal(0, 1)", class = "sd", dpar = dp)
    )
  })
)
fit <- brm(
  y | trials(n) ~ seguro + casa + (1 | edad) + (1 + seguro | rec) +
    (1 | tramo) + (1 | rec:tramo) + (1 | casa:tramo),
  data = dat,
  family = multinomial(),
  prior = priors,
  chains = 4,
  cores = 4,
  iter = 2000,
  seed = 2026,
  backend = "cmdstanr",
  refresh = 100,
  silent = 0,
  file = file.path(dir_salida, "fit_delta_ideologia"),
  file_refit = "on_change"
)
print(summary(fit))

# ---- 2. Voto predicho de los no seguros -----------------------------------------

ns <- enc |>
  filter(seguro == 0) |>
  count(casa, edad, rec, tramo, seguro, decidido = !is.na(grupo))
nuevos <- ns |> distinct(casa, edad, rec, tramo, seguro) |> mutate(n = 1L)
ids <- round(seq(1, ndraws(fit), length.out = nd))
ep <- posterior_epred(
  fit,
  newdata = nuevos,
  draw_ids = ids,
  allow_new_levels = TRUE # alguna combinación rec:tramo sin decididos
) # draw x fila x grupo
i_fila <- match(
  paste(ns$casa, ns$edad, ns$rec, ns$tramo),
  paste(nuevos$casa, nuevos$edad, nuevos$rec, nuevos$tramo)
)
stopifnot(!anyNA(i_fila))

# ---- 3. Corrección por recuerdo -------------------------------------------------

corr <- array(
  0,
  c(nd, length(ds$rec_lv), length(grupos)),
  list(draw = NULL, rec = ds$rec_lv, grupo = grupos)
)
resumen <- list()
for (r in seq_along(ds$rec_lv)) {
  i_dec <- which(ns$rec == ds$rec_lv[r] & ns$decidido)
  i_tod <- which(ns$rec == ds$rec_lv[r])
  if (length(i_dec) == 0) next
  media <- function(i) {
    w <- ns$n[i] / sum(ns$n[i])
    apply(ep[, i_fila[i], , drop = FALSE], c(1, 3), function(z) sum(z * w))
  }
  p_dec <- media(i_dec)
  p_tod <- media(i_tod)
  corr[, r, ] <- log(p_tod) - log(p_dec)
  resumen[[r]] <- tibble(
    rec = ds$rec_lv[r],
    n_dec = sum(ns$n[i_dec]),
    n_ind = sum(ns$n[i_tod]) - sum(ns$n[i_dec]),
    der_dec = 100 * median(p_dec[, "PP"] + p_dec[, "VOX"]),
    der_todos = 100 * median(p_tod[, "PP"] + p_tod[, "VOX"]),
    izq_dec = 100 * median(p_dec[, "PSOE"] + p_dec[, "SUMAR"] + p_dec[, "PODEMOS"]),
    izq_todos = 100 * median(p_tod[, "PSOE"] + p_tod[, "SUMAR"] + p_tod[, "PODEMOS"])
  )
}
cat("\n== no seguros: % PP + VOX e izquierda estatal, decididos frente a todos ==\n")
print(
  bind_rows(resumen) |>
    mutate(across(der_dec:izq_todos, \(z) round(z, 1))) |>
    as.data.frame(),
  row.names = FALSE
)

delta <- ds$delta
for (e in seq_along(ds$edad_lv)) {
  delta[, , e, ] <- delta[, , e, ] + corr
}
saveRDS(
  list(
    delta = delta,
    grupos = grupos,
    rec_lv = ds$rec_lv,
    edad_lv = ds$edad_lv,
    c_mezcla = ds$c_mezcla,
    mezcla = ds$mezcla,
    var_voto = "imp",
    corr = corr
  ),
  file.path(dir_salida, "delta_seguro_imp.rds")
)
cat("\nEscrito", file.path(dir_salida, "delta_seguro_imp.rds"), "\n")
