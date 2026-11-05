# Modelo pequeño para los escenarios de participación (PARTICIPACION.md):
# cuánto se diferencia el voto de los que no están seguros de votar del de los
# que dicen "10, seguro que voto".
#
# Multinomial del partido (PP, PSOE, VOX, SUMAR, PODEMOS, RESTO) sobre seguro,
# con conteos agregados por casa x edad x recuerdo x seguro. El efecto de
# seguro varía por recuerdo y por edad (pooling parcial); la casa es un
# control. Solo las casas con prob_voto (CIS, 40dB, More in Common).
#
# En un logit multinomial, lo que entra solo como efecto principal (casa, el
# nivel de edad o de recuerdo) no cambia la diferencia entre seguros y no
# seguros; solo la cambian las interacciones con seguro. Por eso Δ depende de
# recuerdo y edad, y no de la casa con la que se calcule.
#
# Salida, en DIR_SALIDA (variable de entorno; fuera de datos/ para no tocar el
# post):
#   delta_seguro.rds  lista con
#     delta     array draw x rec x edad x grupo: log P(grupo | no seguro) -
#               log P(grupo | seguro). Solo importan las diferencias entre
#               grupos (la constante se cancela al normalizar)
#     grupos, rec_lv, edad_lv
#     c_mezcla  desplazamiento en logit de P(seguro) a P(seguro | da partido)
#     mezcla    tabla por casa de la que sale c_mezcla
#   fit_delta_seguro.rds  el brmsfit
#
#   DIR_SALIDA=/ruta Rscript 2026/11/mrp-multiencuesta/15_delta_seguro.R
#   DIR_SALIDA=/ruta VAR_VOTO=voto_sim Rscript ...   # con simpatía (_sim)

library(dplyr)
library(tidyr)
library(brms)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_salida <- Sys.getenv("DIR_SALIDA")
stopifnot(nzchar(dir_salida), dir.exists(dir_salida))

grupos <- c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "RESTO")
ndraws <- 1000L
# voto: intención directa, la del modelo de voto. voto_sim: + simpatía, que
# rellena a parte de los indecisos (01_armoniza.R). Salidas con sufijo _sim.
var_voto <- Sys.getenv("VAR_VOTO", "voto")
stopifnot(var_voto %in% c("voto", "voto_sim"))
suf <- if (var_voto == "voto_sim") "_sim" else ""
cat("variable de voto:", var_voto, "\n")

enc <- readRDS(file.path(dir_datos, "encuestas.rds"))
lv <- readRDS(file.path(dir_datos, "datos_stan.rds"))$lv
rec_lv <- lv$rec
edad_lv <- lv$edad

# ---- 1. Mezcla de seguros entre los que dan partido ------------------------

# El modelo de voto entra con todos los que dan partido (03_celdas.R); entre
# ellos hay más seguros que entre todos los entrevistados, que es lo que
# estima mrp_part.stan. c_mezcla pasa de una cosa a la otra en logit.
mezcla <- enc |>
  filter(!is.na(vota_seguro)) |>
  group_by(casa) |>
  summarise(
    n = n(),
    todos = mean(vota_seguro),
    con_partido = mean(vota_seguro[!is.na(voto)]),
    .groups = "drop"
  ) |>
  mutate(dif_logit = qlogis(con_partido) - qlogis(todos))
c_mezcla <- mean(mezcla$dif_logit)
cat("== % seguros, todos frente a los que dan partido ==\n")
print(as.data.frame(mezcla), row.names = FALSE)
cat("c_mezcla (media de dif_logit):", round(c_mezcla, 3), "\n\n")

# ---- 2. Datos agregados ------------------------------------------------------

dat <- enc |>
  mutate(v = .data[[var_voto]]) |>
  filter(
    !is.na(v),
    !is.na(vota_seguro),
    !is.na(rec),
    !is.na(edad)
  ) |>
  mutate(
    grupo = factor(
      if_else(v %in% grupos, as.character(v), "RESTO"),
      levels = grupos
    ),
    rec = factor(rec, levels = rec_lv),
    edad = factor(edad, levels = edad_lv),
    seguro = vota_seguro
  ) |>
  count(casa, edad, rec, seguro, grupo) |>
  pivot_wider(names_from = grupo, values_from = n, values_fill = 0) |>
  mutate(n = rowSums(across(all_of(grupos))))
dat$y <- as.matrix(dat[, grupos])
cat("filas agregadas:", nrow(dat), " entrevistas:", sum(dat$n), "\n")
print(dat |> count(casa, wt = n))

# ---- 3. Modelo ----------------------------------------------------------------

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
  y | trials(n) ~ seguro + casa + (1 + seguro | rec) + (1 + seguro | edad),
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
  file = file.path(dir_salida, paste0("fit_delta_seguro", suf)),
  file_refit = "on_change"
)
print(summary(fit))

# ---- 4. Δ draw a draw ----------------------------------------------------------

rejilla <- expand_grid(
  rec = factor(rec_lv, levels = rec_lv),
  edad = factor(edad_lv, levels = edad_lv),
  seguro = c(0, 1)
) |>
  mutate(casa = dat$casa[1], n = 1L)
ids <- round(seq(1, ndraws(fit), length.out = ndraws))
ep <- posterior_epred(fit, newdata = rejilla, draw_ids = ids) # draw x fila x grupo

i0 <- which(rejilla$seguro == 0)
i1 <- which(rejilla$seguro == 1)
stopifnot(
  identical(rejilla$rec[i0], rejilla$rec[i1]),
  identical(rejilla$edad[i0], rejilla$edad[i1])
)
dif <- log(ep[, i0, , drop = FALSE]) - log(ep[, i1, , drop = FALSE])
# filas de la rejilla: rec por fuera, edad por dentro
delta <- array(
  NA_real_,
  c(ndraws, length(rec_lv), length(edad_lv), length(grupos)),
  list(draw = NULL, rec = rec_lv, edad = edad_lv, grupo = grupos)
)
for (g in seq_along(grupos)) {
  delta[,,, g] <- aperm(
    array(dif[,, g], c(ndraws, length(edad_lv), length(rec_lv))),
    c(1, 3, 2)
  )
}

# comprobación: % de PP + VOX entre seguros y no seguros por recuerdo, con la
# edad 45-54 y la casa de referencia (mediana de los draws)
der <- function(s) {
  i <- which(rejilla$seguro == s & rejilla$edad == "45-54")
  apply(ep[, i, c("PP", "VOX")], c(1, 2), sum)
}
cat("\n== % PP + VOX según el modelo (edad 45-54, casa", dat$casa[1], ") ==\n")
print(
  tibble(
    rec = rec_lv,
    seguros = round(100 * apply(der(1), 2, median), 1),
    no_seguros = round(100 * apply(der(0), 2, median), 1)
  ) |>
    as.data.frame(),
  row.names = FALSE
)

saveRDS(
  list(
    delta = delta,
    grupos = grupos,
    rec_lv = rec_lv,
    edad_lv = edad_lv,
    c_mezcla = c_mezcla,
    mezcla = mezcla,
    var_voto = var_voto
  ),
  file.path(dir_salida, paste0("delta_seguro", suf, ".rds"))
)
cat("\nEscrito", file.path(dir_salida, paste0("delta_seguro", suf, ".rds")), "\n")
