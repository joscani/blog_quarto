# Para cada draw:
#   1. v: voto de la media de la industria el día E (7 categorías).
#   2. e: error del consenso (logit, 4 grupos). Se aplica al logit de PP,
#      PSOE, VOX y SUMAR+PODEMOS (la historia solo conoce a Sumar/Unidas
#      Podemos agregado). El resto (SALF + BLOQUE) no lleva error propio;
#      al renormalizar a 1 absorbe el ajuste en proporción a su tamaño.
#   3. SUMAR+PODEMOS se reparte con el peso de cada lista en el draw (antes
#      del error); SALF + BLOQUE, igual.
combinar <- function(v, e) {
  sp <- v[, "SUMAR"] + v[, "PODEMOS"]
  resto <- v[, "SALF"] + v[, "BLOQUE"]
  g <- cbind(PP = v[, "PP"], PSOE = v[, "PSOE"], VOX = v[, "VOX"], SUMAR = sp)
  g <- plogis(qlogis(g) + e[, colnames(g)])
  tot <- rowSums(g) + resto
  g <- g / tot
  resto <- resto / tot
  cbind(
    g[, c("PP", "PSOE", "VOX")],
    SUMAR = g[, "SUMAR"] * v[, "SUMAR"] / sp,
    PODEMOS = g[, "SUMAR"] * v[, "PODEMOS"] / sp,
    SALF = resto * v[, "SALF"] / (v[, "SALF"] + v[, "BLOQUE"]),
    BLOQUE = resto * v[, "BLOQUE"] / (v[, "SALF"] + v[, "BLOQUE"])
  )
}

# Los dos módulos son independientes: el draw s de A se empareja con el draw
# s de B (los dos con 6.000 draws).
simular_fb <- function(fit_a, fit_b, seed = 2029, var_err = "err_new_casa") {
  K <- length(cats_estado)
  v_T <- unclass(fit_a$draws(
    sprintf("v[%d,%d]", seq_len(K), semana_eleccion),
    format = "draws_matrix"
  ))[, seq_len(K)]
  colnames(v_T) <- cats_estado
  err <- unclass(fit_b$draws(var_err, format = "draws_matrix"))[, 1:4]
  colnames(err) <- cats_error
  n_draws <- nrow(v_T)
  stopifnot(nrow(err) == n_draws)

  v_err <- combinar(v_T, err)
  stopifnot(all(abs(rowSums(v_err) - 1) < 1e-9))

  # composición del BLOQUE: draws del modelo 2 de brms
  set.seed(seed)
  comp <- tibble(time = time_eleccion, bloque = 1) |>
    add_epred_draws(ajustar_bloque(), ndraws = n_draws) |>
    ungroup() |>
    select(.draw, .category, .epred) |>
    pivot_wider(names_from = .category, values_from = .epred)
  comp <- as.matrix(comp[, fams_bloque_fit])
  comp <- comp / rowSums(comp)

  shares_ext <- 100 *
    cbind(v_err[, setdiff(colnames(v_err), "BLOQUE")], v_err[, "BLOQUE"] * comp)
  colnames(shares_ext) <- fams14

  # escaños: shock provincial, reglas del mapa, umbral del 3 % y D'Hondt con
  # el reparto del RD 806/2026
  set.seed(seed)
  sim <- simular(
    n_draws = n_draws,
    listas = "mapa",
    provincial = TRUE,
    shares_ext = shares_ext
  )
  sim$v_err <- v_err
  sim
}
