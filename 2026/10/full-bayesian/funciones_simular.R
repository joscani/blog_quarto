# ============================================================================
# FULL BAYESIAN 29-N · funciones de la combinación draw a draw
#
# Las usan 12_ajustar_conjunto.R, 16_ablacion_error.R y
# 17_sensibilidad_omega.R. Necesitan simular_escanos.R.
#
#   combinar_conjunto(v, e): voto del día E del modelo conjunto + error del
#     consenso, en logit de PP, PSOE, VOX y SUMAR+PODEMOS.
#   simular_conjunto(fit, dc): voto con error, y escaños con el escenario
#     mapa (simular(listas = "mapa")).
#
# Las funciones de la vieja arquitectura de dos módulos (combinar() y
# simular_fb()) están en _archivo_modulos_ab/funciones_modulo_ab.R.
# ============================================================================

# v: draws x 14 familias de 2026 (proporciones, nombres de fams_2026);
# e: draws x 4 (error en logit de PP, PSOE, VOX y SUMAR+PODEMOS).
# Igual que combinar() del módulo A+B: el error va al logit de los cuatro
# grupos; el resto (SALF, regionales y RESTO) no lleva error propio y absorbe
# la renormalización en proporción a su tamaño. Devuelve % en el orden de
# fams14 (simular_escanos.R).
combinar_conjunto <- function(v, e) {
  sp <- v[, "SUMAR"] + v[, "PODEMOS"]
  otras <- setdiff(colnames(v), c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS"))
  resto <- rowSums(v[, otras, drop = FALSE])
  g <- cbind(PP = v[, "PP"], PSOE = v[, "PSOE"], VOX = v[, "VOX"], SUMAR = sp)
  g <- plogis(qlogis(g) + e[, colnames(g)])
  tot <- rowSums(g) + resto
  g <- g / tot
  f_resto <- (1 / tot)
  out <- cbind(
    g[, c("PP", "PSOE", "VOX")],
    SUMAR = g[, "SUMAR"] * v[, "SUMAR"] / sp,
    PODEMOS = g[, "SUMAR"] * v[, "PODEMOS"] / sp,
    v[, otras, drop = FALSE] * f_resto
  )
  colnames(out) <- sub("EH_Bildu", "EH Bildu", colnames(out))
  out <- 100 * out[, fams14]
  stopifnot(all(abs(rowSums(out) - 100) < 1e-6))
  out
}

# escaños con el escenario mapa a partir del ajuste conjunto
simular_conjunto <- function(fit, dc, seed = 2029) {
  Kc <- dc$K[dc$C]
  Tc <- dc$T[dc$C]
  v <- unclass(fit$draws(
    sprintf("v_obj[%d,%d]", seq_len(Kc), Tc),
    format = "draws_matrix"
  ))[, seq_len(Kc)]
  colnames(v) <- fams_campana(dc$objetivo)
  e <- unclass(fit$draws("err_new", format = "draws_matrix"))[, 1:4]
  colnames(e) <- c("PP", "PSOE", "VOX", "SUMAR")
  shares <- combinar_conjunto(v, e)
  set.seed(seed)
  sim <- simular(
    n_draws = nrow(shares),
    listas = "mapa",
    provincial = TRUE,
    shares_ext = shares
  )
  sim$shares <- shares
  sim
}
