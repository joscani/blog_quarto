# =============================================================================
# explica_escanos.R
#
# Script pedagógico que explica pieza a pieza el código del post
# "¿Mayoría absoluta? Escaños con incertidumbre. Andalucía 2026"
#
# Cada sección tiene un ejemplo de juguete que puedes ejecutar sola,
# sin necesidad de los datos reales ni del modelo brms.
# =============================================================================

library(tidyverse)
library(gtools)   # rdirichlet
library(MASS)     # mvrnorm (sin cargar dplyr encima para que select no se enmascare)

# =============================================================================
# 1. D'HONDT: el algoritmo de reparto de escaños
# =============================================================================
#
# D'Hondt asigna escaños uno a uno. En cada ronda, cada partido tiene un
# "cociente" = votos_acumulados / (escaños_ya_asignados + 1).
# El partido con el cociente más alto se lleva el escaño.
#
# Ejemplo mínimo: 3 partidos, 5 escaños.
# Partidos:  A=100 votos,  B=60 votos,  C=40 votos

dhondt <- function(votos, escanos) {
  result <- setNames(integer(length(votos)), names(votos))
  for (i in seq_len(escanos)) {
    ganador <- which.max(votos / (result + 1))
    result[ganador] <- result[ganador] + 1
  }
  result
}

votos_ejemplo <- c(A = 100, B = 60, C = 40)
dhondt(votos_ejemplo, escanos = 5)
# Ronda 1: A=100/1=100 ← gana. result=c(1,0,0)
# Ronda 2: A=100/2=50, B=60/1=60 ← B gana. result=c(1,1,0)
# Ronda 3: A=100/2=50 ← A gana. result=c(2,1,0)
# Ronda 4: A=100/3=33.3, B=60/2=30, C=40/1=40 ← C gana. result=c(2,1,1)
# Ronda 5: A=100/3=33.3 ← A gana. result=c(3,1,1)

# El algoritmo sólo necesita proporciones (no votos absolutos), porque
# lo que importa es el cociente RELATIVO entre partidos:
proporciones_ejemplo <- c(A = 0.5, B = 0.3, C = 0.2)
dhondt(proporciones_ejemplo, escanos = 5)
# Resultado idéntico: c(A=3, B=1, C=1)


# =============================================================================
# 2. DISTRIBUCIÓN DIRICHLET: añadir incertidumbre sobre proporciones
# =============================================================================
#
# Tenemos los resultados provinciales de 2022 como referencia, pero esos
# datos también son inciertos (muestras, no censos). La Dirichlet nos permite
# "sacudir" esas proporciones de forma coherente (siguen sumando 1).
#
# La Dirichlet(alpha) tiene media = alpha / sum(alpha) y varianza inversamente
# proporcional a sum(alpha): alpha grande → distribución concentrada (poco ruido).
#
# En el post, alpha = proporción_2022 * (votos_candidatura / scale_factor)
# Con scale_factor=1000 y votos~500k, alpha~500, que da sd≈2-4pp por partido.

# Ejemplo con 3 "partidos" y proporciones conocidas de referencia:
prop_ref <- c(partido_A = 0.50, partido_B = 0.30, partido_C = 0.20)
votos_cand <- 500000
scale_factor <- 1000

alpha <- prop_ref * (votos_cand / scale_factor)
alpha
# alpha_A=250, alpha_B=150, alpha_C=100  → suma=500 (concentrado cerca de prop_ref)

set.seed(42)
muestras_dir <- rdirichlet(5, alpha)
muestras_dir
# Cada fila es una "realización" de las proporciones con ruido

# Compara con alpha pequeño (scale_factor=50000 → más incertidumbre):
alpha_vago <- prop_ref * (votos_cand / 50000)
muestras_dir_vago <- rdirichlet(5, alpha_vago)
muestras_dir_vago
# Mucho más disperso; las proporciones se alejan bastante de 0.5/0.3/0.2


# =============================================================================
# 3. AJUSTE PROVINCIAL: ratio multiplicativo + normalización
# =============================================================================
#
# El modelo bayesiano nos da el porcentaje REGIONAL de cada partido en el día
# de las elecciones. Necesitamos convertirlo a porcentajes PROVINCIALES.
#
# Idea: si en 2022 el PP sacó en Almería un 10% MÁS que su media regional,
# asumimos que ese "exceso" se mantiene en 2026.
#
# Fórmula:
#   prov_estimado = draw_regional * (prov_2022 / media_regional_2022)
#   y luego normalizamos para que las provincias sumen 1.
#
# Ejemplo de juguete: 2 partidos, 1 provincia

draw_regional <- c(partido_A = 0.45, partido_B = 0.35)  # resto = 1-sum
# (ignoramos "resto" para simplificar)
draw_regional <- draw_regional / sum(draw_regional)       # normalizar

prov_2022       <- c(partido_A = 0.55, partido_B = 0.30) # resultados 2022 en esa provincia
media_reg_2022  <- c(partido_A = 0.50, partido_B = 0.35) # media regional 2022

ratio <- prov_2022 / media_reg_2022
# ratio_A = 0.55/0.50 = 1.10  → en esta provincia, A supera su media en 10%
# ratio_B = 0.30/0.35 = 0.857 → B queda por debajo de su media en ~15%
ratio

raw  <- draw_regional * ratio   # porcentajes "crudos" no normalizados
norm <- raw / sum(raw)          # normalizamos para que sumen 1
norm
# norm_A > draw_regional[A]: la ventaja provincial de A infla su estimación local
# norm_B < draw_regional[B]: la desventaja provincial de B la reduce


# =============================================================================
# 4. sim_escanos_draw(): la función que une todo
# =============================================================================
#
# Para un solo draw del posterior:
#   a) Tomamos los porcentajes regionales del draw.
#   b) Para cada provincia:
#      - Muestreamos proporciones provinciales con Dirichlet (incertidumbre 2022).
#      - Calculamos el ratio provincial vs media regional.
#      - Ajustamos el draw regional por ese ratio y normalizamos.
#      - Aplicamos D'Hondt con los escaños que corresponden a esa provincia.
#   c) Sumamos los escaños de todas las provincias → vector de 6 partidos.
#
# Ejemplo simplificado: 2 provincias, 2 partidos, 3 escaños cada una

partidos_ej <- c("A", "B")

escanos_provs <- tibble(
  provincia   = c("Norte", "Sur"),
  escanos     = c(3, 3),
  A           = c(60, 40),   # % partido A en 2022 en cada provincia
  B           = c(40, 60),   # % partido B en 2022
  votos_cand  = c(100000, 150000)
)

regional_2022_ej <- c(A = 50, B = 50)  # media regional 2022 (en %)

sim_escanos_draw_ej <- function(draw_vec, ref, reg_2022, scale_factor = 1000) {
  escanos_totales <- setNames(integer(length(partidos_ej)), partidos_ej)

  for (i in seq_len(nrow(ref))) {
    # Proporciones de la provincia en 2022
    prov_pct_2022 <- as.numeric(ref[i, partidos_ej]) / 100
    names(prov_pct_2022) <- partidos_ej

    # Dirichlet: ruido sobre el patrón provincial
    alpha            <- prov_pct_2022 * (ref$votos_cand[i] / scale_factor)
    prov_pct_sampled <- rdirichlet(1, alpha)[1, ]
    names(prov_pct_sampled) <- partidos_ej

    # Ratio provincial vs media regional 2022
    ratio <- prov_pct_sampled / (reg_2022[partidos_ej] / 100)

    # Ajuste multiplicativo + normalización
    raw  <- draw_vec[partidos_ej] * ratio
    norm <- raw / sum(raw)

    # D'Hondt
    escanos_totales <- escanos_totales + dhondt(norm, ref$escanos[i])
  }
  escanos_totales
}

set.seed(99)
draw_ejemplo <- c(A = 0.52, B = 0.48)  # un draw del posterior: A un poco por encima
sim_escanos_draw_ej(draw_ejemplo, escanos_provs, regional_2022_ej)
# Norte: A domina → esperable que A se lleve 2-3 escaños ahí
# Sur:   B domina → B se lleva 2-3 allí
# En total, deberían salir exactamente 6 escaños


# =============================================================================
# 5. ERROR HISTÓRICO: MASS::mvrnorm para ruido multivariante correlado
# =============================================================================
#
# Las encuestas se equivocan. En 2022 y 2018 el error no fue independiente
# entre partidos: si el PP se sobreestimó, VOX tendió a subestimarse.
# Capturamos esa correlación con una normal multivariante.
#
# Sigma_reg = covarianza del error histórico + regularización diagonal (evita
# que la matriz sea singular y añade un mínimo de ruido individual).
#
# scale=0.20/1e4 calibra que la desviación típica sea ~2-3pp (0.02-0.03 en escala 0-1).

error_hist <- matrix(
  c(
  # A     B
    -2,    3,   # elección 2020 (simulada)
     1,   -2    # elección 2016 (simulada)
  ),
  nrow = 2, byrow = TRUE,
  dimnames = list(c("2020", "2016"), c("A", "B"))
)

Sigma_reg <- cov(error_hist) + diag(0.25, 2)
Sigma_reg
# Si A se sobreestima, B tiende a subestimarse → covarianza negativa

set.seed(123)
errores <- MASS::mvrnorm(
  n     = 10,           # 10 draws (en el post son ~12.000)
  mu    = rep(0, 2),    # sin sesgo sistemático
  Sigma = Sigma_reg * 0.20 / 1e4
)
errores
# Cada fila: error a sumar al draw regional antes de pasar a D'Hondt
# Filas con A negativo tienden a tener B positivo (correlación negativa)


# =============================================================================
# 6. pmap() + asplit(): aplicar la simulación a todos los draws
# =============================================================================
#
# draws_wide tiene una fila por draw del posterior (e.g. 12.000 filas).
# Para cada fila necesitamos:
#   - Los 6 porcentajes de voto (columnas del tibble).
#   - El vector de error histórico de esa iteración.
#
# pmap() itera en paralelo sobre listas de argumentos.
# asplit(matrix, MARGIN=1) convierte una matriz en una lista de vectores-fila:
#   asplit(error_draws, 1) → lista de 12.000 vectores, uno por draw.
#
# Ejemplo con 3 draws ficticios:

draws_ficticios <- tibble(
  .draw = 1:3,
  A = c(0.52, 0.48, 0.50),
  B = c(0.48, 0.52, 0.50)
)

error_matrix <- matrix(
  c( 0.001, -0.001,
    -0.002,  0.002,
     0.000,  0.000),
  nrow = 3, byrow = TRUE
)

# asplit convierte la matriz en lista de vectores fila
asplit(error_matrix, 1)
# [[1]]  0.001 -0.001
# [[2]] -0.002  0.002
# [[3]]  0.000  0.000

resultados_ej <- pmap(
  list(
    A   = draws_ficticios$A,
    B   = draws_ficticios$B,
    err = asplit(error_matrix, 1)
  ),
  \(A, B, err) {
    draw_vec <- setNames(c(A, B), c("A", "B"))

    # Ajuste por error histórico + renormalizar
    draw_adj <- pmax(draw_vec + err, 0)   # pmax: no permitir negativos
    draw_adj <- draw_adj / sum(draw_adj)

    # Aplicamos D'Hondt con 5 escaños (en el post: sim_escanos_draw sobre 8 provincias)
    dhondt(draw_adj, escanos = 5)
  }
)

resultados_ej
# Lista de 3 vectores, uno por draw, con los escaños de A y B


# =============================================================================
# 7. RESUMEN: la cadena completa de incertidumbre
# =============================================================================
#
# Para entender de dónde viene la variabilidad en los escaños finales,
# hay tres fuentes encadenadas:
#
#   [1] Incertidumbre del modelo bayesiano de encuestas
#       → draws del posterior de brms (columnas de draws_wide)
#       → captura "¿cuánto voto tiene realmente cada partido a nivel regional?"
#
#   [2] Error histórico de las encuestas (MASS::mvrnorm)
#       → perturbación correlada entre partidos en escala de voto regional
#       → captura "las encuestas se equivocan, y de forma correlada"
#
#   [3] Distribución provincial (Dirichlet)
#       → ruido sobre cómo se reparte el voto regional entre provincias
#       → captura "no sabemos exactamente si el patrón 2022 se repetirá"
#
# En el script del post, el orden de aplicación es:
#   draws_wide → (+err) → renormalizar → sim_escanos_draw (Dirichlet + ratio + D'Hondt)
#
# Nota: scale_factor=1000 en Dirichlet da sd≈2-4pp. Si lo subes (ej. 5000)
# el ruido provincial baja; si lo bajas (ej. 100) sube mucho.
# El valor del post está calibrado para que la incertidumbre sea realista
# respecto a los intervalos que ofrecen encuestas con n~1000-2000.
