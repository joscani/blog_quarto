# ============================================================================
# SIMULACIÓN DE ESCAÑOS DEL CONGRESO (desde los draws del meta-análisis)
#
# Replica la maquinaria de escanos-andalucia-2026.qmd adaptada al Congreso:
#
#   1. Draws del posterior del modelo 1 (6 categorías) y del modelo 2
#      (composición del BLOQUE, 8 subceldas) en la fecha objetivo. Ambos
#      con ref_time = 2026-07-01. La predicción va con re_formula = NA
#      (sin efecto casa): el riesgo de que el resultado se aparte de la
#      tendencia (sesgo común del consenso) lo lleva la capa de error.
#   2. Combinación a 13 categorías: 5 grandes + SALF + BLOQUE repartido con
#      la composición del modelo 2 (masa conservada, suman 100).
#   3. Distribución provincial: share_prov = share_nacional x
#      ratio_prov x exp(shock), con el patrón geográfico del 23J
#      (ref_provincial_23J_13.csv). SALF no existía en el 23J: su patrón
#      viene de las Europeas 2024. El ruido es el SHOCK JERÁRQUICO de
#      shock_jerarquico.R, que sustituye a la Dirichlet del divisor 1.000
#      (jubilada tras el backtest provincial: cobertura 0,94 con el shock
#      frente a 0,66 con la Dirichlet, y redundante en modo diagnóstico; ver
#      backtest_provincial.R y el códice del post). En cada draw, el
#      log-ratio del partido f en la provincia p se desvía así:
#        log r^(s)_fp = log r_fp + eta_{f,c(p)} + eps_{f,p}
#        eta ~ N(0, tau_c,f)  (compartido por las provincias de la CCAA)
#        eps ~ N(0, tau_p,f)  (propio de la provincia)
#      Los tau de los 5 grandes salen del posterior brms
#      (tau_shock_draws.csv); SALF hereda el RMS de los tau de los 4 grandes
#      (componente a componente, draw a draw); los regionales
#      (ERC, Junts, PNV, EH Bildu, BNG, CC, UPN) van con los taus en dos
#      etapas de tau_regionales.csv (plug-in); RESTO es residual y no lleva
#      shock. Cada multiplicador se divide por su media ponderada con los
#      votos de la referencia para que el voto nacional que fija la capa de
#      error no se despegue. En provincias donde el partido no tuvo votos en
#      la ref, su ratio 0 anula el multiplicador de todos modos.
#      modo_ruido = "dirichlet" conserva el mecanismo antiguo (divisor
#      1.000, solo para comparar).
#   4. Umbral legal del 3% provincial (LOREG art. 163) antes del D'Hondt.
#      Los votos en blanco cuentan para el umbral legal y aquí no se simulan
#      (~0.5%): el umbral aplicado sobre shares de candidaturas es
#      marginalmente más estricto (3%/1.005). Ceuta y Melilla (1 escaño) no
#      necesitan tratamiento especial: D'Hondt con 1 escaño = mayoría simple.
#   5. D'Hondt provincia a provincia (dhondt), acumulando escaños.
#   6. CAPA DE ERROR DE ENCUESTAS (aquí empírica; en Andalucía era una matriz
#      especificada a mano): cada muestra del posterior recibe un
#      error de encuesta e ~ N(mu, Sigma) sobre los 5 grandes (SALF no tiene
#      historia, sigma = 0); después se reescala el vector completo para que
#      sume 100, de modo que cada partido cede en proporción a su tamaño.
#      El BLOQUE lleva además un shock propio N(0, sigma_bloque): la
#      historia no lo calibra (allí es residual por construcción), es un
#      supuesto explícito. Calibrada con las generales 20D-2015, 26J-2016, 28A-2019,
#      10N-2019 y 23J-2023:
#        - Encuestas individuales: Anexos de Wikipedia de sondeos, campo en la
#          última semana de campaña (14 días para el 26J, campaña corta);
#          ~70 encuestas. Error de cada encuesta = real - encuesta.
#        - Agregado: real - media de la última semana (5 elecciones). mu es esa
#          media (sesgo sistemático del agregado) y sigma su sd muestral.
#        - R (correlaciones): empírica sobre las ~70 encuestas individuales.
#        - Encuentros y saldos: las encuestas infraestiman sistemáticamente al
#          PSOE (+1.32 pp de media, aplicado encogido a la mitad, +0.66) y no
#          hay sesgo estable en el resto. La correlación PP-VOX es positiva
#          en las 3 elecciones con VOX (encogida a +0.5), a diferencia de la
#          -0.5 a mano del post andaluz. Tras el error, el reescalado
#          proporcional conserva la masa sin cargar todo el ajuste sobre el
#          BLOQUE.
#        - El detalle encuesta a encuesta está congelado en
#          2026/10/errores_encuestas_historicas.csv (par encuesta y errores).
#        - Los sigma van de plug-in; el backtest (con el bug del sweep
#          corregido) da una cobertura compatible con el 90% nominal (83%
#          en voto, 94% en escaños con kappa = 0.5). Con t_sigma_df = 4 cada
#          muestra sortea el propio sigma de su posterior (equivale a
#          e ~ t_4): la cobertura del backtest se va al 100% con IC90 de
#          ~33 escaños — demasiado conservador; queda como sensibilidad.
#
# Salidas:
#   simular(n_draws): lista con draws (tibble .draw x 13 partidos), etc.
#   simular(..., provincial = TRUE): añade prov_voto y prov_escanos,
#     arrays draw x provincia x familia (voto = share provincial tras
#     distribuir el nacional, ANTES del umbral del 3%; escaños = D'Hondt
#     provincia a provincia), más prov_ref (lookup codigo/provincia/ccaa)
#   resumen_escanos(sim): mediana + IC por partido
#   prob_mayorias(sim): P(PP>=176), P(PP+VOX>=176), P(izquierda>=176)
#   sim_validacion_23j(): con shares del 23J y ratios de la ref sin ruido
#     debe reproducir los escaños oficiales de los partidos mapeados. Ojo:
#     RESTO (agregado) puede arrastrar 1-2 escaños "fantasma" que en la
#     realidad fueron de PP/PSOE (así era en la calibración del swing).
#
# Uso:
#   source("2026/10/simular_escanos.R")
#   sim <- simular(n_draws = 2000)        # día E (29 nov 2026) por defecto
#   sim <- simular(modo_ruido = "dirichlet")  # ruido antiguo (comparación)
#   sim <- simular(f_antelacion = 1.36)   # error escalado a ~49 días del día E
#   sim <- simular(listas = "separadas")  # SUMAR y PODEMOS separadas (14 fams)
#   sim <- simular(listas = "mapa")       # mapa de coaliciones del 29N (15 fams, con AA)
#   sim <- simular(listas = "mapa", aa_pct_and = 0.06)  # prior de AA más flojo
#   resumen_escanos(sim); prob_mayorias(sim)
#
# Escenario listas = "mapa" (coaliciones registradas el 16 oct 2026):
#   parte del escenario "separadas" (modelo 1 con 7 categorías, ref14, error
#   y shock sobre la suma SUMAR+PODEMOS) y, DESPUÉS de repartir el voto en la
#   provincia y antes del umbral, aplica la regla de cada provincia que viene
#   en la columna regla de coaliciones_izquierda_29N_provincias.csv:
#     - separadas:   SUMAR y PODEMOS compiten por separado (sin cambios);
#     - unida:       lista conjunta; su voto y sus escaños van en la columna
#                    SUMAR y PODEMOS queda a 0 (Canarias, C. Valenciana);
#     - sin_podemos: Podemos no se presenta (Cataluña, acuerdo con ERC). El
#                    reparto de su voto entre ERC, SUMAR y fuera (abstención
#                    u otros) se sortea draw a draw de una Dirichlet con
#                    transfer_cat_alpha, centrada en 0,65/0,30/0,05 (Podemos
#                    hace campaña por ERC); se renormaliza a 100;
#     - con_aa:      Andalucía con Adelante Andalucía como familia propia
#                    (la 15ª). Su share andaluza por draw sale de una
#                    logit-normal centrada en aa_pct_and (el 9,62% de las
#                    autonómicas del 17 may 2026) con sd aa_sd_logit: el
#                    traslado autonómicas->generales es lo incierto. El
#                    patrón provincial de aa_autonomicas2026_provincias.csv
#                    lleva el shock jerárquico con taus heredados (el mismo
#                    trato que SALF). La masa sale primero del RESTO de la
#                    provincia (su casa actual, que no compite en D'Hondt) y
#                    después de PODEMOS/SUMAR con pesos aa_caniba ~
#                    Dirichlet(aa_caniba_alpha); si el pool no llega, cae
#                    en PSOE.
#   AA no sale del modelo 1: fuera de las provincias con_aa su share es 0.
#   Supuesto de la fase 1: el modelo 1 estima a Podemos "como si se
#   presentara en las 52" (las encuestas hasta el 16 oct lo cuentan con
#   Cataluña); la transferencia se aplica aquí, no a las encuestas.
#   sim_validacion_23j()
# ============================================================================

library(brms)
library(dplyr)
library(tidyr)
library(readr)
library(tidybayes)
library(gtools)

source(here::here("2026/10/ajustar_meta.R"))
# shock jerárquico: extraer_tau_draws, mapa_regionales, ...
source(here::here("2026/10/shock_jerarquico.R"))

f_ref13 <- here::here("2026/10/ref_provincial_23J_13.csv")
ref13 <- read_csv(f_ref13, show_col_types = FALSE)

fams_bloque <- c("Junts", "ERC", "PNV", "EH Bildu", "BNG", "CC", "UPN", "RESTO")
# en el fit del modelo 2 la subcelda va sin espacio (prep_bloque la renombra)
fams_bloque_fit <- c(
  "Junts",
  "ERC",
  "PNV",
  "EH_Bildu",
  "BNG",
  "CC",
  "UPN",
  "RESTO"
)
fams13 <- c("PP", "PSOE", "VOX", "SUMAR", "SALF", fams_bloque)
stopifnot(setequal(fams13, unique(ref13$partido)))

# escenario "listas": SUMAR y PODEMOS separadas (14 familias). El peso con
# el que el voto de SUMAR del 23J se parte en cada provincia viene de las
# Europeas 2024 (construir_ref14.R); sin backtest histórico posible: Podemos
# concurrió separado antes del 23J con otra configuración
f_ref14 <- here::here("2026/10/ref_provincial_23J_14.csv")
ref14 <- read_csv(f_ref14, show_col_types = FALSE)
fams14 <- c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "SALF", fams_bloque)
stopifnot(setequal(fams14, unique(ref14$partido)))

# escenario "mapa": regla de cada provincia
# (separadas, unida, sin_podemos, con_aa)
f_mapa_listas <- here::here("2026/10/coaliciones_izquierda_29N_provincias.csv")
reglas_mapa <- c("separadas", "unida", "sin_podemos", "con_aa")

# patrón provincial de AA (autonómicas 5/2026): peso relativo de AA en cada
# provincia andaluza, peso_p = pct AA en p / pct AA andaluz. Es un
# MULTIPLICADOR, no una distribución: la share provincial de AA es
# aa_and x peso_p, y el agregado andaluz ponderado por válidos vuelve a ser
# aa_and (sum_p válidos_p x peso_p = total válidos andaluz)
f_aa_patron <- here::here("2026/10/aa_autonomicas2026_provincias.csv")
patron_aa <- read_csv(f_aa_patron, show_col_types = FALSE)
peso_aa <- patron_aa$votos_aa /
  sum(patron_aa$votos_aa) /
  (patron_aa$votos_validos / sum(patron_aa$votos_validos))
names(peso_aa) <- patron_aa$codigo_provincia

leer_reglas_mapa <- function(f = f_mapa_listas, codigos) {
  m <- read_csv(f, col_types = cols(.default = col_character()))
  stopifnot(
    "regla" %in% names(m),
    all(m$regla %in% reglas_mapa),
    setequal(m$codigo_provincia, codigos),
    !anyDuplicated(m$codigo_provincia)
  )
  m$regla[match(codigos, m$codigo_provincia)]
}

# v: shares provinciales con nombres de fams14 + AA (suman 1)
aplicar_regla_mapa <- function(
  v,
  regla,
  transfer,
  aa_share = 0,
  aa_caniba = c(PODEMOS = 0.5, SUMAR = 0.5)
) {
  if (regla == "unida") {
    v["SUMAR"] <- v["SUMAR"] + v["PODEMOS"]
    v["PODEMOS"] <- 0
  } else if (regla == "sin_podemos") {
    pod <- unname(v["PODEMOS"])
    v[names(transfer)] <- v[names(transfer)] + pod * transfer
    v["PODEMOS"] <- 0
    # la fracción que va a abstención u otros sale del voto válido
    v <- v / sum(v)
  } else if (regla == "con_aa") {
    # AA entra con su share provincial (patrón de las autonómicas 5/2026).
    # La masa sale primero del RESTO de la provincia (su casa en el modelo,
    # que no compite en D'Hondt: si la share de AA lo agota, se toma todo) y
    # después de PODEMOS y SUMAR con los pesos aa_caniba (normalizados),
    # sin pedirle a un partido más de lo que tiene. Si aun así falta, el
    # sobrante cae en PSOE. La masa se conserva; renormalizo por si pmax recorta
    v["AA"] <- aa_share
    de_resto <- min(v["RESTO"], aa_share)
    v["RESTO"] <- v["RESTO"] - de_resto
    quede <- aa_share - de_resto
    pool <- names(aa_caniba)
    w <- aa_caniba / sum(aa_caniba)
    quita <- pmin(quede * w, v[pool])
    v[pool] <- v[pool] - quita
    sobra <- quede - sum(quita)
    if (sobra > 0) {
      v["PSOE"] <- v["PSOE"] - sobra
    }
    v <- pmax(v, 0)
    v <- v / sum(v)
  }
  v
}

# ---- capa de error de encuestas (20D-2015 a 23J-2023) ------------------------
# error en puntos porcentuales sobre el share nacional de los 5 grandes;
# el reescalado posterior lo reparte proporcionalmente (BLOQUE incluido).
# El BLOQUE añade además un shock propio N(0, sigma_bloque), supuesto
# explícito: en la calibración histórica es residual por construcción.
# Calibración con 124 encuestas de la última campaña de 5 elecciones
# (errores = real - encuesta; detalle en errores_encuestas_historicas.csv).
# Quedan fuera los sondeos internos de partidos (PSOE, Podemos...) y la
# ventana usa el fin del TRABAJO DE CAMPO (los rangos con en dash de los
# anexos se parsean al último día).
# Solo se conserva lo robusto:
#   - sigma por partido: 1.3-2.6 pp, estable en todas las elecciones;
#   - mu: solo el PSOE (+1.32 crudo; positivo en 4 de 5 elecciones, estable
#     sin el 26J). PP, VOX y Sumar: mu ~ 0 (cambia de signo según qué
#     elecciones entren). El mu del PSOE se aplica encogido a la mitad
#     (+0.66): el backtest leave-one-out (backtest_pipeline.R) muestra que
#     aplicarlo completo no mejora fuera de muestra;
#   - PP-VOX: errores con el MISMO signo en las 3 elecciones con VOX
#     (correlación +0.92 a nivel de elección). Con n = 3 se encoge a +0.5;
#     el resto de correlaciones es inestable y va a 0.
familias_error <- c("PP", "PSOE", "VOX", "SUMAR", "SALF")
# mu del PSOE encogido a la mitad: el leave-one-out del backtest no valida la
# corrección completa fuera de muestra (κ = 0 gana en voto, κ = 0.5 en
# escaños; 0.5 es el término medio). Los demás: μ = 0
kappa_mu <- 0.5
error_mu <- kappa_mu *
  c(PP = 0, PSOE = 1.32, VOX = 0, SUMAR = 0, SALF = 0)
error_sigma <- c(PP = 2.57, PSOE = 1.32, VOX = 1.29, SUMAR = 1.89, SALF = 0)
R_error <- matrix(
  c(
    #          PP    PSOE   VOX   SUMAR  SALF
    1.00,
    0,
    0.50,
    0,
    0, # PP
    0,
    1.00,
    0,
    0,
    0, # PSOE
    0.50,
    0,
    1.00,
    0,
    0, # VOX
    0,
    0,
    0,
    1.00,
    0, # SUMAR
    0,
    0,
    0,
    0,
    1 # SALF (sin error)
  ),
  nrow = 5,
  byrow = TRUE,
  dimnames = list(familias_error, familias_error)
)
Sigma_error <- outer(error_sigma, error_sigma) * R_error
stopifnot(all(eigen(Sigma_error)$values > -1e-8))

prov_ref <- ref13 |>
  distinct(codigo_provincia, provincia, ccaa, diputados) |>
  arrange(codigo_provincia)

# datos por provincia en el ORDEN de fams (crítico: shares y Dirichlet
# van en el mismo orden). Nombres con sufijo para no pisar columnas de
# ref dentro de summarise (si partido = list(partido[...]) va primero,
# las expresiones siguientes ven la lista, no la columna).
preparar_prov_datos <- function(ref, fams) {
  ref |>
    group_by(codigo_provincia) |>
    summarise(
      partido_f = list(partido[match(fams, partido)]),
      # shares de referencia renormalizados: pct_prov no suma 100 donde SALF
      # (Europeas) se añadió encima del 23J. La Dirichlet se centra en ellos y
      # el multiplicador divide por ellos, así E[mult] = ratio exactamente.
      # (Antes alpha = votos / 1000 mezclaba votos de Europeas con el total del
      # 23J y centraba a SALF en ~0.68 de su ratio.)
      pct23 = list(pct_prov[match(fams, partido)] / sum(pct_prov)),
      ratio_f = list(ratio[match(fams, partido)]),
      # suelo de 0.5 solo para partidos con votos: los que no se presentan
      # (ratio 0) van con alpha 0 y no le quitan masa al resto
      alpha_f = list({
        p <- pct_prov[match(fams, partido)] / sum(pct_prov)
        ifelse(p > 0, pmax(p * sum(votos) / 1000, 0.5), 0)
      }),
      escanos = diputados[1],
      .groups = "drop"
    )
}
prov_datos <- preparar_prov_datos(ref13, fams13)
prov_datos14 <- preparar_prov_datos(ref14, fams14)
stopifnot(
  nrow(prov_datos) == 52,
  sum(prov_datos$escanos) == 350,
  nrow(prov_datos14) == 52,
  sum(prov_datos14$escanos) == 350,
  identical(
    prov_ref,
    ref14 |>
      distinct(codigo_provincia, provincia, ccaa, diputados) |>
      arrange(codigo_provincia)
  )
)

# reparto del 29-N (RD 806/2026, anexo): Madrid 37 -> 38, Cádiz 9 -> 8. La
# validación con el 23J se queda con el mapa de 2023 (prov_datos_23j) para
# reproducir sus 350 escaños; la simulación usa el nuevo.
prov_datos_23j <- prov_datos
escanos_29n <- c("28" = 38, "11" = 8)
aplicar_escanos_29n <- function(d) {
  i <- match(names(escanos_29n), d$codigo_provincia)
  d$escanos[i] <- unname(escanos_29n)
  d
}
prov_datos <- aplicar_escanos_29n(prov_datos)
prov_datos14 <- aplicar_escanos_29n(prov_datos14)
prov_ref <- prov_ref |>
  mutate(
    diputados = coalesce(
      unname(escanos_29n[as.character(codigo_provincia)]),
      diputados
    )
  )
stopifnot(
  sum(prov_datos$escanos) == 350,
  sum(prov_datos14$escanos) == 350,
  sum(prov_ref$diputados) == 350
)

# ---- draws de los dos modelos ------------------------------------------------

extraer_shares13 <- function(
  n_draws = 2000,
  time_obj = time_eleccion,
  con_error = TRUE,
  sigma_bloque = 1.5,
  cor_bloque = 0,
  t_sigma_df = NULL,
  f_antelacion = 1,
  listas = c("unidas", "separadas")
) {
  listas <- match.arg(listas)
  mod_b <- ajustar_bloque()
  if (listas == "unidas") {
    mod_g <- ajustar_grande()
    cats <- c("PP", "PSOE", "VOX", "SUMAR", "SALF", "BLOQUE")
  } else {
    mod_g <- ajustar_grande_sep()
    cats <- c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "SALF", "BLOQUE")
  }
  # re_formula = NA: predicción de la tendencia, sin efecto casa. El
  # resultado no es una encuestadora; el riesgo de sesgo común del consenso
  # (herding) lo lleva la capa de error histórico, y una casa nueva
  # duplicaría esa varianza
  d1 <- tibble(time = time_obj, n = 1) |>
    add_epred_draws(mod_g, ndraws = n_draws, re_formula = NA) |>
    ungroup() |>
    select(.draw, .category, .epred) |>
    pivot_wider(names_from = .category, values_from = .epred)
  d2 <- tibble(time = time_obj, bloque = 1) |>
    add_epred_draws(mod_b, ndraws = n_draws) |>
    ungroup() |>
    select(.draw, .category, .epred) |>
    pivot_wider(names_from = .category, values_from = .epred)
  stopifnot(nrow(d1) == nrow(d2), all(cats %in% names(d1)))
  # epred del modelo 2: proporciones condicionales dentro del BLOQUE
  bloque_comp <- as.matrix(d2[, fams_bloque_fit])
  shares <- as.matrix(d1[, cats])
  # epred puede venir en proporciones (suman 1) o en conteos con n = 1:
  # normalizamos a porcentaje
  sc <- if (max(rowSums(shares)) < 1.5) 100 else 1
  shares7 <- shares * sc
  if (listas == "separadas") {
    # peso de cada lista dentro de la suma SUMAR+PODEMOS en cada draw (del
    # posterior, antes del error): el error de la suma se reparte con él.
    # En draws degenerados (suma 0) el reparte es irrelevante: 0.5
    tot_sp <- shares7[, "SUMAR"] + shares7[, "PODEMOS"]
    s_pod_draw <- ifelse(tot_sp > 0, shares7[, "PODEMOS"] / tot_sp, 0.5)
    # el error de encuesta va sobre la suma (la historia solo conoce a
    # SUMAR agregado): columnas equivalentes al caso unidas
    shares6 <- cbind(
      shares7[, c("PP", "PSOE", "VOX")],
      SUMAR = shares7[, "SUMAR"] + shares7[, "PODEMOS"],
      SALF = shares7[, "SALF"],
      BLOQUE = shares7[, "BLOQUE"]
    )
  } else {
    s_pod_draw <- NULL
    shares6 <- shares7
  }
  if (con_error) {
    # capa de error: e ~ N(mu, Sigma) sobre los 5 grandes. El reescalado
    # posterior reparte el ajuste proporcionalmente al tamaño de cada
    # partido (BLOQUE incluido), en vez de hacerle absorber el residual
    # completo: antes el BLOQUE pagaba todo el error de los grandes.
    # f_antelacion: los sigma están calibrados con la última semana de
    # campaña; a más distancia del día E el error crece. El factor escala
    # los sigma (no las mu: el sesgo no crece con la antelación) y también
    # sigma_bloque; ver factor_antelacion.csv (sigma(36-56d)/sigma(0-7d)
    # = 1.36 agregado, PP 1.13 / PSOE 1.86 / VOX 1.75, en la línea del ~1.4
    # de ciento76 a 49 días).
    stopifnot(f_antelacion >= 1)
    sig_h <- error_sigma * f_antelacion
    if (!is.null(t_sigma_df)) {
      # sensibilidad: los sigma_hat salen de n = 5 elecciones, así que están
      # estimados con ruido (su IC95 va de 0,6x a 2,9x el estimado). Con su
      # posterior — scaled-inv-chi2 con prior de Jeffreys, equivalente a que
      # e siga una t de Student con nu gl — cada muestra trae su propio
      # sigma. Con nu = 4 la varianza esperada se dobla (nu/(nu-2) = 2); el
      # backtest con t_4 da cobertura 100% con IC90 de ~33 escaños: cubre de
      # más. El plug-in calibra mejor (83-94% frente al 90% nominal).
      # z ~ N(0, R) escalado elemento a elemento por c_k * sigma_k da
      # exactamente cov = D_c R D_c.
      nu <- t_sigma_df
      z <- MASS::mvrnorm(n_draws, rep(0, 5), R_error)
      escala <- sqrt(nu / matrix(rchisq(n_draws * 5, nu), n_draws))
      e <- matrix(error_mu, n_draws, 5, byrow = TRUE) +
        z * (escala %*% diag(sig_h))
      colnames(e) <- names(error_mu)
    } else {
      e <- MASS::mvrnorm(n_draws, error_mu, Sigma_error * f_antelacion^2)
    }
    shares6[, 1:5] <- pmax(shares6[, 1:5] + e, 0)
    # error propio del BLOQUE: la historia no lo calibra (allí el BLOQUE es
    # el residual de los grandes por construcción, luego su error sería
    # -suma de los otros); sigma_bloque es un supuesto explícito. Shock
    # de media 0 e independiente de los grandes.
    if (sigma_bloque > 0) {
      # correlación con el RUIDO del PSOE, no con su error completo: si
      # correlara con la corrección sistemática (+1.32), el BLOQUE volvería
      # a pagar parte de esa corrección. cor_bloque es un supuesto (la
      # historia no lo califica: allí el BLOQUE es residual por
      # construcción y la correlación mecánica sería ≈ -1). El denominador
      # lleva el factor de antelación para que el ruido del PSOE quede
      # estandarizado igual que sin él (el factor escala una sola vez,
      # vía sigma_bloque).
      e_b <- sigma_bloque *
        f_antelacion *
        (cor_bloque *
          (e[, "PSOE"] - error_mu["PSOE"]) /
          sig_h["PSOE"] +
          sqrt(1 - cor_bloque^2) * rnorm(n_draws))
      shares6[, 6] <- pmax(shares6[, 6] + e_b, 0)
    }
    shares6 <- 100 * shares6 / rowSums(shares6)
  }
  if (listas == "unidas") {
    out <- cbind(shares6[, 1:5, drop = FALSE], shares6[, 6] * bloque_comp)
    colnames(out) <- fams13
  } else {
    # el error de la suma SUMAR+PODEMOS se reparte entre las dos listas con
    # el peso de cada una en el draw
    s <- pmin(pmax(s_pod_draw, 0), 1)
    total_s <- shares6[, "SUMAR"]
    out <- cbind(
      shares6[, c("PP", "PSOE", "VOX"), drop = FALSE],
      SUMAR = total_s * (1 - s),
      PODEMOS = total_s * s,
      shares6[, "SALF", drop = FALSE],
      shares6[, "BLOQUE"] * bloque_comp
    )
    colnames(out) <- fams14
  }
  stopifnot(all(abs(rowSums(out) - 100) < 0.5))
  out
}

# ---- taus del shock (una fila por draw y familia) ----------------------------

preparar_tau_simular <- function(n_draws, listas = c("unidas", "separadas")) {
  listas <- match.arg(listas)
  fams <- if (listas == "unidas") fams13 else fams14
  # grandes: draws del posterior brms (2.000 por familia, congelados con
  # extraer_tau_draws). Se remuestrean con reemplazo por un vector de índices
  # compartido entre familias, como en el backtest, para conservar la
  # correlación de la incertidumbre de estimación entre familias
  taus_g <- read_csv(
    here::here("2026/10/tau_shock_draws.csv"),
    show_col_types = FALSE
  )
  idx <- sample.int(max(taus_g$draw), n_draws, replace = TRUE)
  grandes <- map_dfr(sort(unique(taus_g$familia)), \(f) {
    tg <- taus_g |> filter(familia == f)
    tibble(
      draw = seq_len(n_draws),
      familia = f,
      tau_c = tg$tau_c[idx],
      tau_p = tg$tau_p[idx]
    )
  })
  # SALF: sin historia propia (no existía antes de 2024). Heredar de SUMAR
  # parecía lo natural por espacio político, pero SALF se suele situar más
  # cerca del bloque de VOX —que además es el partido con mayor tau_c—, así
  # que la herencia por afinidad es traicionera. Se hereda el RMS de los tau
  # de los 4 grandes, componente a componente (tau_c = RMS de los tau_c,
  # tau_p = RMS de los tau_p), la misma solución neutral que con UPN. SALF
  # compite en las 52 provincias, así que conserva la partición
  # comunidad/provincia; el RMS se calcula draw a draw para conservar la
  # incertidumbre de estimación
  sal <- grandes |>
    group_by(draw) |>
    summarise(
      tau_c = sqrt(mean(tau_c^2)),
      tau_p = sqrt(mean(tau_p^2)),
      .groups = "drop"
    ) |>
    mutate(familia = "SALF")
  # regionales: estimación puntual replicada en los draws (plug-in: con 1-2
  # saltos frescos no hay señal para propagar la incertidumbre de estimación)
  reg <- read_csv(
    here::here("2026/10/tau_regionales.csv"),
    show_col_types = FALSE
  ) |>
    mutate(draw = list(seq_len(n_draws))) |>
    unnest(draw) |>
    select(draw, familia, tau_c, tau_p)
  # RESTO: residual por construcción, sin shock
  resto <- tibble(
    draw = seq_len(n_draws),
    familia = "RESTO",
    tau_c = 0,
    tau_p = 0
  )
  tau <- bind_rows(grandes, sal, reg, resto)
  if (listas == "separadas") {
    # PODEMOS hereda los draws de SUMAR: el shock provincial se aplica a la
    # suma (calibrada sobre SUMAR agregado) y de ahí se reparte como el voto
    pod <- tau |>
      filter(familia == "SUMAR") |>
      mutate(familia = "PODEMOS")
    tau <- bind_rows(tau, pod)
  }
  tau |>
    mutate(familia = factor(familia, fams)) |>
    arrange(draw, familia) |>
    mutate(familia = as.character(familia))
}

# ---- D'Hondt -----------------------------------------------------------------

dhondt <- function(votos, escanos) {
  result <- setNames(numeric(length(votos)), names(votos))
  for (i in seq_len(escanos)) {
    ganador <- which.max(votos / (result + 1))
    result[ganador] <- result[ganador] + 1
  }
  result
}

# ---- simulación --------------------------------------------------------------

simular <- function(
  n_draws = 2000,
  time_obj = time_eleccion,
  con_ruido = TRUE,
  modo_ruido = c("shock", "dirichlet"),
  swing = c("proporcional", "logit"),
  con_error = TRUE,
  umbral = 0.03,
  resto_compite = FALSE,
  sigma_bloque = 1.5,
  cor_bloque = 0,
  t_sigma_df = NULL,
  f_antelacion = 1,
  listas = c("unidas", "separadas", "mapa"),
  provincial = FALSE,
  transfer_cat_alpha = c(ERC = 10, SUMAR = 4.5, fuera = 0.75),
  aa_pct_and = 0.0962,
  aa_sd_logit = 0.2,
  aa_caniba_alpha = c(PODEMOS = 4, SUMAR = 4, PSOE = 1),
  f_mapa = f_mapa_listas,
  # voto nacional ya calculado fuera (matriz n_draws x familias del modelo,
  # en %, con el error incluido): sustituye a extraer_shares13(). Lo usa
  # 2026/10/full-bayesian/04_simular.R
  shares_ext = NULL
) {
  modo_ruido <- match.arg(modo_ruido)
  swing <- match.arg(swing)
  listas <- match.arg(listas)
  # "mapa" usa toda la maquinaria de "separadas" y solo cambia el voto
  # provincial al final (aplicar_regla_mapa)
  listas_modelo <- if (listas == "mapa") "separadas" else listas
  regla_prov <- NULL
  aa_prov_k <- pi_caniba <- pi_cat <- NULL
  if (listas == "mapa") {
    stopifnot(
      setequal(names(transfer_cat_alpha), c("ERC", "SUMAR", "fuera")),
      all(transfer_cat_alpha >= 0),
      sum(transfer_cat_alpha) > 0,
      aa_pct_and > 0,
      aa_pct_and < 1,
      aa_sd_logit >= 0,
      all(names(aa_caniba_alpha) %in% fams14),
      !"RESTO" %in% names(aa_caniba_alpha),
      all(aa_caniba_alpha >= 0),
      sum(aa_caniba_alpha) > 0
    )
    regla_prov <- leer_reglas_mapa(f_mapa, prov_datos14$codigo_provincia)
    # prior de la share andaluza de AA: logit-normal centrada en el resultado
    # de las autonómicas de mayo (el traslado a generales es lo incierto)
    aa_and <- plogis(qlogis(aa_pct_and) + rnorm(n_draws, 0, aa_sd_logit))
    # canibalización (PODEMOS/SUMAR/PSOE) y transferencia catalana
    # (ERC/SUMAR/fuera): Dirichlet draw a draw
    pi_caniba <- gtools::rdirichlet(n_draws, aa_caniba_alpha)
    colnames(pi_caniba) <- names(aa_caniba_alpha)
    pi_cat <- gtools::rdirichlet(n_draws, transfer_cat_alpha)
    colnames(pi_cat) <- names(transfer_cat_alpha)
    # patrón provincial base de AA: normalizado dentro de Andalucía conserva
    # el agregado andaluz (share_p(k) = aa_and_k x w_p / sum w)
    g_aa <- peso_aa[match(prov_datos14$codigo_provincia, names(peso_aa))]
    g_aa[is.na(g_aa)] <- 0
    aa_prov_k <- aa_and *
      matrix(g_aa, n_draws, length(g_aa), byrow = TRUE)
  }
  stopifnot(swing == "proporcional" || (con_ruido && modo_ruido == "shock"))
  # escenario: unidas = 13 familias (SUMAR = Sumar + Podemos, como en la
  # historia); separadas = 14 (el peso provincial viene de las Europeas 2024
  # y el error de la suma se reparte con el peso de cada lista en el draw)
  if (listas_modelo == "unidas") {
    fams <- fams13
    ref <- ref13
    prov_d <- prov_datos
  } else {
    fams <- fams14
    ref <- ref14
    prov_d <- prov_datos14
  }
  # en el mapa, AA es la 15ª familia: no sale del modelo 1 (su share la fija
  # la regla provincial), así que no entra en el bucle de shocks
  fams_modelo <- fams
  if (listas == "mapa") {
    fams <- c(fams, "AA")
  }
  if (is.null(shares_ext)) {
    shares_m <- extraer_shares13(
      n_draws,
      time_obj,
      con_error,
      sigma_bloque,
      cor_bloque,
      t_sigma_df,
      f_antelacion,
      listas_modelo
    )
  } else {
    stopifnot(
      nrow(shares_ext) == n_draws,
      identical(colnames(shares_ext), fams_modelo),
      all(abs(rowSums(shares_ext) - 100) < 0.5)
    )
    shares_m <- shares_ext
  }

  if (con_ruido && modo_ruido == "dirichlet") {
    # mecanismo antiguo (divisor 1.000), fuera del default: queda como
    # opción para comparar con el shock jerárquico
    dir_muestras <- lapply(prov_d$alpha_f, \(a) {
      gtools::rdirichlet(n_draws, a)
    })
  }

  if (con_ruido && modo_ruido == "shock") {
    tau <- preparar_tau_simular(n_draws, listas_modelo)
    if (listas == "mapa") {
      # AA: sin historia en generales; hereda los taus de SALF (el RMS de los
      # grandes), el mismo trato
      tau <- bind_rows(
        tau,
        tau |> filter(familia == "SALF") |> mutate(familia = "AA")
      )
    }
    stopifnot(
      setequal(unique(tau$familia), fams),
      all(table(tau$familia) == n_draws)
    )
    ccaa_prov <- prov_ref$ccaa[
      match(prov_d$codigo_provincia, prov_ref$codigo_provincia)
    ]
    ccaa_u <- sort(unique(ccaa_prov))
    ccaa_idx <- match(ccaa_prov, ccaa_u)
    # pesos del re-centrado: votos de las familias por provincia en la
    # ref (23J). Aproxima los votos válidos (~3% menos): el re-centrado es
    # una constante de normalización por draw, no una restricción dura
    w_prov <- ref |>
      group_by(codigo_provincia) |>
      summarise(v = sum(votos), .groups = "drop")
    w <- w_prov$v[match(prov_d$codigo_provincia, w_prov$codigo_provincia)]
    w <- w / sum(w)
    # ratios de la ref: matriz 52 x length(fams) en el orden de fams
    ratios_m <- do.call(rbind, prov_d$ratio_f)
    # shares de referencia en bruto (base del swing en odds): matriz 52 x fams
    pct_prov_m <- ref |>
      select(codigo_provincia, partido, pct_prov) |>
      pivot_wider(names_from = partido, values_from = pct_prov) |>
      arrange(codigo_provincia) |>
      select(all_of(fams_modelo)) |>
      as.matrix()
    pct_nac_ref <- ref |>
      distinct(partido, pct_nac) |>
      slice(match(fams_modelo, partido)) |>
      pull(pct_nac)
    E_fam <- vector("list", length(fams_modelo))
    rc <- matrix(NA_real_, n_draws, length(fams_modelo))
    for (j in seq_along(fams_modelo)) {
      # matrix(b x C) * vector(b) recicla por columnas: fila a fila
      tau_c_k <- tau$tau_c[tau$familia == fams[j]]
      tau_p_k <- tau$tau_p[tau$familia == fams[j]]
      eta <- matrix(rnorm(n_draws * length(ccaa_u)), n_draws) * tau_c_k
      eps <- matrix(rnorm(n_draws * nrow(prov_d)), n_draws) * tau_p_k
      # el shock en escala log-ratio (eta + eps); exp_s es el multiplicador
      log_shock <- eta[, ccaa_idx, drop = FALSE] + eps
      exp_s <- exp(log_shock)
      if (swing == "logit") {
        # swing en odds: la share nacional del draw desplaza la odds de la
        # share provincial de referencia, partido a partido. Los partidos
        # sin votos en la ref de la provincia (logit = -Inf) siguen con 0.
        # El shock jerárquico entra como log-shock sumado al log-odds
        # (odds x exp(shock), el mismo multiplicador de la rama proporcional).
        # Los tau siguen calibrados en log-ratio: reestimarlos con delta
        # logit queda apuntado como pendiente.
        delta_k <- qlogis(pmin(pmax(shares_m[, j] / 100, 1e-6), 1 - 1e-6)) -
          qlogis(pct_nac_ref[j] / 100)
        base <- plogis(
          sweep(
            sweep(log_shock, 2, qlogis(pct_prov_m[, j] / 100), `+`),
            1,
            delta_k,
            `+`
          )
        )
        # el multiplicador lleva ya el ratio del draw (base / P_nac), con
        # suelo para que una share nacional 0 no genere Inf
        M_j <- 100 * sweep(base, 1, pmax(shares_m[, j], 1e-8), `/`)
      } else {
        M_j <- sweep(exp_s, 2, ratios_m[, j], `*`)
      }
      E_fam[[j]] <- M_j
      rc[, j] <- as.numeric(M_j %*% w)^-1
    }
    # por provincia: matriz b x length(fams) (columnas en el orden de fams)
    E_prov <- lapply(seq_len(nrow(prov_d)), \(i) {
      sapply(E_fam, \(M) M[, i])
    })
    if (listas == "mapa") {
      # patrón provincial de AA por draw: multiplicador base x shock log-ratio
      # con los taus de AA (heredados de los grandes). Re-centrado ponderando
      # por válidos: el agregado andaluz del draw sigue siendo aa_and_k
      idx_and <- which(g_aa > 0)
      tau_aa <- tau |> filter(familia == "AA")
      eps_aa <- matrix(rnorm(n_draws * length(idx_and)), n_draws) * tau_aa$tau_p
      g_k <- matrix(g_aa[idx_and], n_draws, length(idx_and), byrow = TRUE) *
        exp(rnorm(n_draws) * tau_aa$tau_c + eps_aa)
      wv_and <- patron_aa$votos_validos[
        match(names(idx_and), patron_aa$codigo_provincia)
      ] /
        sum(patron_aa$votos_validos)
      aa_prov_k[, idx_and] <- aa_and * g_k / as.numeric(g_k %*% wv_and)
    }
  }

  # salida provincial (opcional): arrays draw x provincia x familia.
  # El voto se guarda tras repartir el nacional y ANTES del umbral (la
  # estimación de voto no lo lleva; solo afecta al reparto de escaños)
  prov_voto <- prov_esc <- NULL
  if (provincial) {
    stopifnot(identical(
      sort(unique(ref13$codigo_provincia)),
      prov_ref$codigo_provincia
    ))
    dims <- list(NULL, prov_ref$provincia, fams)
    size <- c(n_draws, nrow(prov_d), length(fams))
    prov_voto <- array(NA_real_, size, dims)
    prov_esc <- array(0, size, dims)
  }

  una <- function(k) {
    e_prov <- matrix(0, nrow(prov_d), length(fams))
    if (provincial) {
      v_prov <- e_prov
    }
    for (i in seq_len(nrow(prov_d))) {
      pd <- prov_d[i, ]
      mult <- if (con_ruido) {
        if (modo_ruido == "dirichlet") {
          ifelse(
            pd$pct23[[1]] > 0,
            pd$ratio_f[[1]] * dir_muestras[[i]][k, ] / pd$pct23[[1]],
            pd$ratio_f[[1]]
          )
        } else {
          # el ratio (fijo o del swing logit) va ya dentro de E_prov
          E_prov[[i]][k, ] * rc[k, ]
        }
      } else {
        ifelse(pd$pct23[[1]] > 0, pd$ratio_f[[1]], 1)
      }
      v <- shares_m[k, ] * mult
      v <- v / sum(v)
      if (listas == "mapa") {
        # AA no sale del modelo: la regla provincial la fija (0 fuera de
        # las provincias con_aa)
        v <- c(v, setNames(0, "AA"))
      }
      if (!is.null(regla_prov)) {
        v <- aplicar_regla_mapa(
          v,
          regla_prov[i],
          pi_cat[k, c("ERC", "SUMAR")],
          aa_prov_k[k, i],
          pi_caniba[k, ]
        )
      }
      if (provincial) {
        v_prov[i, ] <- v
      }
      v[v < umbral] <- 0
      # RESTO agregado gana escaños fantasma (partidos pequeños que en la
      # realidad compiten por separado y casi siempre quedan bajo el umbral);
      # sus votos se tratan como no representados
      if (!resto_compite) {
        v["RESTO"] <- 0
      }
      e_prov[i, ] <- dhondt(v, pd$escanos)
    }
    if (provincial) {
      prov_voto[k, , ] <<- v_prov
      prov_esc[k, , ] <<- e_prov
    }
    colSums(e_prov)
  }

  esc <- purrr::map_dfr(seq_len(n_draws), \(k) {
    una(k) |>
      t() |>
      as_tibble(.name_repair = "minimal") |>
      setNames(fams)
  }) |>
    mutate(.draw = seq_len(n_draws), .before = 1)

  out <- list(
    draws = esc,
    time = time_obj,
    n_draws = n_draws,
    con_ruido = con_ruido,
    modo_ruido = modo_ruido,
    con_error = con_error,
    listas = listas
  )
  if (listas == "mapa") {
    out$transfer_cat_alpha <- transfer_cat_alpha / sum(transfer_cat_alpha)
    out$regla_prov <- setNames(regla_prov, prov_ref$provincia)
    out$aa_pct_and <- aa_pct_and
    out$aa_sd_logit <- aa_sd_logit
    out$aa_caniba_alpha <- aa_caniba_alpha / sum(aa_caniba_alpha)
    out$g_aa <- setNames(g_aa, prov_ref$provincia)
  }
  if (provincial) {
    out$prov_voto <- prov_voto
    out$prov_escanos <- prov_esc
    out$prov_ref <- prov_ref
  }
  out
}

# ---- resúmenes ---------------------------------------------------------------

resumen_escanos <- function(sim, width = 0.8) {
  sim$draws |>
    pivot_longer(-.draw, names_to = "partido", values_to = "escanos") |>
    group_by(partido) |>
    tidybayes::median_qi(escanos, .width = width) |>
    arrange(desc(escanos))
}

# etiqueta de escaños para los gráficos: las medianas por partido no suman
# 350 (cada una sale de draws distintos); las medias sí, porque cada draw
# suma 350. Se redondean por resto mayor para que la suma siga en 350.
escanos_resto_mayor <- function(sim) {
  medias <- colMeans(sim$draws[, setdiff(names(sim$draws), ".draw")])
  base <- floor(medias)
  faltan <- round(sum(medias)) - sum(base)
  extra <- rank(-(medias - base), ties.method = "first") <= faltan
  tibble(partido = names(medias), media = unname(medias),
         escanos = unname(base + extra))
}

bloques <- function(d) {
  d |>
    mutate(
      derecha = PP + VOX,
      # sin SALF: partido a caballo, no lo asigno a ningún bloque. "izquierda"
      # es el bloque de investidura de 2023 (los grupos que apoyaron a
      # Sánchez); UPN no entra (es derecha). En el escenario separadas PODEMOS
      # entra en la izquierda (con SUMAR forman la suma que calibra la
      # historia); AA también (escenario mapa)
      izquierda = PSOE +
        SUMAR +
        (if ("PODEMOS" %in% names(d)) PODEMOS else 0) +
        (if ("AA" %in% names(d)) AA else 0) +
        ERC +
        Junts +
        `EH Bildu` +
        PNV +
        BNG +
        CC
    )
}

prob_mayorias <- function(sim) {
  d <- bloques(sim$draws)
  tibble::tibble(
    escenario = c("PP solo >= 176", "PP + VOX >= 176", "izquierda >= 176"),
    prob = c(
      mean(d$PP >= 176),
      mean(d$derecha >= 176),
      mean(d$izquierda >= 176)
    )
  )
}

# ---- validación con el 23J ---------------------------------------------------

sim_validacion_23j <- function() {
  nac <- ref13 |>
    group_by(partido) |>
    summarise(votos_nac = sum(votos), .groups = "drop") |>
    arrange(match(partido, fams13))
  shares <- setNames(100 * nac$votos_nac / sum(nac$votos_nac), nac$partido)
  shares["SALF"] <- 0

  esc <- vapply(
    seq_len(nrow(prov_datos_23j)),
    \(i) {
      pd <- prov_datos_23j[i, ]
      v <- shares[fams13] * pd$ratio_f[[1]]
      v <- v / sum(v)
      v[v < 0.03] <- 0
      v["RESTO"] <- 0
      dhondt(v, pd$escanos)
    },
    numeric(13)
  ) |>
    t() |>
    colSums()
  setNames(esc, fams13)
}
