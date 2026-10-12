# ============================================================================
# SIMULACIÓN DE ESCAÑOS DEL CONGRESO (desde los draws del meta-análisis)
#
# Replica la maquinaria de escanos-andalucia-2026.qmd adaptada al Congreso:
#
#   1. Draws del posterior del modelo 1 (6 categorías) y del modelo 2
#      (composición del BLOQUE, 8 subceldas) en la fecha objetivo. Ambos
#      con ref_time = 2026-07-01. La predicción usa una casa nueva
#      (allow_new_levels) para que el efecto casa entre como incertidumbre.
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
#      (tau_shock_draws.csv); SALF hereda los de SUMAR; los regionales
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
#          PSOE (+1.8 pp de media) y sobreestiman al PP (+1.0); la correlación
#          PP-VOX es prácticamente nula (-0.10), a diferencia de la -0.5 a mano
#          del post andaluz. Tras el error, el reescalado proporcional
#          conserva la masa sin cargar todo el ajuste sobre el BLOQUE.
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
#   resumen_escanos(sim); prob_mayorias(sim)
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

# datos por provincia en el ORDEN de fams13 (crítico: shares y Dirichlet
# van en el mismo orden). Nombres con sufijo para no pisar columnas de
# ref13 dentro de summarise (si partido = list(partido[...]) va primero,
# las expresiones siguientes ven la lista, no la columna).
prov_datos <- ref13 |>
  group_by(codigo_provincia) |>
  summarise(
    partido13 = list(partido[match(fams13, partido)]),
    # shares de referencia renormalizados: pct_prov no suma 100 donde SALF
    # (Europeas) se añadió encima del 23J. La Dirichlet se centra en ellos y
    # el multiplicador divide por ellos, así E[mult] = ratio exactamente.
    # (Antes alpha = votos / 1000 mezclaba votos de Europeas con el total del
    # 23J y centraba a SALF en ~0.68 de su ratio.)
    pct23 = list(pct_prov[match(fams13, partido)] / sum(pct_prov)),
    ratio13 = list(ratio[match(fams13, partido)]),
    # suelo de 0.5 solo para partidos con votos: los que no se presentan
    # (ratio 0) van con alpha 0 y no le quitan masa al resto
    alpha13 = list({
      p <- pct_prov[match(fams13, partido)] / sum(pct_prov)
      ifelse(p > 0, pmax(p * sum(votos) / 1000, 0.5), 0)
    }),
    escanos = diputados[1],
    .groups = "drop"
  )
stopifnot(nrow(prov_datos) == 52, sum(prov_datos$escanos) == 350)

# ---- draws de los dos modelos ------------------------------------------------

extraer_shares13 <- function(
  n_draws = 2000,
  time_obj = time_eleccion,
  con_error = TRUE,
  sigma_bloque = 1.5,
  cor_bloque = 0,
  t_sigma_df = NULL
) {
  mod_g <- ajustar_grande()
  mod_b <- ajustar_bloque()
  d1 <- tibble(time = time_obj, encuestadora = "prediccion_nueva", n = 1) |>
    add_epred_draws(mod_g, ndraws = n_draws, allow_new_levels = TRUE) |>
    ungroup() |>
    select(.draw, .category, .epred) |>
    pivot_wider(names_from = .category, values_from = .epred)
  d2 <- tibble(time = time_obj, bloque = 1) |>
    add_epred_draws(mod_b, ndraws = n_draws) |>
    ungroup() |>
    select(.draw, .category, .epred) |>
    pivot_wider(names_from = .category, values_from = .epred)
  stopifnot(nrow(d1) == nrow(d2))
  # epred del modelo 2: proporciones condicionales dentro del BLOQUE
  bloque_comp <- as.matrix(d2[, fams_bloque_fit])
  shares <- as.matrix(d1[, c("PP", "PSOE", "VOX", "SUMAR", "SALF", "BLOQUE")])
  # epred puede venir en proporciones (suman 1) o en conteos con n = 1:
  # normalizamos a porcentaje
  sc <- if (max(rowSums(shares)) < 1.5) 100 else 1
  shares6 <- shares * sc
  if (con_error) {
    # capa de error: e ~ N(mu, Sigma) sobre los 5 grandes. El reescalado
    # posterior reparte el ajuste proporcionalmente al tamaño de cada
    # partido (BLOQUE incluido), en vez de hacerle absorber el residual
    # completo: antes el BLOQUE pagaba todo el error de los grandes.
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
        z * (escala %*% diag(error_sigma))
      colnames(e) <- names(error_mu)
    } else {
      e <- MASS::mvrnorm(n_draws, error_mu, Sigma_error)
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
      # construcción y la correlación mecánica sería ≈ -1).
      e_b <- sigma_bloque *
        (cor_bloque *
          (e[, "PSOE"] - error_mu["PSOE"]) /
          error_sigma["PSOE"] +
          sqrt(1 - cor_bloque^2) * rnorm(n_draws))
      shares6[, 6] <- pmax(shares6[, 6] + e_b, 0)
    }
    shares6 <- 100 * shares6 / rowSums(shares6)
  }
  out <- cbind(shares6[, 1:5, drop = FALSE], shares6[, 6] * bloque_comp)
  colnames(out) <- fams13
  stopifnot(all(abs(rowSums(out) - 100) < 0.5))
  out
}

# ---- taus del shock (una fila por draw y familia) ----------------------------

preparar_tau_simular <- function(n_draws) {
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
  bind_rows(grandes, sal, reg, resto) |>
    mutate(familia = factor(familia, fams13)) |>
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
  con_error = TRUE,
  umbral = 0.03,
  resto_compite = FALSE,
  sigma_bloque = 1.5,
  cor_bloque = 0,
  t_sigma_df = NULL
) {
  modo_ruido <- match.arg(modo_ruido)
  shares_m <- extraer_shares13(
    n_draws,
    time_obj,
    con_error,
    sigma_bloque,
    cor_bloque,
    t_sigma_df
  )

  if (con_ruido && modo_ruido == "dirichlet") {
    # mecanismo antiguo (divisor 1.000), fuera del default: queda como
    # opción para comparar con el shock jerárquico
    dir_muestras <- lapply(prov_datos$alpha13, \(a) {
      gtools::rdirichlet(n_draws, a)
    })
  }

  if (con_ruido && modo_ruido == "shock") {
    tau <- preparar_tau_simular(n_draws)
    stopifnot(
      setequal(unique(tau$familia), fams13),
      all(table(tau$familia) == n_draws)
    )
    ccaa_prov <- prov_ref$ccaa[
      match(prov_datos$codigo_provincia, prov_ref$codigo_provincia)
    ]
    ccaa_u <- sort(unique(ccaa_prov))
    ccaa_idx <- match(ccaa_prov, ccaa_u)
    # pesos del re-centrado: votos de las 13 familias por provincia en la
    # ref (23J). Aproxima los votos válidos (~3% menos): el re-centrado es
    # una constante de normalización por draw, no una restricción dura
    w_prov <- ref13 |>
      group_by(codigo_provincia) |>
      summarise(v = sum(votos), .groups = "drop")
    w <- w_prov$v[match(prov_datos$codigo_provincia, w_prov$codigo_provincia)]
    w <- w / sum(w)
    # ratios de la ref: matriz 52 x 13 en el orden de fams13
    ratios_m <- do.call(rbind, prov_datos$ratio13)
    E_fam <- vector("list", length(fams13))
    rc <- matrix(NA_real_, n_draws, length(fams13))
    for (j in seq_along(fams13)) {
      # matrix(b x C) * vector(b) recicla por columnas: fila a fila
      tau_c_k <- tau$tau_c[tau$familia == fams13[j]]
      tau_p_k <- tau$tau_p[tau$familia == fams13[j]]
      eta <- matrix(rnorm(n_draws * length(ccaa_u)), n_draws) * tau_c_k
      eps <- matrix(rnorm(n_draws * nrow(prov_datos)), n_draws) * tau_p_k
      E_fam[[j]] <- exp(eta[, ccaa_idx, drop = FALSE] + eps)
      rc[, j] <- as.numeric(E_fam[[j]] %*% (w * ratios_m[, j]))^-1
    }
    # por provincia: matriz b x 13 (columnas en el orden de fams13)
    E_prov <- lapply(seq_len(nrow(prov_datos)), \(i) {
      sapply(E_fam, \(M) M[, i])
    })
  }

  una <- function(k) {
    vapply(
      seq_len(nrow(prov_datos)),
      \(i) {
        pd <- prov_datos[i, ]
        mult <- if (con_ruido) {
          if (modo_ruido == "dirichlet") {
            ifelse(
              pd$pct23[[1]] > 0,
              pd$ratio13[[1]] * dir_muestras[[i]][k, ] / pd$pct23[[1]],
              pd$ratio13[[1]]
            )
          } else {
            pd$ratio13[[1]] * E_prov[[i]][k, ] * rc[k, ]
          }
        } else {
          ifelse(pd$pct23[[1]] > 0, pd$ratio13[[1]], 1)
        }
        v <- shares_m[k, ] * mult
        v <- v / sum(v)
        v[v < umbral] <- 0
        # RESTO agregado gana escaños fantasma (partidos pequeños que en la
        # realidad compiten por separado y casi siempre quedan bajo el umbral);
        # sus votos se tratan como no representados
        if (!resto_compite) {
          v["RESTO"] <- 0
        }
        dhondt(v, pd$escanos)
      },
      numeric(13)
    ) |>
      t() |>
      colSums()
  }

  esc <- purrr::map_dfr(seq_len(n_draws), \(k) {
    una(k) |>
      t() |>
      as_tibble(.name_repair = "minimal") |>
      setNames(fams13)
  }) |>
    mutate(.draw = seq_len(n_draws), .before = 1)

  list(
    draws = esc,
    time = time_obj,
    n_draws = n_draws,
    con_ruido = con_ruido,
    modo_ruido = modo_ruido,
    con_error = con_error
  )
}

# ---- resúmenes ---------------------------------------------------------------

resumen_escanos <- function(sim, width = 0.8) {
  sim$draws |>
    pivot_longer(-.draw, names_to = "partido", values_to = "escanos") |>
    group_by(partido) |>
    tidybayes::median_qi(escanos, .width = width) |>
    arrange(desc(escanos))
}

bloques <- function(d) {
  d |>
    mutate(
      derecha = PP + VOX,
      # sin SALF: partido a caballo, no lo asigno a ningún bloque
      izquierda = PSOE +
        SUMAR +
        ERC +
        Junts +
        `EH Bildu` +
        PNV +
        BNG +
        CC +
        UPN
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
    seq_len(nrow(prov_datos)),
    \(i) {
      pd <- prov_datos[i, ]
      v <- shares[fams13] * pd$ratio13[[1]]
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
