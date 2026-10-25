# ============================================================================
# BACKTEST DEL PIPELINE COMPLETO: encuestas -> agregado -> error -> escaños
#
# Para cada una de las 5 últimas generales (20D-2015 a 23J-2023) simula lo que
# el pipeline habría predicho el día de la votación y lo compara con el
# resultado real:
#
#   1. Encuestas: las de última semana congeladas en
#      errores_encuestas_historicas.csv (media simple = el "agregado" que
#      calibra la capa de error). El CSV no conserva la n de cada encuesta:
#      el ruido multinomial del agregado asume n = 1.000 por encuesta.
#   2. Sorteo del agregado: bootstrap de encuestas (incertidumbre de casas) +
#      multinomial(n = 1.000) por encuesta remuestreada (ruido muestral).
#   3. Capa de error LEAVE-ONE-OUT: mu recalculado sin la elección testeada
#      (el mu_PSOE de +1.3 no se puede "saber" a priori; esto mide si ayuda
#      fuera de muestra). sigma usa las 5 elecciones (su estimación apenas
#      cambia al quitar una). Correlación PP-VOX +0.5 y shock propio del
#      BLOQUE N(0, 1.5^2), como en producción. kappa encoge el mu hacia 0
#      (1 = plug-in, 0.5 = medio camino, 0 = sin corrección sistemática).
#   4. Reparto provincial con la referencia configurada en d_ref: la PROPIA
#      elección (ratios real_prov/real_nac; circular en la geografía: aquí
#      no se testea el supuesto del ratio) o la ANTERIOR (lo que un
#      pronosticador habría tenido el día E). Los escaños por provincia
#      vienen de la elección objetivo (dato conocido de antemano; el reparto
#      cambió con la reforma de 2019). Los partidos no seguidos compiten con
#      su fuerza real de la elección OBJETIVO: su evolución nacional (p. ej.
#      el hundimiento de C's en 23J) la captaban las encuestas; lo que se
#      testea es la geografía de las familias seguidas. Ruido provincial con
#      modo = "shock" (producción: dos dados comunidad/provincia con los tau
#      draws de brms, tau_shock_draws.csv) o modo = "dirichlet" (el mecanismo
#      antiguo: concentración = votos de la provincia de referencia / 1.000).
#   5. Los partidos no seguidos (C's, regionales, PACMA...) compiten con su
#      fuerza real provincia a provincia — a diferencia del post, donde
#      RESTO va a no representados. Así el backtest evalúa los escaños de las
#      familias seguidas bajo competencia realista.
#   6. Umbral del 3% + D'Hondt provincia a provincia.
#
# Mapeo de familias: los códigos nacionales cambian entre elecciones
# (infoelectoral), así que PP/PSOE/VOX van por patrón de siglas y la familia
# SUMAR (el bloque Podemos + IU de la época) por códigos nacionales curados
# por elección y validados contra los reales congelados en
# errores_encuestas_historicas.csv (los reales del post son % sobre votos
# válidos = candidaturas + blancos; las diferencias del mapeo están por
# debajo de ~0,4 pp y son consistentes entre las 5 elecciones: el backtest
# usa sus propios reales, derivados del mismo mapeo que las ratios).
#
# Salidas:
#   2026/10/backtest_votos.csv   — real, mediana e IC90 de voto por familia
#   2026/10/backtest_escanos.csv — real, mediana e IC90 de escaños
#   2026/10/backtest_resumen.csv — cobertura agregada y P(derecha >= 176)
# ============================================================================

library(tidyverse)
library(infoelectoral)
library(gtools)

source(here::here("2026/10/simular_escanos.R")) # dhondt + la maquinaria común

f_err <- here::here("2026/10/errores_encuestas_historicas.csv")
err <- read_csv(f_err, show_col_types = FALSE)

set.seed(4177)

elecciones <- list(
  list(id = "20d", anno = 2015, mes = "12"),
  list(id = "26j", anno = 2016, mes = "06"),
  list(id = "28a", anno = 2019, mes = "04"),
  list(id = "10n", anno = 2019, mes = "11"),
  list(id = "23j", anno = 2023, mes = "07")
)

# familia SUMAR = el bloque Podemos + IU de cada época; por código nacional
# (las siglas de sus coaliciones son irrecognocibles: EN MAREA, ECP, AHAL
# DUGU, UNIDAD POPULAR...). Validado contra los reales de errores_historicas.
cods_sumar <- list(
  # 2011: no había Podemos; la familia es solo IU
  `11d` = "000088",
  `20d` = c("903736", "905033", "905041", "904850", "905008"),
  `26j` = c("000059", "000012", "000022", "000056", "000055"),
  `28a` = c("000077", "000032", "000075"),
  `10n` = c("000078", "000046", "000051", "000043", "000028", "000074"),
  `23j` = c("000010")
)

asignar_familia <- function(p, id) {
  # sin puntos para el matching: en 2011 van como "P.P."/"P.S.O.E." (las
  # siglas originales se conservan para los nombres x_ de los otros)
  s <- str_remove_all(p$siglas, "\\.")
  p |>
    mutate(
      familia = case_when(
        str_detect(s, regex("^PP", ignore_case = TRUE)) ~ "PP",
        str_detect(s, regex("PSOE|PSC|PSdeG|PSE", ignore_case = TRUE)) ~ "PSOE",
        str_detect(s, regex("^VOX", ignore_case = TRUE)) ~ "VOX",
        codigo_partido_nacional %in% cods_sumar[[id]] ~ "SUMAR",
        TRUE ~ str_c("x_", siglas)
      )
    )
}

# ---- datos reales por elección ----------------------------------------------

prep_eleccion <- function(e) {
  p <- provincias(tipo_eleccion = "congreso", anno = e$anno, mes = e$mes) |>
    filter(codigo_provincia != "99") |>
    asignar_familia(e$id)

  # válidos = candidaturas + blancos (la base de los % de los anexos);
  # n_diputados y blancos son constantes por provincia, no por fila
  den <- p |>
    distinct(
      codigo_provincia,
      votos_candidaturas,
      votos_blancos,
      n_diputados
    ) |>
    mutate(validos = votos_candidaturas + votos_blancos)

  fam <- p |>
    group_by(codigo_provincia, familia) |>
    summarise(votos = sum(votos), escanos = sum(diputados), .groups = "drop") |>
    inner_join(den, by = "codigo_provincia") |>
    mutate(pct_prov = 100 * votos / validos)

  list(
    id = e$id,
    prov = fam,
    real_nac = fam |>
      group_by(familia) |>
      summarise(votos = sum(votos), escanos = sum(escanos), .groups = "drop") |>
      mutate(pct = 100 * votos / sum(den$validos)) |>
      filter(familia %in% c("PP", "PSOE", "VOX", "SUMAR"))
  )
}

# ---- backtest de una elección ------------------------------------------------

backtest_eleccion <- function(
  e,
  preps,
  b = 2000,
  n_mult = 1000,
  kappa = 1,
  d_ref = NULL,
  nu = NULL,
  modo = c("shock", "dirichlet"),
  tau = NULL,
  mapa = NULL,
  # voto nacional ya sorteado fuera (matriz b x c(familias con encuestas,
  # "BLOQUE"), en %, con el error incluido): sustituye al agregado y a la
  # capa de error plug-in. Lo usa 2026/10/full-bayesian/10_backtest.R
  tracked_ext = NULL
) {
  modo <- match.arg(modo)
  id <- e$id
  d <- preps[[id]]
  d_ref <- d_ref %||% d
  enc <- err |> filter(eleccion == id)
  K <- nrow(enc)

  # familias con encuestas (VOX no se reporta en 20D/26J)
  con_datos <- c("PP", "PSOE", "VOX", "SUMAR")[
    map_lgl(c("PP", "PSOE", "VOX", "SUMAR"), \(f) any(!is.na(enc[[f]])))
  ]

  # reales de mi mapeo (voto y escaños) para las familias con encuestas
  real <- d$real_nac |>
    filter(familia %in% con_datos) |>
    select(familia, real_voto = pct, real_esc = escanos)

  if (!is.null(tracked_ext)) {
    stopifnot(
      nrow(tracked_ext) == b,
      identical(colnames(tracked_ext), c(con_datos, "BLOQUE"))
    )
    todo <- tracked_ext
    tracked <- todo[, con_datos, drop = FALSE]
    mu <- NULL
    sg <- NULL
  } else {
    # error del agregado por elección (con MI mapeo, consistente en las 5)
    errs <- bind_rows(map(elecciones, \(x) {
      ex <- err |> filter(eleccion == x$id)
      dx <- preps[[x$id]]
      tibble(
        eleccion = x$id,
        familia = con_datos,
        e = map_dbl(con_datos, \(f) {
          if (x$id != id && !any(!is.na(ex[[f]]))) {
            return(NA_real_)
          }
          dx$real_nac$pct[dx$real_nac$familia == f] - mean(ex[[f]], na.rm = TRUE)
        })
      )
    }))
    # mu leave-one-out, sigma con las 5 (su estimación apenas cambia al quitar una)
    mu_loo <- errs |>
      filter(eleccion != id) |>
      group_by(familia) |>
      summarise(mu = mean(e, na.rm = TRUE), .groups = "drop")
    sigma_5 <- errs |>
      group_by(familia) |>
      summarise(sigma = sd(e, na.rm = TRUE), .groups = "drop")

    mu <- kappa *
      setNames(mu_loo$mu[match(con_datos, mu_loo$familia)], con_datos)
    sg <- setNames(sigma_5$sigma[match(con_datos, sigma_5$familia)], con_datos)
    stopifnot(!any(is.na(mu)), !any(is.na(sg)))

    # matriz de correlaciones: solo PP-VOX a +0.5 (las demás a 0), como en producción
    R <- diag(length(con_datos))
    dimnames(R) <- list(con_datos, con_datos)
    if ("PP" %in% con_datos && "VOX" %in% con_datos) {
      R["PP", "VOX"] <- R["VOX", "PP"] <- 0.5
    }

    # --- sorteo del agregado: bootstrap de encuestas + multinomial(n_mult) ----
    # el multinomial va sobre las 5 categorías (con el BLOQUE como residual):
    # si normalizara solo las familias seguidas, sus shares saldrían
    # re-escalados a 100 y todas infladas
    agg <- t(sapply(seq_len(b), \(k) {
      sel <- sample.int(K, K, replace = TRUE)
      m <- reduce(
        map(sel, \(i) {
          p_i <- as.numeric(enc[i, con_datos])
          p5 <- c(p_i, 100 - sum(p_i))
          rmultinom(1, n_mult, p5)[, 1] * 100 / n_mult
        }),
        `+`
      ) /
        K
      setNames(m[seq_along(con_datos)], con_datos)
    }))

    # --- capa de error LOO + shock del BLOQUE + reescalado ---------------------
    # ojo al sweep: el default de sweep es la RESTA; aquí va la multiplicación
    # (hubo una versión que restaba sigma y metía un sesgo de -1,3 a -2,6 pp
    # por partido). Construcción idéntica a la de producción:
    #   e = mu + (z_corr ⊙ c) ⊙ sigma,  z_corr ~ N(0, R),  c = sqrt(nu/chi2_nu)
    z <- matrix(rnorm(b * length(con_datos)), b, length(con_datos))
    zc <- z %*% t(chol(R)) # cov = R, como el mvrnorm de producción
    # con nu (4 por defecto) cada muestra sortea su propio sigma — los
    # sigma_hat vienen de 5 elecciones y están estimados con ruido
    if (!is.null(nu)) {
      zc <- zc * sqrt(nu / matrix(rchisq(b * length(con_datos), nu), b))
    }
    e_err <- sweep(zc, 2, sg, `*`) |> sweep(2, mu, "+")
    tracked <- pmax(agg + e_err, 0)
    bloque <- pmax(100 - rowSums(agg) + rnorm(b, 0, 1.5), 0)
    todo <- cbind(tracked, bloque)
    todo <- 100 * todo / rowSums(todo)
    colnames(todo) <- c(con_datos, "BLOQUE")
    tracked <- todo[, con_datos, drop = FALSE]
  }

  # --- geografía: ratios y Dirichlet con la referencia -----------------------
  # d_ref es la elección previa (test del supuesto del ratio) o la propia
  # (circular). Los escaños por provincia vienen de la elección objetivo
  # (dato conocido de antemano: el reparto cambió con la reforma de 2019).
  dip_tgt <- d$prov |> distinct(codigo_provincia, n_diputados)
  fam_prov <- d_ref$prov |>
    select(-n_diputados, -escanos) |>
    left_join(dip_tgt, by = "codigo_provincia") |>
    filter(familia %in% con_datos) |>
    group_by(codigo_provincia) |>
    nest()
  otros <- d$prov |>
    filter(!familia %in% con_datos) |>
    group_by(codigo_provincia) |>
    summarise(otros_pct = list(setNames(pct_prov, familia)), .groups = "drop")
  otros_list <- setNames(otros$otros_pct, otros$codigo_provincia)

  pct23_prov <- lapply(seq_len(nrow(fam_prov)), \(i) {
    fp <- fam_prov$data[[i]]
    p23 <- setNames(fp$pct_prov, fp$familia)[con_datos]
    p23[is.na(p23)] <- 0 # familias sin fila en esa provincia (VOX en 2015...)
    p23
  })
  # shares DENTRO de las familias seguidas (suman 1): la Dirichlet se centra
  # en ellas y mult = ratio * dir/q23 tiene esperanza ratio exacta (en
  # producción pct23 ya venía como proporción dentro de las 13 familias)
  q23_prov <- lapply(pct23_prov, \(p23) {
    q <- p23 / sum(p23)
    q[!is.finite(q)] <- 0
    q
  })
  # el denominador del ratio es el nacional de la REFERENCIA, no el real
  # objetivo: ratio = share_prov_ref/share_nac_ref y el share nacional nuevo
  # lo aporta el agregado. Con d_ref = d son el mismo número y el ratio
  # reproduce la geografía real; con d_ref = previa, usar el nacional
  # objetivo anularía los cambios uniformes entre elecciones (un partido que
  # duplica su voto nacional quedaría con la mitad de ratio y mantendría su
  # voto de la elección anterior, como le pasaba al VOX en 28A)
  nac_ref <- d_ref$real_nac |>
    filter(familia %in% con_datos)
  ratios <- lapply(pct23_prov, \(p23) {
    pct_nac <- nac_ref$pct[match(con_datos, nac_ref$familia)]
    ifelse(p23 > 0, p23 / pct_nac, 0)
  })
  prov_i <- fam_prov$codigo_provincia # nest() saca la clave del sub-df

  if (modo == "shock") {
    # shock jerárquico comunidad→provincia, idéntico a backtest_prov()
    # (backtest_provincial.R): tau una fila por draw y familia, re-centrado
    # con los votos válidos de la REFERENCIA para conservar el voto nacional
    stopifnot(!is.null(tau), !is.null(mapa))
    tau_f <- tau |> filter(familia %in% con_datos)
    stopifnot(
      setequal(unique(tau_f$familia), con_datos),
      all(table(tau_f$familia) == b)
    )
    ccaa_prov <- mapa$codigo_ccaa[match(prov_i, mapa$codigo_provincia)]
    ccaa_unicos <- sort(unique(ccaa_prov))
    ccaa_idx <- match(ccaa_prov, ccaa_unicos)
    pesos <- vapply(
      seq_len(nrow(fam_prov)),
      \(i) fam_prov$data[[i]]$validos[1],
      numeric(1)
    )
    pesos <- pesos / sum(pesos)
    E <- vector("list", length(con_datos))
    rc <- vector("list", length(con_datos))
    for (j in seq_along(con_datos)) {
      # matrix(b x C) * vector(b) recicla por columnas: fila a fila
      tau_c_k <- tau_f$tau_c[tau_f$familia == con_datos[j]]
      tau_p_k <- tau_f$tau_p[tau_f$familia == con_datos[j]]
      eta <- matrix(rnorm(b * length(ccaa_unicos)), b, length(ccaa_unicos)) *
        tau_c_k
      eps <- matrix(rnorm(b * length(prov_i)), b, length(prov_i)) * tau_p_k
      Ef <- exp(eta[, ccaa_idx, drop = FALSE] + eps)
      ratio_f <- vapply(ratios, \(r) r[[j]], numeric(1))
      E[[j]] <- Ef
      rc[[j]] <- as.numeric(Ef %*% (pesos * ratio_f))^-1
    }
  } else {
    dir_m <- lapply(seq_len(nrow(fam_prov)), \(i) {
      fp <- fam_prov$data[[i]]
      v_prov <- fp$votos_candidaturas[1] + fp$votos_blancos[1]
      alpha <- ifelse(
        pct23_prov[[i]] > 0,
        pmax(pct23_prov[[i]] * v_prov / 1000, 0.5),
        0
      )
      # gtools:: explícito: brms enmascara rdirichlet y alteraría los draws
      gtools::rdirichlet(b, alpha) # b x n_fam, orden con_datos
    })
  }

  esc <- matrix(0, b, length(con_datos), dimnames = list(NULL, con_datos))
  derecha <- logical(b)
  for (k in seq_len(b)) {
    for (i in seq_len(nrow(fam_prov))) {
      fp <- fam_prov$data[[i]]
      if (modo == "shock") {
        mult <- ratios[[i]] *
          vapply(
            seq_along(con_datos),
            \(j) E[[j]][k, i] * rc[[j]][k],
            numeric(1)
          )
      } else {
        mult <- ifelse(
          q23_prov[[i]] > 0,
          ratios[[i]] * dir_m[[i]][k, ] / q23_prov[[i]],
          0
        )
      }
      v <- c(tracked[k, con_datos] * mult, otros_list[[prov_i[i]]])
      v <- v / sum(v)
      v[v < 0.03] <- 0
      h <- dhondt(v, fp$n_diputados[1])
      esc[k, ] <- esc[k, ] + h[con_datos]
    }
    # PP + VOX en el total nacional del draw (antes se comparaba con 176 el
    # reparto de cada provincia, y salía siempre 0)
    v_der <- esc[k, "PP"] + if ("VOX" %in% con_datos) esc[k, "VOX"] else 0
    derecha[k] <- v_der >= 176
  }
  esc <- as_tibble(esc)

  # --- métricas ---------------------------------------------------------------
  tab_votos <- bind_rows(lapply(con_datos, \(f) {
    d0 <- todo[, f]
    r <- real$real_voto[real$familia == f]
    tibble(
      eleccion = id,
      familia = f,
      real = r,
      mediana = median(d0),
      q05 = unname(quantile(d0, 0.05)),
      q95 = unname(quantile(d0, 0.95)),
      dentro = r >= quantile(d0, 0.05) && r <= quantile(d0, 0.95)
    )
  }))
  tab_esc <- bind_rows(lapply(con_datos, \(f) {
    d0 <- esc[[f]]
    r <- real$real_esc[real$familia == f]
    tibble(
      eleccion = id,
      familia = f,
      real = r,
      mediana = median(d0),
      q05 = unname(quantile(d0, 0.05)),
      q95 = unname(quantile(d0, 0.95)),
      dentro = r >= quantile(d0, 0.05) && r <= quantile(d0, 0.95)
    )
  }))
  list(
    id = id,
    votos = tab_votos,
    escanos = tab_esc,
    p_derecha = mean(derecha),
    real_derecha = sum(real$real_esc[real$familia %in% c("PP", "VOX")]),
    n_enc = K,
    familias = con_datos,
    kappa = kappa,
    mu_loo = mu,
    sigma = sg
  )
}

# ---- orquestación ------------------------------------------------------------

# elecciones de referencia geográfica: la anterior a cada una de las 5
refs_def <- list(
  list(id = "11d", anno = 2011, mes = "11"),
  list(id = "20d", anno = 2015, mes = "12"),
  list(id = "26j", anno = 2016, mes = "06"),
  list(id = "28a", anno = 2019, mes = "04"),
  list(id = "10n", anno = 2019, mes = "11")
)
prev_de <- c(
  `20d` = "11d",
  `26j` = "20d",
  `28a` = "26j",
  `10n` = "28a",
  `23j` = "10n"
)

preparar_preps <- function() {
  setNames(
    map(c(elecciones, refs_def), prep_eleccion),
    map_chr(c(elecciones, refs_def), "id")
  )
}

# set.seed dentro: variantes emparejadas (mismos draws de agregado y de
# geografía), la comparación no se contamina del ruido Monte Carlo
correr_bt <- function(
  preps,
  e,
  d_ref = NULL,
  b = 2000,
  kappa = 1,
  nu = NULL,
  modo = "shock",
  tau = NULL,
  mapa = NULL
) {
  set.seed(4177)
  backtest_eleccion(
    e,
    preps,
    b = b,
    kappa = kappa,
    d_ref = d_ref,
    nu = nu,
    modo = modo,
    tau = tau,
    mapa = mapa
  )
}

resumen_de <- function(res) {
  votos <- map_dfr(res, "votos")
  escanos <- map_dfr(res, "escanos")
  bind_rows(
    votos |> mutate(tipo = "voto"),
    escanos |> mutate(tipo = "escaños")
  ) |>
    group_by(tipo) |>
    summarise(
      celdas = n(),
      cobertura = mean(dentro),
      ancho_medio_ic90 = mean(q95 - q05),
      error_mediano = median(abs(mediana - real)),
      .groups = "drop"
    )
}

# grid: geografía (propia, circular vs previa) x shrinkage del mu
# (diagnóstico del mecanismo Dirichlet; el shock se valida en producción)
comparar_variantes <- function(b = 2000, nu = NULL, modo = "dirichlet") {
  preps <- preparar_preps()
  configs <- list(
    list(geo = "propia", kappa = 0.5),
    list(geo = "previa", kappa = 1),
    list(geo = "previa", kappa = 0.5),
    list(geo = "previa", kappa = 0)
  )
  map_dfr(configs, \(cfg) {
    res <- map(elecciones, \(e) {
      d_ref <- if (cfg$geo == "previa") preps[[prev_de[[e$id]]]] else NULL
      correr_bt(
        preps,
        e,
        d_ref = d_ref,
        b = b,
        kappa = cfg$kappa,
        nu = nu,
        modo = modo
      )
    })
    bind_rows(
      resumen_de(res) |>
        mutate(geo = cfg$geo, kappa = cfg$kappa, .before = 1),
      tibble(
        geo = cfg$geo,
        kappa = cfg$kappa,
        tipo = "P(derecha>=176)",
        celdas = length(res),
        cobertura = mean(map_dbl(res, "p_derecha")),
        ancho_medio_ic90 = NA,
        error_mediano = NA
      )
    )
  })
}

# guarda los CSV del backtest con la configuración elegida.
# modo = "shock" (producción): los tau draws salen del fit completo
# (mod_tau_shock.rds) con extraer_tau_draws(), los mismos del backtest
# provincial in-sample de producción
backtest_guardar <- function(
  geo = "previa",
  kappa = 0.5,
  b = 2000,
  nu = NULL,
  modo = "shock"
) {
  preps <- preparar_preps()
  tau <- NULL
  mapa <- NULL
  if (modo == "shock") {
    fit_full <- readRDS(here::here("2026/10/mod_tau_shock.rds"))
    tau <- extraer_tau_draws(fit_full, b = b)
    mapa <- mapa_ccaa_congreso()
  }
  res <- map(elecciones, \(e) {
    d_ref <- if (geo == "previa") preps[[prev_de[[e$id]]]] else NULL
    correr_bt(
      preps,
      e,
      d_ref = d_ref,
      b = b,
      kappa = kappa,
      nu = nu,
      modo = modo,
      tau = tau,
      mapa = mapa
    )
  })
  votos <- map_dfr(res, "votos")
  escanos <- map_dfr(res, "escanos")
  write_csv(votos, here::here("2026/10/backtest_votos.csv"))
  write_csv(escanos, here::here("2026/10/backtest_escanos.csv"))
  resumen <- resumen_de(res)
  mayores <- map_dfr(res, \(r) {
    tibble(
      eleccion = r$id,
      n_encuestas = r$n_enc,
      p_derecha_176 = r$p_derecha,
      real_derecha = r$real_derecha
    )
  })
  write_csv(
    resumen |>
      bind_rows(
        mayores |>
          mutate(tipo = "P(derecha)")
      ),
    here::here("2026/10/backtest_resumen.csv")
  )
  cat("\n== COBERTURA ==")
  print(resumen |> mutate(across(where(is.numeric), \(x) round(x, 2))))
  cat("\n== P(PP+VOX >= 176) ==")
  print(mayores |> mutate(across(where(is.numeric), \(x) round(x, 2))))
  invisible(list(votos = votos, escanos = escanos, res = res))
}

if (sys.nframe() == 0) {
  print(comparar_variantes())
}
