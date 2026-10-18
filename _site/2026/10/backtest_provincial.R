# ============================================================================
# VALIDACIÓN PROVINCIAL DEL BACKTEST (baseline del "punto 5")
#
# Misma configuración que backtest_guardar(geo = "previa", kappa = 0.5):
# geografía de la elección previa, mu del PSOE encogido a la mitad, misma
# semilla por elección (draws emparejados con el baseline nacional).
#
# Única diferencia con backtest_pipeline.R: en vez de escaños se registran
# los shares de voto provinciales de las familias seguidas en cada draw
# (v/sum(v) dentro de la provincia, incluidos los partidos no seguidos con
# su fuerza real). El umbral del 3% y D'Hondt no afectan a los shares de
# voto, así que se omiten.
#
# Cada celda (elección × familia × provincia) se compara con el pct_prov
# real de la elección objetivo y se marca si cae dentro del IC90.
#
# ref_ratio_cero marca las celdas donde el ratio de la REFERENCIA es 0
# (p. ej. VOX en la referencia de 28A): el modelo predice 0 por
# construcción, así que si el real es > 0 son fallos estructurales del
# supuesto del ratio, no del ruido.
#
# Salida:
#   2026/10/backtest_prov_celdas.csv — una fila por celda
# ============================================================================

library(tidyverse)
library(infoelectoral)
library(gtools)

err <- read_csv(
  here::here("2026/10/errores_encuestas_historicas.csv"),
  show_col_types = FALSE
)

elecciones <- list(
  list(id = "20d", anno = 2015, mes = "12"),
  list(id = "26j", anno = 2016, mes = "06"),
  list(id = "28a", anno = 2019, mes = "04"),
  list(id = "10n", anno = 2019, mes = "11"),
  list(id = "23j", anno = 2023, mes = "07")
)

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

prep_eleccion <- function(e) {
  p <- provincias(tipo_eleccion = "congreso", anno = e$anno, mes = e$mes) |>
    filter(codigo_provincia != "99") |>
    asignar_familia(e$id)

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
    mutate(pct_prov = 100 * votos / validos) |>
    # validos viaja con la provincia para poder agrupar celdas por tamaño
    group_by(codigo_provincia) |>
    mutate(validos_prov = first(validos)) |>
    ungroup()

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

# ---- backtest provincial de una elección ------------------------------------

# modo:
#   "dirichlet"        — baseline: re-sorteo Dirichlet del 23J (divisor 1.000)
#   "shock"            — shock jerárquico comunidad→provincia en log-ratio,
#                        calibrado con la deriva histórica (tau, de
#                        shock_jerarquico.R), sustituye a la Dirichlet y va
#                        re-centrado para conservar el voto nacional
#   "dirichlet+shock"  — ambos superpuestos (diagnóstico de doble conteo)
backtest_prov <- function(
  e,
  preps,
  b = 2000,
  n_mult = 1000,
  kappa = 0.5,
  modo = "dirichlet",
  swing = "proporcional",
  tau = NULL,
  mapa = NULL
) {
  swing <- match.arg(swing, c("proporcional", "logit"))
  stopifnot(swing == "proporcional" || modo %in% c("shock", "dirichlet+shock"))
  id <- e$id
  d <- preps[[id]]
  d_ref <- preps[[prev_de[[id]]]]
  enc <- err |> filter(eleccion == id)
  K <- nrow(enc)

  con_datos <- c("PP", "PSOE", "VOX", "SUMAR")[
    map_lgl(c("PP", "PSOE", "VOX", "SUMAR"), \(f) any(!is.na(enc[[f]])))
  ]

  # capa de error: idéntica al backtest nacional (mu LOO, sigma con las 5)
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

  R <- diag(length(con_datos))
  dimnames(R) <- list(con_datos, con_datos)
  if ("PP" %in% con_datos && "VOX" %in% con_datos) {
    R["PP", "VOX"] <- R["VOX", "PP"] <- 0.5
  }

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

  z <- matrix(rnorm(b * length(con_datos)), b, length(con_datos))
  zc <- z %*% t(chol(R))
  e_err <- sweep(zc, 2, sg, `*`) |> sweep(2, mu, "+")
  tracked <- pmax(agg + e_err, 0)
  bloque <- pmax(100 - rowSums(agg) + rnorm(b, 0, 1.5), 0)
  todo <- cbind(tracked, bloque)
  todo <- 100 * todo / rowSums(todo)
  colnames(todo) <- c(con_datos, "BLOQUE")
  tracked <- todo[, con_datos, drop = FALSE]

  # geografía: ratios y Dirichlet de la elección previa, como en el nacional
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
    p23[is.na(p23)] <- 0
    p23
  })
  q23_prov <- lapply(pct23_prov, \(p23) {
    q <- p23 / sum(p23)
    q[!is.finite(q)] <- 0
    q
  })
  nac_ref <- d_ref$real_nac |>
    filter(familia %in% con_datos)
  ratios <- lapply(pct23_prov, \(p23) {
    pct_nac <- nac_ref$pct[match(con_datos, nac_ref$familia)]
    ifelse(p23 > 0, p23 / pct_nac, 0)
  })
  prov_i <- fam_prov$codigo_provincia

  usa_dirichlet <- modo %in% c("dirichlet", "dirichlet+shock")
  usa_shock <- modo %in% c("shock", "dirichlet+shock")

  if (usa_shock) {
    stopifnot(!is.null(tau), !is.null(mapa))
    # tau: UNA FILA POR DRAW DE SIMULACIÓN Y FAMILIA (columnas draw, familia,
    # tau_c, tau_p). Con estimación puntual va replicada b veces; con brms
    # son draws del posterior muestreados a b, así la incertidumbre de los
    # tau entra en el shock
    tau_f <- tau |> filter(familia %in% con_datos)
    stopifnot(
      setequal(unique(tau_f$familia), con_datos),
      all(table(tau_f$familia) == b)
    )
    ccaa_prov <- mapa$codigo_ccaa[match(prov_i, mapa$codigo_provincia)]
    ccaa_unicos <- sort(unique(ccaa_prov))
    ccaa_idx <- match(ccaa_prov, ccaa_unicos)
    # pesos del re-centrado: votos válidos de la REFERENCIA (disponibles el
    # día del pronóstico). Tras exponenciar el shock, cada ratio se divide
    # por su media ponderada para que el voto nacional implícito no se
    # despegue del que fija la capa de error
    pesos_ref <- d_ref$prov |>
      distinct(codigo_provincia, w = validos_prov)
    pesos <- pesos_ref$w[match(prov_i, pesos_ref$codigo_provincia)]
    pesos <- pesos / sum(pesos)
    # E[[j]]: multiplicador (ratio x shock) por draw y provincia; rc[[j]]:
    # constante de re-centrado por draw. Con swing = "logit" el ratio es
    # dinámico: la share nacional del draw desplaza la odds de la share
    # provincial de referencia, partido a partido (los celdas con ratio 0
    # en la ref siguen con 0); el shock jerárquico entra como log-shock
    # sumado al log-odds (odds x exp(shock), el mismo multiplicador),
    # con los tau aún calibrados en log-ratio (recalibrar queda pendiente)
    p0_m <- do.call(rbind, pct23_prov)
    nac_ref_pct <- nac_ref$pct[match(con_datos, nac_ref$familia)]
    E <- vector("list", length(con_datos))
    rc <- vector("list", length(con_datos))
    for (j in seq_along(con_datos)) {
      # el tau de cada draw escala por filas: matrix(b x C) * vector(b)
      # recicla el vector por columnas, o sea, fila a fila
      tau_c_k <- tau_f$tau_c[tau_f$familia == con_datos[j]]
      tau_p_k <- tau_f$tau_p[tau_f$familia == con_datos[j]]
      eta <- matrix(rnorm(b * length(ccaa_unicos)), b, length(ccaa_unicos)) *
        tau_c_k
      eps <- matrix(rnorm(b * length(prov_i)), b, length(prov_i)) *
        tau_p_k
      # el log-shock (eta + eps) es lo que se suma al log-odds en logit;
      # Ef = exp(log_shock) es el multiplicador de la rama proporcional
      log_shock <- eta[, ccaa_idx, drop = FALSE] + eps
      Ef <- exp(log_shock)
      if (swing == "logit") {
        delta_k <- qlogis(pmin(pmax(tracked[, j] / 100, 1e-6), 1 - 1e-6)) -
          qlogis(nac_ref_pct[j] / 100)
        base <- plogis(
          sweep(
            sweep(log_shock, 2, qlogis(p0_m[, j] / 100), `+`),
            1,
            delta_k,
            `+`
          )
        )
        # division con suelo: tracked = 0 en algún draw no debe dar Inf
        Ej <- Ef * 100 * sweep(base, 1, pmax(tracked[, j], 1e-8), `/`)
      } else {
        ratio_f <- vapply(ratios, \(r) r[[j]], numeric(1))
        Ej <- sweep(Ef, 2, ratio_f, `*`)
      }
      E[[j]] <- Ej
      rc[[j]] <- as.numeric(Ej %*% pesos)^-1
    }
  }

  if (usa_dirichlet) {
    dir_m <- lapply(seq_len(nrow(fam_prov)), \(i) {
      fp <- fam_prov$data[[i]]
      v_prov <- fp$votos_candidaturas[1] + fp$votos_blancos[1]
      alpha <- ifelse(
        pct23_prov[[i]] > 0,
        pmax(pct23_prov[[i]] * v_prov / 1000, 0.5),
        0
      )
      # gtools:: explícito: brms enmascara rdirichlet y alteraría los draws
      gtools::rdirichlet(b, alpha)
    })
  }

  # shares provinciales simulados: una matriz b x n_fam por provincia
  prov_shares <- setNames(
    replicate(length(prov_i), matrix(NA_real_, b, length(con_datos)), FALSE),
    prov_i
  )
  for (k in seq_len(b)) {
    for (i in seq_along(prov_i)) {
      if (usa_shock) {
        mult <- vapply(
          seq_along(con_datos),
          \(j) {
            # el ratio (fijo o dinámico del swing logit) va dentro de E
            m <- E[[j]][k, i] * rc[[j]][k]
            if (usa_dirichlet && q23_prov[[i]][j] > 0) {
              m <- m * dir_m[[i]][k, j] / q23_prov[[i]][j]
            }
            if (is.na(m)) 0 else m
          },
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
      prov_shares[[i]][k, ] <- v[con_datos]
    }
  }

  # celdas: real (pct_prov del objetivo) vs IC90 de los draws
  bind_rows(lapply(seq_along(prov_i), \(i) {
    p <- prov_i[i]
    m <- prov_shares[[i]]
    real_fam <- d$prov |>
      filter(codigo_provincia == p, familia %in% con_datos)
    bind_rows(lapply(con_datos, \(f) {
      r <- real_fam$pct_prov[real_fam$familia == f]
      if (length(r) == 0) {
        r <- 0
      }
      # draws en proporción (0-1), real en %: escala común
      d0 <- 100 * m[, match(f, con_datos)]
      q05 <- unname(quantile(d0, 0.05))
      q95 <- unname(quantile(d0, 0.95))
      tibble(
        eleccion = id,
        codigo_provincia = p,
        familia = f,
        real = r,
        mediana = median(d0),
        q05 = q05,
        q95 = q95,
        # por posición: al indexar por un nombre ausente, R crea un elemento
        # con nombre NA aunque el valor ya se haya rellenado a 0
        ref_ratio_cero = pct23_prov[[i]][match(f, con_datos)] == 0,
        dentro = r >= q05 && r <= q95
      )
    }))
  })) |>
    left_join(
      d$prov |> distinct(codigo_provincia, validos_prov),
      by = "codigo_provincia"
    )
}
