# ============================================================================
# FULL BAYESIAN 29-N · 11: datos del modelo conjunto
#
# Un único voto latente por campaña (las 5 generales pasadas y la de 2026) y
# un único formato para todas las encuestas: cada encuesta tiene celdas y
# cada celda suma una o varias familias de su campaña.
#
#   - Encuestas de los grandes (2026 y campañas pasadas): celdas que cubren
#     TODAS las familias de la campaña. En 2026 la celda "BLOQUE" suma las 8
#     familias regionales y RESTO.
#   - Desglose regional (2026, encuestas_meta_bloque.csv): celdas solo con
#     las 8 familias del BLOQUE. La probabilidad de cada celda se normaliza
#     dentro de las celdas de la encuesta: es la multinomial CONDICIONAL al
#     total del BLOQUE. Así las 22 encuestas que están en los dos ficheros no
#     cuentan dos veces el nivel del BLOQUE, y ElectoPanel (fuera de la
#     estimación de los grandes) solo aporta el reparto regional.
#
# Familias por campaña (la última es la referencia del log-ratio):
#   2026:       PP PSOE VOX SUMAR PODEMOS SALF Junts ERC PNV EH_Bildu BNG CC
#               UPN RESTO
#   2019-2023:  PP PSOE VOX SUMAR RESTO   (RESTO incluye a Cs en 2019)
#   2015-2016:  PP PSOE SUMAR RESTO       (VOX dentro de RESTO)
#
# Resultados (solo campañas pasadas): logit del resultado real de los grupos
# PP, PSOE, VOX y SUMAR; en 2026 el grupo SUMAR es SUMAR + PODEMOS.
#
# Uso: source("2026/10/full-bayesian/11_datos_conjunto.R");
#      dc <- datos_conjunto()
#      dc <- datos_conjunto(fuera = "23j")   # backtest: sin el 23J ni nada
#                                            # posterior a él
# ============================================================================

source(here::here("2026/10/full-bayesian/01_datos.R"))
suppressMessages(library(purrr))

fams_2026 <- c(
  "PP",
  "PSOE",
  "VOX",
  "SUMAR",
  "PODEMOS",
  "SALF",
  "Junts",
  "ERC",
  "PNV",
  "EH_Bildu",
  "BNG",
  "CC",
  "UPN",
  "RESTO"
)
fams_bloque_cj <- c(
  "Junts",
  "ERC",
  "PNV",
  "EH_Bildu",
  "BNG",
  "CC",
  "UPN",
  "RESTO"
)
grupos_error <- c("PP", "PSOE", "VOX", "SUMAR")
inicio_2026 <- as.Date("2025-06-02") # lunes de la semana del primer desglose
semanas_campana <- 8 # las 8 últimas semanas antes del día E son "campaña"
n_hist <- 1000 # tamaño supuesto de las encuestas históricas sin n conocido

fams_campana <- function(id) {
  if (id == "2026") {
    fams_2026
  } else if (id %in% c("20d", "26j")) {
    c("PP", "PSOE", "SUMAR", "RESTO")
  } else {
    c("PP", "PSOE", "VOX", "SUMAR", "RESTO")
  }
}

# ---- encuestas de cada campaña, en formato largo (encuesta x celda x familia) ----

# 2026, grandes: 7 celdas; la del BLOQUE suma las 8 familias regionales.
# Encuestas agregadas (preparar_meta.R: casas_agregadas, fichero
# encuestas_meta_agregadas.csv): la fuente publica PP, PSOE, VOX y Sumar +
# Podemos juntos, y SALF sin separar. Entran con celdas propias: la celda
# IZQ suma las familias SUMAR y PODEMOS (el mismo agregado que el modelo de
# error usa para la historia) y la residual lleva SALF, el BLOQUE y RESTO,
# que quedan sin observar por separado.
encuestas_2026_agregadas <- function() {
  f <- here::here("2026/10/encuestas_meta_agregadas.csv")
  if (!file.exists(f)) {
    return(NULL)
  }
  a <- read_csv(f, show_col_types = FALSE) |>
    filter(!is.na(muestra))
  if (nrow(a) == 0) {
    return(NULL)
  }
  celdas <- tibble(
    celda = c("PP", "PSOE", "VOX", "IZQ", "RESTO"),
    familias = list(
      "PP",
      "PSOE",
      "VOX",
      c("SUMAR", "PODEMOS"),
      c("SALF", fams_bloque_cj, "RESTO")
    )
  )
  map_dfr(seq_len(nrow(a)), \(i) {
    resto <- 100 - (a$PP[i] + a$PSOE[i] + a$VOX[i] + a$izq[i])
    stopifnot(resto > 0)
    tibble(
      campana = "2026",
      encuesta = paste("2026-ag", i),
      casa = norm_casa(a$encuestadora[i]),
      fecha = a$fecha[i],
      celda = celdas$celda,
      y = as.integer(round(
        c(a$PP[i], a$PSOE[i], a$VOX[i], a$izq[i], resto) / 100 * a$muestra[i]
      )),
      familias = celdas$familias
    )
  })
}

encuestas_2026_grandes <- function() {
  g <- prep_grande(f_grande_sep, cats_estado)
  celdas <- tibble(
    celda = cats_estado,
    familias = c(as.list(cats_estado[1:6]), list(fams_bloque_cj))
  )
  map_dfr(seq_len(nrow(g)), \(i) {
    tibble(
      campana = "2026",
      encuesta = paste("2026-g", i),
      casa = norm_casa(g$encuestadora[i]),
      fecha = g$fecha[i],
      celda = celdas$celda,
      y = as.integer(unlist(g[i, cats_estado])),
      familias = celdas$familias
    )
  })
}

# 2026, desglose regional: 8 celdas, condicional al BLOQUE
encuestas_2026_desglose <- function() {
  b <- prep_bloque()
  map_dfr(seq_len(nrow(b)), \(i) {
    tibble(
      campana = "2026",
      encuesta = paste("2026-d", i),
      casa = norm_casa(b$encuestadora[i]),
      fecha = b$fecha[i],
      celda = fams_bloque_cj,
      y = as.integer(b$cell_counts[i, ]),
      familias = as.list(fams_bloque_cj)
    )
  })
}

# tamaño de la muestra escrito en el nombre de la casa ("CELESTE-TEL (1.100)",
# anexo del 20D); NA si no está
n_de_casa <- function(x) {
  n <- str_match(x, "\\(([0-9][0-9.]*)\\)")[, 2]
  suppressWarnings(as.integer(str_remove_all(n, "\\.")))
}

# campañas pasadas: errores_encuestas_ventana.csv, publicables (fin de campo
# al menos 6 días antes del día E). n: el de la columna n o el del nombre de
# la casa; si no hay, 1.000. Sin el CIS, como en 2026.
encuestas_pasadas <- function(id, min_dias = 6) {
  f <- setdiff(fams_campana(id), "RESTO")
  read_csv(
    here::here("2026/10/errores_encuestas_ventana.csv"),
    show_col_types = FALSE
  ) |>
    filter(eleccion == id, dias >= min_dias) |>
    filter(if_all(all_of(f), \(x) !is.na(x))) |>
    mutate(RESTO = 100 - rowSums(across(all_of(f)))) |>
    filter(RESTO > 0) |>
    mutate(
      n_enc = coalesce(as.integer(n), n_de_casa(casa), as.integer(n_hist)),
      encuesta = paste(id, row_number()),
      casa = norm_casa(casa)
    ) |>
    filter(casa != "cis") |>
    pivot_longer(
      all_of(c(f, "RESTO")),
      names_to = "celda",
      values_to = "pct"
    ) |>
    transmute(
      campana = id,
      encuesta,
      casa,
      fecha,
      celda,
      y = as.integer(round(pct / 100 * n_enc)),
      familias = as.list(celda)
    )
}

# ---- datos para Stan ---------------------------------------------------------------

# fuera: general pasada que se predice en el backtest. Se quita su resultado
# y todo lo posterior a su día E (campañas, autonómicas); su campaña se
# conserva sin el resultado y hace el papel de "2026".
# corte_dias: horizonte del backtest. De la campaña objetivo solo entran las
# encuestas con fin de campo al menos corte_dias antes del día E (y las
# autonómicas anteriores a esa fecha). 0 = la víspera (las de 6 días o más).
datos_conjunto <- function(fuera = NULL, corte_dias = 0) {
  pasadas <- elecciones_hist
  objetivo <- "2026"
  dia_obj <- as.Date(fecha_elecciones)
  if (!is.null(fuera)) {
    objetivo <- fuera
    dia_obj <- fechas_hist[[fuera]]
    pasadas <- elecciones_hist[fechas_hist < dia_obj]
  }
  campanas <- c(pasadas, objetivo)
  dia_e <- c(fechas_hist[pasadas], setNames(dia_obj, objetivo))

  enc <- bind_rows(
    map_dfr(setdiff(campanas, "2026"), \(c) {
      encuestas_pasadas(c, if (c == objetivo) max(6, corte_dias) else 6)
    }),
    if ("2026" %in% campanas) {
      bind_rows(
        encuestas_2026_grandes(),
        encuestas_2026_agregadas(),
        encuestas_2026_desglose()
      )
    }
  )

  # semanas: desde el primer lunes con encuestas de cada campaña hasta el día E
  inicio <- enc |>
    group_by(campana) |>
    summarise(inicio = min(fecha), .groups = "drop")
  inicio$inicio[inicio$campana == "2026"] <- inicio_2026
  inicio <- setNames(inicio$inicio, inicio$campana)[campanas]
  semana_de <- \(c, f) {
    as.integer(floor(as.numeric(as.Date(f) - inicio[c]) / 7)) + 1L
  }
  T_c <- map_int(campanas, \(c) semana_de(c, dia_e[[c]]))

  K_c <- map_int(campanas, \(c) length(fams_campana(c)))
  # casas: un efecto por (campaña, casa)
  slots <- enc |>
    distinct(campana, casa) |>
    arrange(match(campana, campanas), casa)
  slot_de <- \(c, h) match(paste(c, h), paste(slots$campana, slots$casa))

  encs <- enc |> distinct(encuesta, campana, casa, fecha)
  cel <- enc |> mutate(i = match(encuesta, encs$encuesta))
  memb <- cel |>
    mutate(j = row_number()) |>
    select(j, campana, familias) |>
    unnest(familias) |>
    mutate(k = map2_int(campana, familias, \(c, f) match(f, fams_campana(c))))
  stopifnot(!anyNA(memb$k))

  # resultados reales de las campañas pasadas (grupos PP, PSOE, VOX, SUMAR)
  real <- errores_hist()$real |> filter(eleccion %in% pasadas)
  logit_real <- sapply(grupos_error, \(g) qlogis(real[[g]] / 100))
  if (is.null(dim(logit_real))) {
    logit_real <- matrix(logit_real, 1)
  }
  obs_real <- !is.na(logit_real)
  logit_real[!obs_real] <- 0
  grupo_memb <- map_dfr(seq_along(campanas), \(c) {
    f <- fams_campana(campanas[c])
    map_dfr(seq_along(grupos_error), \(g) {
      miembros <- if (grupos_error[g] == "SUMAR") {
        c("SUMAR", "PODEMOS")
      } else {
        grupos_error[g]
      }
      k <- match(miembros, f)
      tibble(c = c, g = g, k = k[!is.na(k)])
    })
  })

  # autonómicas anteriores al día E del objetivo
  db <- datos_error(
    antes_de = dia_obj - corte_dias,
    casas_obj = character(0),
    con_autonomicas = TRUE
  )
  aut <- db$tipo[db$eleccion] == 2

  list(
    # campañas
    C = length(campanas),
    K = K_c,
    T = T_c,
    theta1_mu = unlist(map(campanas, \(c) lr_previa(c))),
    campana_sem = unlist(map2(T_c, campanas, \(t, c) {
      as.integer(seq_len(t) > t - semanas_campana)
    })),
    # encuestas y celdas
    N = nrow(encs),
    S = nrow(slots),
    enc_campana = match(encs$campana, campanas),
    enc_semana = map2_int(encs$campana, encs$fecha, semana_de),
    enc_slot = map2_int(encs$campana, encs$casa, slot_de),
    J = nrow(cel),
    celda_enc = cel$i,
    y = cel$y,
    M = nrow(memb),
    memb_celda = memb$j,
    memb_fam = memb$k,
    # resultados
    P = length(pasadas),
    G = length(grupos_error),
    logit_real = logit_real,
    obs_real = matrix(as.integer(obs_real), nrow(obs_real)),
    GM = nrow(grupo_memb),
    gm_campana = grupo_memb$c,
    gm_grupo = grupo_memb$g,
    gm_fam = grupo_memb$k,
    # autonómicas (errores en logit de la última semana)
    Ja = sum(aut),
    za = db$z[aut, , drop = FALSE],
    obsa = db$obs[aut, , drop = FALSE],
    Ea = length(unique(db$eleccion[aut])),
    elec_a = as.integer(factor(db$eleccion[aut])),
    Ha = length(unique(db$casa[aut])),
    casa_a = as.integer(factor(db$casa[aut])),
    # para leer la salida
    campanas = campanas,
    objetivo = objetivo,
    dia_e = dia_e,
    slots = slots,
    encuestas = encs
  )
}

# prior de la semana 1 de cada campaña: log del voto de la general anterior
# (una entrada por familia, sin referencia; el modelo lo centra). Para 2026 se usa ref_provincial_23J_14.csv (Sumar y
# Podemos repartidos, SALF de las Europeas 2024, regionales del 23J).
lr_previa <- function(c) {
  f <- fams_campana(c)
  if (c == "2026") {
    r <- read_csv(
      here::here("2026/10/ref_provincial_23J_14.csv"),
      show_col_types = FALSE
    ) |>
      distinct(partido, pct_nac) |>
      mutate(partido = sub("EH Bildu", "EH_Bildu", partido))
    pct <- setNames(r$pct_nac, r$partido)[f]
  } else {
    previa <- c(
      `20d` = NA,
      `26j` = "20d",
      `28a` = "26j",
      `10n` = "28a",
      `23j` = "10n"
    )[[c]]
    real <- errores_hist()$real
    if (is.na(previa)) {
      # 20D: no hay general anterior en los datos; prior centrado en el
      # resultado del propio 20D con la misma sd (0,5): es un prior débil
      # sobre el punto de partida, no sobre el resultado
      previa <- "20d"
    }
    rp <- real |> filter(eleccion == previa)
    pct <- c(
      PP = rp$PP,
      PSOE = rp$PSOE,
      VOX = rp$VOX,
      SUMAR = rp$SUMAR
    )[setdiff(f, "RESTO")]
    pct[is.na(pct)] <- 1 # VOX en 2015-2016 no existe como categoría
    pct <- c(pct, RESTO = 100 - sum(pct))
  }
  unname(log(pct))
}

# grupos de sigma_rw (y de sigma_casa): los cuatro grandes por separado,
# Podemos y SALF juntos, los regionales juntos y RESTO (con Cs en 2015-2019)
grupo_rw <- c(
  PP = 1,
  PSOE = 2,
  VOX = 3,
  SUMAR = 4,
  PODEMOS = 5,
  SALF = 5,
  Junts = 6,
  ERC = 6,
  PNV = 6,
  EH_Bildu = 6,
  BNG = 6,
  CC = 6,
  UPN = 6,
  RESTO = 7
)

priors_conjunto <- list(
  eta1_sd = 0.5,
  sd_rw = 0.05,
  sd_kappa = 0.5,
  sd_casa = 0.2,
  # phi: concentración de la Dirichlet-multinomial. Centrado en 2.000 (del
  # orden de una muestra): con el 90 % entre ~200 y ~20.000, deja que los
  # datos digan si hay sobredispersión (phi pequeño) o no (phi grande)
  log_phi_mu = log(2000),
  sd_log_phi = 1.4,
  sd_mu = 0.1,
  sd_log_s0 = 0.5,
  sd_tau_s = 0.3,
  sd_gamma = 0.5,
  sd_enc_a = 0.2,
  sd_b = 0.1,
  eta_lkj = 2,
  # volatilidad propia de cada campaña (1, el modelo principal desde el 18
  # oct: tau_rw = 0,70, las campañas difieren mucho) o común (0, la v2);
  # tau_rw ~ N+(0, 0,5):
  # el 90 % del prior deja que una campaña tenga entre ~0,4 y ~2,3 veces la
  # volatilidad media
  sigma_rw_campana = 1L,
  sd_tau_rw = 0.5
)

# lista para Stan: añade a datos_conjunto() los índices de inicio y longitud
# (encuesta -> celdas, celda -> pertenencias) y las escalas de los priors
stan_conjunto <- function(dc, prior_only = 0L, priors = priors_conjunto) {
  enc_c_n <- as.integer(table(factor(dc$celda_enc, seq_len(dc$N))))
  cel_m_n <- as.integer(table(factor(dc$memb_celda, seq_len(dc$J))))
  stopifnot(
    !is.unsorted(dc$celda_enc),
    !is.unsorted(dc$memb_celda),
    all(enc_c_n > 0),
    all(cel_m_n > 0)
  )
  rwgrp <- unlist(map(dc$campanas, \(c) unname(grupo_rw[fams_campana(c)])))
  c(
    dc[c(
      "C",
      "K",
      "T",
      "theta1_mu",
      "campana_sem",
      "N",
      "S",
      "enc_campana",
      "enc_semana",
      "enc_slot",
      "J",
      "y",
      "M",
      "memb_fam",
      "P",
      "G",
      "logit_real",
      "obs_real",
      "GM",
      "gm_campana",
      "gm_grupo",
      "gm_fam",
      "Ja",
      "za",
      "obsa",
      "Ea",
      "elec_a",
      "Ha",
      "casa_a"
    )],
    list(
      L1 = sum(dc$K),
      rwgrp = as.integer(rwgrp),
      R = 7L,
      enc_c_ini = as.integer(cumsum(c(1L, head(enc_c_n, -1)))),
      enc_c_n = enc_c_n,
      slot_campana = match(dc$slots$campana, dc$campanas),
      cel_m_ini = as.integer(cumsum(c(1L, head(cel_m_n, -1)))),
      cel_m_n = cel_m_n
    ),
    priors,
    prior_only = as.integer(prior_only)
  )
}

if (sys.nframe() == 0) {
  dc <- datos_conjunto()
  cat("Campañas:", paste(dc$campanas, collapse = ", "), "\n")
  print(tibble(
    campana = dc$campanas,
    familias = dc$K,
    semanas = dc$T,
    encuestas = as.integer(table(factor(dc$enc_campana, seq_len(dc$C)))),
    casas = as.integer(table(factor(dc$slots$campana, dc$campanas)))
  ))
  cat(
    "\nCeldas:",
    dc$J,
    " pertenencias celda-familia:",
    dc$M,
    " resultados pasados:",
    dc$P,
    "\nAutonómicas: ",
    dc$Ja,
    "encuestas en",
    dc$Ea,
    "elecciones,",
    dc$Ha,
    "casas\n"
  )
}
