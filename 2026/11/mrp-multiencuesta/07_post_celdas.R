# Postestratificación por celda de población, casa media y último mes: para
# cada draw de la posterior, la probabilidad de voto válido en cada celda
# provincia x sexo x edad x edu x rec de la población. Es el artefacto del que
# saldrán los escaños (08_escanos.R), la matriz de transición 23J->hoy y las
# curvas de edad.
#
# Dos fuentes, mismo formato de salida (post_celdas_casa_media.rds):
#   - brms (mod_brms_nuts.rds): posterior_epred con la fórmula COMPLETA
#     (re_formula = NULL, incluidos (1 | casa) y (1 | casa:rec)), un epred por
#     casa y media entre casas (la casa no se conoce en la población). OJO: no
#     recortar la fórmula: en brms los interceptos de cada dpar van libres y el
#     modo común solo se cancela con todos los efectos; sin los de casa el eta
#     de celda se infla (sd 0.20 -> 0.60) y los escaños salen con intervalos
#     2-3 veces más anchos (v. PLAN.md, 2026-10-25). Sin modelo de participación.
#   - stan (fit_separado_<modo>_voto.rds, de 05b_separado.R): se reproducen en
#     R los draws de mrp_voto.stan, con la casa media = media de las 4 casas
#     (como la fila n_casa + 1 de v_casa en el modelo). OJO: no es exactamente
#     el efecto casa = 0 de brms (la media de 4 efectos estimados no es 0).
#     Con "part" se añade además pv, el peso de participación de mrp_part.stan
#     por celda y draw, para ponderar las celdas (como v_part en 05b).
#
#   Rscript 2026/11/mrp-multiencuesta/07_post_celdas.R                      # brms, 1000 draws
#   Rscript 2026/11/mrp-multiencuesta/07_post_celdas.R prueba               # brms, 50 draws, sin guardar
#   Rscript 2026/11/mrp-multiencuesta/07_post_celdas.R nuts stan            # stan, 1000 draws
#   Rscript 2026/11/mrp-multiencuesta/07_post_celdas.R nuts stan part       # stan con participación
#   Rscript 2026/11/mrp-multiencuesta/07_post_celdas.R prueba stan part     # prueba con los ajustes prueba
#   Rscript 2026/11/mrp-multiencuesta/07_post_celdas.R nuts rtmb part       # RTMB (rtmb/05b_rtmb.R)
#
# Fuente rtmb: los mismos cálculos que stan con las muestras de Laplace de
# datos/fit_rtmb_voto_<modo>.rds y fit_rtmb_part_<modo>.rds (1.000 muestras
# independientes; ver rtmb/mrp_voto_rtmb.R). Salidas con el sufijo _rtmb.
#
# Salida: datos/post_celdas_casa_media.rds (stan, o _part con participación) o
# datos/post_celdas_casa_media_brms.rds (brms), con
#   p       array draws x celda x partido (probabilidad de voto válido)
#   post    tibble con la metainformación de la celda (prov, ccaa, sexo, edad,
#           edu, rec, N) y, si "part", la columna pv_media (media de pv por
#           celda; el pv draw a draw va en el array pv)
#   pv      array draws x celda (solo con "part")

library(dplyr)
library(brms)

args <- commandArgs(trailingOnly = TRUE)
modo <- if (length(args) >= 1) args[1] else "nuts"
fuente <- if (length(args) >= 2) args[2] else "brms"
con_part <- if (length(args) >= 3) args[3] == "part" else FALSE
stopifnot(modo %in% c("nuts", "prueba"), fuente %in% c("brms", "stan", "rtmb"))
stopifnot(!con_part || fuente != "brms") # el modelo de participación no existe en brms
ndraws <- if (modo == "prueba") 50L else 1000L
cat(
  "modo:",
  modo,
  " fuente:",
  fuente,
  " participación:",
  con_part,
  " draws:",
  ndraws,
  "\n"
)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
source(file.path(dir_proy, "00_partidos.R"))

tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))
voto_lv <- setdiff(partidos_lv, "AC")
territorio_mod <- territorio[names(territorio) != "AC"]

# Celdas de población, igual que en 04_brms.R y 04b (y que 05_stan.R)
post <- as.data.frame.table(tabla, responseName = "N") |>
  filter(N > 1e-6) |>
  left_join(prov_ccaa, by = "prov")
stopifnot(nrow(post) == 12264)

p_celda <- NULL # draws x celda x partido
pv <- NULL # draws x celda
mes_salida <- NULL

if (fuente == "brms") {
  mod <- readRDS(file.path(dir_datos, "mod_brms_nuts.rds"))
  celdas <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
  prov_lv <- levels(prov_ccaa$prov)
  K <- 9
  post <- post |> mutate(n = 1)
  for (k in seq_len(K)) {
    post[[paste0("p", k)]] <- factor(post$prov, levels = prov_lv)
    post[[paste0("w", k)]] <- as.numeric(k == 1)
  }
  for (p in names(territorio_mod)) {
    post[[paste0("off_", p)]] <- ifelse(
      post$ccaa %in% territorio_mod[[p]],
      0,
      -20
    )
  }
  post$mes <- factor(max(levels(celdas$mes)), levels = levels(celdas$mes))

  mascara <- sapply(voto_lv, function(p) {
    if (p %in% names(territorio_mod)) {
      as.numeric(post$ccaa %in% territorio_mod[[p]])
    } else {
      rep(1, nrow(post))
    }
  })

  casas <- levels(celdas$casa)
  p_celda <- NULL
  for (cs in casas) {
    post_c <- post
    post_c$casa <- factor(cs, levels = casas)
    # misma submuestra de draws para las 4 casas
    set.seed(2026)
    pc <- posterior_epred(
      mod,
      newdata = post_c,
      ndraws = ndraws,
      allow_new_levels = TRUE,
      sample_new_levels = "gaussian"
    )
    p_celda <- if (is.null(p_celda)) pc else p_celda + pc
    cat("  casa", cs, "hecha\n")
  }
  p_celda <- p_celda / length(casas)
  for (d in seq_len(dim(p_celda)[1])) {
    p_celda[d, , ] <- p_celda[d, , ] * mascara
    p_celda[d, , ] <- p_celda[d, , ] / rowSums(p_celda[d, , ])
  }
  mes_salida <- max(levels(celdas$mes))
  nd <- ndraws
} else {
  # ---- stan: mrp_voto.stan (+ mrp_part.stan con "part"), como 05b_separado.R --

  # los datos de los ajustes salen de 05_stan.R (SOLO_DATOS, sin ajustar)
  f_datos <- tempfile(fileext = ".rds")
  st <- system2(
    "Rscript",
    c(file.path(dir_proy, "05_stan.R"), modo, "nuevo"),
    env = paste0("SOLO_DATOS=", f_datos),
    stdout = FALSE
  )
  stopifnot(st == 0, file.exists(f_datos))
  x <- readRDS(f_datos)
  datos <- x$datos
  mes_salida <- x$lv$mes[datos$mes_post]

  # las celdas de 05_stan.R y las de aquí tienen que ser las mismas, en el mismo orden
  stopifnot(
    identical(as.numeric(datos$P_N), as.numeric(post$N)),
    identical(as.integer(datos$P_prov), as.integer(as.character(post$prov))),
    identical(as.integer(datos$P_edad), match(post$edad, x$lv$edad)),
    identical(as.integer(datos$P_edu), match(post$edu, x$lv$edu)),
    identical(as.integer(datos$P_rec), match(post$rec, x$lv$rec)),
    identical(as.integer(datos$P_ccaa), match(post$ccaa, x$lv$ccaa))
  )

  matriz <- function(dm) {
    matrix(dm, nrow(dm), dimnames = list(NULL, colnames(dm)))
  }
  # thinning determinista: ndraws índices equiespaciados
  adelgaza <- function(dm, n) {
    dm[round(seq(1, nrow(dm), length.out = n)), , drop = FALSE]
  }
  n_casa <- datos$n_casa
  n_rec <- datos$n_rec
  P <- nrow(post)
  J <- datos$J

  arr <- function(dm, v, n) {
    m <- dm[, paste0(v, "[", rep(1:n, J), ",", rep(1:J, each = n), "]")]
    array(m, c(nrow(dm), n, J))
  }
  vecs <- function(dm, v, n) dm[, paste0(v, "[", 1:n, "]"), drop = FALSE]

  if (fuente == "rtmb") {
    # muestras independientes: se toman ndraws equiespaciadas, como adelgaza()
    A <- readRDS(file.path(dir_datos, paste0("fit_rtmb_voto_", modo, ".rds")))$A
    i_d <- round(seq(1, nrow(A$alpha), length.out = ndraws))
    A <- lapply(A, function(a) if (length(dim(a)) == 3) a[i_d, , , drop = FALSE] else a[i_d, , drop = FALSE])
    nd <- ndraws
  } else {
  av <- readRDS(file.path(
    dir_datos,
    paste0("fit_separado_", modo, "_voto.rds")
  ))
  dv <- matriz(av$fit$draws(format = "draws_matrix"))
  dv <- adelgaza(dv, ndraws)
  nd <- nrow(dv)
  A <- list(
    alpha = vecs(dv, "alpha", J),
    b_mujer = vecs(dv, "b_mujer", J),
    mes = arr(dv, "a_mes", datos$n_mes),
    edad = arr(dv, "a_edad", datos$n_edad),
    edu = arr(dv, "a_edu", datos$n_edu),
    rec = arr(dv, "a_rec", n_rec),
    ccaa = arr(dv, "a_ccaa", datos$n_ccaa),
    prov = arr(dv, "a_prov", datos$n_prov),
    ccaa_rec = arr(dv, "a_ccaa_rec", datos$n_ccaa * n_rec),
    casa = arr(dv, "a_casa", n_casa),
    casa_rec = arr(dv, "a_casa_rec", n_casa * n_rec)
  )
  }
  stopifnot(length(datos$P_mascara) == P * J)
  mascara <- matrix(datos$P_mascara, P, J) # P_mascara viene columna a columna

  pv <- NULL
  if (con_part && fuente == "rtmb") {
    Tp <- readRDS(file.path(dir_datos, paste0("fit_rtmb_part_", modo, ".rds")))$Tp
    i_p <- round(seq(1, length(Tp$delta), length.out = ndraws))
    Tp <- lapply(Tp, function(a) if (is.null(dim(a))) a[i_p] else a[i_p, , drop = FALSE])
  } else if (con_part) {
    ap <- readRDS(file.path(
      dir_datos,
      paste0("fit_separado_", modo, "_part.rds")
    ))
    dp <- matriz(ap$fit$draws(format = "draws_matrix"))
    dp <- adelgaza(dp, ndraws)
    Tp <- list(
      alpha = dp[, "alpha_t"],
      b = dp[, "b_mujer_t"],
      mes = vecs(dp, "a_mes_t", datos$n_mes),
      edad = vecs(dp, "a_edad_t", datos$n_edad),
      edu = vecs(dp, "a_edu_t", datos$n_edu),
      rec = vecs(dp, "a_rec_t", n_rec),
      ccaa = vecs(dp, "a_ccaa_t", datos$n_ccaa),
      prov = vecs(dp, "a_prov_t", datos$n_prov),
      delta = dp[, "delta"]
    )
  }

  # índices de las celdas de población (los mismos que ya verificó el stopifnot)
  P_mujer <- datos$P_mujer
  P_edad <- datos$P_edad
  P_edu <- datos$P_edu
  P_rec <- datos$P_rec
  P_ccaa <- datos$P_ccaa
  P_prov <- datos$P_prov
  i_cr <- (P_ccaa - 1) * n_rec + P_rec

  # prob_valida() por filas, como en 05b
  prob <- function(eta) {
    p <- exp(eta - apply(eta, 1, max)) * mascara
    p / rowSums(p)
  }

  p_celda <- array(
    NA,
    c(nd, P, J),
    list(draw = seq_len(nd), celda = NULL, partido = voto_lv)
  )
  pv <- if (con_part) matrix(NA, nd, P) else NULL
  for (d in seq_len(nd)) {
    base <- outer(rep(1, P), A$alpha[d, ]) +
      outer(P_mujer, A$b_mujer[d, ]) +
      matrix(A$mes[d, datos$mes_post, ], P, J, byrow = TRUE) +
      A$edad[d, P_edad, ] +
      A$edu[d, P_edu, ] +
      A$rec[d, P_rec, ] +
      A$ccaa[d, P_ccaa, ] +
      A$prov[d, P_prov, ] +
      A$ccaa_rec[d, i_cr, ]
    # casa media: media de los efectos de las n_casa casas (como 05b)
    ef_media <- 0
    for (h in seq_len(n_casa)) {
      ef_media <- ef_media +
        (matrix(A$casa[d, h, ], P, J, byrow = TRUE) +
          A$casa_rec[d, (h - 1) * n_rec + P_rec, ]) /
          n_casa
    }
    p_celda[d, , ] <- prob(base + ef_media)
    if (con_part) {
      eta_t <- Tp$alpha[d] +
        P_mujer * Tp$b[d] +
        Tp$mes[d, datos$mes_post] +
        Tp$edad[d, P_edad] +
        Tp$edu[d, P_edu] +
        Tp$rec[d, P_rec] +
        Tp$ccaa[d, P_ccaa] +
        Tp$prov[d, P_prov]
      pv[d, ] <- plogis(eta_t + Tp$delta[d])
    }
  }
}

# comprobación: el agregado nacional de la casa media (con participación, el
# electorado; sin ella, la población en edad de votar)
w <- if (is.null(pv)) post$N else colMeans(pv) * post$N
dr <- t(apply(p_celda, 1, function(q) colSums(q * w) / sum(w)))
cat(
  "\n== casa media,",
  fuente,
  ",",
  mes_salida,
  if (is.null(pv)) {
    ", si votara todo el mundo (% voto válido) ==\n"
  } else {
    ", electorado (% voto válido) ==\n"
  }
)
print(setNames(round(100 * colMeans(dr), 1), voto_lv))

if (modo == "nuts") {
  meta <- post |> select(prov, ccaa, sexo, edad, edu, rec, N)
  if (con_part) {
    meta$pv_media <- colMeans(pv)
  }
  salida <- paste0(
    "post_celdas_casa_media",
    if (fuente == "brms") {
      "_brms"
    } else {
      paste0(if (fuente == "rtmb") "_rtmb", if (con_part) "_part")
    },
    ".rds"
  )
  saveRDS(
    list(
      p = p_celda,
      post = meta,
      voto_lv = voto_lv,
      pv = pv,
      mes = mes_salida,
      fuente = fuente,
      ndraws = nd,
      seed = 2026
    ),
    file.path(dir_datos, salida)
  )
  cat(
    "\nguardado",
    salida,
    "(",
    nd,
    "draws x",
    nrow(post),
    "celdas x",
    length(voto_lv),
    "partidos )\n"
  )
}
