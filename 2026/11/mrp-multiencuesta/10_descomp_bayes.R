# Descomposición de la varianza draw a draw del predictor, Stan vs brms.
#
# Motivación: los IC90 de escaños de brms son 2-3x más anchos que los de Stan,
# con la misma media. Descomponemos la varianza del share en una celda de
# referencia por término del modelo: con el gradiente del softmax, la
# contribución de cada término es g' C_t g, con C_t su covarianza entre
# partidos. También medimos el ruido común de provincia a provincia.
#
#   Rscript 2026/11/mrp-multiencuesta/10_descomp_bayes.R
#
# No escribe ficheros.

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
})

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

fit_stan <- readRDS(file.path(dir_datos, "fit_separado_nuts_voto.rds"))
mod_brms <- readRDS(file.path(dir_datos, "mod_brms_nuts.rds"))
celdas <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))

J <- 12
# orden interno de los dos fits: el de voto_lv (00_partidos.R), que es el
# orden de colnames(celdas$y) y el de cats en brms
partidos_stan <- c(
  "PP",
  "PSOE",
  "VOX",
  "SUMAR",
  "PODEMOS",
  "ERC",
  "JUNTS",
  "BILDU",
  "PNV",
  "BNG",
  "CCA",
  "OTROS"
)
lv <- lapply(
  celdas[c("casa", "mes", "ccaa", "prov", "sexo", "edad", "edu", "rec")],
  levels
)
n_mes <- length(lv$mes)
n_casa <- length(lv$casa)
n_rec <- length(lv$rec)
n_ccaa <- length(lv$ccaa)
n_prov <- length(lv$prov)
n_edad <- length(lv$edad)
n_edu <- length(lv$edu)
mes_post <- n_mes

# Celda de referencia: Madrid, mujer, modal en edad/edu/rec, casa media,
# último mes
cel <- list(
  prov = "28",
  ccaa = "Madrid",
  sexo = 1,
  edad = "21-24",
  edu = "baja",
  rec = "ABST"
)
stopifnot(as.character(prov_ccaa$ccaa[prov_ccaa$prov == cel$prov]) == cel$ccaa)
cel$i_cr <- (which(lv$ccaa == cel$ccaa) - 1) * n_rec + which(lv$rec == cel$rec)

# ---------------------------------------------------------------------------
# Stan
# ---------------------------------------------------------------------------
dv <- fit_stan$fit$draws(format = "draws_matrix")
nd <- nrow(dv)
get_arr <- function(v, n) {
  m <- dv[, paste0(v, "[", rep(1:n, J), ",", rep(1:J, each = n), "]")]
  array(m, c(nd, n, J))
}
A <- list(
  alpha = dv[, paste0("alpha[", 1:J, "]")],
  b = dv[, paste0("b_mujer[", 1:J, "]")],
  mes = get_arr("a_mes", n_mes),
  edad = get_arr("a_edad", n_edad),
  edu = get_arr("a_edu", n_edu),
  rec = get_arr("a_rec", n_rec),
  ccaa = get_arr("a_ccaa", n_ccaa),
  prov = get_arr("a_prov", n_prov),
  ccaa_rec = get_arr("a_ccaa_rec", n_ccaa * n_rec),
  casa = get_arr("a_casa", n_casa),
  casa_rec = get_arr("a_casa_rec", n_casa * n_rec)
)
nombres_term <- c(
  "alpha",
  "mujer",
  "mes",
  "edad",
  "edu",
  "rec",
  "ccaa",
  "prov",
  "ccaa:rec",
  "casa (media)"
)
i_ccaa <- which(lv$ccaa == cel$ccaa)
i_prov <- which(lv$prov == cel$prov)
i_edad <- which(lv$edad == cel$edad)
i_edu <- which(lv$edu == cel$edu)
i_rec <- which(lv$rec == cel$rec)
T_stan <- list(
  alpha = A$alpha,
  mujer = A$b * cel$sexo,
  mes = A$mes[, mes_post, ],
  edad = A$edad[, i_edad, ],
  edu = A$edu[, i_edu, ],
  rec = A$rec[, i_rec, ],
  ccaa = A$ccaa[, i_ccaa, ],
  prov = A$prov[, i_prov, ],
  `ccaa:rec` = A$ccaa_rec[, cel$i_cr, ],
  `casa (media)` = Reduce(
    `+`,
    lapply(seq_len(n_casa), function(h) {
      A$casa[, h, ] + A$casa_rec[, (h - 1) * n_rec + i_rec, ]
    })
  ) /
    n_casa
)
E_stan <- Reduce(`+`, T_stan)

# ---------------------------------------------------------------------------
# brms
# ---------------------------------------------------------------------------
vb <- as_draws_matrix(mod_brms, variable = c("^b_", "^r_"), regex = TRUE)
cn <- colnames(vb)
dpars <- c(
  "PSOE",
  "VOX",
  "SUMAR",
  "PODEMOS",
  "ERC",
  "JUNTS",
  "BILDU",
  "PNV",
  "BNG",
  "CCA",
  "OTROS"
) # PP = refcat: eta identicamente 0
idx_stan <- match(dpars, partidos_stan)
# en filas de provincia única, el id del nivel mm es el número de provincia
niv_mm_celda <- as.integer(cel$prov)

col_r <- function(grupo, dp, nivel) {
  # brms hace los niveles sintácticos: los espacios se vuelven puntos
  nivel <- gsub(" ", ".", as.character(nivel))
  v <- paste0("r_", grupo, "__mu", dp, "[", nivel, ",Intercept]")
  col <- match(v, cn)
  stopifnot(!is.na(col), !anyNA(nivel))
  as.numeric(vb[, col])
}
T_brms <- setNames(
  lapply(dpars, function(dp) {
    list(
      alpha = as.numeric(vb[, paste0("b_mu", dp, "_Intercept")]),
      mujer = as.numeric(vb[, paste0("b_mu", dp, "_sexoM")]) * cel$sexo,
      mes = col_r("mes", dp, lv$mes[mes_post]),
      edad = col_r("edad", dp, cel$edad),
      edu = col_r("edu", dp, cel$edu),
      rec = col_r("rec", dp, cel$rec),
      ccaa = col_r("ccaa", dp, cel$ccaa),
      prov = col_r("mmp1p2p3p4p5p6p7p8p9", dp, niv_mm_celda),
      `ccaa:rec` = col_r("ccaa:rec", dp, paste0(cel$ccaa, "_", cel$rec)),
      `casa (media)` = Reduce(
        `+`,
        lapply(seq_len(n_casa), function(h) {
          col_r("casa", dp, lv$casa[h]) +
            col_r("casa:rec", dp, paste0(lv$casa[h], "_", cel$rec))
        })
      ) /
        n_casa
    )
  }),
  dpars
)
mat_termino <- function(nm) {
  m <- matrix(0, nd, J)
  m[, idx_stan] <- sapply(T_brms, function(t) t[[nm]])
  m
}
Tb <- setNames(lapply(nombres_term, mat_termino), nombres_term)
E_brms <- Reduce(`+`, Tb)

# Comprobación: eta montado a mano == posterior_linpred en filas reales
fila_val <- c(
  which(celdas$prov == "28" & celdas$w1 == 1 & celdas$w2 == 0)[1],
  which(celdas$prov == "8" & celdas$w1 == 1 & celdas$w2 == 0)[1],
  which(celdas$w2 > 0)[1]
)
eta_lp <- posterior_linpred(
  mod_brms,
  newdata = celdas[fila_val, ],
  dpar = "muPSOE",
  ndraws = nd
)
eta_mano <- sapply(fila_val, function(f) {
  w <- as.numeric(celdas[f, paste0("w", 1:9)])
  p <- as.numeric(celdas[f, paste0("p", 1:9)])
  act <- w > 0
  mm_i <- Reduce(
    `+`,
    Map(
      function(pi, wi) {
        wi * col_r("mmp1p2p3p4p5p6p7p8p9", "PSOE", pi)
      },
      p[act],
      w[act]
    )
  ) /
    sum(w[act])
  col_r("mes", "PSOE", as.character(celdas$mes[f])) +
    col_r("edad", "PSOE", as.character(celdas$edad[f])) +
    col_r("edu", "PSOE", as.character(celdas$edu[f])) +
    col_r("rec", "PSOE", as.character(celdas$rec[f])) +
    col_r("ccaa", "PSOE", as.character(celdas$ccaa[f])) +
    mm_i +
    col_r(
      "ccaa:rec",
      "PSOE",
      paste0(as.character(celdas$ccaa[f]), "_", as.character(celdas$rec[f]))
    ) +
    col_r("casa", "PSOE", as.character(celdas$casa[f])) +
    col_r(
      "casa:rec",
      "PSOE",
      paste0(as.character(celdas$casa[f]), "_", as.character(celdas$rec[f]))
    ) +
    as.numeric(vb[, "b_muPSOE_Intercept"]) +
    as.numeric(vb[, "b_muPSOE_sexoM"]) * (as.character(celdas$sexo[f]) == "M")
})
cat("== comprobación eta brms (máx dif. absoluta por fila):\n")
print(apply(abs(eta_lp - eta_mano), 2, max))

# ---------------------------------------------------------------------------
# Descomposición de la varianza del share por término
# ---------------------------------------------------------------------------
share <- function(E) {
  P <- exp(E - apply(E, 1, max))
  P / rowSums(P)
}
descomp <- function(E, Tterms, partido) {
  i_p <- which(partidos_stan == partido)
  P <- share(E)
  pm <- colMeans(P)
  p <- pm[i_p]
  g <- -p * pm
  g[i_p] <- p * (1 - p)
  V_lineal <- as.numeric(g %*% cov(E) %*% g)
  V_exacta <- var(P[, i_p])
  contrib <- sapply(Tterms, function(t) as.numeric(g %*% cov(t) %*% g))
  data.frame(
    termino = c(names(Tterms), "TOTAL linealizado", "TOTAL exacto"),
    sd_pp = c(sqrt(contrib), sqrt(V_lineal), sqrt(V_exacta)) * 100
  )
}

cat("\n===== sd del share por término (puntos), celda Madrid modal =====\n")
for (pp in c("PSOE", "VOX")) {
  ds <- descomp(E_stan, T_stan, pp)
  db <- descomp(E_brms, Tb, pp)
  out <- merge(ds, db, by = "termino", suffixes = c("_stan", "_brms"))
  out$dif <- out$sd_pp_brms - out$sd_pp_stan
  cat("\n--", pp, ":\n")
  print(out, digits = 2, row.names = FALSE)
}

# ---------------------------------------------------------------------------
# Ruido común entre provincias: correlación media de las desviaciones draw a
# draw del share entre las 52 provincias, con el mismo perfil modal
# ---------------------------------------------------------------------------
ccaa_p <- as.character(prov_ccaa$ccaa)[match(
  lv$prov,
  as.character(prov_ccaa$prov)
)]
i_ccaa_p <- match(ccaa_p, lv$ccaa)
stopifnot(!anyNA(i_ccaa_p))
i_cr_p <- (i_ccaa_p - 1) * n_rec + which(lv$rec == cel$rec)

# Stan: términos dependientes de provincia, ya como arrays nd x n_prov x J
prov_s <- A$prov
ccaa_s <- A$ccaa[, i_ccaa_p, ]
cr_s <- A$ccaa_rec[, i_cr_p, ]

# brms: idem, extrayendo columna a columna (arrays nd x n_niveles x J, con
# la columna de PP en cero)
get_arr_b <- function(grupo, niveles) {
  arr <- array(0, c(nd, length(niveles), J))
  for (d in seq_along(dpars)) {
    arr[,, idx_stan[d]] <- sapply(niveles, function(nv) {
      col_r(grupo, dpars[d], nv)
    })
  }
  arr
}
prov_b <- get_arr_b("mmp1p2p3p4p5p6p7p8p9", as.integer(lv$prov))
ccaa_b <- get_arr_b("ccaa", lv$ccaa[i_ccaa_p])
cr_b <- get_arr_b("ccaa:rec", paste0(ccaa_p, "_", cel$rec))

base_s <- E_stan - T_stan$prov - T_stan$ccaa - T_stan$`ccaa:rec`
base_b <- E_brms - Tb$prov - Tb$ccaa - Tb$`ccaa:rec`

ruido <- function(base, prov, ccaa, cr, partido) {
  i_p <- which(partidos_stan == partido)
  S <- matrix(NA, nd, n_prov)
  for (ip in seq_len(n_prov)) {
    E <- base + prov[, ip, ] + ccaa[, ip, ] + cr[, ip, ]
    P <- share(E)
    S[, ip] <- P[, i_p]
  }
  desv <- sweep(S, 2, colMeans(S))
  c(
    sd_media_prov = mean(apply(S, 2, sd)) * 100,
    corr_media = mean(as.matrix(cor(desv))[lower.tri(cor(desv))])
  )
}
cat("\n===== ruido común entre provincias (perfil modal, casa media) =====\n")
for (pp in c("PSOE", "VOX")) {
  rs <- ruido(base_s, prov_s, ccaa_s, cr_s, pp)
  rb <- ruido(base_b, prov_b, ccaa_b, cr_b, pp)
  cat("\n--", pp, ":\n")
  print(rbind(stan = rs, brms = rb), digits = 3)
}

# ---------------------------------------------------------------------------
# sd_ de cada grupo, Stan vs brms (media posterior, por partido)
# ---------------------------------------------------------------------------
sd_b <- as_draws_matrix(mod_brms, variable = "^sd_", regex = TRUE)
cn_sd <- colnames(sd_b)
tabla_sd <- function(g_stan, g_brms, etiqueta) {
  out <- data.frame(
    partido = partidos_stan,
    stan = colMeans(dv[, paste0("sd_", g_stan, "[", 1:J, "]")]),
    brms = NA
  )
  for (dp in dpars) {
    col <- grep(
      paste0("^sd_", g_brms, "__mu", dp, "_Intercept$"),
      cn_sd,
      value = TRUE
    )
    if (length(col) == 1) out$brms[out$partido == dp] <- mean(sd_b[, col])
  }
  names(out)[2:3] <- paste0(etiqueta, c("_stan", "_brms"))
  out
}
cat("\n===== sd_ por grupo y partido (media posterior) =====\n")
for (par in list(
  c("casa", "casa"),
  c("casa_rec", "casa:rec"),
  c("rec", "rec"),
  c("prov", "mmp1p2p3p4p5p6p7p8p9"),
  c("ccaa", "ccaa"),
  c("ccaa_rec", "ccaa:rec"),
  c("edad", "edad"),
  c("edu", "edu"),
  c("mes", "mes")
)) {
  print(tabla_sd(par[1], par[2], par[1]), row.names = FALSE, digits = 2)
  cat("\n")
}
