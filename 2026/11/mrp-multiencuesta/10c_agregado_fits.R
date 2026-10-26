# Agregado nacional y provincial del share, calculado DIRECTAMENTE de los
# fits (sin pasar por post_celdas), con el ensamblado validado en
# 10_descomp_bayes.R. Patrón oro para comparar Stan vs brms.
#
#   Rscript 2026/11/mrp-multiencuesta/10c_agregado_fits.R

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
})

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
source(file.path(dir_proy, "00_partidos.R"))

fit_stan <- readRDS(file.path(dir_datos, "fit_separado_nuts_voto.rds"))
mod_brms <- readRDS(file.path(dir_datos, "mod_brms_nuts.rds"))
celdas <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
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
voto_lv <- setdiff(partidos_lv, "AC")
territorio_mod <- territorio[names(territorio) != "AC"]
stopifnot(identical(voto_lv, partidos_stan))

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

# Celdas de población (igual que 07)
post <- as.data.frame.table(tabla, responseName = "N") |>
  subset(N > 1e-6)
names(post)[1:5] <- c("prov", "sexo", "edad", "edu", "rec")
post$ccaa <- as.character(prov_ccaa$ccaa)[
  match(as.character(post$prov), as.character(prov_ccaa$prov))
]
P <- nrow(post)
cat("celdas de población:", P, "\n")

i_edad <- match(as.character(post$edad), lv$edad)
i_edu <- match(as.character(post$edu), lv$edu)
i_rec <- match(as.character(post$rec), lv$rec)
i_ccaa <- match(post$ccaa, lv$ccaa)
i_prov <- match(as.character(post$prov), lv$prov)
mujer <- as.numeric(as.character(post$sexo) == "M")
w <- post$N
i_cr <- (i_ccaa - 1) * n_rec + i_rec
stopifnot(!anyNA(i_edad), !anyNA(i_edu), !anyNA(i_rec), !anyNA(i_prov))

# máscara territorial: celda x partido (columnas en orden partidos_stan)
mascara <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) {
    as.numeric(post$ccaa %in% territorio_mod[[p]])
  } else {
    rep(1, P)
  }
})

ndraws <- 1000
adelgaza <- function(dm) {
  dm[round(seq(1, nrow(dm), length.out = ndraws)), , drop = FALSE]
}

# ---------------------------------------------------------------------------
# Stan
# ---------------------------------------------------------------------------
dv <- adelgaza(fit_stan$fit$draws(format = "draws_matrix"))
# a matriz plana: el subset de draws_matrix con índices de fila repetidos
# (expand_rows) deduplica/reordena y escramblea el ensamblado
dv <- matrix(dv, nrow(dv), dimnames = list(NULL, colnames(dv)))
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

# ---------------------------------------------------------------------------
# brms: matrices de niveles por grupo y dpar (nd x n_niveles)
# ---------------------------------------------------------------------------
vb <- adelgaza(as_draws_matrix(
  mod_brms,
  variable = c("^b_", "^r_"),
  regex = TRUE
))
vb <- matrix(vb, nrow(vb), dimnames = list(NULL, colnames(vb)))
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

col_r_vec <- function(grupo, dp, niveles) {
  cols <- paste0(
    "r_",
    grupo,
    "__mu",
    dp,
    "[",
    gsub(" ", ".", niveles),
    ",Intercept]"
  )
  stopifnot(all(cols %in% cn))
  m <- vb[, cols]
  matrix(as.numeric(m), nrow(m), length(cols))
}
L <- list(
  edad = lapply(setNames(dpars, dpars), function(dp) {
    col_r_vec("edad", dp, lv$edad)
  }),
  edu = lapply(setNames(dpars, dpars), function(dp) {
    col_r_vec("edu", dp, lv$edu)
  }),
  rec = lapply(setNames(dpars, dpars), function(dp) {
    col_r_vec("rec", dp, lv$rec)
  }),
  ccaa = lapply(setNames(dpars, dpars), function(dp) {
    col_r_vec("ccaa", dp, lv$ccaa)
  }),
  mes = lapply(setNames(dpars, dpars), function(dp) {
    col_r_vec("mes", dp, lv$mes)
  }),
  mm = lapply(setNames(dpars, dpars), function(dp) {
    col_r_vec("mmp1p2p3p4p5p6p7p8p9", dp, as.character(seq_len(n_prov)))
  }),
  casa = lapply(setNames(dpars, dpars), function(dp) {
    col_r_vec("casa", dp, lv$casa)
  }),
  `casa:rec` = lapply(setNames(dpars, dpars), function(dp) {
    col_r_vec(
      "casa:rec",
      dp,
      paste0(rep(lv$casa, each = n_rec), "_", rep(lv$rec, times = n_casa))
    )
  }),
  alpha = lapply(setNames(dpars, dpars), function(dp) {
    as.numeric(vb[, paste0("b_mu", dp, "_Intercept")])
  }),
  mujer = lapply(setNames(dpars, dpars), function(dp) {
    as.numeric(vb[, paste0("b_mu", dp, "_sexoM")])
  })
)

# ccaa:rec: brms solo tiene niveles para las combinaciones presentes en las
# encuestas. Las celdas de población con combinaciones ausentes son niveles
# nuevos: 07 las muestreaba gaussiano con la sd del modelo; aquí igual.
niv_ccr <- unique(sub(
  "^r_ccaa:rec__muPSOE\\[(.*),Intercept\\]$",
  "\\1",
  cn[grepl("^r_ccaa:rec__muPSOE", cn)]
))
lab_ccr <- as.vector(outer(lv$ccaa, lv$rec, paste, sep = "_"))
existe_ccr <- lab_ccr %in% niv_ccr
cat(
  "combinaciones ccaa:rec existentes:",
  sum(existe_ccr),
  "de",
  length(existe_ccr),
  "\n"
)
L$`ccaa:rec` <- lapply(setNames(dpars, dpars), function(dp) {
  col_r_vec("ccaa:rec", dp, niv_ccr)
})
sd_b <- adelgaza(as_draws_matrix(mod_brms, variable = "^sd_", regex = TRUE))
sd_b <- matrix(sd_b, nrow(sd_b), dimnames = list(NULL, colnames(sd_b)))
sd_ccr <- setNames(
  lapply(dpars, function(dp) {
    as.numeric(sd_b[, paste0("sd_ccaa:rec__mu", dp, "_Intercept")])
  }),
  dpars
)

# ---------------------------------------------------------------------------
# Agregado por batches de celdas
# ---------------------------------------------------------------------------
res_nac <- list(stan = matrix(0, nd, J), brms = matrix(0, nd, J))
res_prov <- list(stan = list(), brms = list())
for (pv in lv$prov) {
  res_prov$stan[[pv]] <- matrix(0, nd, J)
  res_prov$brms[[pv]] <- matrix(0, nd, J)
}
prov_char <- as.character(post$prov)

softmax_mask <- function(E, mask_b) {
  # E: nd x B x J ; mask_b: B x J (columnas en orden partidos_stan)
  m3 <- array(rep(mask_b, each = nd), dim(E))
  mx <- array(rep(apply(E, c(1, 2), max), times = J), dim(E))
  p <- exp(E - mx) * m3
  den <- array(rep(apply(p, c(1, 2), sum), times = J), dim(E))
  p / den
}

B <- 500
nb <- ceiling(P / B)
for (b in seq_len(nb)) {
  batch <- ((b - 1) * B + 1):min(b * B, P)
  B_b <- length(batch)
  w_b <- w[batch]
  mask_b <- mascara[batch, , drop = FALSE]
  ie <- i_edad[batch]
  iu <- i_edu[batch]
  ir <- i_rec[batch]
  ic <- i_ccaa[batch]
  ip <- i_prov[batch]
  icr <- i_cr[batch]
  mu_b <- mujer[batch]
  lab_b <- paste0(lv$ccaa[ic], "_", lv$rec[ir])

  # ---- Stan ----
  # ojo: los draws deben quedar como dim más RÁPIDA del array (column-major):
  # rep con times, no with each
  expand_rows <- function(m) m[rep(seq_len(nd), times = B_b), , drop = FALSE]
  casa_arr <- array(0, c(nd, B_b, J))
  for (h in seq_len(n_casa)) {
    casa_arr <- casa_arr +
      array(A$casa[, h, ][rep(seq_len(nd), times = B_b), ], c(nd, B_b, J)) +
      A$casa_rec[, (h - 1) * n_rec + ir, ]
  }
  E_stan <-
    array(expand_rows(A$alpha), c(nd, B_b, J)) +
    array(expand_rows(A$b), c(nd, B_b, J)) *
      array(rep(rep(mu_b, each = nd), J), c(nd, B_b, J)) +
    array(expand_rows(A$mes[, mes_post, ]), c(nd, B_b, J)) +
    A$edad[, ie, ] +
    A$edu[, iu, ] +
    A$rec[, ir, ] +
    A$ccaa[, ic, ] +
    A$prov[, ip, ] +
    A$ccaa_rec[, icr, ] +
    casa_arr / n_casa
  p_stan <- softmax_mask(E_stan, mask_b)

  # ---- brms ----
  E_brms <- array(0, c(nd, B_b, J))
  for (d in seq_along(dpars)) {
    dp <- dpars[d]
    t_dp <-
      matrix(L$alpha[[dp]], nd, B_b) +
      matrix(L$mujer[[dp]], nd, B_b) * rep(mu_b, each = nd) +
      L$mes[[dp]][, rep(mes_post, B_b), drop = FALSE] +
      L$edad[[dp]][, ie, drop = FALSE] +
      L$edu[[dp]][, iu, drop = FALSE] +
      L$rec[[dp]][, ir, drop = FALSE] +
      L$ccaa[[dp]][, ic, drop = FALSE] +
      L$mm[[dp]][, ip, drop = FALSE]
    # ccaa:rec: nivel existente o gaussiano nuevo (como 07)
    pos <- match(lab_b, niv_ccr)
    ex <- !is.na(pos)
    t_ccr <- matrix(0, nd, B_b)
    if (any(ex)) {
      t_ccr[, ex] <- L$`ccaa:rec`[[dp]][, pos[ex], drop = FALSE]
    }
    if (any(!ex)) {
      t_ccr[, !ex] <- matrix(rnorm(nd * sum(!ex)), nd, sum(!ex)) * sd_ccr[[dp]]
    }
    t_dp <- t_dp + t_ccr
    # casa media: media sobre las casas de casa + casa:rec
    cm <- matrix(0, nd, B_b)
    for (h in seq_len(n_casa)) {
      cm <- cm + L$casa[[dp]][, h] + L$`casa:rec`[[dp]][, (h - 1) * n_rec + ir]
    }
    t_dp <- t_dp + cm / n_casa
    E_brms[,, idx_stan[d]] <- t_dp
  }
  p_brms <- softmax_mask(E_brms, mask_b)

  # ---- agregación (filas de la matriz plana: d + nd*(i-1)) ----
  agrega <- function(p3, rn, rp) {
    s <- matrix(p3, nrow = nd * B_b)
    rn <- rn + rowsum(rep(w_b, each = nd) * s, rep(seq_len(nd), times = B_b))
    for (pv in unique(prov_char[batch])) {
      ib <- which(prov_char[batch] == pv)
      m <- length(ib)
      s2 <- p3[, ib, , drop = FALSE]
      dim(s2) <- c(nd * m, J)
      rp[[pv]] <- rp[[pv]] +
        rowsum(rep(w_b[ib], each = nd) * s2, rep(seq_len(nd), times = m))
    }
    list(nac = rn, prov = rp)
  }
  a <- agrega(p_stan, res_nac$stan, res_prov$stan)
  res_nac$stan <- a$nac
  res_prov$stan <- a$prov
  a <- agrega(p_brms, res_nac$brms, res_prov$brms)
  res_nac$brms <- a$nac
  res_prov$brms <- a$prov
  cat("batch", b, "/", nb, "\n")
}

W <- sum(w)
saveRDS(
  list(nac = res_nac, prov = res_prov, w = w, post = post),
  file.path(dir_datos, "agregado_fits.rds")
)
share_nac <- list(
  stan = sweep(res_nac$stan, 2, W, "/") * 100,
  brms = sweep(res_nac$brms, 2, W, "/") * 100
)
colnames(share_nac$stan) <- colnames(share_nac$brms) <- partidos_stan

cat("\n== voto nacional (casa media, sin participación): media (sd)\n")
for (pp in partidos_stan) {
  cat(sprintf(
    "%-8s stan: %6.2f (%5.2f)   brms: %6.2f (%5.2f)\n",
    pp,
    mean(share_nac$stan[, pp]),
    sd(share_nac$stan[, pp]),
    mean(share_nac$brms[, pp]),
    sd(share_nac$brms[, pp])
  ))
}

w_prov <- sapply(lv$prov, function(pv) sum(w[prov_char == pv]))
sd_prov <- function(res, pp) {
  k <- which(partidos_stan == pp)
  sapply(lv$prov, function(pv) sd(res[[pv]][, k] / w_prov[pv]) * 100)
}
cat("\n== sd del share provincial (pp), media en las 52 y Madrid\n")
for (pp in c("PSOE", "VOX", "PP")) {
  ss <- sd_prov(res_prov$stan, pp)
  sb <- sd_prov(res_prov$brms, pp)
  cat(sprintf(
    "%-6s media: stan %5.2f  brms %5.2f   | Madrid: %5.2f vs %5.2f\n",
    pp,
    mean(ss),
    mean(sb),
    ss["28"],
    sb["28"]
  ))
}

cat("\n== ruido común: correlación media de desviaciones entre provincias\n")
corr_media <- function(res, pp) {
  k <- which(partidos_stan == pp)
  S <- sapply(lv$prov, function(pv) res[[pv]][, k] / w_prov[pv])
  desv <- sweep(S, 2, colMeans(S))
  mean(cor(desv)[lower.tri(cor(desv))])
}
for (pp in c("PSOE", "VOX", "PP")) {
  cat(sprintf(
    "%-6s stan: %5.3f   brms: %5.3f\n",
    pp,
    corr_media(res_prov$stan, pp),
    corr_media(res_prov$brms, pp)
  ))
}
