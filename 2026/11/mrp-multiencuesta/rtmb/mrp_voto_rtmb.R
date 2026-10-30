# Modelo de voto de mrp_voto.stan en RTMB: aproximación de Laplace.
#
# Mismo modelo y mismos priors que mrp_voto.stan (priors "nuevo"). Cambia la
# inferencia:
#  - Efectos aleatorios (z_*), alpha y b_mujer: se integran con Laplace.
#  - Las 9 x J desviaciones típicas: moda de la posterior marginal aproximada,
#    en escala log (prior half-normal + jacobiano de log).
#  - Incertidumbre: muestras de la normal de Laplace de los aleatorios
#    (precisión = Hessiano en la moda). Las sd quedan fijas en su moda: su
#    incertidumbre no se propaga (diferencia con Stan).
# La verosimilitud omite las constantes de la multinomial, como mrp_voto_vec.stan.
#
#   Rscript 2026/11/mrp-multiencuesta/rtmb/mrp_voto_rtmb.R
#   (constantes abajo: modo prueba | nuts; HILOS en rtmb/comun.R)
#
# Salidas: datos/fit_rtmb_voto_<modo>.rds: muestras de los efectos (A, como en
# stan/ajusta.R), v_casa, sd en la moda y tiempos. La postestratificación con
# participación la hace rtmb/ajusta.R.
# Compara con datos/fit_separado_<modo>_voto.rds (Stan) si existe.

source("2026/11/mrp-multiencuesta/rtmb/comun.R")

# constantes: editar aquí, no hay argumentos (HILOS en rtmb/comun.R)
modo <- "nuts" # prueba | nuts
stopifnot(modo %in% c("prueba", "nuts"))
n_draws <- 1000L
hilos <- configura_hilos()

# 1. Datos
x <- carga_datos(modo)
d <- x$datos
voto_lv <- x$voto_lv
s_b_mujer <- 1 # como en stan/ajusta.R

J <- d$J
Y <- matrix(as.numeric(d$y), d$N, J)
n_y <- rowSums(Y)
ccaa_rec <- (d$ccaa - 1) * d$n_rec + d$rec
casa_rec <- (d$casa - 1) * d$n_rec + d$rec
esc_casa <- sqrt(d$n_casa / (d$n_casa - 1))

# AA: las filas de encuestas que no la ofrecían (d$sin_aa) condicionan su
# verosimilitud a no-AA: sus ceros de AA son de diseño, no evidencia
j_aa <- d$j_aa
stopifnot(!is.na(j_aa), voto_lv[j_aa] == "AA")

# Filas mezcla: hasta K entradas por fila; las que faltan, con peso 0
K <- max(d$len)
mix_prov <- matrix(1L, d$N2, K)
mix_w <- matrix(0, d$N2, K)
for (m in seq_len(d$N2)) {
  e <- d$ini[m] + seq_len(d$len[m]) - 1
  mix_prov[m, seq_along(e)] <- d$ent_prov[e]
  mix_w[m, seq_along(e)] <- d$ent_w[e]
}
Y1 <- Y[d$fila1, , drop = FALSE]
Y2 <- Y[d$fila2, , drop = FALSE]

Q_J <- q_suma0(J)
Q_casa <- q_suma0(d$n_casa)

# Medias por fila sin rowMeans (RTMB no lo admite con números AD)
centra <- function(a) a - matrix(rep(a %*% rep(1 / J, J), J), nrow(a), J)

niveles <- c(
  casa = d$n_casa,
  mes = d$n_mes,
  edad = d$n_edad,
  edu = d$n_edu,
  rec = d$n_rec,
  ccaa = d$n_ccaa,
  prov = d$n_prov,
  ccaa_rec = d$n_ccaa * d$n_rec,
  casa_rec = d$n_casa * d$n_rec
)
s_sd <- c(
  casa = d$s_sd_main,
  mes = d$s_sd_main,
  edad = d$s_sd_main,
  edu = d$s_sd_main,
  rec = d$s_sd_rec,
  ccaa = d$s_sd_main,
  prov = d$s_sd_main,
  ccaa_rec = d$s_sd_int,
  casa_rec = d$s_sd_int
)

# 2. Efectos a partir de los parámetros. Sirve para números AD y para numéricos
efectos <- function(p) {
  sd <- exp(p$log_sd) # 9 x J, filas en el orden de niveles
  a <- list()
  a$casa <- centra(
    esc_casa *
      (Q_casa %*% p$u_casa) *
      matrix(rep(sd[1, ], each = d$n_casa), d$n_casa, J)
  )
  for (k in 2:9) {
    nm <- names(niveles)[k]
    z <- p[[paste0("z_", nm)]]
    a[[nm]] <- centra(
      z * matrix(rep(sd[k, ], each = niveles[k]), niveles[k], J)
    )
  }
  a$alpha <- Q_J %*% p$u_alpha # J x 1 (as.vector pierde la clase AD)
  a$b_mujer <- Q_J %*% p$u_b_mujer
  a
}

nll <- function(p) {
  a <- efectos(p)
  # priors (signo cambiado: nll)
  res <- -sum(dnorm(p$u_alpha, 0, d$s_alpha, log = TRUE)) -
    sum(dnorm(p$u_b_mujer, 0, s_b_mujer, log = TRUE)) -
    sum(dnorm(p$u_casa, 0, 1, log = TRUE))
  for (nm in names(niveles)[-1]) {
    res <- res - sum(dnorm(p[[paste0("z_", nm)]], 0, 1, log = TRUE))
  }
  # sd ~ half-normal(0, s) en escala log: + log(sd) por el jacobiano
  sd <- exp(p$log_sd)
  res <- res -
    sum(dnorm(sd, 0, matrix(rep(s_sd, J), 9, J), log = TRUE)) -
    sum(p$log_sd)

  # predictor de todas las filas, sin provincia
  eta <- matrix(rep(a$alpha, each = d$N), d$N, J) +
    matrix(rep(d$mujer, J), d$N, J) *
      matrix(rep(a$b_mujer, each = d$N), d$N, J) +
    a$casa[d$casa, , drop = FALSE] +
    a$mes[d$mes, , drop = FALSE] +
    a$edad[d$edad, , drop = FALSE] +
    a$edu[d$edu, , drop = FALSE] +
    a$rec[d$rec, , drop = FALSE] +
    a$ccaa[d$ccaa, , drop = FALSE] +
    a$ccaa_rec[ccaa_rec, , drop = FALSE] +
    a$casa_rec[casa_rec, , drop = FALSE] +
    d$off
  # filas con provincia; en las sin AA, el normalizador excluye AA
  eta1 <- eta[d$fila1, , drop = FALSE] + a$prov[d$prov1, , drop = FALSE]
  masc1 <- matrix(1, length(d$fila1), J)
  i1 <- which(d$sin_aa[d$fila1] == 1)
  if (length(i1)) {
    masc1[i1, j_aa] <- 0
  }
  lse1 <- log((exp(eta1) * masc1) %*% rep(1, J))
  ll <- sum(Y1 * eta1) - sum(n_y[d$fila1] * lse1)
  # filas mezcla: p = sum_l w_l softmax(eta + a_prov_l); en las sin AA,
  # renormalizada sin AA (sus y de AA son 0)
  eta2 <- eta[d$fila2, , drop = FALSE]
  p2 <- 0
  for (l in seq_len(K)) {
    e <- exp(eta2 + a$prov[mix_prov[, l], , drop = FALSE])
    w <- mix_w[, l] / (e %*% rep(1, J))
    p2 <- p2 + e * matrix(rep(w, J), d$N2, J)
  }
  p2 <- p2 / matrix(rep(p2 %*% rep(1, J), J), d$N2, J)
  # en las sin AA la verosimilitud se condiciona a no-AA (como multinomial_cond
  # en Stan): softmax completo y corrección -n*log(1 - p_AA). Renormalizar con
  # la máscara dejaría p_AA = 0 con y_AA = 0: 0*log(0) = NaN (todas las filas
  # de mezcla son sin AA: el CIS no tiene filas sin provincia)
  pen2 <- ifelse(d$sin_aa[d$fila2] == 1, n_y[d$fila2] * log1p(-p2[, j_aa]), 0)
  ll <- ll + sum(Y2 * log(p2)) - sum(pen2)
  res - ll
}

par0 <- list(
  u_alpha = rep(0, J - 1),
  u_b_mujer = rep(0, J - 1),
  u_casa = matrix(0, d$n_casa - 1, J)
)
for (nm in names(niveles)[-1]) {
  par0[[paste0("z_", nm)]] <- matrix(0, niveles[nm], J)
}
par0$log_sd <- matrix(log(0.5 * s_sd), 9, J) # mediana del half-normal ~ 0,67 s
aleatorios <- setdiff(names(par0), "log_sd")

# 3. Ajuste
cat(
  "modo:",
  modo,
  " N:",
  d$N,
  " J:",
  J,
  " filas mezcla:",
  d$N2,
  " efectos integrados:",
  sum(lengths(par0[aleatorios])),
  " sd optimizadas:",
  length(par0$log_sd),
  "\n"
)
aj <- ajusta(nll, par0, aleatorios)
obj <- aj$obj
opt <- aj$opt

# 4. Muestras de los efectos (numéricos), con el formato de A en stan/ajusta.R
t0 <- Sys.time()
pars <- muestras_laplace(obj, opt, n_draws)
ef <- parallel::mclapply(pars, efectos, mc.cores = hilos)
A <- lapply(setNames(nm = c("alpha", "b_mujer", names(niveles))), function(nm) {
  apila(lapply(ef, `[[`, nm))
})
t_sdr <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("muestras de los efectos:", round(t_sdr, 1), "s\n")

# 5. v_casa: voto válido (%) por casa y casa media, como en stan/ajusta.R
prob <- function(eta) {
  p <- exp(eta - apply(eta, 1, max)) * d$P_mascara
  p / rowSums(p)
}
i_cr <- (d$P_ccaa - 1) * d$n_rec + d$P_rec
P <- d$P
t0 <- Sys.time()
v_casa_k <- function(k) {
  out <- matrix(NA, d$n_casa + 1, J)
  a <- ef[[k]]
  base <- outer(rep(1, P), c(a$alpha)) +
    outer(d$P_mujer, c(a$b_mujer)) +
    matrix(a$mes[d$mes_post, ], P, J, byrow = TRUE) +
    a$edad[d$P_edad, ] +
    a$edu[d$P_edu, ] +
    a$rec[d$P_rec, ] +
    a$ccaa[d$P_ccaa, ] +
    a$prov[d$P_prov, ] +
    a$ccaa_rec[i_cr, ]
  ef_media <- 0
  for (h in seq_len(d$n_casa + 1)) {
    if (h <= d$n_casa) {
      ef <- matrix(a$casa[h, ], P, J, byrow = TRUE) +
        a$casa_rec[(h - 1) * d$n_rec + d$P_rec, ]
      ef_media <- ef_media + ef / d$n_casa
    } else {
      ef <- ef_media
    }
    out[h, ] <- 100 * colSums(d$P_N * prob(base + ef)) / sum(d$P_N)
  }
  out
}
lv <- parallel::mclapply(seq_len(n_draws), v_casa_k, mc.cores = hilos)
v_casa <- apila(lv) # draws x casa x partido
t_post <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("postestratificación:", n_draws, "muestras,", round(t_post, 1), "s\n")

nombres_casa <- c(x$lv$casa, "casa media")
resume <- function(v) {
  data.frame(
    partido = voto_lv,
    media = round(colMeans(v), 1),
    q05 = round(apply(v, 2, quantile, 0.05), 1),
    q95 = round(apply(v, 2, quantile, 0.95), 1)
  )
}
h_media <- d$n_casa + 1
res_rtmb <- resume(v_casa[, h_media, ])

sd_moda <- matrix(
  exp(opt$par[names(opt$par) == "log_sd"]),
  9,
  J,
  dimnames = list(names(niveles), voto_lv)
)

f_out <- file.path(dir_datos, paste0("fit_rtmb_voto_", modo, ".rds"))
saveRDS(
  list(
    A = A,
    v_casa = v_casa,
    sd_moda = sd_moda,
    opt = opt,
    hilos = hilos,
    tiempos = c(
      optimizacion = aj$t_opt,
      muestras = t_sdr,
      postestratificacion = t_post
    )
  ),
  f_out
)
cat("guardado", f_out, "\n")

# 6. Comparación con Stan, si está el ajuste del mismo modo (y con las mismas
# dimensiones: si el fit de Stan es de una versión anterior del modelo, se salta)
f_stan <- file.path(dir_datos, paste0("fit_separado_", modo, "_voto.rds"))
cat("\n== casa media, % voto válido (sin participación) ==\n")
stan_ok <- FALSE
if (file.exists(f_stan)) {
  fit <- readRDS(f_stan)$fit
  vs <- posterior::as_draws_matrix(fit$draws("v_casa"))
  stan_ok <- any(grepl(
    paste0("v_casa[", h_media, ",", J, "]"),
    colnames(vs),
    fixed = TRUE
  ))
  if (!stan_ok) {
    cat("(fit de Stan de una versión anterior del modelo: sin comparación)\n")
  }
}
if (stan_ok) {
  vs <- vs[, paste0("v_casa[", h_media, ",", 1:J, "]")]
  res_stan <- resume(vs)
  print(
    data.frame(
      partido = voto_lv,
      stan = res_stan$media,
      rtmb = res_rtmb$media,
      stan_ancho90 = res_stan$q95 - res_stan$q05,
      rtmb_ancho90 = res_rtmb$q95 - res_rtmb$q05
    ),
    row.names = FALSE
  )

  sd_stan <- sapply(names(niveles), function(nm) {
    s <- fit$summary(paste0("sd_", nm), "median")$median
  })
  cat(
    "\n== sd: mediana Stan frente a moda RTMB, media sobre los",
    J,
    "partidos ==\n"
  )
  print(
    data.frame(
      nivel = names(niveles),
      stan = round(colMeans(sd_stan), 3),
      rtmb = round(rowMeans(sd_moda), 3),
      sd_cero_rtmb = rowSums(sd_moda < 0.01)
    ),
    row.names = FALSE
  )
} else {
  print(res_rtmb, row.names = FALSE)
  cat("(sin ajuste de Stan para comparar:", f_stan, ")\n")
}
cat(
  "\ntiempo total RTMB (sin preparar datos):",
  round((aj$t_opt + t_sdr + t_post) / 60, 1),
  "min\n"
)
