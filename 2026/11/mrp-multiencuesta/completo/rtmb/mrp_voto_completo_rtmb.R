# Modelo completo, parte 3: modelo de voto de las 4 casas en RTMB (Laplace),
# para iterar la estructura rápido (decisión de José, 2026-11-02: el ajuste
# final, con la estructura decidida, en Stan).
#
# Parte de rtmb/mrp_voto_rtmb.R (mismo esquema: aleatorios integrados por
# Laplace, sd en su moda, muestras de la normal de Laplace). Cambios:
#  - datos de completo/09_datos_voto.R: voto + simpatía, BLANCO aparte;
#  - efectos nuevos: ideología (5 tramos) y nivel de participación (3), y sus
#    interacciones con la casa (casa x ideología, casa x nivel): cada casa
#    usa a su manera la escala de ideología (el CIS, de 1 a 10) y la de
#    probabilidad de votar (en el CIS, muchos más "10");
#  - sin la interacción CCAA x recuerdo (decisión de José; mediana de
#    |efecto| 0,003 en mrp_voto.stan);
#  - mezcla provincia x nivel en las filas a las que les falta alguna de las
#    dos (pesos fijos de 09_datos_voto.R).
#
#   Rscript 2026/11/mrp-multiencuesta/completo/rtmb/mrp_voto_completo_rtmb.R
#
# Salida: completo/datos/fit_rtmb_voto.rds: muestras de los efectos (A), sd en
# la moda, optimización y tiempos.

source("2026/11/mrp-multiencuesta/rtmb/comun.R") # RTMB, q_suma0, ajusta, muestras_laplace, apila

n_draws <- 1000L
# hilos de TMB: variable de entorno HILOS (por defecto, los 8 de comun.R)
HILOS <- as.integer(Sys.getenv("HILOS", HILOS))
hilos <- configura_hilos()
dir_completo <- file.path(dir_proy, "completo", "datos")

# 1. Datos
x <- readRDS(file.path(dir_completo, "datos_voto.rds"))
d <- x$datos
lv <- x$lv
J <- d$J
Y <- matrix(as.numeric(d$y), d$N, J)
n_y <- rowSums(Y)
n_niv <- d$n_nivel
casa_rec <- (d$casa - 1) * d$n_rec + d$rec
casa_ideol <- (d$casa - 1) * d$n_ideol + d$ideol
esc_casa <- sqrt(d$n_casa / (d$n_casa - 1))
j_aa <- d$j_aa
stopifnot(lv$voto[j_aa] == "AA")

f1 <- d$fila1
f2 <- d$fila2
Y1 <- Y[f1, , drop = FALSE]
Y2 <- Y[f2, , drop = FALSE]
casa2 <- d$casa[f2]

# Mezcla: fuera las combinaciones con peso < 0,01 (renormalizando) y, en la
# columna l, solo las filas que tienen al menos l combinaciones. Sin esto, el
# bucle recorría 3.463 x 27 = 93.501 fila x combinación, el 74 % de relleno
# con peso 0, y la cinta de AD agotó los 48 GB (primer intento, 2026-11-02).
mix_w <- d$mix_w
mix_w[mix_w < 0.01] <- 0
mix_w <- mix_w / rowSums(mix_w)
orden <- t(apply(mix_w, 1, function(w) order(w > 0, decreasing = TRUE)))
reordena <- function(m) t(sapply(seq_len(nrow(m)), function(i) m[i, orden[i, ]]))
mix_w <- reordena(mix_w)
mix_prov <- reordena(d$mix_prov)
mix_niv <- reordena(d$mix_niv)
n_comb <- rowSums(mix_w > 0)
M <- max(n_comb)
filas_l <- lapply(seq_len(M), function(l) which(n_comb >= l))
cat(
  "mezcla: fila x combinación con peso >= 0,01:", sum(n_comb),
  " (sin filtrar:", sum(d$mix_w > 0), ")\n"
)

Q_J <- q_suma0(J)
Q_casa <- q_suma0(d$n_casa)
centra <- function(a) a - matrix(rep(a %*% rep(1 / J, J), J), nrow(a), J)

niveles <- c(
  casa = d$n_casa,
  mes = d$n_mes,
  edad = d$n_edad,
  edu = d$n_edu,
  rec = d$n_rec,
  ccaa = d$n_ccaa,
  prov = d$n_prov,
  ideol = d$n_ideol,
  nivel = n_niv,
  casa_rec = d$n_casa * d$n_rec,
  casa_ideol = d$n_casa * d$n_ideol,
  casa_nivel = d$n_casa * n_niv
)
s_sd <- c(
  casa = d$s_sd_main,
  mes = d$s_sd_main,
  edad = d$s_sd_main,
  edu = d$s_sd_main,
  rec = d$s_sd_rec,
  ccaa = d$s_sd_main,
  prov = d$s_sd_main,
  ideol = d$s_sd_rec, # la ideología puede pesar casi como el recuerdo
  nivel = d$s_sd_main,
  casa_rec = d$s_sd_int,
  casa_ideol = d$s_sd_int,
  casa_nivel = d$s_sd_int
)
G <- length(niveles)

# 2. Efectos a partir de los parámetros (AD y numéricos)
efectos <- function(p) {
  sd <- exp(p$log_sd) # G x J
  a <- list()
  a$casa <- centra(
    esc_casa *
      (Q_casa %*% p$u_casa) *
      matrix(rep(sd[1, ], each = d$n_casa), d$n_casa, J)
  )
  for (k in 2:G) {
    nm <- names(niveles)[k]
    z <- p[[paste0("z_", nm)]]
    a[[nm]] <- centra(
      z * matrix(rep(sd[k, ], each = niveles[k]), niveles[k], J)
    )
  }
  a$alpha <- Q_J %*% p$u_alpha
  a$b_mujer <- Q_J %*% p$u_b_mujer
  a
}

nll <- function(p) {
  a <- efectos(p)
  res <- -sum(dnorm(p$u_alpha, 0, d$s_alpha, log = TRUE)) -
    sum(dnorm(p$u_b_mujer, 0, d$s_b_mujer, log = TRUE)) -
    sum(dnorm(p$u_casa, 0, 1, log = TRUE))
  for (nm in names(niveles)[-1]) {
    res <- res - sum(dnorm(p[[paste0("z_", nm)]], 0, 1, log = TRUE))
  }
  sd <- exp(p$log_sd)
  res <- res -
    sum(dnorm(sd, 0, matrix(rep(s_sd, J), G, J), log = TRUE)) -
    sum(p$log_sd)

  # predictor común (sin provincia ni nivel)
  eta <- matrix(rep(a$alpha, each = d$N), d$N, J) +
    matrix(rep(d$mujer, J), d$N, J) *
      matrix(rep(a$b_mujer, each = d$N), d$N, J) +
    a$casa[d$casa, , drop = FALSE] +
    a$mes[d$mes, , drop = FALSE] +
    a$edad[d$edad, , drop = FALSE] +
    a$edu[d$edu, , drop = FALSE] +
    a$rec[d$rec, , drop = FALSE] +
    a$ccaa[d$ccaa, , drop = FALSE] +
    a$ideol[d$ideol, , drop = FALSE] +
    a$casa_rec[casa_rec, , drop = FALSE] +
    a$casa_ideol[casa_ideol, , drop = FALSE] +
    d$off

  # filas con provincia y nivel conocidos; en las sin AA, el normalizador
  # excluye AA
  eta1 <- eta[f1, , drop = FALSE] +
    a$prov[d$prov1, , drop = FALSE] +
    a$nivel[d$nivel1, , drop = FALSE] +
    a$casa_nivel[(d$casa[f1] - 1) * n_niv + d$nivel1, , drop = FALSE]
  masc1 <- matrix(1, length(f1), J)
  i1 <- which(d$sin_aa[f1] == 1)
  if (length(i1)) {
    masc1[i1, j_aa] <- 0
  }
  lse1 <- log((exp(eta1) * masc1) %*% rep(1, J))
  ll <- sum(Y1 * eta1) - sum(n_y[f1] * lse1)

  # filas con mezcla: p = sum_l w_l softmax(eta + prov_l + nivel_l)
  eta2 <- eta[f2, , drop = FALSE]
  p2 <- 0 * eta2
  for (l in seq_len(M)) {
    r <- filas_l[[l]]
    e <- exp(
      eta2[r, , drop = FALSE] +
        a$prov[mix_prov[r, l], , drop = FALSE] +
        a$nivel[mix_niv[r, l], , drop = FALSE] +
        a$casa_nivel[(casa2[r] - 1) * n_niv + mix_niv[r, l], , drop = FALSE]
    )
    w <- mix_w[r, l] / (e %*% rep(1, J))
    p2[r, ] <- p2[r, , drop = FALSE] + e * matrix(rep(w, J), length(r), J)
  }
  p2 <- p2 / matrix(rep(p2 %*% rep(1, J), J), length(f2), J)
  # sin AA: verosimilitud condicionada a no-AA (como en rtmb/mrp_voto_rtmb.R)
  pen2 <- ifelse(d$sin_aa[f2] == 1, n_y[f2] * log1p(-p2[, j_aa]), 0)
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
par0$log_sd <- matrix(log(0.5 * s_sd), G, J)
aleatorios <- setdiff(names(par0), "log_sd")

# 3. Ajuste
cat(
  "N:", d$N, " J:", J, " filas con mezcla:", length(f2),
  " efectos integrados:", sum(lengths(par0[aleatorios])),
  " sd optimizadas:", length(par0$log_sd), "\n"
)
aj <- ajusta(nll, par0, aleatorios)
obj <- aj$obj
opt <- aj$opt

# 4. Muestras de los efectos
t0 <- Sys.time()
pars <- muestras_laplace(obj, opt, n_draws)
ef <- parallel::mclapply(pars, efectos, mc.cores = hilos)
A <- lapply(setNames(nm = c("alpha", "b_mujer", names(niveles))), function(nm) {
  apila(lapply(ef, `[[`, nm))
})
t_mue <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("muestras de los efectos:", round(t_mue, 1), "s\n")

sd_moda <- matrix(
  exp(opt$par[names(opt$par) == "log_sd"]),
  G,
  J,
  dimnames = list(names(niveles), lv$voto)
)

saveRDS(
  list(
    A = A,
    sd_moda = sd_moda,
    opt = opt,
    lv = lv,
    tiempos = c(optimizacion = aj$t_opt, muestras = t_mue)
  ),
  file.path(dir_completo, "fit_rtmb_voto.rds")
)
cat("guardado", file.path(dir_completo, "fit_rtmb_voto.rds"), "\n")

# 5. Resumen
cat("\n== sd en la moda: media y máximo sobre los", J, "partidos ==\n")
print(data.frame(
  grupo = names(niveles),
  media = round(rowMeans(sd_moda), 3),
  max = round(apply(sd_moda, 1, max), 3),
  partido_max = lv$voto[apply(sd_moda, 1, which.max)],
  sd_cero = rowSums(sd_moda < 0.01)
), row.names = FALSE)

grandes <- c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "OTROS", "BLANCO")
media_ef <- function(nm, niv) {
  m <- apply(A[[nm]], c(2, 3), mean)
  dimnames(m) <- list(niv, lv$voto)
  round(m[, grandes], 2)
}
cat("\n== efecto de la ideología (log-odds, media) ==\n")
print(media_ef("ideol", lv$ideol))
cat("\n== efecto del nivel de participación (log-odds, media) ==\n")
print(media_ef("nivel", lv$nivel))
cat("\ntiempo total:", round((aj$t_opt + t_mue) / 60, 1), "min\n")
