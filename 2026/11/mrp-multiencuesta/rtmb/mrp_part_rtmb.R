# Modelo de participación de mrp_part.stan en RTMB: aproximación de Laplace.
#
# Binomial de decir "10, seguro que voto", con efecto casa (solo las casas que
# lo preguntan), mes, sexo, edad, estudios, recuerdo, CCAA y provincia, y la
# misma mezcla de provincias que el voto. Mismos priors que mrp_part.stan.
#  - Efectos aleatorios, alpha_t y b_mujer_t: integrados con Laplace.
#  - Las 7 sd: moda de la posterior marginal aproximada, en escala log.
#  - El efecto casa reproduce la densidad de Stan, a_casa_t ~ normal(0, esc * sd)
#    sobre los n_casa_t componentes, aunque el vector solo tenga n_casa_t - 1
#    libres (ver PLAN.md, 2026-10-26).
#  - Como Stan (generated quantities): part_bruta y delta (calibra) por muestra.
#
#   Rscript 2026/11/mrp-multiencuesta/rtmb/mrp_part_rtmb.R
#   (constantes abajo: modo prueba | nuts; HILOS en rtmb/comun.R)
#
# Salida: datos/fit_rtmb_part_<modo>.rds: muestras de los efectos (Tp, como en
# stan/ajusta.R), sd en la moda y tiempos. Compara part_bruta y delta con
# datos/fit_separado_<modo>_part.rds (Stan) si existe.

source("2026/11/mrp-multiencuesta/rtmb/comun.R")

# constantes: editar aquí, no hay argumentos (HILOS en rtmb/comun.R)
modo <- "nuts" # prueba | nuts
stopifnot(modo %in% c("prueba", "nuts"))
n_draws <- 1000L
hilos <- configura_hilos()

# 1. Datos
x <- carga_datos(modo)
d <- x$datos
m_alpha_t <- 1 # como en stan/ajusta.R
s_alpha_t <- 1.5
s_b_mujer_t <- 1
esc_casa_t <- sqrt(d$n_casa_t / (d$n_casa_t - 1))
Q_casa_t <- q_suma0(d$n_casa_t)

# Filas mezcla: hasta K entradas por fila; las que faltan, con peso 0
K <- max(d$t_len)
mix_prov <- matrix(1L, d$NT2, K)
mix_w <- matrix(0, d$NT2, K)
for (m in seq_len(d$NT2)) {
  e <- d$t_ini[m] + seq_len(d$t_len[m]) - 1
  mix_prov[m, seq_along(e)] <- d$t_ent_prov[e]
  mix_w[m, seq_along(e)] <- d$t_ent_w[e]
}
k1 <- d$t_k[d$t_fila1]
n1 <- d$t_n[d$t_fila1]
k2 <- d$t_k[d$t_fila2]
n2 <- d$t_n[d$t_fila2]

niveles <- c(
  casa = d$n_casa_t,
  mes = d$n_mes,
  edad = d$n_edad,
  edu = d$n_edu,
  rec = d$n_rec,
  ccaa = d$n_ccaa,
  prov = d$n_prov
)
s_sd <- c(
  casa = d$s_sd_main,
  mes = d$s_sd_main,
  edad = d$s_sd_main,
  edu = d$s_sd_main,
  rec = d$s_sd_rec,
  ccaa = d$s_sd_main,
  prov = d$s_sd_main
)

# 2. Efectos a partir de los parámetros (números AD o numéricos)
efectos <- function(p) {
  sd <- exp(p$log_sd)
  list(
    alpha = p$alpha_t,
    b = p$b_mujer_t,
    casa = Q_casa_t %*% p$u_casa, # suma 0; ya en su escala (centrado)
    mes = p$a_mes,
    edad = p$a_edad,
    edu = p$a_edu,
    rec = p$a_rec,
    ccaa = sd[6] * p$z_ccaa,
    prov = sd[7] * p$z_prov
  )
}

nll <- function(p) {
  a <- efectos(p)
  sd <- exp(p$log_sd)
  res <- -dnorm(p$alpha_t, m_alpha_t, s_alpha_t, log = TRUE) -
    dnorm(p$b_mujer_t, 0, s_b_mujer_t, log = TRUE) -
    sum(dnorm(a$casa, 0, esc_casa_t * sd[1], log = TRUE)) -
    sum(dnorm(p$a_mes, 0, sd[2], log = TRUE)) -
    sum(dnorm(p$a_edad, 0, sd[3], log = TRUE)) -
    sum(dnorm(p$a_edu, 0, sd[4], log = TRUE)) -
    sum(dnorm(p$a_rec, 0, sd[5], log = TRUE)) -
    sum(dnorm(p$z_ccaa, 0, 1, log = TRUE)) -
    sum(dnorm(p$z_prov, 0, 1, log = TRUE))
  # sd ~ half-normal(0, s) en escala log: + log(sd) por el jacobiano
  res <- res - sum(dnorm(sd, 0, s_sd, log = TRUE)) - sum(p$log_sd)

  eta <- a$alpha +
    d$t_mujer * a$b +
    a$casa[d$t_casa] +
    a$mes[d$t_mes] +
    a$edad[d$t_edad] +
    a$edu[d$t_edu] +
    a$rec[d$t_rec] +
    a$ccaa[d$t_ccaa]
  # filas con provincia: binomial_logit (sin la constante)
  x1 <- eta[d$t_fila1] + a$prov[d$t_prov1]
  ll <- sum(k1 * x1 - n1 * log(1 + exp(x1)))
  # filas mezcla: logit p = log sum w inv_logit(x) - log sum w (1 - inv_logit(x))
  eta2 <- eta[d$t_fila2]
  s1 <- 0
  s0 <- 0
  for (l in seq_len(K)) {
    xl <- eta2 + a$prov[mix_prov[, l]]
    s1 <- s1 + mix_w[, l] / (1 + exp(-xl))
    s0 <- s0 + mix_w[, l] / (1 + exp(xl))
  }
  ll <- ll + sum(k2 * log(s1) + (n2 - k2) * log(s0) - n2 * log(s1 + s0))
  res - ll
}

par0 <- list(
  alpha_t = 0,
  b_mujer_t = 0,
  u_casa = rep(0, d$n_casa_t - 1),
  a_mes = rep(0, d$n_mes),
  a_edad = rep(0, d$n_edad),
  a_edu = rep(0, d$n_edu),
  a_rec = rep(0, d$n_rec),
  z_ccaa = rep(0, d$n_ccaa),
  z_prov = rep(0, d$n_prov),
  log_sd = log(0.5 * s_sd)
)
aleatorios <- setdiff(names(par0), "log_sd")

# 3. Ajuste
cat(
  "modo:",
  modo,
  " filas:",
  d$NT,
  " filas mezcla:",
  d$NT2,
  " efectos integrados:",
  sum(lengths(par0[aleatorios])),
  " sd optimizadas:",
  length(par0$log_sd),
  "\n"
)
aj <- ajusta(nll, par0, aleatorios)

# 4. Muestras y cantidades generadas (part_bruta y delta, como en mrp_part.stan)
# delta tal que sum(N * inv_logit(eta + delta)) = obj * sum(N), por Newton;
# el paso se limita a +-2, como calibra() en funciones.stan
calibra <- function(eta, N, obj) {
  delta <- 0
  for (it in 1:50) {
    p <- plogis(eta + delta)
    f <- sum(N * p) / sum(N) - obj
    df <- sum(N * p * (1 - p)) / sum(N)
    delta <- delta - min(max(f / df, -2), 2)
    if (abs(f) < 1e-10) break
  }
  delta
}
t0 <- Sys.time()
pars <- muestras_laplace(aj$obj, aj$opt, n_draws)
gq <- function(p) {
  a <- efectos(p)
  eta_pt <- c(a$alpha) +
    d$P_mujer * c(a$b) +
    a$mes[d$mes_post] +
    a$edad[d$P_edad] +
    a$edu[d$P_edu] +
    a$rec[d$P_rec] +
    a$ccaa[d$P_ccaa] +
    a$prov[d$P_prov]
  c(
    a[c("alpha", "b", "mes", "edad", "edu", "rec", "ccaa", "prov")],
    part_bruta = 100 * sum(d$P_N * plogis(eta_pt)) / sum(d$P_N),
    delta = calibra(eta_pt, d$P_N, d$part_obj)
  )
}
g <- parallel::mclapply(pars, gq, mc.cores = hilos)
Tp <- lapply(setNames(nm = names(g[[1]])), function(nm) {
  apila(lapply(g, `[[`, nm))
})
t_gq <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("muestras y cantidades generadas:", round(t_gq, 1), "s\n")

sd_moda <- setNames(
  exp(aj$opt$par[names(aj$opt$par) == "log_sd"]),
  names(niveles)
)
f_out <- file.path(dir_datos, paste0("fit_rtmb_part_", modo, ".rds"))
saveRDS(
  list(
    Tp = Tp,
    sd_moda = sd_moda,
    opt = aj$opt,
    hilos = hilos,
    tiempos = c(optimizacion = aj$t_opt, muestras = t_gq)
  ),
  f_out
)
cat("guardado", f_out, "\n")

# 5. Comparación con Stan
resumen <- function(v) {
  round(
    c(
      media = mean(v),
      q05 = quantile(v, 0.05, names = FALSE),
      q95 = quantile(v, 0.95, names = FALSE)
    ),
    3
  )
}
f_stan <- file.path(dir_datos, paste0("fit_separado_", modo, "_part.rds"))
if (file.exists(f_stan)) {
  fit <- readRDS(f_stan)$fit
  ds <- posterior::as_draws_matrix(fit$draws(c("part_bruta", "delta")))
  cat("\n== Stan frente a RTMB ==\n")
  print(rbind(
    part_bruta_stan = resumen(ds[, "part_bruta"]),
    part_bruta_rtmb = resumen(Tp$part_bruta),
    delta_stan = resumen(ds[, "delta"]),
    delta_rtmb = resumen(Tp$delta)
  ))
  sd_stan <- sapply(names(niveles), function(nm) {
    fit$summary(paste0("sd_", nm, "_t"), "median")$median
  })
  cat("\n== sd: mediana Stan frente a moda RTMB ==\n")
  print(
    data.frame(
      nivel = names(niveles),
      stan = round(sd_stan, 3),
      rtmb = round(sd_moda, 3)
    ),
    row.names = FALSE
  )
} else {
  print(rbind(part_bruta = resumen(Tp$part_bruta), delta = resumen(Tp$delta)))
}
cat(
  "\ntiempo total RTMB (sin preparar datos):",
  round((aj$t_opt + t_gq) / 60, 1),
  "min\n"
)
