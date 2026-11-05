# Modelo de voto seguro / dudoso con glmmTMB: truco multinomial–Poisson
# (GUIA.md, "Idea para explorar"; decisión de José, 2026-11-03: probarlo).
#
# Formato largo: una fila por celda x partido, con los ceros. Fuera las
# casillas imposibles (regionales fuera de su territorio, AA en encuestas sin
# AA). Solo filas con provincia y seguro / dudoso conocidos (sin la mezcla:
# fuera Ateneo y las de More in Common sin provincia).
#
#   y ~ Poisson(exp(phi_celda + eta_partido))
# phi_celda: efecto aleatorio con la sd FIJA en 10 (casi plano): con Laplace,
# la verosimilitud de eta es la del multinomial (GUIA.md). Lo que no
# interactúa con partido lo absorbe phi_celda.
# Cada efecto, con una sd por partido y sin correlaciones: diag(0 + partido | g).
# Mismos efectos que mrp_voto_seg.stan: casa, mes, edad, edu, rec, ccaa,
# prov, casa x rec, dudoso x edad, dudoso x rec; fijos: partido, partido x
# mujer, partido x dudoso.
#
# Diferencias con el bayesiano: máxima verosimilitud con Laplace (sd en su
# estimación, sin incertidumbre); los efectos no suman cero entre niveles (lo
# hace su media 0); sin las filas con mezcla.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/13_glmmtmb_voto_seg.R
#   MODO=prueba Rscript ...   # 20 % de las celdas, para medir tiempo y memoria
#   HILOS=8 (por defecto)
#
# Salida: completo/datos/glmmtmb_voto_seg[_prueba].rds (coeficientes, sd,
# tiempos, comparación con NumPyro).

library(dplyr)
library(tidyr)
library(glmmTMB)
library(reticulate)
use_virtualenv("~/.virtualenvs/r-mrp-numpyro", required = TRUE)

modo <- Sys.getenv("MODO", "final")
stopifnot(modo %in% c("final", "prueba"))
hilos <- as.integer(Sys.getenv("HILOS", "8"))
suf <- if (modo == "prueba") "_prueba" else ""
dir_completo <- here::here("2026/11/mrp-multiencuesta/completo/datos")

# ---- 1. Datos en formato largo ----------------------------------------------------

x <- readRDS(file.path(dir_completo, "datos_voto_seg.rds"))
d <- x$datos
lv <- x$lv
J <- d$J
r <- d$fila1 # filas con provincia y seg conocidos
celdas <- tibble(
  celda = seq_along(r),
  casa = lv$casa[d$casa[r]],
  mes = lv$mes[d$mes[r]],
  edad = lv$edad[d$edad[r]],
  edu = lv$edu[d$edu[r]],
  rec = lv$rec[d$rec[r]],
  ccaa = lv$ccaa[d$ccaa[r]],
  prov = lv$prov[d$prov1],
  mujer = d$mujer[r],
  dudoso = as.numeric(d$seg1 == 1),
  sin_aa = d$sin_aa[r]
)
Y <- d$y[r, , drop = FALSE]
OFF <- d$off[r, , drop = FALSE]
if (modo == "prueba") {
  set.seed(2026)
  k <- sort(sample(nrow(celdas), round(0.2 * nrow(celdas))))
  celdas <- celdas[k, ] |> mutate(celda = seq_along(k))
  Y <- Y[k, , drop = FALSE]
  OFF <- OFF[k, , drop = FALSE]
}
n <- nrow(celdas)
largo <- celdas[rep(seq_len(n), each = J), ] |>
  mutate(
    partido = factor(rep(lv$voto, n), levels = lv$voto),
    y = as.vector(t(Y)),
    off = as.vector(t(OFF))
  ) |>
  filter(off == 0, !(sin_aa == 1 & partido == "AA")) |>
  mutate(
    celda = factor(celda),
    casa_rec = interaction(casa, rec, drop = TRUE),
    across(c(casa, mes, edad, edu, rec, ccaa, prov), factor)
  )
cat("modo:", modo, "  celdas:", n, "  entrevistas:", sum(Y), "  filas en largo:",
    nrow(largo), "\n")
stopifnot(sum(largo$y) == sum(Y))

# ---- 2. Modelo, con la sd de (1 | celda) fija en 10 --------------------------------

f <- y ~ partido * (mujer + dudoso) + (1 | celda) +
  diag(0 + partido | casa) + diag(0 + partido | mes) + diag(0 + partido | edad) +
  diag(0 + partido | edu) + diag(0 + partido | rec) + diag(0 + partido | ccaa) +
  diag(0 + partido | prov) + diag(0 + partido | casa_rec) +
  diag(0 + partido:dudoso | edad) + diag(0 + partido:dudoso | rec)

# estructura sin ajustar, para saber dónde queda la sd de celda en theta
est <- glmmTMB(f, family = poisson, data = largo, doFit = FALSE)
n_theta <- length(est$parameters$theta)
blq <- vapply(est$condReStruc, function(s) s$blockNumTheta, numeric(1))
i_celda <- which(names(est$condReStruc) == "1 | celda")
stopifnot(length(i_celda) == 1, blq[i_celda] == 1)
pos_celda <- sum(blq[seq_len(i_celda - 1)]) + 1
cat("parámetros de varianza:", n_theta, "  sd de celda en la posición", pos_celda, "\n")
theta_ini <- rep(log(0.3), n_theta)
theta_ini[pos_celda] <- log(10)
mapa <- seq_len(n_theta)
mapa[pos_celda] <- NA
mapa <- factor(mapa)

t0 <- Sys.time()
cat("inicio:", format(t0, "%H:%M:%S"), "\n")
m <- glmmTMB(
  f, family = poisson, data = largo,
  start = list(theta = theta_ini), map = list(theta = mapa),
  control = glmmTMBControl(parallel = hilos, optCtrl = list(iter.max = 2000, eval.max = 4000))
)
minutos <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
cat("fin:", format(Sys.time(), "%H:%M:%S"), "  minutos:", round(minutos, 1), "\n")
cat("convergencia (0 = bien):", m$fit$convergence, " ", m$fit$message, "\n")
cat("pdHess:", m$sdr$pdHess, "\n")

# ---- 3. Comparación con NumPyro ----------------------------------------------------

# b_dud: partido:dudoso relativo a PP; se centra entre partidos, como b_dud
fe <- fixef(m)$cond
b <- setNames(numeric(J), lv$voto)
nm <- paste0("partido", lv$voto[-1], ":dudoso")
b[-1] <- fe[nm]
b_tmb <- b - mean(b)

# sd por bloque: VarCorr da las sd de cada término en el orden de la fórmula
vc <- VarCorr(m)$cond
sd_tmb <- lapply(vc, function(s) attr(s, "stddev"))
nombres_blq <- names(est$condReStruc)
sd_dud_edad <- sd_tmb[[which(nombres_blq == "0 + partido:dudoso | edad")]]
sd_dud_rec <- sd_tmb[[which(nombres_blq == "0 + partido:dudoso | rec")]]

np <- import("numpy", convert = FALSE)
z <- np$load(file.path(dir_completo, "voto_seg_numpyro_draws_nuts.npz"))
g <- function(k) py_to_r(z$`__getitem__`(k))
comp <- tibble(
  partido = lv$voto,
  b_dud_numpyro = colMeans(g("b_dud")),
  b_dud_numpyro_sd = apply(g("b_dud"), 2, sd),
  b_dud_glmmtmb = unname(b_tmb),
  sd_dud_rec_numpyro = colMeans(g("sd_dud_rec")),
  sd_dud_rec_glmmtmb = unname(sd_dud_rec),
  sd_dud_edad_numpyro = colMeans(g("sd_dud_edad")),
  sd_dud_edad_glmmtmb = unname(sd_dud_edad)
)
cat("\n== b_dud y sd de dudoso x recuerdo / x edad: NumPyro (media posterior) frente a glmmTMB ==\n")
print(comp |> mutate(across(where(is.numeric), \(v) round(v, 2))) |> as.data.frame(),
      row.names = FALSE)
cat("correlación de b_dud:", round(cor(comp$b_dud_numpyro, comp$b_dud_glmmtmb), 3), "\n")

saveRDS(
  list(fixef = fe, sd = sd_tmb, comp = comp, minutos = minutos, modo = modo,
       convergencia = m$fit$convergence, pdHess = m$sdr$pdHess,
       n_celdas = n, n_largo = nrow(largo)),
  file.path(dir_completo, paste0("glmmtmb_voto_seg", suf, ".rds"))
)
cat("guardado", file.path(dir_completo, paste0("glmmtmb_voto_seg", suf, ".rds")), "\n")
