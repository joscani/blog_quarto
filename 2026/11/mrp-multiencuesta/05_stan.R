# MRP en Stan puro (cmdstanr). Con las 4 casas solo prepara los datos (los usa
# 05b_separado.R); el modelo conjunto mrp.stan está en obsoleto/. Con "cis"
# ajusta mrp_cis.stan. Los usos de abajo con "todas" ya no ajustan.
# Usa las celdas de 03b_celdas.R (datos/celdas_brms.rds y celdas_part.rds), las mismas que 04_brms.R, para poder
# comparar los dos ajustes.
#
# Uso:
#   Rscript 2026/11/mrp-multiencuesta/05_stan.R prueba   # 10 % de las filas, 1 cadena, 300 iteraciones
#   Rscript 2026/11/mrp-multiencuesta/05_stan.R nuts     # 4 cadenas x 2000 iteraciones
#   Rscript 2026/11/mrp-multiencuesta/05_stan.R prior    # prior predictive check: sin verosimilitud, 4 cadenas x 250 muestras
#
# Segundo argumento, los priors: "nuevo" (por defecto, los del prior predictive
# check) o "brms" (exponential(1) en todas las sd, como 04_brms.R). Ejemplo:
#   Rscript 2026/11/mrp-multiencuesta/05_stan.R prior brms
#
# Tercer argumento, la muestra: "todas" (por defecto, modelo mrp.stan) o "cis"
# (solo barómetros del CIS, modelo mrp_cis.stan: sin efecto casa ni filas mezcla).
#   Rscript 2026/11/mrp-multiencuesta/05_stan.R nuts nuevo cis
#
# Modo "gq": no ajusta. Carga el ajuste nuts y postestratifica sus muestras en R
# con otra tabla de población (variable de entorno TABLA_POST). Para la prueba de
# sensibilidad a la semilla del raking:
#   TABLA_POST=2026/11/mrp-multiencuesta/datos/sensibilidad/tabla_CIS3567_a0.5.rds \
#     Rscript 2026/11/mrp-multiencuesta/05_stan.R gq nuevo cis

library(dplyr)
library(tidyr)
library(cmdstanr)

args <- commandArgs(trailingOnly = TRUE)
modo <- if (length(args) >= 1) args[1] else "prueba"
stopifnot(modo %in% c("prueba", "nuts", "prior", "gq"))
priors <- if (length(args) >= 2) args[2] else "nuevo"
stopifnot(priors %in% c("nuevo", "brms"))
muestra <- if (length(args) >= 3) args[3] else "todas"
stopifnot(muestra %in% c("todas", "cis"))
solo_cis <- muestra == "cis"
cat("modo:", modo, " priors:", priors, " muestra:", muestra, "\n")
# nombre de los ficheros de salida
# (gq usa el ajuste nuts)
version <- paste0(if (modo == "gq") "nuts" else modo, "_", priors, if (solo_cis) "_cis" else "")

dir_proy  <- "2026/11/mrp-multiencuesta"
dir_datos <- file.path(dir_proy, "datos")
source(file.path(dir_proy, "00_partidos.R"))   # partidos_lv, rec_lv, territorio

celdas    <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
f_tabla   <- Sys.getenv("TABLA_POST", file.path(dir_datos, "tabla_post_prov.rds"))
tabla     <- readRDS(f_tabla)
cat("tabla de población:", f_tabla, "\n")
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))

voto_lv <- setdiff(partidos_lv, "AC")
territorio_mod <- territorio[names(territorio) != "AC"]
stopifnot(identical(colnames(celdas$y), voto_lv))
K <- 9

if (solo_cis) {
  celdas <- celdas %>% filter(casa == "CIS") %>% droplevels()
  stopifnot(all(celdas$w1 == 1))   # el CIS siempre trae la provincia
}

if (modo == "prueba") {
  # las mismas filas que la prueba de 04_brms.R
  set.seed(2026)
  celdas <- celdas[sample(nrow(celdas), round(0.1 * nrow(celdas))), ]
}

# Niveles fijos: Stan solo ve enteros 1..n
lv <- list(casa = levels(celdas$casa), mes = levels(celdas$mes),
           edad = levels(celdas$edad), edu = levels(celdas$edu), rec = rec_lv,
           ccaa = levels(prov_ccaa$ccaa), prov = levels(prov_ccaa$prov))
idx <- function(x, niveles) {
  i <- match(as.character(x), niveles)
  stopifnot(!anyNA(i))
  i
}

# ---------------------------------------------------------------------------
# 1. Provincia: filas conocidas y filas mezcla
# ---------------------------------------------------------------------------

conocida <- celdas$w1 == 1
fila1 <- which(conocida)
fila2 <- which(!conocida)

W  <- as.matrix(celdas[fila2, paste0("w", 1:K)])
Pm <- sapply(1:K, function(k) as.character(celdas[[paste0("p", k)]][fila2]))
usa <- W > 0
len <- rowSums(usa)
ini <- cumsum(c(1, head(len, -1)))
# t(): recorre la matriz fila a fila, así las provincias de cada fila quedan juntas
ent_w    <- t(W)[t(usa)]
ent_prov <- idx(t(Pm)[t(usa)], lv$prov)
stopifnot(all(abs(tapply(ent_w, rep(seq_along(len), len), sum) - 1) < 1e-8))

cat("filas con provincia:", length(fila1), " sin provincia:", length(fila2),
    " entradas de la mezcla:", length(ent_w), "\n")

off <- sapply(voto_lv, function(p) {   # una columna por partido (sin referencia)
  if (p %in% names(territorio_mod)) celdas[[paste0("off_", p)]] else rep(0, nrow(celdas))
})

# ---------------------------------------------------------------------------
# 2. Población para postestratificar
# ---------------------------------------------------------------------------

post <- as.data.frame.table(tabla, responseName = "N") %>%
  filter(N > 1e-6) %>%
  left_join(prov_ccaa, by = "prov")

mascara <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) as.numeric(post$ccaa %in% territorio_mod[[p]]) else rep(1, nrow(post))
})

# ---------------------------------------------------------------------------
# 3. Datos para Stan
# ---------------------------------------------------------------------------

datos <- list(
  prior_only = as.integer(modo == "prior"),
  sd_exponencial = as.integer(priors == "brms"),
  s_alpha = if (priors == "brms") 1.5 else 1,
  s_sd_rec = 2, s_sd_main = 0.5, s_sd_int = 0.25,   # solo con priors = "nuevo"
  N = nrow(celdas), J = length(voto_lv), y = unname(celdas$y),
  mujer = as.numeric(celdas$sexo == "M"),
  n_casa = length(lv$casa), n_mes = length(lv$mes), n_edad = length(lv$edad),
  n_edu = length(lv$edu), n_rec = length(lv$rec), n_ccaa = length(lv$ccaa),
  n_prov = length(lv$prov),
  casa = idx(celdas$casa, lv$casa), mes = idx(celdas$mes, lv$mes),
  edad = idx(celdas$edad, lv$edad), edu = idx(celdas$edu, lv$edu),
  rec = idx(celdas$rec, lv$rec), ccaa = idx(celdas$ccaa, lv$ccaa),
  off = unname(off),
  N1 = length(fila1), fila1 = fila1, prov1 = idx(celdas$p1[fila1], lv$prov),
  N2 = length(fila2), fila2 = fila2, ini = ini, len = len,
  E = length(ent_w), ent_prov = ent_prov, ent_w = ent_w,
  P = nrow(post), P_N = post$N, P_mujer = as.numeric(post$sexo == "M"),
  P_edad = idx(post$edad, lv$edad), P_edu = idx(post$edu, lv$edu),
  P_rec = idx(post$rec, lv$rec), P_ccaa = idx(post$ccaa, lv$ccaa),
  P_prov = idx(post$prov, lv$prov), P_mascara = unname(mascara),
  mes_post = length(lv$mes)
)
# Participación: celdas de 03b_celdas.R. Se calibra a la participación del
# 23-J entre residentes: votos emitidos / censo INE = 70,4 %
# (2026/10/mrp-raking/datos/mun_23j.rds).
cp <- readRDS(file.path(dir_datos, "celdas_part.rds"))
if (solo_cis) cp <- cp %>% filter(casa == "CIS")
cp <- cp %>% mutate(mes = factor(as.character(mes), levels = lv$mes))
stopifnot(!anyNA(cp$mes))
datos <- c(datos, list(
  con_part = 1L, NT = nrow(cp), t_n = cp$n, t_k = cp$k,
  t_mujer = as.numeric(cp$sexo == "M"), t_mes = idx(cp$mes, lv$mes),
  t_edad = idx(cp$edad, lv$edad), t_edu = idx(cp$edu, lv$edu),
  t_rec = idx(cp$rec, lv$rec), t_ccaa = idx(cp$ccaa, lv$ccaa),
  part_obj = as.numeric(Sys.getenv("PART_OBJ", "0.704"))   # escenario: solo en modo gq
))
cat("celdas de participación:", nrow(cp), " entrevistas:", sum(cp$n),
    " dicen 10:", round(100 * sum(cp$k) / sum(cp$n), 1), "%\n")

if (solo_cis) {
  stopifnot(all(cp$w1 == 1))
  datos$t_prov <- idx(cp$p1, lv$prov)
  # mrp_cis.stan: sin casa, sin filas mezcla y con la provincia de cada fila
  datos$prov <- idx(celdas$p1, lv$prov)
  datos[c("n_casa", "casa", "N1", "fila1", "prov1", "N2", "fila2", "ini", "len",
          "E", "ent_prov", "ent_w")] <- NULL
} else {
  # mrp.stan: participación con casa y con la misma mezcla de provincias
  datos$con_part <- NULL
  t_con <- cp$w1 == 1
  t_fila2 <- which(!t_con)
  tW  <- as.matrix(cp[t_fila2, paste0("w", 1:K)])
  tPm <- sapply(1:K, function(k) as.character(cp[[paste0("p", k)]][t_fila2]))
  t_usa <- tW > 0
  t_len <- rowSums(t_usa)
  # El efecto casa de participación solo lleva las casas con prob. de votar,
  # que son las primeras de lv$casa (Ateneo del Dato, la última, no la tiene)
  t_casa <- idx(cp$casa, lv$casa)
  n_casa_t <- max(t_casa)
  stopifnot(setequal(unique(t_casa), seq_len(n_casa_t)))
  cat("casas en participación:", paste(lv$casa[seq_len(n_casa_t)], collapse = ", "), "\n")
  datos <- c(datos, list(
    n_casa_t = n_casa_t, t_casa = t_casa,
    NT1 = sum(t_con), t_fila1 = which(t_con), t_prov1 = idx(cp$p1[t_con], lv$prov),
    NT2 = length(t_fila2), t_fila2 = t_fila2,
    t_ini = cumsum(c(1, head(t_len, -1))), t_len = t_len,
    TE = sum(t_usa), t_ent_prov = idx(t(tPm)[t(t_usa)], lv$prov), t_ent_w = t(tW)[t(t_usa)]
  ))
  cat("participación: filas con provincia:", datos$NT1, " sin provincia:", datos$NT2, "\n")
}

# Solo los datos, sin ajustar: los usa 05b_separado.R (voto y participación por separado)
if (Sys.getenv("SOLO_DATOS") != "") {
  saveRDS(list(datos = datos, lv = lv, voto_lv = voto_lv), Sys.getenv("SOLO_DATOS"))
  quit(save = "no")
}
if (!solo_cis) stop("El modelo conjunto mrp.stan está en obsoleto/. Para las 4 casas usa 05b_separado.R.")

# ---------------------------------------------------------------------------
# 4. Ajuste
# ---------------------------------------------------------------------------
# Se guarda el ajuste con una huella del modelo y de los datos: si ninguno de
# los dos cambia, se carga en vez de reajustar (como file_refit = "on_change").

f_stan <- file.path(dir_proy, if (solo_cis) "mrp_cis.stan" else "mrp.stan")
huella <- digest::digest(list(readLines(f_stan), datos, version))
f_fit  <- file.path(dir_datos, paste0("fit_stan_", version, ".rds"))

mod <- cmdstan_model(f_stan)

if (modo == "gq") {
  # Misma posterior, otra tabla. Se postestratifica en R: generate_quantities
  # de cmdstan relee las muestras con 6 cifras y sum_to_zero_vector las
  # rechaza (la suma da 3e-8, no 0). Solo mrp_cis.stan. 1.000 de las 4.000 muestras.
  stopifnot(solo_cis)
  fit <- readRDS(f_fit)$fit
  dm <- posterior::thin_draws(fit$draws(format = "draws_matrix"), 4)
  J <- datos$J
  mat <- function(v, d, n) matrix(dm[d, paste0(v, "[", rep(1:n, J), ",", rep(1:J, each = n), "]")], n, J)
  vec <- function(v, d, n) as.numeric(dm[d, paste0(v, "[", 1:n, "]")])
  i_cr <- (datos$P_ccaa - 1) * datos$n_rec + datos$P_rec
  v_casa <- v_part <- matrix(NA, nrow(dm), J)
  for (d in seq_len(nrow(dm))) {
    eta <- outer(rep(1, datos$P), vec("alpha", d, J)) + outer(datos$P_mujer, vec("b_mujer", d, J)) +
      mat("a_mes", d, datos$n_mes)[rep(datos$mes_post, datos$P), ] +
      mat("a_edad", d, datos$n_edad)[datos$P_edad, ] + mat("a_edu", d, datos$n_edu)[datos$P_edu, ] +
      mat("a_rec", d, datos$n_rec)[datos$P_rec, ] + mat("a_ccaa", d, datos$n_ccaa)[datos$P_ccaa, ] +
      mat("a_prov", d, datos$n_prov)[datos$P_prov, ] +
      mat("a_ccaa_rec", d, datos$n_ccaa * datos$n_rec)[i_cr, ]
    p <- exp(eta - apply(eta, 1, max)) * datos$P_mascara   # prob_valida()
    p <- p / rowSums(p)
    eta_t <- dm[d, "alpha_t"] + datos$P_mujer * dm[d, "b_mujer_t"] +
      vec("a_mes_t", d, datos$n_mes)[datos$mes_post] + vec("a_edad_t", d, datos$n_edad)[datos$P_edad] +
      vec("a_edu_t", d, datos$n_edu)[datos$P_edu] + vec("a_rec_t", d, datos$n_rec)[datos$P_rec] +
      vec("a_ccaa_t", d, datos$n_ccaa)[datos$P_ccaa] + vec("a_prov_t", d, datos$n_prov)[datos$P_prov]
    delta <- uniroot(function(x) sum(datos$P_N * plogis(eta_t + x)) / sum(datos$P_N) - datos$part_obj,
                     c(-10, 10), tol = 1e-10)$root                              # calibra()
    pv <- plogis(eta_t + delta)
    v_casa[d, ] <- 100 * colSums(datos$P_N * p) / sum(datos$P_N)
    v_part[d, ] <- 100 * colSums(datos$P_N * pv * p) / sum(datos$P_N * pv)
  }
  res <- bind_rows(lapply(list(v_casa = v_casa, v_part = v_part), function(m)
    tibble(partido = voto_lv, mrp = colMeans(m),
           q05 = apply(m, 2, quantile, 0.05), q95 = apply(m, 2, quantile, 0.95))),
    .id = "estimacion")
  res$tabla <- basename(f_tabla)
  res$part_obj <- datos$part_obj
  print(as.data.frame(res %>% mutate(across(where(is.numeric), ~round(.x, 1)))), row.names = FALSE)
  dir.create(file.path(dir_datos, "sensibilidad"), showWarnings = FALSE)
  saveRDS(res, file.path(dir_datos, "sensibilidad",
                         paste0("estim_", version, "_", sub("[.]rds$", "", basename(f_tabla)),
                                if (datos$part_obj != 0.704) paste0("_part", datos$part_obj) else "", ".rds")))
  quit(save = "no")
}

if (file.exists(f_fit) && identical(readRDS(f_fit)$huella, huella)) {
  cat("modelo y datos sin cambios: cargo", f_fit, "\n")
  fit <- readRDS(f_fit)$fit
} else {
  t0 <- Sys.time()
  fit <- mod$sample(
    data = datos, seed = 2026,
    chains = if (modo == "prueba") 1 else 4, parallel_chains = 4,
    iter_warmup = switch(modo, prueba = 150, prior = 250, nuts = 1000),
    iter_sampling = switch(modo, prueba = 150, prior = 250, nuts = 1000),
    init = 0, adapt_delta = 0.95, max_treedepth = 8,   # igual que 04_brms.R
    refresh = if (modo == "prueba") 25 else 100
  )
  cat("\ntiempo de ajuste:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
  fit$save_object(file.path(dir_datos, "tmp_fit.rds"))   # lee todas las muestras en memoria
  saveRDS(list(fit = fit, huella = huella), f_fit)
  file.remove(file.path(dir_datos, "tmp_fit.rds"))
}

if (modo == "prior") {
  # Qué dice el prior en la escala de los datos, antes de ver ningún voto
  pm <- as.vector(fit$draws("p_max", format = "draws_matrix"))
  cat("\n== prior: probabilidad del partido más votado en una celda ==\n")
  print(round(quantile(pm, c(0.05, 0.25, 0.5, 0.75, 0.95)), 2))
  cat("celdas con un partido > 90 %:", round(100 * mean(pm > 0.9), 1), "%\n")
  cat("celdas con un partido > 99 %:", round(100 * mean(pm > 0.99), 1), "%\n")

  v <- fit$draws("v_casa", format = "draws_matrix")
  fila <- if (solo_cis) 1 else length(lv$casa) + 1
  m <- v[, paste0("v_casa[", fila, ",", seq_along(voto_lv), "]"), drop = FALSE]
  cat("\n== prior: voto en España según", if (solo_cis) "el CIS" else "la casa media", "(%) ==\n")
  print(data.frame(partido = voto_lv,
                   q05 = round(apply(m, 2, quantile, 0.05), 1),
                   mediana = round(apply(m, 2, median), 1),
                   q95 = round(apply(m, 2, quantile, 0.95), 1)), row.names = FALSE)
  saveRDS(list(p_max = pm, v_casa = v), file.path(dir_datos, paste0("prior_check_", priors, if (solo_cis) "_cis" else "", ".rds")))
  quit(save = "no")
}

# ---------------------------------------------------------------------------
# 5. Diagnóstico y resultados
# ---------------------------------------------------------------------------

cat("\n== diagnóstico del muestreador ==\n")
print(fit$diagnostic_summary())
td <- fit$sampler_diagnostics(format = "draws_df")
cat("iteraciones en max_treedepth (8):", sum(td$treedepth__ >= 8), "de", nrow(td), "\n")
cat("pasos leapfrog por iteración (mediana):", median(td$n_leapfrog__), "\n")

sds <- fit$summary(variables = c(if (!solo_cis) "sd_casa", "sd_mes", "sd_edad", "sd_edu", "sd_rec",
                                 "sd_ccaa", "sd_prov", "sd_ccaa_rec", if (!solo_cis) "sd_casa_rec"))
cat("\nRhat máximo de las sd:", round(max(sds$rhat), 3),
    " ESS bulk mínimo:", round(min(sds$ess_bulk)), "\n")

v <- fit$draws("v_casa", format = "draws_matrix")
nombres_casa <- if (solo_cis) "CIS solo" else c(lv$casa, "casa media")
res <- bind_rows(lapply(seq_along(nombres_casa), function(h) {
  m <- v[, paste0("v_casa[", h, ",", seq_along(voto_lv), "]"), drop = FALSE]
  tibble(casa = nombres_casa[h], partido = voto_lv,
         mrp = colMeans(m),
         q05 = apply(m, 2, quantile, 0.05),
         q95 = apply(m, 2, quantile, 0.95))
}))

cat("\n== MRP por casa,", max(lv$mes), ", si votara todo el mundo (% voto válido) ==\n")
print(as.data.frame(res %>% select(casa, partido, mrp) %>%
                      pivot_wider(names_from = casa, values_from = mrp) %>%
                      mutate(across(-partido, ~round(.x, 1)))), row.names = FALSE)

cat("\n== participación ==\n")
print(fit$summary(variables = c("part_bruta", "delta")), width = 120)
vp <- fit$draws("v_part", format = "draws_matrix")
res_part <- bind_rows(lapply(seq_along(nombres_casa), function(h) {
  m <- vp[, paste0("v_part[", h, ",", seq_along(voto_lv), "]"), drop = FALSE]
  tibble(casa = paste0(nombres_casa[h], ", con participación"), partido = voto_lv,
         mrp = colMeans(m), q05 = apply(m, 2, quantile, 0.05), q95 = apply(m, 2, quantile, 0.95))
}))
cat("\n== MRP con participación calibrada al 70,4 % (% voto válido) ==\n")
print(as.data.frame(res_part %>% select(casa, partido, mrp) %>%
                      pivot_wider(names_from = casa, values_from = mrp) %>%
                      mutate(across(-partido, ~round(.x, 1)))), row.names = FALSE)
res <- bind_rows(res, res_part)

saveRDS(res, file.path(dir_datos, paste0("estim_stan_", version, ".rds")))
cat("\nguardado", paste0("estim_stan_", version, ".rds"), "\n")
