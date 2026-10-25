# MRP conjunto de las 4 casas: voto (mrp_voto.stan, con reduce_sum) y
# participación (mrp_part.stan) en dos modelos separados. No comparten
# parámetros: la posterior es la misma que la del antiguo mrp.stan conjunto
# (obsoleto/). Los datos los prepara 05_stan.R. La postestratificación con
# participación se hace en R, emparejando las muestras de los dos ajustes.
#
#   Rscript 2026/11/mrp-multiencuesta/05b_separado.R prueba 3   # 10 % del voto, 1 cadena, 3 hilos
#   Rscript 2026/11/mrp-multiencuesta/05b_separado.R nuts 2     # 4 cadenas x 2000 it., 2 hilos por cadena
#
# Primer argumento: prueba | nuts. Segundo: hilos por cadena del voto
# (4 cadenas x hilos núcleos). Solo priors "nuevo".
# Salidas: datos/fit_separado_<modo>_voto.rds, _part.rds y estim_separado_<modo>.rds.

library(dplyr)
library(tidyr)
library(cmdstanr)

args <- commandArgs(trailingOnly = TRUE)
modo <- if (length(args) >= 1) args[1] else "prueba"
stopifnot(modo %in% c("prueba", "nuts"))
hilos <- if (length(args) >= 2) as.integer(args[2]) else 1L
cat("modo:", modo, " hilos por cadena:", hilos, "\n")

dir_proy <- "2026/11/mrp-multiencuesta"
dir_datos <- file.path(dir_proy, "datos")

# 1. Datos: los de 05_stan.R, sin ajustar
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
voto_lv <- x$voto_lv
nombres_casa <- c(x$lv$casa, "casa media")

# Lo nuevo: priors que antes estaban escritos en el código, y el tamaño de trozo
datos <- c(
  datos,
  list(
    s_b_mujer = 1, # b_mujer ~ normal(0, 1)
    m_alpha_t = 1,
    s_alpha_t = 1.5,
    s_b_mujer_t = 1, # alpha_t ~ normal(1, 1.5)
    grainsize = max(1L, as.integer(ceiling(datos$N / (4 * hilos))))
  )
)
cat(
  "filas de voto:",
  datos$N,
  " grainsize:",
  datos$grainsize,
  " filas de participación:",
  datos$NT,
  "\n"
)

# 2. Ajustes (CmdStan ignora los datos que un modelo no declara)
compila <- function(f) {
  cmdstan_model(
    file.path(dir_proy, f),
    include_paths = file.path(dir_proy, "stan"),
    cpp_options = list(stan_threads = TRUE)
  )
}
ajusta <- function(mod, h) {
  t0 <- Sys.time()
  fit <- mod$sample(
    data = datos,
    seed = 2026,
    chains = if (modo == "prueba") 1 else 4,
    parallel_chains = 4,
    threads_per_chain = h,
    iter_warmup = if (modo == "prueba") 150 else 1000,
    iter_sampling = if (modo == "prueba") 150 else 1000,
    init = 0,
    adapt_delta = 0.95,
    max_treedepth = 8, # igual que 05_stan.R
    refresh = if (modo == "prueba") 25 else 100
  )
  list(fit = fit, min = as.numeric(difftime(Sys.time(), t0, units = "mins")))
}
diagnostico <- function(fit, nombre, min) {
  td <- fit$sampler_diagnostics(format = "draws_df")
  tibble(
    modelo = nombre,
    minutos = round(min, 1),
    divergencias = sum(td$divergent__),
    en_max_treedepth = sum(td$treedepth__ >= 8),
    leapfrog_mediana = median(td$n_leapfrog__)
  )
}

# Cada ajuste se guarda con una huella del modelo (y funciones.stan) y de los
# datos: si no cambian, se carga en vez de reajustar (como en 05_stan.R)
ajusta_o_carga <- function(f_stan, h, nombre) {
  huella <- digest::digest(list(
    readLines(file.path(dir_proy, f_stan)),
    readLines(file.path(dir_proy, "stan", "funciones.stan")),
    datos[names(datos) != "grainsize"],
    modo
  ))
  f_fit <- file.path(
    dir_datos,
    paste0("fit_separado_", modo, "_", nombre, ".rds")
  )
  if (file.exists(f_fit) && identical(readRDS(f_fit)$huella, huella)) {
    cat("modelo y datos sin cambios: cargo", f_fit, "\n")
    return(readRDS(f_fit))
  }
  a <- ajusta(compila(f_stan), h)
  a$fit$draws()
  a$fit$sampler_diagnostics() # en memoria antes de guardar
  a$huella <- huella
  saveRDS(a, f_fit)
  cat("guardado", f_fit, "\n")
  a
}

av <- ajusta_o_carga("mrp_voto.stan", hilos, "voto")
ap <- ajusta_o_carga("mrp_part.stan", 1L, "part") # la participación es barata: un hilo

cat("\n== tiempos y diagnóstico ==\n")
print(
  as.data.frame(bind_rows(
    diagnostico(av$fit, paste0("voto, ", hilos, " hilos"), av$min),
    diagnostico(ap$fit, "participación", ap$min)
  )),
  row.names = FALSE
)
sds <- bind_rows(
  av$fit$summary(
    variables = c(
      "sd_casa",
      "sd_mes",
      "sd_edad",
      "sd_edu",
      "sd_rec",
      "sd_ccaa",
      "sd_prov",
      "sd_ccaa_rec",
      "sd_casa_rec"
    )
  ),
  ap$fit$summary(
    variables = c(
      "sd_casa_t",
      "sd_mes_t",
      "sd_edad_t",
      "sd_edu_t",
      "sd_rec_t",
      "sd_ccaa_t",
      "sd_prov_t"
    )
  )
)
cat(
  "Rhat máximo de las sd:",
  round(max(sds$rhat), 3),
  " ESS bulk mínimo:",
  round(min(sds$ess_bulk)),
  "\n"
)

# ---------------------------------------------------------------------------
# 3. Postestratificación con participación, en R
# ---------------------------------------------------------------------------
# Los dos ajustes son independientes: la muestra d del voto se empareja con la
# muestra d de la participación. Para cada una:
#   p[q, j]  = prob_valida() de la celda q (por casa y para la casa media)
#   pv[q]    = inv_logit(eta_t[q] + delta), con delta de mrp_part.stan
#   v_part   = sum_q N_q pv_q p[q, j] / sum_q N_q pv_q
# Se rehace también v_casa (sin participación) para comprobar que el cálculo
# en R da lo mismo que el de Stan.

J <- datos$J
P <- datos$P
n_rec <- datos$n_rec
n_casa <- datos$n_casa
matriz <- function(dm) matrix(dm, nrow(dm), dimnames = list(NULL, colnames(dm))) # sin la clase draws
dv <- matriz(av$fit$draws(format = "draws_matrix"))
dp <- matriz(ap$fit$draws(format = "draws_matrix"))
nd <- min(nrow(dv), nrow(dp))
arr <- function(dm, v, n) {
  # draws x n x J
  m <- dm[
    seq_len(nd),
    paste0(v, "[", rep(1:n, J), ",", rep(1:J, each = n), "]")
  ]
  array(m, c(nd, n, J))
}
vecs <- function(dm, v, n) {
  dm[seq_len(nd), paste0(v, "[", 1:n, "]"), drop = FALSE]
}

A <- list(
  alpha = vecs(dv, "alpha", J),
  b_mujer = vecs(dv, "b_mujer", J),
  casa = arr(dv, "a_casa", n_casa),
  mes = arr(dv, "a_mes", datos$n_mes),
  edad = arr(dv, "a_edad", datos$n_edad),
  edu = arr(dv, "a_edu", datos$n_edu),
  rec = arr(dv, "a_rec", n_rec),
  ccaa = arr(dv, "a_ccaa", datos$n_ccaa),
  prov = arr(dv, "a_prov", datos$n_prov),
  ccaa_rec = arr(dv, "a_ccaa_rec", datos$n_ccaa * n_rec),
  casa_rec = arr(dv, "a_casa_rec", n_casa * n_rec)
)
Tp <- list(
  alpha = dp[seq_len(nd), "alpha_t"],
  b = dp[seq_len(nd), "b_mujer_t"],
  mes = vecs(dp, "a_mes_t", datos$n_mes),
  edad = vecs(dp, "a_edad_t", datos$n_edad),
  edu = vecs(dp, "a_edu_t", datos$n_edu),
  rec = vecs(dp, "a_rec_t", n_rec),
  ccaa = vecs(dp, "a_ccaa_t", datos$n_ccaa),
  prov = vecs(dp, "a_prov_t", datos$n_prov),
  delta = dp[seq_len(nd), "delta"]
)

i_cr <- (datos$P_ccaa - 1) * n_rec + datos$P_rec
N_q <- datos$P_N
prob <- function(eta) {
  # prob_valida() por filas
  p <- exp(eta - apply(eta, 1, max)) * datos$P_mascara
  p / rowSums(p)
}
v_casa_r <- v_part <- array(NA, c(nd, n_casa + 1, J))
t0 <- Sys.time()
for (d in seq_len(nd)) {
  base <- outer(rep(1, P), A$alpha[d, ]) +
    outer(datos$P_mujer, A$b_mujer[d, ]) +
    matrix(A$mes[d, datos$mes_post, ], P, J, byrow = TRUE) +
    A$edad[d, datos$P_edad, ] +
    A$edu[d, datos$P_edu, ] +
    A$rec[d, datos$P_rec, ] +
    A$ccaa[d, datos$P_ccaa, ] +
    A$prov[d, datos$P_prov, ] +
    A$ccaa_rec[d, i_cr, ]
  eta_t <- Tp$alpha[d] +
    datos$P_mujer * Tp$b[d] +
    Tp$mes[d, datos$mes_post] +
    Tp$edad[d, datos$P_edad] +
    Tp$edu[d, datos$P_edu] +
    Tp$rec[d, datos$P_rec] +
    Tp$ccaa[d, datos$P_ccaa] +
    Tp$prov[d, datos$P_prov]
  pv <- plogis(eta_t + Tp$delta[d])
  ef_media <- 0
  for (h in seq_len(n_casa + 1)) {
    if (h <= n_casa) {
      ef <- matrix(A$casa[d, h, ], P, J, byrow = TRUE) +
        A$casa_rec[d, (h - 1) * n_rec + datos$P_rec, ]
      ef_media <- ef_media + ef / n_casa
    } else {
      ef <- ef_media
    }
    p <- prob(base + ef)
    v_casa_r[d, h, ] <- 100 * colSums(N_q * p) / sum(N_q)
    v_part[d, h, ] <- 100 * colSums(N_q * pv * p) / sum(N_q * pv)
  }
}
cat(
  "\npostestratificación en R:",
  nd,
  "muestras,",
  round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
  "min\n"
)

v_stan <- array(
  dv[
    seq_len(nd),
    paste0(
      "v_casa[",
      rep(1:(n_casa + 1), J),
      ",",
      rep(1:J, each = n_casa + 1),
      "]"
    )
  ],
  c(nd, n_casa + 1, J)
)
cat(
  "comprobación v_casa, R frente a Stan: diferencia máx. =",
  signif(max(abs(v_casa_r - v_stan)), 2),
  "puntos\n"
)
# el orden de las sumas difiere de Stan: con PODEMOS aparte (J = 12) el ruido
# de redondeo pasa de 1e-6 a ~1e-5; un error real sería del orden de 0,1
stopifnot(max(abs(v_casa_r - v_stan)) < 1e-4)

resume <- function(a, etiqueta) {
  bind_rows(lapply(seq_along(nombres_casa), function(h) {
    tibble(
      casa = paste0(nombres_casa[h], etiqueta),
      partido = voto_lv,
      mrp = colMeans(a[, h, ]),
      q05 = apply(a[, h, ], 2, quantile, 0.05),
      q95 = apply(a[, h, ], 2, quantile, 0.95)
    )
  }))
}
res <- bind_rows(resume(v_stan, ""), resume(v_part, ", con participación"))

cat(
  "\n== MRP por casa,",
  max(x$lv$mes),
  ", si votara todo el mundo (% voto válido) ==\n"
)
print(
  as.data.frame(
    res %>%
      filter(!grepl("participación", casa)) %>%
      select(casa, partido, mrp) %>%
      pivot_wider(names_from = casa, values_from = mrp) %>%
      mutate(across(-partido, ~ round(.x, 1)))
  ),
  row.names = FALSE
)
cat("\n== participación ==\n")
print(ap$fit$summary(variables = c("part_bruta", "delta")), width = 120)
cat(
  "\n== MRP con participación calibrada al",
  100 * datos$part_obj,
  "% (% voto válido) ==\n"
)
print(
  as.data.frame(
    res %>%
      filter(grepl("participación", casa)) %>%
      mutate(casa = sub(", con participación", "", casa)) %>%
      select(casa, partido, mrp) %>%
      pivot_wider(names_from = casa, values_from = mrp) %>%
      mutate(across(-partido, ~ round(.x, 1)))
  ),
  row.names = FALSE
)

saveRDS(res, file.path(dir_datos, paste0("estim_separado_", modo, ".rds")))
cat("\nguardado", paste0("estim_separado_", modo, ".rds"), "\n")
