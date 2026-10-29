# Datos de modelo para los tres backends (Stan, numpyro y RTMB): las listas
# que esperan mrp_voto.stan / mrp_part.stan / mrp_cis.stan y sus ports. Sale
# de lo que antes era el antiguo 05_stan.R con la variable SOLO_DATOS.
#
# Usa las celdas de 03_celdas.R (datos/celdas_brms.rds y celdas_part.rds), las
# mismas que 05_brms_mm.R, para poder comparar los ajustes.
#
# Guarda DOS juegos de datos en datos/datos_stan.rds:
#   datos     : las 4 casas (voto con filas mezcla; participación con casa)
#   datos_cis : solo el CIS (sin efecto casa ni filas mezcla; stan/ajusta_cis.R)
# con sus niveles (lv) y voto_lv.
#
# Para una prueba rápida: modo <- "prueba" aquí y en el script de ajuste
# (10 % de las filas, 1 cadena). En producción: modo <- "nuts".

library(dplyr)
library(tidyr)

modo <- "nuts" # prueba | nuts
priors <- "nuevo" # nuevo | brms (exponential(1) en las sd, como 05_brms_mm.R)

dir_proy <- "2026/11/mrp-multiencuesta"
dir_datos <- file.path(dir_proy, "datos")
source(file.path(dir_proy, "00_partidos.R")) # partidos_lv, rec_lv, territorio

celdas_todas <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))

voto_lv <- setdiff(partidos_lv, "AC")
territorio_mod <- territorio[names(territorio) != "AC"]
stopifnot(identical(colnames(celdas_todas$y), voto_lv))
K <- 9

cat("modo:", modo, "  priors:", priors, "\n")
if (modo == "prueba") {
  set.seed(2026)
  celdas_todas <- celdas_todas[
    sample(nrow(celdas_todas), round(0.1 * nrow(celdas_todas))),
  ]
}

# ---------------------------------------------------------------------------
# Población para postestratificar (común a los dos juegos de datos)
# ---------------------------------------------------------------------------

post <- as.data.frame.table(tabla, responseName = "N") |>
  filter(N > 1e-6) |>
  left_join(prov_ccaa, by = "prov")

mascara <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) {
    as.numeric(post$ccaa %in% territorio_mod[[p]])
  } else {
    rep(1, nrow(post))
  }
})

# ---------------------------------------------------------------------------
# Preparación de los datos de voto y participación
# ---------------------------------------------------------------------------

prep <- function(celdas, solo_cis) {
  if (solo_cis) {
    celdas <- celdas |> filter(casa == "CIS") |> droplevels()
    stopifnot(all(celdas$w1 == 1)) # el CIS siempre trae la provincia
  }

  # Niveles fijos: Stan solo ve enteros 1..n
  lv <- list(
    casa = levels(celdas$casa),
    mes = levels(celdas$mes),
    edad = levels(celdas$edad),
    edu = levels(celdas$edu),
    rec = rec_lv,
    ccaa = levels(prov_ccaa$ccaa),
    prov = levels(prov_ccaa$prov)
  )
  idx <- function(x, niveles) {
    i <- match(as.character(x), niveles)
    stopifnot(!anyNA(i))
    i
  }

  # 1. Provincia: filas conocidas y filas mezcla
  conocida <- celdas$w1 == 1
  fila1 <- which(conocida)
  fila2 <- which(!conocida)

  W <- as.matrix(celdas[fila2, paste0("w", 1:K)])
  Pm <- sapply(1:K, function(k) as.character(celdas[[paste0("p", k)]][fila2]))
  usa <- W > 0
  len <- rowSums(usa)
  ini <- cumsum(c(1, head(len, -1)))
  # t(): recorre la matriz fila a fila, así las provincias de cada fila quedan juntas
  ent_w <- t(W)[t(usa)]
  ent_prov <- idx(t(Pm)[t(usa)], lv$prov)
  stopifnot(all(abs(tapply(ent_w, rep(seq_along(len), len), sum) - 1) < 1e-8))

  cat(
    "filas con provincia:",
    length(fila1),
    "  sin provincia:",
    length(fila2),
    "  entradas de la mezcla:",
    length(ent_w),
    "\n"
  )

  off <- sapply(voto_lv, function(p) {
    # una columna por partido (sin referencia)
    if (p %in% names(territorio_mod)) {
      celdas[[paste0("off_", p)]]
    } else {
      rep(0, nrow(celdas))
    }
  })

  # 2. Datos para Stan
  datos <- list(
    prior_only = 0L,
    sd_exponencial = as.integer(priors == "brms"),
    s_alpha = if (priors == "brms") 1.5 else 1,
    s_sd_rec = 2,
    s_sd_main = 0.5,
    s_sd_int = 0.25, # solo con priors = "nuevo"
    N = nrow(celdas),
    J = length(voto_lv),
    y = unname(celdas$y),
    mujer = as.numeric(celdas$sexo == "M"),
    n_casa = length(lv$casa),
    n_mes = length(lv$mes),
    n_edad = length(lv$edad),
    n_edu = length(lv$edu),
    n_rec = length(lv$rec),
    n_ccaa = length(lv$ccaa),
    n_prov = length(lv$prov),
    casa = idx(celdas$casa, lv$casa),
    mes = idx(celdas$mes, lv$mes),
    edad = idx(celdas$edad, lv$edad),
    edu = idx(celdas$edu, lv$edu),
    rec = idx(celdas$rec, lv$rec),
    ccaa = idx(celdas$ccaa, lv$ccaa),
    off = unname(off),
    N1 = length(fila1),
    fila1 = fila1,
    prov1 = idx(celdas$p1[fila1], lv$prov),
    N2 = length(fila2),
    fila2 = fila2,
    ini = ini,
    len = len,
    E = length(ent_w),
    ent_prov = ent_prov,
    ent_w = ent_w,
    P = nrow(post),
    P_N = post$N,
    P_mujer = as.numeric(post$sexo == "M"),
    P_edad = idx(post$edad, lv$edad),
    P_edu = idx(post$edu, lv$edu),
    P_rec = idx(post$rec, lv$rec),
    P_ccaa = idx(post$ccaa, lv$ccaa),
    P_prov = idx(post$prov, lv$prov),
    P_mascara = unname(mascara),
    mes_post = length(lv$mes)
  )

  # AA: filas de encuestas que no la ofrecían (00_partidos.R). Su verosimilitud
  # se condiciona a no-AA (multinomial_cond en funciones.stan). El CIS siempre
  # la pregunta, así que en datos_cis no hace falta.
  if (!solo_cis) {
    datos$sin_aa <- as.integer(celdas$sin_aa)
    datos$j_aa <- match("AA", voto_lv)
    stopifnot(!is.na(datos$j_aa))
  }

  # 3. Participación: celdas de 03_celdas.R. Se calibra a la participación del
  # 23-J entre residentes: votos emitidos / censo INE = 70,4 %
  # (2026/10/mrp-raking/datos/mun_23j.rds).
  cp <- readRDS(file.path(dir_datos, "celdas_part.rds"))
  if (solo_cis) {
    cp <- cp |> filter(casa == "CIS")
  }
  cp <- cp |> mutate(mes = factor(as.character(mes), levels = lv$mes))
  stopifnot(!anyNA(cp$mes))
  datos <- c(
    datos,
    list(
      con_part = 1L,
      NT = nrow(cp),
      t_n = cp$n,
      t_k = cp$k,
      t_mujer = as.numeric(cp$sexo == "M"),
      t_mes = idx(cp$mes, lv$mes),
      t_edad = idx(cp$edad, lv$edad),
      t_edu = idx(cp$edu, lv$edu),
      t_rec = idx(cp$rec, lv$rec),
      t_ccaa = idx(cp$ccaa, lv$ccaa),
      part_obj = as.numeric(Sys.getenv("PART_OBJ", "0.704")) # escenario
    )
  )
  cat(
    "celdas de participación:",
    nrow(cp),
    "  entrevistas:",
    sum(cp$n),
    "  dicen 10:",
    round(100 * sum(cp$k) / sum(cp$n), 1),
    "%\n"
  )

  if (solo_cis) {
    stopifnot(all(cp$w1 == 1))
    datos$t_prov <- idx(cp$p1, lv$prov)
    # mrp_cis.stan: sin casa, sin filas mezcla y con la provincia de cada fila
    datos$prov <- idx(celdas$p1, lv$prov)
    datos[c(
      "n_casa",
      "casa",
      "N1",
      "fila1",
      "prov1",
      "N2",
      "fila2",
      "ini",
      "len",
      "E",
      "ent_prov",
      "ent_w"
    )] <- NULL
  } else {
    # participación con casa y con la misma mezcla de provincias
    datos$con_part <- NULL
    t_con <- cp$w1 == 1
    t_fila2 <- which(!t_con)
    tW <- as.matrix(cp[t_fila2, paste0("w", 1:K)])
    tPm <- sapply(1:K, function(k) as.character(cp[[paste0("p", k)]][t_fila2]))
    t_usa <- tW > 0
    t_len <- rowSums(t_usa)
    # El efecto casa de participación solo lleva las casas con prob. de votar,
    # que son las primeras de lv$casa (Ateneo del Dato, la última, no la tiene)
    t_casa <- idx(cp$casa, lv$casa)
    n_casa_t <- max(t_casa)
    stopifnot(setequal(unique(t_casa), seq_len(n_casa_t)))
    cat(
      "casas en participación:",
      paste(lv$casa[seq_len(n_casa_t)], collapse = ", "),
      "\n"
    )
    datos <- c(
      datos,
      list(
        n_casa_t = n_casa_t,
        t_casa = t_casa,
        NT1 = sum(t_con),
        t_fila1 = which(t_con),
        t_prov1 = idx(cp$p1[t_con], lv$prov),
        NT2 = length(t_fila2),
        t_fila2 = t_fila2,
        t_ini = cumsum(c(1, head(t_len, -1))),
        t_len = t_len,
        TE = sum(t_usa),
        t_ent_prov = idx(t(tPm)[t(t_usa)], lv$prov),
        t_ent_w = t(tW)[t(t_usa)]
      )
    )
    cat(
      "participación: filas con provincia:",
      datos$NT1,
      "  sin provincia:",
      datos$NT2,
      "\n"
    )
  }

  list(datos = datos, lv = lv)
}

cat("\n== datos de las 4 casas ==\n")
casas <- prep(celdas_todas, solo_cis = FALSE)
cat("\n== datos solo CIS ==\n")
cis <- prep(celdas_todas, solo_cis = TRUE)

saveRDS(
  list(
    modo = modo,
    priors = priors,
    datos = casas$datos,
    lv = casas$lv,
    datos_cis = cis$datos,
    lv_cis = cis$lv,
    voto_lv = voto_lv
  ),
  file.path(dir_datos, "datos_stan.rds")
)
cat("\nguardado datos/datos_stan.rds\n")
