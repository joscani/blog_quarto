# MRP con xgboost: mismo pipeline que 05_mrp.R, cambiando solo la M.
#
# Se mantiene igual:
#   - datos y recodificación (00_partidos.R)
#   - tabla de postestratificación rakeada (04_raking.R)
#   - modelo de participación (brms, cacheado en datos/) y su calibración al
#     23-J, para que la comparación aísle el modelo de voto
#   - ceros estructurales por territorio (máscara, como en el post)
#
# Cambia:
#   - el multinomial multinivel de brms por un clasificador multiclase de
#     xgboost, sobre las mismas variables (sexo, edad, edu, rec, ccaa)
#   - xgboost no tiene efectos aleatorios ni offsets: los ceros estructurales
#     se imponen a posteriori con la máscara
#   - no hay posterior: estimación puntual, sin intervalos
#
# Aviso: con 7 casos de AC y 6 de CCA, el multinomial de xgboost rara vez
# predirá esas categorías; es parte de lo que la comparación tiene que mostrar.

library(dplyr)
library(tidyr)
library(haven)
library(brms)
library(xgboost)

options(mc.cores = 4, brms.backend = "cmdstanr")
dir_datos <- "2026/10/mrp-raking/datos"

cis <- readRDS(file.path(dir_datos, "cis_imputado.rds"))
tabla <- readRDS(file.path(dir_datos, "tabla_postestratificacion.rds"))
tg <- readRDS(file.path(dir_datos, "targets.rds"))

source("2026/10/mrp-raking/00_partidos.R") # partidos_lv, rec_lv, territorio,
# mascara_territorio
voto_lv <- partidos_lv
edu_lv <- c("baja", "media", "alta")

# ---------------------------------------------------------------------------
# 1. Datos y celdas de postestratificación
# ---------------------------------------------------------------------------

dat <- cis %>% filter(!is.na(voto), !is.na(edu), !is.na(edad), !is.na(rec))
cat("muestra utilizable:", nrow(dat), "de", nrow(cis), "\n")

post <- as.data.frame.table(tabla, responseName = "N") %>%
  filter(N > 1e-6) %>%
  mutate(
    ccaa = factor(ccaa, levels = levels(dat$ccaa)),
    sexo = factor(sexo, levels = levels(dat$sexo)),
    edad = factor(edad, levels = levels(dat$edad)),
    edu = factor(edu, levels = edu_lv),
    rec = factor(rec, levels = rec_lv),
    n = 1
  )
cat("celdas de la tabla con población:", nrow(post), "\n")

# ---------------------------------------------------------------------------
# 2. Modelo de voto: xgboost multinomial
# ---------------------------------------------------------------------------

# Una-hot para los árboles: la misma matriz para la muestra y para las celdas
mm <- function(d) {
  model.matrix(~ sexo + edad + edu + rec + ccaa, data = d)[, -1]
}

set.seed(2026)
dtrain <- xgb.DMatrix(mm(dat), label = as.integer(dat$voto) - 1)

params <- list(
  objective = "multi:softprob",
  num_class = length(voto_lv),
  eval_metric = "mlogloss",
  eta = 0.1,
  max_depth = 3,
  min_child_weight = 5,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Número de árboles: el mínimo del mlogloss en validación cruzada (con early
# stopping de 20 rondas). xgboost 3.2 no rellena cv$best_iteration, así que
# lo sacamos del log de la cv.
cv <- xgb.cv(
  params,
  dtrain,
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 0
)
nrounds <- which.min(cv$evaluation_log$test_mlogloss_mean)
cat("árboles (mínimo del mlogloss en cv):", nrounds, "\n")

mod_xgb <- xgb.train(params, dtrain, nrounds = nrounds)

cat("\n== variables que más usan los árboles ==\n")
print(head(xgb.importance(model = mod_xgb), 8))

# Predicción en las celdas de la tabla. softprob devuelve un vector aplastado
# (num_class valores por fila, en orden); algunas versiones lo devuelven ya
# como matriz.
p <- predict(mod_xgb, xgb.DMatrix(mm(post)))
theta <- if (is.matrix(p)) p else matrix(p, nrow = nrow(post), byrow = TRUE)
colnames(theta) <- voto_lv

# Ceros estructurales: mismo truco que en el post. Fuera de su territorio,
# los partidos regionales se anulan y se renormaliza cada celda.
mascara <- mascara_territorio(post$ccaa, voto_lv)
theta <- theta * mascara
theta <- theta / rowSums(theta)

# ---------------------------------------------------------------------------
# 3. Participación: la misma de 05_mrp.R
# ---------------------------------------------------------------------------

celdas_part <- cis %>%
  filter(!is.na(vota_seguro)) %>%
  group_by(ccaa, sexo, edad, edu, rec) %>%
  summarise(vota = sum(vota_seguro), n = n(), .groups = "drop")

mod_part <- readRDS(file.path(dir_datos, "mod_participacion.rds"))

set.seed(2026) # los mismos 500 draws que en el post
t_draws <- posterior_epred(mod_part, newdata = post, ndraws = 500)

rec_pob <- tg$rec_ccaa %>% group_by(rec) %>% summarise(N = sum(N))
participacion_23j <- with(
  rec_pob,
  sum(N[!rec %in% c("ABST", "NO_PODIA")]) / sum(N[rec != "NO_PODIA"])
)

calibra <- function(p, N, objetivo) {
  f <- function(delta) sum(N * plogis(qlogis(p) + delta)) / sum(N) - objetivo
  plogis(qlogis(p) + uniroot(f, c(-10, 10))$root)
}
t_cal <- t(apply(t_draws, 1, calibra, N = post$N, objetivo = participacion_23j))

# Pesos de celda con la participación media posterior
w <- post$N * colMeans(t_cal)

# ---------------------------------------------------------------------------
# 4. Estimaciones y comparación
# ---------------------------------------------------------------------------

est_xgb <- 100 * colSums(theta * w) / sum(w)
names(est_xgb) <- voto_lv

# Lo que guardó 05_mrp.R, para ponerlo al lado
est <- readRDS(file.path(dir_datos, "estimaciones.rds"))

comparacion_xgb <- est$comparacion %>%
  mutate(xgb = est_xgb[partido]) %>%
  select(partido, bruto, peso_cis, raking, xgb, mrp, cis)

cat("\n== estimación nacional (%) ==\n")
print(
  as.data.frame(
    comparacion_xgb %>% mutate(across(-partido, ~ round(.x, 1)))
  ),
  row.names = FALSE
)

# Por comunidad: donde el MRP debería lucir (y donde xgboost no puede)
ccaa_reg <- c(
  "Andalucía",
  "Cataluña",
  "País Vasco",
  "Navarra",
  "Galicia",
  "Canarias"
)

est_xgb_ccaa <- purrr::map_dfr(ccaa_reg, function(cc) {
  i <- post$ccaa == cc
  tibble(
    ccaa = cc,
    partido = voto_lv,
    xgb = 100 * colSums(theta[i, ] * w[i]) / sum(w[i]),
    mrp = est$est_ccaa[cc, ]
  )
}) %>%
  filter(xgb >= 1 | mrp >= 1 | (partido == "AC" & ccaa == "Cataluña"))

cat("\n== por CCAA (%) ==\n")
print(
  as.data.frame(
    est_xgb_ccaa %>% mutate(across(where(is.numeric), ~ round(.x, 1)))
  ),
  row.names = FALSE
)

saveRDS(
  list(
    comparacion_xgb = comparacion_xgb,
    est_xgb = est_xgb,
    est_xgb_ccaa = est_xgb_ccaa
  ),
  file.path(dir_datos, "estimaciones_xgb.rds")
)
cat("\nguardado estimaciones_xgb.rds\n")
