# Modelo de voto con seguro / dudoso (decisión de José, 2026-11-03): datos de
# las 4 casas para completo/stan/mrp_voto_seg.stan.
#
# Respecto a 09_datos_voto.R: sin ideología, y el nivel de participación pasa
# a dos niveles: seguro (dice "10") y dudoso (0-9). La participación sale del
# modelo ya ajustado del post (II) (stan/mrp_part.stan,
# datos/fit_separado_nuts_part.rds); no se usan las partes 1 y 2 de completo/.
#
# Celdas casa x estudio x mes x ccaa x prov x sexo x edad x edu x rec x seg
# con el recuento de cada partido (voto + simpatía, BLANCO aparte, Aliança
# Catalana con OTROS).
#
# Dos cosas pueden faltar en una fila, y entonces se promedia (mezcla):
#   - la provincia (More in Common y Ateneo fuera de las uniprovinciales):
#     pesos de cada provincia de su CCAA en esa celda de la tabla de
#     postestratificación (como 03_celdas.R);
#   - seg (Ateneo no pregunta la probabilidad de votar; unas 230 entrevistas
#     de las otras casas no contestan): P(seguro | celda, provincia, casa) de
#     mrp_part.stan con las medias posteriores, desplazada en logit con
#     c_mezcla para pasar de "entre todos" a "entre los que dan partido"
#     (como 15_delta_seguro.R). Ateneo, con la casa media. Pesos fijos: su
#     incertidumbre no se propaga.
# Una fila sin provincia ni seg mezcla provincia x seg (máximo 2 x 9).
#
# Lee datos/ (solo lectura) y completo/datos/encuestas_completo.rds; escribe
# completo/datos/datos_voto_seg.rds.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/10_datos_voto_seg.R

library(dplyr)
library(tidyr)
library(posterior)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_completo <- file.path(dir_proy, "completo", "datos")
source(file.path(dir_proy, "00_partidos.R")) # partidos_lv, rec_lv, territorio, encuestas_con_aa

enc <- readRDS(file.path(dir_completo, "encuestas_completo.rds"))
tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))
lv_ii <- readRDS(file.path(dir_datos, "datos_stan.rds"))$lv # niveles del (II)
prov_lv <- levels(prov_ccaa$prov)
voto_lv <- c(setdiff(partidos_lv, "AC"), "BLANCO")
territorio_mod <- territorio[names(territorio) != "AC"]
seg_lv <- c("dudoso", "seguro")

# ---- 1. Entrevistas -------------------------------------------------------------

dat <- enc |>
  filter(
    !is.na(voto_sim),
    !is.na(sexo),
    !is.na(edad),
    !is.na(edu),
    !is.na(rec),
    !is.na(ccaa)
  ) |>
  mutate(
    voto = factor(
      if_else(voto_sim == "AC", "OTROS", as.character(voto_sim)),
      levels = voto_lv
    ),
    seg = case_when(
      is.na(vota_seguro) ~ NA_character_,
      vota_seguro == 1 ~ "seguro",
      TRUE ~ "dudoso"
    ),
    prov = factor(prov, levels = prov_lv)
  )
stopifnot(!anyNA(dat$voto), all(dat$mes %in% lv_ii$mes))
cat("entrevistas:", nrow(dat), "\n")
print(as.data.frame(dat |> count(casa, seg)))

celdas <- dat |>
  count(casa, estudio, mes, ccaa, prov, sexo, edad, edu, rec, seg, voto) |>
  pivot_wider(names_from = voto, values_from = n, values_fill = 0)
for (p in setdiff(voto_lv, names(celdas))) {
  celdas[[p]] <- 0L
}
celdas$y <- as.matrix(celdas[, voto_lv])
celdas$sin_aa <- !(as.character(celdas$estudio) %in% encuestas_con_aa)
stopifnot(all(celdas$AA[celdas$sin_aa] == 0))
cat("filas:", nrow(celdas), "\n")

# ---- 2. Pesos de provincia (copia de 09_datos_voto.R) ------------------------------

cuota <- as.data.frame.table(tabla, responseName = "N") |>
  left_join(prov_ccaa, by = "prov") |>
  group_by(ccaa, sexo, edad, edu, rec) |>
  mutate(w = N / sum(N)) |>
  ungroup()
n_prov <- apply(tabla, 1, sum)
cuota_ccaa <- tibble(
  prov = factor(names(n_prov), levels = prov_lv),
  N = as.numeric(n_prov)
) |>
  left_join(prov_ccaa, by = "prov") |>
  group_by(ccaa) |>
  mutate(w_ccaa = N / sum(N)) |>
  ungroup() |>
  select(prov, w_ccaa)

pesos_prov <- function(fila) {
  if (!is.na(celdas$prov[fila])) {
    return(tibble(prov = as.character(celdas$prov[fila]), w = 1))
  }
  wf <- cuota |>
    filter(
      ccaa == celdas$ccaa[fila],
      sexo == celdas$sexo[fila],
      edad == celdas$edad[fila],
      edu == celdas$edu[fila],
      rec == celdas$rec[fila]
    ) |>
    left_join(cuota_ccaa, by = "prov")
  if (!(all(is.finite(wf$w)) && sum(wf$w) > 0)) {
    wf$w <- wf$w_ccaa
  }
  tibble(prov = as.character(wf$prov), w = wf$w)
}

# ---- 3. Pesos de seg: mrp_part.stan del (II), medias posteriores -------------------

# c_mezcla: de P(seguro) entre todos a P(seguro) entre los que dan partido
# (voto + simpatía), en logit; media de las tres casas que preguntan
mezcla <- enc |>
  filter(!is.na(vota_seguro)) |>
  group_by(casa) |>
  summarise(
    todos = mean(vota_seguro),
    con_partido = mean(vota_seguro[!is.na(voto_sim)]),
    .groups = "drop"
  ) |>
  mutate(dif_logit = qlogis(con_partido) - qlogis(todos))
c_mezcla <- mean(mezcla$dif_logit)
cat("\n== % seguros, todos frente a los que dan partido (voto_sim) ==\n")
print(as.data.frame(mezcla), row.names = FALSE)
cat("c_mezcla:", round(c_mezcla, 3), "\n\n")

fp <- readRDS(file.path(dir_datos, "fit_separado_nuts_part.rds"))$fit
m <- colMeans(as_draws_matrix(fp$draws(
  c("alpha_t", "b_mujer_t", "a_casa_t", "a_mes_t", "a_edad_t", "a_edu_t",
    "a_rec_t", "a_ccaa_t", "a_prov_t")
)))
# a_casa_t solo lleva las casas que preguntan: las primeras de lv_ii$casa
n_casa_t <- sum(grepl("^a_casa_t\\[", names(m)))
g <- function(v, x, niveles) {
  m[paste0(v, "[", match(as.character(x), niveles), "]")]
}
# P(seguro | da partido) de la fila 'fila' en las provincias 'provs'
p_seguro <- function(fila, provs) {
  h <- match(as.character(celdas$casa[fila]), lv_ii$casa)
  ef_casa <- if (h <= n_casa_t) m[[paste0("a_casa_t[", h, "]")]] else 0 # casa media
  eta <- m[["alpha_t"]] + (celdas$sexo[fila] == "M") * m[["b_mujer_t"]] + ef_casa +
    g("a_mes_t", celdas$mes[fila], lv_ii$mes) +
    g("a_edad_t", celdas$edad[fila], lv_ii$edad) +
    g("a_edu_t", celdas$edu[fila], lv_ii$edu) +
    g("a_rec_t", celdas$rec[fila], lv_ii$rec) +
    g("a_ccaa_t", celdas$ccaa[fila], lv_ii$ccaa) +
    g("a_prov_t", provs, lv_ii$prov)
  stopifnot(!anyNA(eta))
  plogis(eta + c_mezcla)
}

# ---- 4. Combinaciones provincia x seg de cada fila ----------------------------------

conocida <- !is.na(celdas$prov) & !is.na(celdas$seg)
fila1 <- which(conocida)
fila2 <- which(!conocida)
combos <- lapply(fila2, function(f) {
  pp <- pesos_prov(f) |> filter(w > 0) # como usa = W > 0 en 06_stan_datos.R
  if (!is.na(celdas$seg[f])) {
    return(pp |> mutate(seg = celdas$seg[f]))
  }
  ps <- p_seguro(f, pp$prov)
  bind_rows(
    pp |> mutate(seg = "seguro", w = w * ps),
    pp |> mutate(seg = "dudoso", w = w * (1 - ps))
  )
})
len <- vapply(combos, nrow, integer(1))
ent <- bind_rows(combos)
stopifnot(
  all(abs(vapply(combos, function(cb) sum(cb$w), numeric(1)) - 1) < 1e-8),
  all(ent$w > 0)
)
cat(
  "filas con provincia y seg:", length(fila1),
  "  con mezcla:", length(fila2),
  "  entradas de mezcla:", nrow(ent),
  "  combinaciones máx:", max(len), "\n"
)

# ---- 5. Datos para Stan -------------------------------------------------------------

lv <- list(
  casa = lv_ii$casa,
  mes = lv_ii$mes,
  edad = levels(enc$edad),
  edu = levels(enc$edu),
  rec = rec_lv,
  ccaa = levels(prov_ccaa$ccaa),
  prov = prov_lv,
  seg = seg_lv,
  voto = voto_lv
)
idx <- function(x, niveles) {
  i <- match(as.character(x), niveles)
  stopifnot(!anyNA(i))
  i
}
off <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) {
    ifelse(celdas$ccaa %in% territorio_mod[[p]], 0, -20)
  } else {
    rep(0, nrow(celdas))
  }
})
stopifnot(all(celdas$y[off < 0] == 0))

datos <- list(
  prior_only = 0L,
  s_alpha = 1,
  s_b_mujer = 1,
  s_b_dud = 1,
  s_sd_rec = 2,
  s_sd_main = 0.5,
  s_sd_int = 0.25,
  s_sd_dud = 0.5,
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
  sin_aa = as.integer(celdas$sin_aa),
  j_aa = match("AA", voto_lv),
  N1 = length(fila1),
  fila1 = fila1,
  prov1 = idx(celdas$prov[fila1], lv$prov),
  seg1 = idx(celdas$seg[fila1], lv$seg),
  N2 = length(fila2),
  fila2 = fila2,
  ini = as.integer(cumsum(c(1L, len))[seq_along(len)]),
  len = len,
  E = nrow(ent),
  ent_prov = idx(ent$prov, lv$prov),
  ent_seg = idx(ent$seg, lv$seg),
  ent_w = ent$w
)

saveRDS(
  list(datos = datos, lv = lv, c_mezcla = c_mezcla, mezcla = mezcla),
  file.path(dir_completo, "datos_voto_seg.rds")
)
cat("Escrito", file.path(dir_completo, "datos_voto_seg.rds"), "\n")
