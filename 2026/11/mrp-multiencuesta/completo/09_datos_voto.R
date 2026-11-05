# Modelo completo, parte 3: datos del modelo de voto de las 4 casas para RTMB
# (completo/rtmb/mrp_voto_completo_rtmb.R).
#
# Celdas casa x estudio x mes x ccaa x prov x sexo x edad x edu x rec x
# ideología x nivel con el recuento de cada partido (voto + simpatía, BLANCO
# aparte, Aliança Catalana con OTROS). Solo entrevistas con partido e
# ideología (como las partes 1 y 2).
#
# Dos cosas pueden faltar en una fila, y entonces se promedia (mezcla):
#   - la provincia (More in Common y Ateneo fuera de las uniprovinciales):
#     pesos de cada provincia de su CCAA en esa celda de la tabla de
#     postestratificación (pesos_mm de 03_celdas.R);
#   - el nivel de participación (Ateneo no lo pregunta; unas pocas
#     entrevistas de las otras casas no contestan): pesos = P(nivel) de la
#     parte 2 (completo/datos/fit_part.rds), con las medias posteriores y los
#     umbrales de su casa (Ateneo, los de la casa media). Pesos fijos: su
#     incertidumbre no se propaga.
# Una fila sin provincia ni nivel mezcla provincia x nivel.
#
# Lee datos/ (solo lectura) y completo/datos/; escribe
# completo/datos/datos_voto.rds.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/09_datos_voto.R

library(dplyr)
library(tidyr)
library(posterior)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_completo <- file.path(dir_proy, "completo", "datos")
source(file.path(dir_proy, "00_partidos.R")) # partidos_lv, territorio, encuestas_con_aa

enc <- readRDS(file.path(dir_completo, "encuestas_completo.rds"))
tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))
part <- readRDS(file.path(dir_completo, "datos_part.rds"))
prov_lv <- levels(prov_ccaa$prov)
voto_lv <- c(setdiff(partidos_lv, "AC"), "BLANCO")
territorio_mod <- territorio[names(territorio) != "AC"]
niveles_part <- c("improbable", "probable", "seguro")
K_prov <- 9

# ---- 1. Entrevistas -------------------------------------------------------------

dat <- enc |>
  filter(
    !is.na(voto_sim),
    !is.na(ideol5),
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
    nivel = case_when(
      is.na(prob_voto) ~ NA_character_,
      prob_voto <= 4 ~ "improbable",
      prob_voto <= 9 ~ "probable",
      TRUE ~ "seguro"
    ),
    prov = factor(prov, levels = prov_lv)
  )
stopifnot(!anyNA(dat$voto))
cat("entrevistas:", nrow(dat), "\n")
print(dat |> count(casa, sin_nivel = is.na(nivel), sin_prov = is.na(prov)))

celdas <- dat |>
  count(casa, estudio, mes, ccaa, prov, sexo, edad, edu, rec, ideol5, nivel, voto) |>
  pivot_wider(names_from = voto, values_from = n, values_fill = 0)
for (p in setdiff(voto_lv, names(celdas))) {
  celdas[[p]] <- 0L
}
celdas$y <- as.matrix(celdas[, voto_lv])
celdas$sin_aa <- !(as.character(celdas$estudio) %in% encuestas_con_aa)
stopifnot(all(celdas$AA[celdas$sin_aa] == 0))
cat("filas:", nrow(celdas), "\n")

# ---- 2. Pesos de provincia (copia de 03_celdas.R) ---------------------------------

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

# ---- 3. Pesos de nivel (parte 2, medias posteriores) ------------------------------

fp <- readRDS(file.path(dir_completo, "fit_part.rds"))
m <- colMeans(as_draws_matrix(fp$draws(
  c("c", "c_media", "b_mujer", "a_ideol", "a_mes", "a_edad", "a_edu", "a_rec", "a_ccaa")
)))
lvp <- part$lv
pesos_nivel <- function(fila) {
  if (!is.na(celdas$nivel[fila])) {
    return(tibble(nivel = celdas$nivel[fila], w = 1))
  }
  g <- function(v, x, niveles) m[[paste0(v, "[", match(as.character(x), niveles), "]")]]
  mes <- as.character(celdas$mes[fila])
  eta <- (celdas$sexo[fila] == "M") * m[["b_mujer"]] +
    g("a_ideol", celdas$ideol5[fila], lvp$ideol) +
    (if (mes %in% lvp$mes) g("a_mes", mes, lvp$mes) else 0) +
    g("a_edad", celdas$edad[fila], lvp$edad) +
    g("a_edu", celdas$edu[fila], lvp$edu) +
    g("a_rec", celdas$rec[fila], lvp$rec) +
    g("a_ccaa", celdas$ccaa[fila], lvp$ccaa)
  h <- match(as.character(celdas$casa[fila]), lvp$casa)
  cc <- if (is.na(h)) {
    c(m[["c_media[1]"]], m[["c_media[2]"]])
  } else {
    c(m[[paste0("c[", h, ",1]")]], m[[paste0("c[", h, ",2]")]])
  }
  p <- c(1 - plogis(eta - cc[1]), plogis(eta - cc[1]) - plogis(eta - cc[2]), plogis(eta - cc[2]))
  tibble(nivel = niveles_part, w = p)
}

# ---- 4. Combinaciones provincia x nivel de cada fila -------------------------------

conocida <- !is.na(celdas$prov) & !is.na(celdas$nivel)
fila1 <- which(conocida)
fila2 <- which(!conocida)
combos <- lapply(fila2, function(f) {
  tidyr::crossing(
    pesos_prov(f) |> rename(w_prov = w),
    pesos_nivel(f) |> rename(w_niv = w)
  ) |>
    mutate(w = w_prov * w_niv)
})
M <- max(sapply(combos, nrow))
mix_prov <- matrix(1L, length(fila2), M)
mix_niv <- matrix(1L, length(fila2), M)
mix_w <- matrix(0, length(fila2), M)
for (i in seq_along(fila2)) {
  cb <- combos[[i]]
  k <- seq_len(nrow(cb))
  mix_prov[i, k] <- match(cb$prov, prov_lv)
  mix_niv[i, k] <- match(cb$nivel, niveles_part)
  mix_w[i, k] <- cb$w
}
stopifnot(!anyNA(mix_prov), !anyNA(mix_niv), all(abs(rowSums(mix_w) - 1) < 1e-8))
cat(
  "filas con provincia y nivel:",
  length(fila1),
  "  con mezcla:",
  length(fila2),
  "  combinaciones máx:",
  M,
  "\n"
)

# ---- 5. Datos para RTMB -----------------------------------------------------------

lv <- list(
  casa = c("CIS", "40dB", "More in Common", "Ateneo del Dato"),
  mes = sort(unique(as.character(celdas$mes))),
  edad = levels(enc$edad),
  edu = levels(enc$edu),
  rec = rec_lv,
  ccaa = levels(prov_ccaa$ccaa),
  prov = prov_lv,
  ideol = levels(enc$ideol5),
  nivel = niveles_part,
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
  s_alpha = 1,
  s_b_mujer = 1,
  s_sd_rec = 2,
  s_sd_main = 0.5,
  s_sd_int = 0.25,
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
  n_ideol = length(lv$ideol),
  n_nivel = length(lv$nivel),
  casa = idx(celdas$casa, lv$casa),
  mes = idx(celdas$mes, lv$mes),
  edad = idx(celdas$edad, lv$edad),
  edu = idx(celdas$edu, lv$edu),
  rec = idx(celdas$rec, lv$rec),
  ccaa = idx(celdas$ccaa, lv$ccaa),
  ideol = idx(celdas$ideol5, lv$ideol),
  off = unname(off),
  sin_aa = as.integer(celdas$sin_aa),
  j_aa = match("AA", voto_lv),
  fila1 = fila1,
  prov1 = idx(celdas$prov[fila1], lv$prov),
  nivel1 = idx(celdas$nivel[fila1], lv$nivel),
  fila2 = fila2,
  mix_prov = mix_prov,
  mix_niv = mix_niv,
  mix_w = mix_w
)

saveRDS(list(datos = datos, lv = lv), file.path(dir_completo, "datos_voto.rds"))
cat("Escrito", file.path(dir_completo, "datos_voto.rds"), "\n")
