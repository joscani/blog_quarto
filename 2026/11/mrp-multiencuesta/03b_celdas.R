# Celdas para los modelos: agrega las entrevistas de datos/encuestas.rds.
#   datos/celdas_brms.rds  votos por celda (voto válido), para 04_brms.R y 05_stan.R
#   datos/celdas_part.rds  participación por celda ("10, seguro que voto"), para 05_stan.R
# More in Common sin provincia: pesos de cada provincia de su CCAA (p1..p9, w1..w9).
#
#   Rscript 2026/11/mrp-multiencuesta/03b_celdas.R

library(dplyr)
library(tidyr)

dir_datos <- "2026/11/mrp-multiencuesta/datos"
source("2026/11/mrp-multiencuesta/00_partidos.R")   # partidos_lv, rec_lv, territorio



enc       <- readRDS(file.path(dir_datos, "encuestas.rds"))
tabla     <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))

# Aliança Catalana solo la pregunta el CIS: en las otras casas está en "Otro".
# En el modelo conjunto va con OTROS.
voto_lv <- setdiff(partidos_lv, "AC")
territorio_mod <- territorio[names(territorio) != "AC"]
prov_lv <- levels(prov_ccaa$prov)
K <- 9   # máximo de provincias por CCAA (Castilla y León)

# ---------------------------------------------------------------------------
# 1. Datos agregados por celda
# ---------------------------------------------------------------------------

dat <- enc %>%
  filter(!is.na(voto), !is.na(sexo), !is.na(edad), !is.na(edu), !is.na(rec)) %>%
  mutate(voto = factor(ifelse(voto == "AC", "OTROS", as.character(voto)), levels = voto_lv),
         prov = factor(prov, levels = prov_lv))

cat("\n== entrevistas usadas ==\n")
print(dat %>% count(casa, estudio))
cat("total:", nrow(dat), "de", nrow(enc), "\n")

celdas <- dat %>%
  count(casa, mes, ccaa, prov, sexo, edad, edu, rec, voto) %>%
  pivot_wider(names_from = voto, values_from = n, values_fill = 0)
for (p in setdiff(voto_lv, names(celdas))) celdas[[p]] <- 0L
celdas$y <- as.matrix(celdas[, voto_lv])
celdas$n <- rowSums(celdas$y)
cat("\nfilas agregadas:", nrow(celdas), " (", round(nrow(celdas) / nrow(dat), 2), "por entrevista )\n")
print(celdas %>% count(casa, prov_conocida = !is.na(prov)))

# ---------------------------------------------------------------------------
# 2. Pesos de provincia para mm()
# ---------------------------------------------------------------------------
# Provincia conocida: peso 1 en su provincia. Desconocida (More in Common):
# peso de cada provincia de su CCAA en esa misma celda sexo x edad x estudios
# x recuerdo, según la tabla de postestratificación.

cuota <- as.data.frame.table(tabla, responseName = "N") %>%
  left_join(prov_ccaa, by = "prov") %>%
  group_by(ccaa, sexo, edad, edu, rec) %>%
  mutate(w = N / sum(N)) %>% ungroup()

# Si una celda de la CCAA no tiene población (no debería pasar con datos
# observados), se usa el peso de la provincia en el electorado de la CCAA
n_prov <- apply(tabla, 1, sum)
cuota_ccaa <- tibble(prov = factor(names(n_prov), levels = prov_lv), N = as.numeric(n_prov)) %>%
  left_join(prov_ccaa, by = "prov") %>%
  group_by(ccaa) %>% mutate(w_ccaa = N / sum(N)) %>% ungroup() %>% select(prov, w_ccaa)

provs_de <- split(as.character(prov_ccaa$prov), prov_ccaa$ccaa)

pesos_mm <- function(d) {
  P <- matrix(NA_character_, nrow(d), K)
  W <- matrix(0, nrow(d), K)
  conocida <- !is.na(d$prov)
  P[conocida, ] <- as.character(d$prov[conocida])
  W[conocida, 1] <- 1
  idx <- which(!conocida)
  if (length(idx) > 0) {
    clave <- d[idx, ] %>% mutate(fila = idx) %>%
      select(fila, ccaa, sexo, edad, edu, rec)
    w_largo <- clave %>%
      left_join(cuota %>% select(ccaa, sexo, edad, edu, rec, prov, w),
                by = c("ccaa", "sexo", "edad", "edu", "rec"), relationship = "many-to-many") %>%
      left_join(cuota_ccaa, by = "prov") %>%
      group_by(fila) %>%
      mutate(w = if (all(is.finite(w)) && sum(w) > 0) w else w_ccaa) %>%
      ungroup()
    for (f in idx) {
      wf <- w_largo %>% filter(fila == f)
      k <- nrow(wf)
      P[f, ] <- c(as.character(wf$prov), rep(as.character(wf$prov[1]), K - k))
      W[f, ] <- c(wf$w, rep(0, K - k))
    }
  }
  stopifnot(!anyNA(P), all(abs(rowSums(W) - 1) < 1e-8))
  for (k in seq_len(K)) {
    d[[paste0("p", k)]] <- factor(P[, k], levels = prov_lv)
    d[[paste0("w", k)]] <- W[, k]
  }
  d
}

celdas <- pesos_mm(celdas)

cat("\n== ejemplo: una fila de More in Common de Castilla y León ==\n")
ej <- celdas %>% filter(casa == "More in Common", ccaa == "Castilla y León") %>% slice(1)
print(data.frame(prov = sapply(1:K, function(k) as.character(ej[[paste0("p", k)]])),
                 w = round(sapply(1:K, function(k) ej[[paste0("w", k)]]), 3)))

con_offsets <- function(d) {
  for (p in names(territorio_mod)) {
    d[[paste0("off_", p)]] <- ifelse(d$ccaa %in% territorio_mod[[p]], 0, -20)
  }
  d
}
celdas <- con_offsets(celdas)

# Casa como efecto aleatorio, (1 | casa): cada casa se desvía de una casa
# "típica" y las desviaciones se encogen hacia ella. (La primera versión usaba
# un efecto fijo con contr.sum(3); el usuario prefiere este, más didáctico.)
celdas$casa <- factor(celdas$casa, levels = c("CIS", "40dB", "More in Common", "Ateneo del Dato"))
celdas$mes <- factor(celdas$mes)

saveRDS(celdas, file.path(dir_datos, "celdas_brms.rds"))

# Celdas de participación: todas las entrevistas con probabilidad de votar,
# también las que no dan voto válido. k = cuántas dicen "10, seguro que voto".
celdas_part <- enc %>%
  filter(!is.na(vota_seguro), !is.na(sexo), !is.na(edad), !is.na(edu), !is.na(rec)) %>%
  mutate(prov = factor(prov, levels = prov_lv)) %>%
  group_by(casa, mes, ccaa, prov, sexo, edad, edu, rec) %>%
  summarise(n = n(), k = sum(vota_seguro), .groups = "drop") %>%
  pesos_mm()
celdas_part$casa <- factor(celdas_part$casa, levels = levels(celdas$casa))
celdas_part$mes <- factor(celdas_part$mes, levels = levels(celdas$mes))
cat("\n== celdas de participación ==\n")
print(celdas_part %>% group_by(casa) %>%
        summarise(entrevistas = sum(n), pct_seguro = round(100 * sum(k) / sum(n), 1), filas = n()))
saveRDS(celdas_part, file.path(dir_datos, "celdas_part.rds"))
