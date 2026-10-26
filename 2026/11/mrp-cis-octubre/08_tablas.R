# Tablas de comparación, para España y por comunidad: bruto, pesos del CIS,
# MRP con intervalo al 90 % y, como referencia, la estimación publicada por
# el CIS (solo España) o el 23-J (comunidades). Lee lo que guarda 05_mrp.R.

library(dplyr)
library(tidyr)
library(readr)

dir_datos <- "2026/11/mrp-cis-octubre/datos"
dir_pob   <- "2026/10/mrp-raking/datos"   # población: la misma que en el post de octubre
source("2026/11/mrp-cis-octubre/00_partidos.R")   # partidos_lv
voto_lv <- partidos_lv

est <- readRDS(file.path(dir_datos, "estimaciones.rds"))
tg  <- readRDS(file.path(dir_pob, "targets.rds"))
dat <- est$dat   # la muestra con la que se ajusta el MRP

# Porcentaje de cada partido dentro de cada grupo, con o sin pesos
reparto <- function(d, peso = NULL) {
  d$w <- if (is.null(peso)) 1 else d[[peso]]
  d %>%
    group_by(ccaa, partido = as.character(voto)) %>%
    summarise(w = sum(w), .groups = "drop_last") %>%
    mutate(pct = 100 * w / sum(w)) %>%
    ungroup() %>%
    select(-w)
}

resumen_draws <- function(m) {
  tibble(partido = colnames(m),
         mrp = 100 * colMeans(m),
         q05 = 100 * apply(m, 2, quantile, 0.05),
         q95 = 100 * apply(m, 2, quantile, 0.95))
}

# ---------------------------------------------------------------------------
# 1. España
# ---------------------------------------------------------------------------

tabla_nac <- tibble(partido = voto_lv) %>%
  left_join(reparto(mutate(dat, ccaa = "España")) %>% select(partido, bruto = pct),
            by = "partido") %>%
  left_join(reparto(mutate(dat, ccaa = "España"), "peso") %>% select(partido, pesos_cis = pct),
            by = "partido") %>%
  left_join(resumen_draws(est$mrp_nac), by = "partido") %>%
  left_join(est$comparacion %>% select(partido, cis_publicado = cis), by = "partido") %>%
  mutate(across(c(bruto, pesos_cis), ~replace_na(.x, 0)))

cat("== España (% sobre voto válido) ==\n")
print(as.data.frame(tabla_nac %>% mutate(across(-partido, ~round(.x, 1)))),
      row.names = FALSE)

# ---------------------------------------------------------------------------
# 2. Comunidades
# ---------------------------------------------------------------------------
# PESO es la ponderación nacional del CIS. Dentro de cada comunidad solo
# importan los pesos relativos, que incluyen el ajuste por estudios.

res_23j <- tg$rec_ccaa %>%
  filter(!rec %in% c("ABST", "NO_PODIA")) %>%
  group_by(ccaa) %>%
  mutate(pct_23j = 100 * N / sum(N)) %>%
  ungroup() %>%
  transmute(ccaa = as.character(ccaa), partido = as.character(rec), pct_23j)

mrp_ccaa <- est$mrp_ccaa   # draws x ccaa x partido
est_ccaa <- bind_rows(lapply(dimnames(mrp_ccaa)[[2]], function(cc) {
  resumen_draws(mrp_ccaa[, cc, ]) %>% mutate(ccaa = cc)
}))

entrevistas <- dat %>% count(ccaa, name = "entrevistas") %>%
  mutate(ccaa = as.character(ccaa))

tabla_ccaa <- est_ccaa %>%
  left_join(reparto(mutate(dat, ccaa = as.character(ccaa))) %>% rename(bruto = pct),
            by = c("ccaa", "partido")) %>%
  left_join(reparto(mutate(dat, ccaa = as.character(ccaa)), "peso") %>% rename(pesos_cis = pct),
            by = c("ccaa", "partido")) %>%
  left_join(res_23j, by = c("ccaa", "partido")) %>%
  left_join(entrevistas, by = "ccaa") %>%
  mutate(
    across(c(bruto, pesos_cis, pct_23j), ~replace_na(.x, 0)),
    pct_23j = if_else(partido == "AC", NA_real_, pct_23j)   # no se presentó
  ) %>%
  # Solo los partidos que pintan algo en cada comunidad
  filter(mrp >= 1 | replace_na(pct_23j, 0) >= 1 | bruto >= 1) %>%
  mutate(ccaa = factor(ccaa, levels = dimnames(mrp_ccaa)[[2]]),
         partido = factor(partido, levels = voto_lv)) %>%
  arrange(ccaa, partido) %>%
  select(ccaa, entrevistas, partido, bruto, pesos_cis, mrp, q05, q95, pct_23j)

cat("\n== Comunidades (% sobre voto válido; 23-J sobre votos emitidos) ==\n")
for (cc in levels(tabla_ccaa$ccaa)) {
  t <- tabla_ccaa %>% filter(ccaa == cc)
  cat("\n--", cc, "(", t$entrevistas[1], "entrevistas ) --\n")
  print(as.data.frame(t %>% select(-ccaa, -entrevistas) %>%
                        mutate(across(-partido, ~round(.x, 1)))),
        row.names = FALSE)
}

write_csv(tabla_nac, file.path(dir_datos, "tabla_nacional_3582.csv"))
write_csv(tabla_ccaa, file.path(dir_datos, "tabla_ccaa_3582.csv"))
cat("\nguardadas tabla_nacional_3582.csv y tabla_ccaa_3582.csv\n")
