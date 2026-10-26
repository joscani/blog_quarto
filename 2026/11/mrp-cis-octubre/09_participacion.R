# La sección "¿Quién va a votar?" del post de octubre, con el estudio 3582.
# Usa los modelos que ajusta 05_mrp.R (los lee del disco, no los reajusta).

library(dplyr)
library(tidyr)
library(readr)
library(brms)

options(mc.cores = 4, brms.backend = "cmdstanr")
dir_datos <- "2026/11/mrp-cis-octubre/datos"
dir_pob   <- "2026/10/mrp-raking/datos"   # población: la misma que en el post de octubre

cis   <- readRDS(file.path(dir_datos, "cis_imputado.rds"))
tabla <- readRDS(file.path(dir_datos, "tabla_postestratificacion.rds"))
tg    <- readRDS(file.path(dir_pob, "targets.rds"))
mod      <- readRDS(file.path(dir_datos, "mod_mrp.rds"))
mod_part <- readRDS(file.path(dir_datos, "mod_participacion.rds"))

source("2026/11/mrp-cis-octubre/00_partidos.R")   # partidos_lv, rec_lv, territorio
voto_lv <- partidos_lv
edu_lv  <- c("baja", "media", "alta")

con_offsets <- function(d) {
  for (p in names(territorio)) {
    d[[paste0("off_", p)]] <- ifelse(d$ccaa %in% territorio[[p]], 0, -20)
  }
  d
}

# La tabla de postestratificación y los mismos 500 draws que en 05_mrp.R
post <- as.data.frame.table(tabla, responseName = "N") %>%
  filter(N > 1e-6) %>%
  mutate(edu = factor(edu, levels = edu_lv),
         rec = factor(rec, levels = rec_lv),
         n = 1) %>%
  con_offsets()

set.seed(2026)
ep <- posterior_epred(mod, newdata = post, allow_new_levels = TRUE, ndraws = 500)
mascara <- mascara_territorio(post$ccaa, voto_lv)
for (d in seq_len(dim(ep)[1])) {
  ep[d, , ] <- ep[d, , ] * mascara
  ep[d, , ] <- ep[d, , ] / rowSums(ep[d, , ])
}

resumen <- function(m) {
  tibble(partido = voto_lv,
         mrp = 100 * colMeans(m),
         q05 = 100 * apply(m, 2, quantile, 0.05),
         q95 = 100 * apply(m, 2, quantile, 0.95))
}
redondea <- function(d) as.data.frame(d %>% mutate(across(where(is.numeric), ~round(.x, 1))))

# ---------------------------------------------------------------------------
# 1. Si votara todo el mundo
# ---------------------------------------------------------------------------

mrp_draws_todos <- t(apply(ep, 1, function(p) colSums(p * post$N) / sum(post$N)))
est_todos <- resumen(mrp_draws_todos)

# Cuántos puntos de PP y VOX aporta cada grupo de recuerdo en Cataluña
theta_media <- apply(ep, c(2, 3), mean)
i_cat <- post$ccaa == "Cataluña"

diag_cat <- post[i_cat, ] %>%
  mutate(pp = theta_media[i_cat, "PP"] * N,
         vox = theta_media[i_cat, "VOX"] * N) %>%
  group_by(rec) %>%
  summarise(N = sum(N), pp = sum(pp), vox = sum(vox)) %>%
  mutate(peso = 100 * N / sum(N),
         puntos_pp = 100 * pp / sum(N),
         puntos_vox = 100 * vox / sum(N)) %>%
  filter(peso > 0.5) %>%
  select(rec, peso, puntos_pp, puntos_vox)

cat("== Cataluña, si votara todo el mundo: % del electorado y puntos que aporta cada recuerdo ==\n")
print(redondea(diag_cat), row.names = FALSE)

# ---------------------------------------------------------------------------
# 2. Quién dice que votará seguro (PROBVOTO = 10)
# ---------------------------------------------------------------------------

prob_voto <- cis %>%
  filter(!is.na(vota_seguro)) %>%
  group_by(rec) %>%
  summarise(entrevistas = n(), seguro = 100 * mean(vota_seguro))

cat("\n== % que dice que votará con toda seguridad, por recuerdo ==\n")
print(redondea(prob_voto), row.names = FALSE)
cat("entrevistas en el modelo de participación:", sum(prob_voto$entrevistas), "\n")

# ---------------------------------------------------------------------------
# 3. Calibración a la participación del 23-J
# ---------------------------------------------------------------------------

set.seed(2026)
t_draws <- posterior_epred(mod_part, newdata = post, ndraws = 500)

rec_pob <- tg$rec_ccaa %>% group_by(rec) %>% summarise(N = sum(N))
participacion_23j <- with(rec_pob,
  sum(N[!rec %in% c("ABST", "NO_PODIA")]) / sum(N[rec != "NO_PODIA"]))

calibra <- function(p, N, objetivo) {
  f <- function(delta) sum(N * plogis(qlogis(p) + delta)) / sum(N) - objetivo
  plogis(qlogis(p) + uniroot(f, c(-10, 10))$root)
}
t_cal <- t(apply(t_draws, 1, calibra, N = post$N, objetivo = participacion_23j))

cat("\n== participación (%) ==\n")
print(round(100 * c(
  objetivo_23j = participacion_23j,
  sin_calibrar = sum(colMeans(t_draws) * post$N) / sum(post$N)
), 1))

part_rec <- post %>%
  mutate(t = colMeans(t_cal)) %>%
  group_by(rec) %>%
  summarise(participacion = 100 * sum(t * N) / sum(N))

cat("\n== participación calibrada por recuerdo (%) ==\n")
print(redondea(part_rec), row.names = FALSE)

# ---------------------------------------------------------------------------
# 4. Con participación
# ---------------------------------------------------------------------------

w <- sweep(t_cal, 2, post$N, "*")   # draws x celdas: N_j * t_j
mrp_draws <- t(sapply(seq_len(nrow(w)), function(d) {
  colSums(ep[d, , ] * w[d, ]) / sum(w[d, ])
}))
est_mrp <- resumen(mrp_draws)

todos_vs_votan <- est_todos %>%
  select(partido, todos = mrp) %>%
  left_join(est_mrp %>% select(partido, votan = mrp), by = "partido") %>%
  mutate(diferencia = votan - todos)

cat("\n== España (% sobre voto válido): si votara todo el mundo vs con participación ==\n")
print(redondea(todos_vs_votan), row.names = FALSE)

saveRDS(list(diag_cat = diag_cat, prob_voto = prob_voto,
             participacion_23j = participacion_23j, part_rec = part_rec,
             est_todos = est_todos, todos_vs_votan = todos_vs_votan),
        file.path(dir_datos, "participacion.rds"))
cat("\nguardado participacion.rds\n")
