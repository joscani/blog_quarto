# Modelo multinivel sobre la encuesta, postestratificación con la tabla
# rakeada, y comparación con el raking clásico de pesos.

library(dplyr)
library(tidyr)
library(haven)
library(brms)
library(survey)

options(mc.cores = 4, brms.backend = "cmdstanr")
dir_datos <- "2026/10/mrp-raking/datos"

cis   <- readRDS(file.path(dir_datos, "cis_imputado.rds"))
tabla <- readRDS(file.path(dir_datos, "tabla_postestratificacion.rds"))
tg    <- readRDS(file.path(dir_datos, "targets.rds"))

source("2026/10/mrp-raking/00_partidos.R")   # partidos_lv, rec_lv, territorio
voto_lv <- partidos_lv
edu_lv  <- c("baja", "media", "alta")

# La estimación del CIS para este mismo estudio, sobre voto válido, para
# tenerla al lado. Está en datos/3577_Estimacion.pdf.
cis_publicado <- tibble::tibble(
  partido = voto_lv,
  # SUMAR incluye a Podemos, como en 00_partidos.R.
  # OTROS = Se Acabó la Fiesta + UPN + otros partidos + en blanco
  cis = c(PP = 25.5, PSOE = 31.0, VOX = 16.6, SUMAR = 5.7 + 3.7,
          ERC = 2.5, JUNTS = 0.7, AC = NA, BILDU = 1.3, PNV = 0.7, BNG = 0.7, CCA = 0.2,
          OTROS = 1.8 + 0.1 + 8.6 + 1.0)[voto_lv]
)

cat("== intención recodificada (voto válido) ==\n")
print(table(cis$voto, useNA = "ifany"))

dat <- cis %>% filter(!is.na(voto), !is.na(edu), !is.na(edad), !is.na(rec))
cat("\nmuestra utilizable:", nrow(dat), "de", nrow(cis), "\n")

# ---------------------------------------------------------------------------
# 2. Modelo multinivel
# ---------------------------------------------------------------------------
# Se agrega por celda y se usa la multinomial con trials(): hay muchas
# combinaciones repetidas, así que no tiene sentido tratar cada entrevista
# por separado.
#
# Los partidos regionales llevan un offset: 0 donde se presentan y -20 donde
# no. Así su probabilidad fuera de su comunidad es prácticamente 0 y esos
# ceros estructurales no arrastran hacia abajo el efecto de su comunidad.

celdas_enc <- dat %>%
  count(ccaa, sexo, edad, edu, rec, voto) %>%
  pivot_wider(names_from = voto, values_from = n, values_fill = 0)
celdas_enc$y <- as.matrix(celdas_enc[, voto_lv])
celdas_enc$n <- rowSums(celdas_enc$y)

con_offsets <- function(d) {
  for (p in names(territorio)) {
    d[[paste0("off_", p)]] <- ifelse(d$ccaa %in% territorio[[p]], 0, -20)
  }
  d
}
celdas_enc <- con_offsets(celdas_enc)

cat("celdas con datos:", nrow(celdas_enc), "\n")

priors <- Reduce(`+`, lapply(paste0("mu", voto_lv[-1]), function(dp) {
  prior_string("normal(0, 1.5)", class = "Intercept", dpar = dp) +
    prior_string("normal(0, 1)", class = "b", dpar = dp) +
    prior_string("exponential(1)", class = "sd", dpar = dp)
}))

f_comun <- "sexo + (1 | edad) + (1 | edu) + (1 | rec) + (1 | ccaa) + (1 | ccaa:rec)"
f_partidos <- lapply(voto_lv[-1], function(p) {
  offset <- if (p %in% names(territorio)) paste0(" + offset(off_", p, ")") else ""
  as.formula(paste0("mu", p, " ~ ", f_comun, offset))
})
formula_mrp <- do.call(bf, c(list(as.formula(paste("y | trials(n) ~", f_comun))),
                             f_partidos))

mod <- brm(
  formula_mrp,
  data = celdas_enc, family = multinomial(), prior = priors,
  chains = 4, iter = 2000, refresh = 0, silent = 2, seed = 2026,
  control = list(adapt_delta = 0.95),
  file = file.path(dir_datos, "mod_mrp")
)
print(mod, digits = 2)

# ---------------------------------------------------------------------------
# 3. Postestratificación
# ---------------------------------------------------------------------------

post <- as.data.frame.table(tabla, responseName = "N") %>%
  filter(N > 1e-6) %>%
  mutate(edu = factor(edu, levels = edu_lv),
         rec = factor(rec, levels = rec_lv),
         n = 1) %>%
  con_offsets()

cat("\nceldas de la tabla con población:", nrow(post), "\n")

set.seed(2026)   # los mismos 500 draws que en el post
ep <- posterior_epred(mod, newdata = post, allow_new_levels = TRUE, ndraws = 500)
# ep: draws x celdas x categorías

# Ceros estructurales exactos: el offset deja casi a 0 a los partidos
# regionales fuera de su territorio, pero no siempre a 0 exacto (fuera de su
# comunidad el efecto de ccaa solo lo informa la prior). Se anulan y se
# renormaliza cada celda.
mascara <- mascara_territorio(post$ccaa, voto_lv)
for (d in seq_len(dim(ep)[1])) {
  ep[d, , ] <- ep[d, , ] * mascara
  ep[d, , ] <- ep[d, , ] / rowSums(ep[d, , ])
}

# Participación: binomial multinivel sobre "seguro que vota" (PROBVOTO = 10),
# calibrado para que la participación total sea la del 23-J (residentes).
celdas_part <- cis %>%
  filter(!is.na(vota_seguro)) %>%
  group_by(ccaa, sexo, edad, edu, rec) %>%
  summarise(vota = sum(vota_seguro), n = n(), .groups = "drop")

mod_part <- brm(
  vota | trials(n) ~ sexo + (1 | edad) + (1 | edu) + (1 | rec) + (1 | ccaa),
  data = celdas_part, family = binomial(),
  prior = prior(normal(0, 1.5), class = "Intercept") +
    prior(normal(0, 1), class = "b") + prior(exponential(1), class = "sd"),
  chains = 4, iter = 2000, refresh = 0, silent = 2, seed = 2026,
  file = file.path(dir_datos, "mod_participacion")
)

set.seed(2026)   # los mismos draws que en el post
t_draws <- posterior_epred(mod_part, newdata = post, ndraws = 500)

rec_pob <- tg$rec_ccaa %>% group_by(rec) %>% summarise(N = sum(N))
participacion_23j <- with(rec_pob,
  sum(N[!rec %in% c("ABST", "NO_PODIA")]) / sum(N[rec != "NO_PODIA"]))

calibra <- function(p, N, objetivo) {
  f <- function(delta) sum(N * plogis(qlogis(p) + delta)) / sum(N) - objetivo
  plogis(qlogis(p) + uniroot(f, c(-10, 10))$root)
}
t_cal <- t(apply(t_draws, 1, calibra, N = post$N, objetivo = participacion_23j))
W <- sweep(t_cal, 2, post$N, "*")   # draws x celdas: N_j * t_j

cat("\nparticipación sin calibrar:",
    round(sum(colMeans(t_draws) * post$N) / sum(post$N), 3),
    " objetivo:", round(participacion_23j, 3), "\n")

agrega <- function(ep, W, grupo = NULL) {
  # W: pesos de cada celda en cada draw. Devuelve draws x categorías
  # (o draws x grupo x categorías).
  una <- function(idx) {
    t(sapply(seq_len(dim(ep)[1]), function(d) {
      colSums(ep[d, idx, ] * W[d, idx]) / sum(W[d, idx])
    }))
  }
  if (is.null(grupo)) return(una(seq_len(dim(ep)[2])))
  lv <- levels(grupo)
  out <- array(NA_real_, c(dim(ep)[1], length(lv), dim(ep)[3]),
               dimnames = list(NULL, lv, dimnames(ep)[[3]]))
  for (g in lv) out[, g, ] <- una(which(grupo == g))
  out
}

mrp_nac <- agrega(ep, W)
est_mrp <- tibble(
  partido = voto_lv,
  mrp = colMeans(mrp_nac) * 100,
  q05 = apply(mrp_nac, 2, quantile, 0.05) * 100,
  q95 = apply(mrp_nac, 2, quantile, 0.95) * 100
)

# ---------------------------------------------------------------------------
# 4. Comparación: bruto, ponderación del CIS y raking clásico de pesos
# ---------------------------------------------------------------------------

dis_cis <- svydesign(ids = ~1, weights = ~peso, data = dat)

# Raking clásico: mismos targets, pero sobre los pesos en vez de sobre la tabla
pop_sexo_edad_ccaa <- tg$sexo_edad_ccaa %>%
  group_by(ccaa, sexo, edad) %>% summarise(Freq = sum(N), .groups = "drop") %>%
  as.data.frame()
pop_edu_sexo_ccaa <- tg$edu_sexo_ccaa %>%
  group_by(ccaa, sexo, edu) %>% summarise(Freq = sum(N), .groups = "drop") %>%
  as.data.frame()
pop_rec_ccaa <- tg$rec_ccaa %>%
  group_by(ccaa, rec) %>% summarise(Freq = sum(N), .groups = "drop") %>%
  as.data.frame()

# Aquí aparece la diferencia de fondo entre los dos enfoques. El raking de
# pesos solo puede repartir peso entre celdas que existan en la muestra: si
# no hay ninguna mujer de 18-20 años de Aragón entrevistada, esa población no
# tiene a quién asignarse y desaparece de la estimación. Hay que pedirle
# partial = TRUE para que siga adelante ignorando esos estratos.
# El MRP no tiene ese problema: el modelo predice también en las celdas
# vacías, tomando prestada información del resto.

estratos_vacios <- function(d, vars, pop) {
  obs <- d %>% count(across(all_of(vars)))
  pop %>% anti_join(obs, by = vars) %>% summarise(estratos = n(), poblacion = sum(Freq))
}

cat("\n== estratos sin ningún entrevistado ==\n")
print(bind_rows(
  estratos_vacios(dat, c("ccaa", "sexo", "edad"), pop_sexo_edad_ccaa) %>% mutate(margen = "ccaa x sexo x edad"),
  estratos_vacios(dat, c("ccaa", "sexo", "edu"), pop_edu_sexo_ccaa) %>% mutate(margen = "ccaa x sexo x edu"),
  estratos_vacios(dat, c("ccaa", "rec"), pop_rec_ccaa) %>% mutate(margen = "ccaa x recuerdo")
) %>% select(margen, estratos, poblacion))

# Con los márgenes de tres vías el raking de pesos no arranca, así que hay
# que bajar a márgenes más gruesos. Esa es su limitación de fondo: para
# ponderar a ccaa x recuerdo harían falta muchas más entrevistas. El MRP
# usa ese cruce sin problema porque lo que postestratifica es una tabla, no
# los pesos de la muestra.
marg <- function(d, vars) {
  d %>% group_by(across(all_of(vars))) %>%
    summarise(Freq = sum(N), .groups = "drop") %>% as.data.frame()
}

pop_ccaa      <- tg$total_ccaa %>% transmute(ccaa, Freq = total) %>% as.data.frame()
pop_sexo_edad <- marg(tg$sexo_edad_ccaa, c("sexo", "edad"))
pop_sexo_edu  <- marg(tg$edu_sexo_ccaa, c("sexo", "edu"))
pop_rec       <- marg(tg$rec_ccaa, "rec")

dis_rake <- rake(
  svydesign(ids = ~1, weights = ~1, data = dat),
  sample.margins = list(~ccaa, ~sexo + edad, ~sexo + edu, ~rec),
  population.margins = list(pop_ccaa, pop_sexo_edad, pop_sexo_edu, pop_rec),
  control = list(maxit = 100, epsilon = 1e-6)
)

cat("\npoblación representada por el raking:",
    format(round(sum(weights(dis_rake))), big.mark = " "), "\n")
cat("rango de los pesos:", paste(round(range(weights(dis_rake))), collapse = " - "), "\n")

pct <- function(x) as.numeric(x) * 100

comparacion <- tibble(
  partido = voto_lv,
  bruto = pct(prop.table(table(dat$voto))[voto_lv]),
  peso_cis = pct(coef(svymean(~voto, dis_cis))[paste0("voto", voto_lv)]),
  raking = pct(coef(svymean(~voto, dis_rake))[paste0("voto", voto_lv)]),
  mrp = est_mrp$mrp
) %>% left_join(cis_publicado, by = "partido")

cat("\n== estimación nacional (%) ==\n")
print(as.data.frame(comparacion %>% mutate(across(-partido, ~round(.x, 1)))),
      row.names = FALSE)

cat("\n== MRP con intervalo al 90% ==\n")
print(as.data.frame(est_mrp %>% mutate(across(-partido, ~round(.x, 1)))),
      row.names = FALSE)

# Por comunidad: donde el MRP debería lucir
mrp_ccaa <- agrega(ep, W, post$ccaa)
est_ccaa <- apply(mrp_ccaa, c(2, 3), mean) * 100

cat("\n== MRP por CCAA (%) ==\n")
print(round(est_ccaa, 1))

saveRDS(list(comparacion = comparacion, est_mrp = est_mrp,
             est_ccaa = est_ccaa, pesos_rake = weights(dis_rake)),
        file.path(dir_datos, "estimaciones.rds"))
cat("\nguardado estimaciones.rds\n")
# Por comunidad: donde el MRP debería lucir
mrp_edu <- agrega(ep, W, post$edu)
est_edu <- apply(mrp_edu, c(2, 3), mean) * 100
