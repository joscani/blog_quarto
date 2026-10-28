# MRP del CIS con mgcv::gam(family = multinom): la misma estructura que
# mrp_cis.stan (05_stan.R con muestra "cis") pero con verosimilitud
# multinomial directa — sin truco multinomial-Poisson y sin efectos fijos
# por celda (la normalización softmax va dentro de la verosimilitud).
#
# Equivalencias con el modelo Stan:
#   * intercepto por partido  -> cada fórmula de la lista lleva la suya
#     (referencia = PP, la primera columna de y; en Stan hay sum-to-zero,
#     aquí contrastes contra la referencia: mismo modelo, otra parametrización)
#   * b_mujer[J]              -> mujer en cada fórmula
#   * a_mes/a_edad/a_edu/a_rec/a_ccaa/a_ccaa_rec/a_prov por partido
#                             -> s(grupo, bs = "re") en cada fórmula
#   * off de los partidos regionales -> offset(off_<partido>) en su fórmula
# Diferencias (importantes):
#   * sin priors: las sd de los RE se estiman por ML penalizado. El prior
#     normal(0, 0.5) de las sd del Stan no tiene equivalente.
#   * los RE de niveles sin datos en la encuesta valen 0 (penalización), no
#     se muestrean del prior como en Stan.
#   * sin posterior: point estimates; los intervalos requerirían bootstrap.
#
# Uso: Rscript 2026/11/mrp-multiencuesta/05d_mgcv_cis.R
# Salida: datos/estim_gam_cis.rds

library(dplyr)
library(tidyr)
library(mgcv)

dir_proy <- "2026/11/mrp-multiencuesta"
dir_datos <- file.path(dir_proy, "datos")
source(file.path(dir_proy, "00_partidos.R")) # partidos_lv, rec_lv, territorio

celdas <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))

voto_lv <- setdiff(partidos_lv, "AC")
territorio_mod <- territorio[names(territorio) != "AC"]
stopifnot(identical(colnames(celdas$y), voto_lv))
J <- length(voto_lv)

cis <- celdas %>% filter(casa == "CIS")
stopifnot(all(cis$w1 == 1)) # en CIS la provincia siempre está
partidos <- voto_lv

# niveles fijos para que encuesta y población compartan factores
lv_ccaa <- levels(prov_ccaa$ccaa)
lv_prov <- levels(prov_ccaa$prov)

prep <- function(d) {
  d %>%
    mutate(
      ccaa = factor(ccaa, levels = lv_ccaa),
      prov = factor(prov, levels = lv_prov),
      rec = factor(rec, levels = rec_lv),
      ccaa_rec = interaction(ccaa, rec, drop = FALSE),
      mujer = as.numeric(sexo == "M")
    )
}
cis <- prep(cis)

# ---------------------------------------------------------------------------
# 1. Expandir a entrevistas: multinom() quiere etiquetas 0..K por entrevista
# ---------------------------------------------------------------------------

y_mat <- cis$y
idx_cell <- rep(seq_len(nrow(cis)), times = rowSums(y_mat))
lab <- unlist(lapply(seq_len(nrow(cis)), function(i) {
  rep(seq_len(J), times = as.integer(y_mat[i, ]))
}))
dd <- cis[idx_cell, ] %>%
  select(mes, ccaa, ccaa_rec, prov, sexo, edad, edu, rec, mujer) %>%
  mutate(y = lab - 1) # 0 = PP (referencia), 1..12 = el resto

# columnas de offset, una por partido (0 para los no regionales)
off_full <- matrix(0, nrow(cis), J, dimnames = list(NULL, partidos))
off_cols <- paste0("off_", partidos)
off_cols <- off_cols[off_cols %in% names(cis)]
off_full[, sub("^off_", "", off_cols)] <- as.matrix(cis[, off_cols])
off_exp <- off_full[idx_cell, , drop = FALSE]
colnames(off_exp) <- paste0("off_", partidos)
dd <- cbind(dd, off_exp)

cat("celdas:", nrow(cis), " entrevistas:", nrow(dd), "\n")
cat("partidos con offset:", sub("^off_", "", off_cols), "\n")

# ---------------------------------------------------------------------------
# 2. Ajuste: una fórmula por categoría (excepto la referencia, PP)
# ---------------------------------------------------------------------------

terminos <- c(
  "mujer",
  "s(mes, bs = 're')",
  "s(edad, bs = 're')",
  "s(edu, bs = 're')",
  "s(rec, bs = 're')",
  "s(ccaa, bs = 're')",
  "s(ccaa_rec, bs = 're')",
  "s(prov, bs = 're')"
)
fmlas <- lapply(seq_len(J - 1), function(j) {
  as.formula(paste(
    "y ~",
    paste(
      c(terminos, paste0("offset(off_", partidos[j + 1], ")")),
      collapse = " + "
    )
  ))
})

cat("ajustando gam multinomial (", J - 1, " predictores)...\n")
t0 <- Sys.time()
fit <- gam(fmlas, family = multinom(K = J - 1), data = dd)
cat(
  "tiempo de ajuste:",
  round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
  "min\n"
)

# sd de los RE = sqrt(1/sp) por término (verificado contra glmer en la demo)
nombres_re <- c("mes", "edad", "edu", "rec", "ccaa", "ccaa_rec", "prov")
sds_re <- data.frame(
  grupo = rep(nombres_re, each = J - 1),
  partido = rep(partidos[-1], times = length(nombres_re)),
  sp = fit$sp
) %>%
  mutate(sd = sqrt(1 / sp))
cat("\n== sd de los efectos aleatorios (equivalente a sd_* de Stan) ==\n")
print(
  sds_re %>%
    select(grupo, partido, sd) %>%
    tidyr::pivot_wider(names_from = grupo, values_from = sd) %>%
    mutate(across(-partido, ~ round(.x, 2))),
  row.names = FALSE
)

# ---------------------------------------------------------------------------
# 3. Postestratificación con la tabla de población
# ---------------------------------------------------------------------------

tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
post <- as.data.frame.table(tabla, responseName = "N") %>%
  filter(N > 1e-6) %>%
  left_join(prov_ccaa, by = "prov") %>%
  prep() %>%
  mutate(mes = max(levels(celdas$mes))) # el mes más reciente de la encuesta

cat("\nceldas de población:", nrow(post), "\n")

p_post <- predict(fit, newdata = post, type = "response") # P x J
mascara <- sapply(partidos, function(p) {
  if (p %in% names(territorio_mod)) {
    as.numeric(post$ccaa %in% territorio_mod[[p]])
  } else {
    rep(1, nrow(post))
  }
})
p_post <- p_post * mascara
p_post <- p_post / rowSums(p_post)

est_gam <- tibble(
  partido = partidos,
  gam = 100 * colSums(post$N * p_post) / sum(post$N)
)

# ---------------------------------------------------------------------------
# 4. Comparación con el ajuste Stan (05_stan.R, versión nuts_nuevo_cis)
# ---------------------------------------------------------------------------

f_stan <- file.path(dir_datos, "estim_stan_nuts_nuevo_cis.rds")
if (file.exists(f_stan)) {
  est_stan <- readRDS(f_stan) %>%
    filter(casa == "CIS solo") %>%
    select(partido, stan = mrp)
  comparacion <- est_gam %>%
    left_join(est_stan, by = "partido") %>%
    mutate(dif = gam - stan)
  cat("\n== voto válido estimado (%): gam vs Stan ==\n")
  print(
    as.data.frame(comparacion %>% mutate(across(-partido, ~ round(.x, 1)))),
    row.names = FALSE
  )
} else {
  cat("\n(sin estim_stan_nuts_nuevo_cis.rds para comparar)\n")
  print(
    as.data.frame(est_gam %>% mutate(across(-partido, ~ round(.x, 1)))),
    row.names = FALSE
  )
}

saveRDS(
  list(est = est_gam, fit = fit, sds_re = sds_re),
  file.path(dir_datos, "estim_gam_cis.rds")
)
cat("\nguardado estim_gam_cis.rds\n")
