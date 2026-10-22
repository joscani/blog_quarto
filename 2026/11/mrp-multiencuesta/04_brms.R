# MRP con las 9 encuestas en brms. Provincia con multi-membership (mm()):
# More in Common no tiene provincia fuera de las uniprovinciales, y su efecto
# provincia es la media ponderada de las provincias de su comunidad.
#
# Uso:
#   Rscript 2026/11/mrp-multiencuesta/04_brms.R prueba       # 10 % de las filas, 1 cadena: prueba el script
#   Rscript 2026/11/mrp-multiencuesta/04_brms.R pathfinder   # no sirve: Pareto k = 12. Además brms 2.23.0 (CRAN) falla al leer path__ de cmdstan >= 2.37; corregido en brms 2.23.2 de GitHub (issue #1830, PR #1831)
#   Rscript 2026/11/mrp-multiencuesta/04_brms.R nuts         # ajuste final, horas

library(dplyr)
library(tidyr)
library(brms)

args <- commandArgs(trailingOnly = TRUE)
modo <- if (length(args) >= 1) args[1] else "prueba"
stopifnot(modo %in% c("prueba", "pathfinder", "nuts"))
cat("modo:", modo, "\n")

options(mc.cores = 4, brms.backend = "cmdstanr")
dir_datos <- "2026/11/mrp-multiencuesta/datos"
source("2026/11/mrp-multiencuesta/00_partidos.R")   # partidos_lv, rec_lv, territorio

tabla     <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))

# Aliança Catalana solo la pregunta el CIS: en las otras casas está en "Otro".
# En el modelo conjunto va con OTROS.
voto_lv <- setdiff(partidos_lv, "AC")
territorio_mod <- territorio[names(territorio) != "AC"]
prov_lv <- levels(prov_ccaa$prov)
K <- 9   # máximo de provincias por CCAA (Castilla y León)

# Celdas de 03b_celdas.R
celdas <- readRDS(file.path(dir_datos, "celdas_brms.rds"))

con_offsets <- function(d) {
  for (p in names(territorio_mod)) {
    d[[paste0("off_", p)]] <- ifelse(d$ccaa %in% territorio_mod[[p]], 0, -20)
  }
  d
}

# ---------------------------------------------------------------------------
# 3. Modelo
# ---------------------------------------------------------------------------

termino_mm <- paste0("(1 | mm(", paste0("p", 1:K, collapse = ", "),
                     ", weights = cbind(", paste0("w", 1:K, collapse = ", "),
                     "), scale = FALSE))")
f_comun <- paste("sexo + (1 | casa) + (1 | edad) + (1 | edu) + (1 | rec) + (1 | ccaa) +",
                 "(1 | ccaa:rec) + (1 | casa:rec) + (1 | mes) +", termino_mm)
cat("\nfórmula común:", f_comun, "\n")

f_partidos <- lapply(voto_lv[-1], function(p) {
  offset <- if (p %in% names(territorio_mod)) paste0(" + offset(off_", p, ")") else ""
  as.formula(paste0("mu", p, " ~ ", f_comun, offset))
})
formula_mrp <- do.call(bf, c(list(as.formula(paste("y | trials(n) ~", f_comun))), f_partidos))

priors <- Reduce(`+`, lapply(paste0("mu", voto_lv[-1]), function(dp) {
  prior_string("normal(0, 1.5)", class = "Intercept", dpar = dp) +
    prior_string("normal(0, 1)", class = "b", dpar = dp) +
    prior_string("exponential(1)", class = "sd", dpar = dp)
}))

if (modo == "prueba") {
  set.seed(2026)
  celdas <- celdas[sample(nrow(celdas), round(0.1 * nrow(celdas))), ]
  cat("\nprueba: ", nrow(celdas), "filas\n")
}

t0 <- Sys.time()
if (modo == "pathfinder") {
  mod <- brm(formula_mrp, data = celdas, family = multinomial(), prior = priors,
             algorithm = "pathfinder", seed = 2026, refresh = 100, silent = 0,
             file = file.path(dir_datos, "mod_brms_pathfinder"))
} else if (modo == "prueba") {
  mod <- brm(formula_mrp, data = celdas, family = multinomial(), prior = priors,
             chains = 1, iter = 300, seed = 2026, refresh = 50, silent = 0,
             file = file.path(dir_datos, "mod_brms_prueba"), file_refit = "always")
} else {
  mod <- brm(formula_mrp, data = celdas, family = multinomial(), prior = priors,
             chains = 4, iter = 2000, threads = threading(2), seed = 2026,
             refresh = 100, silent = 0, init = 0,
             # init = 0 y max_treedepth = 8: con los valores por defecto (inicio
             # aleatorio y profundidad 10), el primer intento seguía en la iteración
             # 1 tras 18 minutos
             control = list(adapt_delta = 0.95, max_treedepth = 8),
             file = file.path(dir_datos, "mod_brms_nuts"), file_refit = "on_change")
}
cat("\ntiempo de ajuste:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")

# ---------------------------------------------------------------------------
# 4. Postestratificación por casa, en el último mes (sin participación)
# ---------------------------------------------------------------------------

post <- as.data.frame.table(tabla, responseName = "N") %>%
  filter(N > 1e-6) %>%
  left_join(prov_ccaa, by = "prov") %>%
  mutate(n = 1)
for (k in seq_len(K)) {
  post[[paste0("p", k)]] <- factor(post$prov, levels = prov_lv)
  post[[paste0("w", k)]] <- as.numeric(k == 1)
}
post <- con_offsets(post)
post$mes <- factor(max(levels(celdas$mes)), levels = levels(celdas$mes))

mascara <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) as.numeric(post$ccaa %in% territorio_mod[[p]]) else rep(1, nrow(post))
})

por_casa <- lapply(levels(celdas$casa), function(cs) {
  nd <- post %>% mutate(casa = factor(cs, levels = levels(celdas$casa)))
  set.seed(2026)
  ep <- posterior_epred(mod, newdata = nd, ndraws = min(200, ndraws(mod)),
                        allow_new_levels = TRUE, sample_new_levels = "gaussian")
  for (d in seq_len(dim(ep)[1])) {
    ep[d, , ] <- ep[d, , ] * mascara
    ep[d, , ] <- ep[d, , ] / rowSums(ep[d, , ])
  }
  dr <- t(apply(ep, 1, function(p) colSums(p * post$N) / sum(post$N)))
  colnames(dr) <- voto_lv
  dr
})
names(por_casa) <- levels(celdas$casa)

res <- bind_rows(lapply(names(por_casa), function(cs) {
  m <- por_casa[[cs]]
  tibble(casa = cs, partido = voto_lv,
         mrp = 100 * colMeans(m),
         q05 = 100 * apply(m, 2, quantile, 0.05),
         q95 = 100 * apply(m, 2, quantile, 0.95))
}))

cat("\n== MRP por casa,", max(levels(celdas$mes)), ", si votara todo el mundo (% voto válido) ==\n")
print(as.data.frame(res %>% select(casa, partido, mrp) %>%
                      pivot_wider(names_from = casa, values_from = mrp) %>%
                      mutate(across(-partido, ~round(.x, 1)))), row.names = FALSE)

saveRDS(list(res = res, draws = por_casa), file.path(dir_datos, paste0("estim_brms_", modo, ".rds")))
cat("\nguardado", paste0("estim_brms_", modo, ".rds"), "\n")
