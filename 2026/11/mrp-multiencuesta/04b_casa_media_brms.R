# Casa media con el ajuste brms (datos/mod_brms_nuts.rds): postestratificación
# sin los términos de casa, (1 | casa) y (1 | casa:rec), es decir, con efecto
# casa = 0. Último mes, si votara todo el mundo (% voto válido).
#
#   Rscript 2026/11/mrp-multiencuesta/04b_casa_media_brms.R

library(dplyr)
library(tidyr)
library(brms)

dir_datos <- "2026/11/mrp-multiencuesta/datos"
source("2026/11/mrp-multiencuesta/00_partidos.R")

mod       <- readRDS(file.path(dir_datos, "mod_brms_nuts.rds"))
celdas    <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
tabla     <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))

voto_lv <- setdiff(partidos_lv, "AC")
territorio_mod <- territorio[names(territorio) != "AC"]
prov_lv <- levels(prov_ccaa$prov)
K <- 9

# Tabla de población, igual que en 04_brms.R
post <- as.data.frame.table(tabla, responseName = "N") %>%
  filter(N > 1e-6) %>%
  left_join(prov_ccaa, by = "prov") %>%
  mutate(n = 1)
for (k in seq_len(K)) {
  post[[paste0("p", k)]] <- factor(post$prov, levels = prov_lv)
  post[[paste0("w", k)]] <- as.numeric(k == 1)
}
for (p in names(territorio_mod)) {
  post[[paste0("off_", p)]] <- ifelse(post$ccaa %in% territorio_mod[[p]], 0, -20)
}
post$mes <- factor(max(levels(celdas$mes)), levels = levels(celdas$mes))
post$casa <- factor("CIS", levels = levels(celdas$casa))   # no se usa: sin términos de casa

mascara <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) as.numeric(post$ccaa %in% territorio_mod[[p]]) else rep(1, nrow(post))
})

termino_mm <- paste0("(1 | mm(", paste0("p", 1:K, collapse = ", "),
                     ", weights = cbind(", paste0("w", 1:K, collapse = ", "),
                     "), scale = FALSE))")
re_sin_casa <- as.formula(paste("~ (1 | edad) + (1 | edu) + (1 | rec) + (1 | ccaa) +",
                                "(1 | ccaa:rec) + (1 | mes) +", termino_mm))

set.seed(2026)
ep <- posterior_epred(mod, newdata = post, re_formula = re_sin_casa,
                      ndraws = 400, allow_new_levels = TRUE, sample_new_levels = "gaussian")
for (d in seq_len(dim(ep)[1])) {
  ep[d, , ] <- ep[d, , ] * mascara
  ep[d, , ] <- ep[d, , ] / rowSums(ep[d, , ])
}
dr <- 100 * t(apply(ep, 1, function(p) colSums(p * post$N) / sum(post$N)))
colnames(dr) <- voto_lv

res <- tibble(partido = voto_lv, mrp = colMeans(dr),
              q05 = apply(dr, 2, quantile, 0.05), q95 = apply(dr, 2, quantile, 0.95))
cat("\n== casa media (brms),", max(levels(celdas$mes)), ", si votara todo el mundo (% voto válido) ==\n")
print(as.data.frame(res %>% mutate(across(-partido, ~round(.x, 1)))), row.names = FALSE)

saveRDS(list(res = res, draws = dr), file.path(dir_datos, "estim_brms_nuts_casa_media.rds"))
cat("\nguardado estim_brms_nuts_casa_media.rds\n")
