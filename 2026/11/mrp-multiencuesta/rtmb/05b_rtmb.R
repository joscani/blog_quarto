# Flujo de 05b_separado.R con los ajustes RTMB: postestratifica el voto con
# participación (v_part) a partir de datos/fit_rtmb_voto_<modo>.rds y
# datos/fit_rtmb_part_<modo>.rds, que se ajustan aquí si no existen.
# Mismas fórmulas que la sección 3 de 05b_separado.R. La muestra d del voto se
# empareja con la d de la participación (los dos ajustes son independientes).
#
#   RTMB_HILOS=8 Rscript 2026/11/mrp-multiencuesta/rtmb/05b_rtmb.R nuts
#
# Salida: datos/estim_rtmb_<modo>.rds, con el formato de estim_separado_<modo>.rds.
# Compara con datos/estim_separado_<modo>.rds (Stan) si existe.

library(dplyr)
library(tidyr)
source("2026/11/mrp-multiencuesta/rtmb/comun.R")

args <- commandArgs(trailingOnly = TRUE)
modo <- if (length(args) >= 1) args[1] else "prueba"
stopifnot(modo %in% c("prueba", "nuts"))
hilos <- configura_hilos()

# 1. Ajustes (si faltan) y datos
for (m in c("voto", "part")) {
  f <- file.path(dir_datos, paste0("fit_rtmb_", m, "_", modo, ".rds"))
  if (!file.exists(f)) {
    cat("ajusto", m, "\n")
    st <- system2("Rscript", c(file.path(dir_proy, "rtmb", paste0("mrp_", m, "_rtmb.R")), modo))
    stopifnot(st == 0)
  }
}
av <- readRDS(file.path(dir_datos, paste0("fit_rtmb_voto_", modo, ".rds")))
ap <- readRDS(file.path(dir_datos, paste0("fit_rtmb_part_", modo, ".rds")))
x <- carga_datos(modo)
datos <- x$datos
voto_lv <- x$voto_lv
nombres_casa <- c(x$lv$casa, "casa media")
A <- av$A
Tp <- ap$Tp

# 2. Postestratificación (sección 3 de 05b_separado.R)
J <- datos$J
P <- datos$P
n_rec <- datos$n_rec
n_casa <- datos$n_casa
nd <- min(nrow(A$alpha), length(Tp$delta))
i_cr <- (datos$P_ccaa - 1) * n_rec + datos$P_rec
N_q <- datos$P_N
prob <- function(eta) {
  p <- exp(eta - apply(eta, 1, max)) * datos$P_mascara
  p / rowSums(p)
}
t0 <- Sys.time()
una <- function(d) {
  vc <- vp <- matrix(NA, n_casa + 1, J)
  base <- outer(rep(1, P), A$alpha[d, ]) +
    outer(datos$P_mujer, A$b_mujer[d, ]) +
    matrix(A$mes[d, datos$mes_post, ], P, J, byrow = TRUE) +
    A$edad[d, datos$P_edad, ] +
    A$edu[d, datos$P_edu, ] +
    A$rec[d, datos$P_rec, ] +
    A$ccaa[d, datos$P_ccaa, ] +
    A$prov[d, datos$P_prov, ] +
    A$ccaa_rec[d, i_cr, ]
  eta_t <- Tp$alpha[d] +
    datos$P_mujer * Tp$b[d] +
    Tp$mes[d, datos$mes_post] +
    Tp$edad[d, datos$P_edad] +
    Tp$edu[d, datos$P_edu] +
    Tp$rec[d, datos$P_rec] +
    Tp$ccaa[d, datos$P_ccaa] +
    Tp$prov[d, datos$P_prov]
  pv <- plogis(eta_t + Tp$delta[d])
  ef_media <- 0
  for (h in seq_len(n_casa + 1)) {
    if (h <= n_casa) {
      ef <- matrix(A$casa[d, h, ], P, J, byrow = TRUE) +
        A$casa_rec[d, (h - 1) * n_rec + datos$P_rec, ]
      ef_media <- ef_media + ef / n_casa
    } else {
      ef <- ef_media
    }
    p <- prob(base + ef)
    vc[h, ] <- 100 * colSums(N_q * p) / sum(N_q)
    vp[h, ] <- 100 * colSums(N_q * pv * p) / sum(N_q * pv)
  }
  list(vc = vc, vp = vp)
}
r <- parallel::mclapply(seq_len(nd), una, mc.cores = hilos)
v_casa_r <- apila(lapply(r, `[[`, "vc"))
v_part <- apila(lapply(r, `[[`, "vp"))
cat("postestratificación:", nd, "muestras,",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s\n")
cat("comprobación v_casa, aquí frente a mrp_voto_rtmb.R: diferencia máx. =",
    signif(max(abs(v_casa_r - av$v_casa[seq_len(nd), , ])), 2), "puntos\n")
stopifnot(max(abs(v_casa_r - av$v_casa[seq_len(nd), , ])) < 1e-8)

resume <- function(a, etiqueta) {
  bind_rows(lapply(seq_along(nombres_casa), function(h) {
    tibble(
      casa = paste0(nombres_casa[h], etiqueta),
      partido = voto_lv,
      mrp = colMeans(a[, h, ]),
      q05 = apply(a[, h, ], 2, quantile, 0.05),
      q95 = apply(a[, h, ], 2, quantile, 0.95)
    )
  }))
}
res <- bind_rows(resume(v_casa_r, ""), resume(v_part, ", con participación"))
saveRDS(res, file.path(dir_datos, paste0("estim_rtmb_", modo, ".rds")))
cat("guardado", file.path(dir_datos, paste0("estim_rtmb_", modo, ".rds")), "\n")

cat("\n== participación (RTMB) ==\n")
cat("part_bruta:", round(mean(Tp$part_bruta), 2), " delta:", round(mean(Tp$delta), 3), "\n")
cat("\n== MRP con participación calibrada al", 100 * datos$part_obj,
    "% (% voto válido), RTMB ==\n")
print(
  as.data.frame(
    res %>%
      filter(grepl("participación", casa)) %>%
      mutate(casa = sub(", con participación", "", casa)) %>%
      select(casa, partido, mrp) %>%
      pivot_wider(names_from = casa, values_from = mrp) %>%
      mutate(across(-partido, ~ round(.x, 1)))
  ),
  row.names = FALSE
)

# 3. Comparación con Stan
f_stan <- file.path(dir_datos, paste0("estim_separado_", modo, ".rds"))
if (file.exists(f_stan)) {
  cmp <- inner_join(readRDS(f_stan), res, by = c("casa", "partido"),
                    suffix = c("_stan", "_rtmb")) %>%
    mutate(dif = mrp_rtmb - mrp_stan,
           ancho = (q95_rtmb - q05_rtmb) / (q95_stan - q05_stan))
  cat("\n== RTMB frente a Stan, las", nrow(cmp), "cifras (casas x partidos, con y sin participación) ==\n")
  cat("máx |dif. media|:", round(max(abs(cmp$dif)), 2), "puntos;",
      " ancho del 90 % RTMB / Stan: mediana", round(median(cmp$ancho), 2),
      ", rango", paste(round(range(cmp$ancho), 2), collapse = "–"), "\n")
  cat("\n== casa media con participación ==\n")
  print(as.data.frame(
    cmp %>% filter(casa == "casa media, con participación") %>%
      transmute(partido, stan = round(mrp_stan, 1), rtmb = round(mrp_rtmb, 1),
                stan_ancho90 = round(q95_stan - q05_stan, 1),
                rtmb_ancho90 = round(q95_rtmb - q05_rtmb, 1))
  ), row.names = FALSE)
}
