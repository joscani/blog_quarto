# Tabla de postestratificación por provincia: semilla, ceros estructurales e
# IPF a dos targets. Mismo esquema que 04_raking.R del post de octubre, pero
# con provincia en vez de CCAA.

library(dplyr)
library(tidyr)
library(mipfp)

# Prueba de sensibilidad a la semilla (opcional):
#   Rscript 2026/11/mrp-multiencuesta/03_raking.R "CIS 3567" 0.5   # semilla sin ese barómetro
#   Rscript 2026/11/mrp-multiencuesta/03_raking.R ninguno 0.25     # otro alpha
# Sin argumentos: semilla con todo el CIS, alpha = 0,5 y datos/tabla_post_prov.rds.
# Con argumentos, la tabla va a datos/sensibilidad/tabla_<excluye>_a<alpha>.rds.
args <- commandArgs(trailingOnly = TRUE)
excluye <- if (length(args) >= 1) args[1] else "ninguno"
alpha <- if (length(args) >= 2) as.numeric(args[2]) else 0.5
cat("semilla sin:", excluye, " alpha:", alpha, "\n")

dir_datos <- "2026/11/mrp-multiencuesta/datos"
source("2026/11/mrp-multiencuesta/00_partidos.R")   # partidos_lv, rec_lv, territorio

enc <- readRDS(file.path(dir_datos, "encuestas.rds"))
tg  <- readRDS(file.path(dir_datos, "targets_prov.rds"))

prov_lv <- levels(tg$pob_prov$prov)
edad_lv <- levels(tg$pob_prov$edad)
edu_lv  <- levels(tg$pob_prov$edu)

# Provincia -> CCAA, para los ceros de los partidos regionales
prov_ccaa <- enc %>% filter(casa == "CIS") %>% distinct(prov, ccaa) %>%
  mutate(prov = factor(prov, levels = prov_lv)) %>% arrange(prov)
stopifnot(nrow(prov_ccaa) == 52, !anyDuplicated(prov_ccaa$prov))

# ---------------------------------------------------------------------------
# 1. Semilla
# ---------------------------------------------------------------------------
# Solo el CIS: tiene provincia en todas las entrevistas y estudios con la
# codificación exacta. La semilla solo aporta la asociación entre variables
# dentro de cada provincia; los totales los fijan los targets.

sem_dat <- enc %>%
  filter(casa == "CIS", estudio != excluye, !is.na(prov), !is.na(sexo), !is.na(edad), !is.na(edu), !is.na(rec)) %>%
  mutate(prov = factor(prov, levels = prov_lv))

cat("== semilla ==\n")
cat("entrevistas del CIS con todas las variables:", nrow(sem_dat), "\n")

semilla_encuesta <- xtabs(~ prov + sexo + edad + edu + rec, sem_dat)
dims <- dim(semilla_encuesta)
dn <- dimnames(semilla_encuesta)
cat("celdas:", length(semilla_encuesta), " vacías:", sum(semilla_encuesta == 0),
    sprintf("(%.0f%%)\n", 100 * mean(semilla_encuesta == 0)))

# Con 28.000 celdas y 11.000 entrevistas casi todo está vacío: se suaviza
# mezclando con la independencia, como en el post de octubre
semilla <- alpha * array(1, dims, dn) / length(semilla_encuesta) +
  (1 - alpha) * semilla_encuesta / sum(semilla_encuesta)

# Ceros estructurales: 18-20 años <-> no podía votar
i18 <- which(dn$edad == "18-20")
inp <- which(dn$rec == "NO_PODIA")
semilla[, , i18, , -inp] <- 0
semilla[, , -i18, , inp] <- 0

# Partidos regionales fuera de su comunidad
ccaa_de_prov <- as.character(prov_ccaa$ccaa[match(dn$prov, prov_ccaa$prov)])
for (p in intersect(names(territorio), dn$rec)) {
  semilla[!ccaa_de_prov %in% territorio[[p]], , , , p] <- 0
}
cat("ceros estructurales:", sum(semilla == 0), "celdas\n")

# ---------------------------------------------------------------------------
# 2. Targets
# ---------------------------------------------------------------------------
#   1,2,3,4 -> provincia x sexo x edad x estudios  (Censo 2024)
#   1,5     -> provincia x recuerdo                (23-J + nuevos electores)

t1 <- xtabs(N ~ prov + sexo + edad + edu, tg$pob_prov)
t2 <- xtabs(N ~ prov + rec, tg$rec_prov)

# Ipfp empareja por posición: los nombres tienen que coincidir en orden
stopifnot(
  identical(dimnames(t1)$prov, dn$prov), identical(dimnames(t1)$sexo, dn$sexo),
  identical(dimnames(t1)$edad, dn$edad), identical(dimnames(t1)$edu, dn$edu),
  identical(dimnames(t2)$prov, dn$prov), identical(dimnames(t2)$rec, dn$rec)
)

# Un target positivo donde la semilla es cero estructural no se puede ajustar
t2_cero <- apply(semilla, c(1, 5), sum) == 0 & t2 > 0
if (any(t2_cero)) {
  cat("\n== votos donde el partido no se presentaba (se pasan a OTROS) ==\n")
  print(which(t2_cero, arr.ind = TRUE))
}

# ---------------------------------------------------------------------------
# 3. IPF
# ---------------------------------------------------------------------------
# En proporciones y reescalado al final: mipfp considera inconsistentes
# targets que difieren en 1e-08 por redondeo.

total <- sum(t1)
fit <- Ipfp(seed = semilla / sum(semilla),
            target.list = list(c(1, 2, 3, 4), c(1, 5)),
            target.data = list(t1 / sum(t1), t2 / sum(t2)),
            iter = 1000, tol = 1e-10, print = FALSE)
tabla <- fit$x.hat * total

cat("\n== convergencia ==\n")
cat("iteraciones:", length(fit$evol.stp.crit), " criterio final:",
    signif(tail(fit$evol.stp.crit, 1), 3), "\n")

# ---------------------------------------------------------------------------
# 4. ¿Cuadra?
# ---------------------------------------------------------------------------

err1 <- max(abs(apply(tabla, 1:4, sum) - t1))
err2 <- max(abs(apply(tabla, c(1, 5), sum) - t2))
cat("error máximo en T1:", signif(err1, 3), " en T2:", signif(err2, 3), "personas\n")

cat("\n== recuerdo nacional: objetivo vs tabla (miles) ==\n")
print(data.frame(rec = rec_lv,
                 objetivo = round(colSums(t2)[rec_lv] / 1000),
                 tabla = round(apply(tabla, 5, sum)[rec_lv] / 1000)), row.names = FALSE)

# Comparación con la tabla por CCAA del post de octubre, agregando provincias
tabla_oct <- readRDS("2026/10/mrp-raking/datos/tabla_postestratificacion.rds")
ccaa_prov <- apply(tabla, c(1, 5), sum) %>% as.data.frame.table(responseName = "N") %>%
  left_join(prov_ccaa, by = "prov") %>%
  group_by(ccaa, rec) %>% summarise(N = sum(N), .groups = "drop")
ccaa_oct <- apply(tabla_oct, c(1, 5), sum) %>% as.data.frame.table(responseName = "N_oct")
comp <- ccaa_prov %>% mutate(ccaa = as.character(ccaa), rec = as.character(rec)) %>%
  left_join(ccaa_oct %>% mutate(ccaa = as.character(ccaa), rec = as.character(rec)),
            by = c("ccaa", "rec"))
cat("\n== CCAA x recuerdo: esta tabla vs la del post de octubre ==\n")
cat("correlación:", round(cor(comp$N, comp$N_oct), 5),
    " diferencia relativa máxima en celdas > 10.000:",
    round(100 * max(abs(comp$N / comp$N_oct - 1)[comp$N_oct > 1e4]), 1), "%\n")

celdas <- as.data.frame.table(tabla, responseName = "N") %>% filter(N > 1e-6)
cat("\nceldas con población:", nrow(celdas), "de", length(tabla), "\n")
cat("total de la tabla:", format(round(sum(tabla)), big.mark = " "), "\n")

if (length(args) == 0) {
  saveRDS(tabla, file.path(dir_datos, "tabla_post_prov.rds"))
  saveRDS(prov_ccaa, file.path(dir_datos, "prov_ccaa.rds"))
  cat("guardadas tabla_post_prov.rds y prov_ccaa.rds\n")
} else {
  dir.create(file.path(dir_datos, "sensibilidad"), showWarnings = FALSE)
  f_alt <- file.path(dir_datos, "sensibilidad",
                     paste0("tabla_", gsub(" ", "", excluye), "_a", alpha, ".rds"))
  saveRDS(tabla, f_alt)
  cat("guardada", f_alt, "\n")
}
