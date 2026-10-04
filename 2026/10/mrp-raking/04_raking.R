# Semilla, ceros estructurales y raking. Sale la tabla de postestratificación.

library(dplyr)
library(tidyr)
library(mipfp)
library(nnet)

dir_datos <- "2026/10/mrp-raking/datos"

cis <- readRDS(file.path(dir_datos, "cis_recodificado.rds"))
tg  <- readRDS(file.path(dir_datos, "targets.rds"))

ccaa_lv <- levels(cis$ccaa)
edad_lv <- levels(cis$edad)
edu_lv  <- c("baja", "media", "alta")
source("2026/10/mrp-raking/00_partidos.R")   # partidos_lv, rec_lv, territorio

# ---------------------------------------------------------------------------
# 1. El recuerdo oculto
# ---------------------------------------------------------------------------
# 345 personas no dan recuerdo: 295 dicen haber votado pero no a quién, 32 no
# tenían derecho a voto en 2023, y el resto son N.R. de participación. No se
# pueden dejar fuera sin más: quien esconde el voto no lo esconde al azar.
# Se imputa con un multinomial que usa la intención de voto declarada y la
# autoubicación ideológica, que es lo que de verdad informa sobre el recuerdo.

cat("== casos sin recuerdo ==\n")
print(cis %>% count(sin_recuerdo = is.na(rec)))

cis_mod <- cis %>%
  filter(!is.na(edu), !is.na(edad), !is.na(sexo), !is.na(ccaa)) %>%
  mutate(intencion = ifelse(is.na(intencion), "NC", intencion),
         ideol_f = cut(ideol, breaks = c(0, 3, 5, 7, 10),
                       labels = c("izq", "centroizq", "centroder", "der")),
         ideol_f = factor(ifelse(is.na(ideol_f), "NC", as.character(ideol_f))))

ajuste_imp <- multinom(rec ~ intencion + ideol_f + edad + edu + sexo,
                       data = cis_mod %>% filter(!is.na(rec), rec != "NO_PODIA"),
                       trace = FALSE, maxit = 500)

falta <- is.na(cis_mod$rec)
set.seed(2026)
if (any(falta)) {
  p_imp <- predict(ajuste_imp, newdata = cis_mod[falta, ], type = "probs")
  # Ceros estructurales: un partido regional no se pudo votar fuera de su
  # comunidad. Se anula esa probabilidad y se renormaliza cada fila.
  p_imp <- p_imp * mascara_territorio(cis_mod$ccaa[falta], colnames(p_imp))
  p_imp <- p_imp / rowSums(p_imp)
  sorteo <- apply(p_imp, 1, function(p) sample(colnames(p_imp), 1, prob = p))
  cis_mod$rec[falta] <- factor(sorteo, levels = rec_lv)
}

cat("\n== recuerdo tras imputar (y lo que dicen las urnas) ==\n")
urnas <- tg$rec_ccaa %>% group_by(rec) %>% summarise(N = sum(N)) %>%
  mutate(urnas = round(100 * N / sum(N), 1)) %>% select(rec, urnas)
print(cis_mod %>% count(rec) %>% mutate(encuesta = round(100 * n / sum(n), 1)) %>%
        left_join(urnas, by = "rec"))

# ---------------------------------------------------------------------------
# 2. Semilla
# ---------------------------------------------------------------------------
# La semilla es la propia encuesta. Con 4.000 casos y 5.586 celdas está
# llena de huecos, así que se suaviza mezclándola con la tabla de
# independencia; si no, el IPF dejaría a cero celdas perfectamente posibles.

semilla_encuesta <- xtabs(~ ccaa + sexo + edad + edu + rec, cis_mod)
dims <- dim(semilla_encuesta)
dn <- dimnames(semilla_encuesta)

cat("\n== celdas vacías en la semilla cruda ==\n")
cat(sum(semilla_encuesta == 0), "de", length(semilla_encuesta),
    sprintf("(%.0f%%)\n", 100 * mean(semilla_encuesta == 0)))

alpha <- 0.5
semilla <- alpha * array(1, dims, dn) / length(semilla_encuesta) +
  (1 - alpha) * semilla_encuesta / sum(semilla_encuesta)

# Ceros estructurales: quien tiene 18-20 años hoy no pudo votar el 23-J, y
# quien pudo votar no tiene 18-20 años. Esto importa porque el IPF respeta
# los ceros: sin ellos, "no podía votar" se repartiría por todas las edades.
i18 <- which(dn$edad == "18-20")
inp <- which(dn$rec == "NO_PODIA")
semilla[, , i18, , -inp] <- 0
semilla[, , -i18, , inp] <- 0

# Y los partidos regionales fuera de su comunidad. Los objetivos ya valen 0
# ahí, pero ponerlos en la semilla deja claro de dónde salen esos ceros.
for (p in names(territorio)) {
  semilla[!dn$ccaa %in% territorio[[p]], , , , p] <- 0
}

cat("ceros estructurales añadidos:", sum(semilla == 0), "celdas\n")

# ---------------------------------------------------------------------------
# 3. Raking
# ---------------------------------------------------------------------------
# Tres targets, de tres fuentes distintas:
#   1,2,3 -> sexo x edad x ccaa   (Estadística Continua de Población)
#   1,2,4 -> edu  x sexo x ccaa   (EPA)
#   1,5   -> recuerdo x ccaa      (resultados del 23-J + nuevos electores)
# El orden de las dimensiones del array es ccaa, sexo, edad, edu, rec.

a_array <- function(d, vars) {
  f <- as.formula(paste("N ~", paste(vars, collapse = " + ")))
  xtabs(f, d)
}

t1 <- a_array(tg$sexo_edad_ccaa, c("ccaa", "sexo", "edad"))
t2 <- a_array(tg$edu_sexo_ccaa,  c("ccaa", "sexo", "edu"))
t3 <- a_array(tg$rec_ccaa,       c("ccaa", "rec"))

# Los tres targets suman el mismo total por construcción, pero no bit a bit:
# quedan diferencias de 1e-08 por redondeo y mipfp las considera
# inconsistentes. Se rakea en proporciones y se reescala al final.
stopifnot(
  identical(dimnames(t1)$ccaa, dn$ccaa), identical(dimnames(t1)$sexo, dn$sexo),
  identical(dimnames(t1)$edad, dn$edad), identical(dimnames(t2)$edu, dn$edu),
  identical(dimnames(t3)$rec, dn$rec)
)

total <- sum(tg$total_ccaa$total)
fit <- Ipfp(seed = semilla / sum(semilla),
            target.list = list(c(1, 2, 3), c(1, 2, 4), c(1, 5)),
            target.data = list(t1 / sum(t1), t2 / sum(t2), t3 / sum(t3)),
            iter = 1000, tol = 1e-10, print = FALSE)

tabla <- fit$x.hat * total
cat("\n== convergencia ==\n")
cat("iteraciones:", length(fit$evol.stp.crit), " criterio final:",
    signif(tail(fit$evol.stp.crit, 1), 3), "\n")

# ---------------------------------------------------------------------------
# 4. ¿Cuadra?
# ---------------------------------------------------------------------------

cat("\n== marginal de recuerdo: objetivo vs tabla ==\n")
comp <- data.frame(
  rec = rec_lv,
  objetivo = round(apply(t3, 2, sum)[rec_lv] / 1000),
  tabla = round(apply(tabla, 5, sum)[rec_lv] / 1000)
)
print(comp, row.names = FALSE)

cat("\n== estudios: encuesta vs tabla ==\n")
print(data.frame(
  edu = edu_lv,
  encuesta = as.numeric(round(100 * prop.table(table(cis_mod$edu))[edu_lv], 1)),
  tabla = as.numeric(round(100 * prop.table(apply(tabla, 4, sum))[edu_lv], 1))
), row.names = FALSE)

cat("\ntotal de la tabla:", format(round(sum(tabla)), big.mark = " "), "\n")

saveRDS(tabla, file.path(dir_datos, "tabla_postestratificacion.rds"))
saveRDS(cis_mod, file.path(dir_datos, "cis_imputado.rds"))
cat("guardada tabla_postestratificacion.rds\n")
