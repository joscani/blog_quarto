# Población y raking: de la encuesta unida a la tabla de postestratificación.
#
#   1. Targets por provincia:
#      T1: provincia x sexo x edad x estudios (Censo anual de Población, INE)
#      T2: provincia x recuerdo del 23-J      (resultados por municipio, solo CER)
#      A diferencia del post de octubre, el Censo da la conjunta de sexo, edad
#      y estudios, así que no hace falta juntar ECP y EPA.
#   2. Semilla (solo CIS) con ceros estructurales e IPF a los dos targets.
#      Mismo esquema que 04_raking.R del post de octubre, pero con provincia.
#
# Salidas: datos/targets_prov.rds, datos/tabla_post_prov.rds y datos/prov_ccaa.rds.
#
# Para probar la sensibilidad a la semilla, cambia las constantes de abajo
# (excluye = "CIS 3567", alpha = 0.25) y guarda la tabla con otro nombre.

library(ineapir)
library(dplyr)
library(tidyr)
library(readr)
library(mipfp)

dir_datos <- "2026/11/mrp-multiencuesta/datos"
dir_pob <- "2026/10/mrp-raking/datos" # mun_23j.rds, ccaa_codigos.csv, targets.rds
source("2026/11/mrp-multiencuesta/00_partidos.R") # partidos_lv, rec_lv, territorio

edad_lv <- c("18-20", "21-24", "25-34", "35-44", "45-54", "55-64", "65+")
edu_lv <- c("baja", "media", "alta")

# sensibilidad de la semilla (por defecto, sin excluir nada y alpha = 0,5)
excluye <- "ninguno"
alpha <- 0.5

# ---------------------------------------------------------------------------
# 1. Censo: tabla 66600, provincia a provincia
# ---------------------------------------------------------------------------
# La API no devuelve la tabla entera ("restricciones de volumen"), así que se
# pide filtrada por provincia (variable 115). La 67573 es idéntica.
# Último periodo: 1 de enero de 2024.

f_censo <- file.path(dir_datos, "censo_66600.rds")
if (!file.exists(f_censo)) {
  vv <- get_metadata_table_varval(idTable = 66600)
  provs <- vv |> filter(Fk_Variable == 115)
  censo_raw <- bind_rows(lapply(provs$Id, function(id) {
    get_data_table(
      idTable = 66600,
      nlast = 1,
      unnest = TRUE,
      metanames = TRUE,
      metacodes = TRUE,
      tip = "AM",
      filter = list("115" = as.character(id))
    )
  }))
  saveRDS(censo_raw, f_censo)
}
censo_raw <- readRDS(f_censo)
cat("== Censo 66600 ==\n")
cat(
  "filas:",
  nrow(censo_raw),
  "  periodo:",
  unique(substr(censo_raw$Fecha, 1, 10)),
  "\n"
)
cat(
  "provincias:",
  length(unique(censo_raw$Comunidad.Autónoma.y.Provincia.Codigo)),
  "\n"
)
print(unique(censo_raw$Nivel.de.estudios))

# Estudios a los tres niveles del post. La FP de grado superior es "alta",
# como en la CNED-2014 (y como la EPA en el post de octubre).
recod_edu_censo <- function(x) {
  case_when(
    x %in%
      c(
        "Sin estudios",
        "Educación primaria",
        "Primera etapa de Educación Secundaria y similar"
      ) ~ "baja",
    grepl("^Segunda etapa", x) ~ "media",
    grepl(
      "^Enseñanzas de formación profesional|^Grados universitarios|^Másteres|^Doctorado",
      x
    ) ~ "alta",
    TRUE ~ NA_character_ # "Total"
  )
}

censo <- censo_raw |>
  transmute(
    prov = as.integer(Comunidad.Autónoma.y.Provincia.Codigo),
    sexo_txt = Sexo,
    edad_txt = Edad,
    nivel = Nivel.de.estudios,
    valor = Valor
  ) |>
  filter(
    sexo_txt %in% c("Hombres", "Mujeres"),
    grepl("^[0-9]+ años?$", edad_txt)
  ) |> # fuera "15 y más" y "90 y más"
  mutate(
    edad_n = as.integer(sub(" .*", "", edad_txt)),
    sexo = ifelse(sexo_txt == "Hombres", "H", "M"),
    edu = recod_edu_censo(nivel),
    fp_superior = grepl("^Enseñanzas de formación profesional", nivel)
  ) |>
  filter(edad_n >= 18, !is.na(edu))

# "90 y más años" es un tramo abierto: se añade aparte
censo_90 <- censo_raw |>
  transmute(
    prov = as.integer(Comunidad.Autónoma.y.Provincia.Codigo),
    sexo_txt = Sexo,
    edad_txt = Edad,
    nivel = Nivel.de.estudios,
    valor = Valor
  ) |>
  filter(sexo_txt %in% c("Hombres", "Mujeres"), edad_txt == "90 y más años") |>
  mutate(
    edad_n = 90L,
    sexo = ifelse(sexo_txt == "Hombres", "H", "M"),
    edu = recod_edu_censo(nivel),
    fp_superior = grepl("^Enseñanzas de formación profesional", nivel)
  ) |>
  filter(!is.na(edu))

# Comprobación: si las edades simples llegan hasta 89, "90 y más" no se solapa
cat("edad máxima en años simples:", max(censo$edad_n), "\n")
censo <- bind_rows(censo, censo_90) |>
  mutate(
    edad = cut(
      edad_n,
      breaks = c(18, 21, 25, 35, 45, 55, 65, Inf),
      labels = edad_lv,
      right = FALSE
    )
  )
stopifnot(!anyNA(censo$edad), !anyNA(censo$valor))

pob_prov <- censo |>
  group_by(prov, sexo, edad, edu) |>
  summarise(N = sum(valor), .groups = "drop")

cat(
  "\npoblación 18+ (Censo 2024):",
  format(sum(pob_prov$N), big.mark = " "),
  "\n"
)
cat(
  "celdas provincia x sexo x edad x estudios:",
  nrow(pob_prov),
  "  vacías:",
  sum(pob_prov$N == 0),
  "\n"
)

cat("\n== estudios en población (%) ==\n")
print(
  pob_prov |>
    group_by(edu) |>
    summarise(N = sum(N)) |>
    mutate(pct = round(100 * N / sum(N), 1))
)

# Para repartir después las categorías ambiguas de 40dB y More in Common:
# qué parte de "alta" es FP de grado superior en cada celda
fp_sup <- censo |>
  filter(edu == "alta") |>
  group_by(prov, sexo, edad) |>
  summarise(
    N_alta = sum(valor),
    N_fp_superior = sum(valor[fp_superior]),
    .groups = "drop"
  )

# ---------------------------------------------------------------------------
# 2. Resultados del 23-J por provincia (solo residentes en España)
# ---------------------------------------------------------------------------
# Igual que en el post de octubre: por municipio, que no incluye el voto CERA.

mun <- readRDS(file.path(dir_pob, "mun_23j.rds"))

agrupa_siglas <- function(siglas) {
  case_when(
    siglas == "PP" ~ "PP",
    grepl("^PSOE$|^PSC$|PSdeG|^PSE-EE|^PSIB|^PSN-PSOE|^PSOE-", siglas) ~ "PSOE",
    siglas == "VOX" ~ "VOX",
    grepl("SUMAR", siglas) ~ "SUMAR",
    siglas == "ERC" ~ "ERC",
    siglas == "JxCAT - JUNTS" ~ "JUNTS",
    siglas == "EH Bildu" ~ "BILDU",
    siglas == "EAJ-PNV" ~ "PNV",
    siglas == "B.N.G." ~ "BNG",
    siglas == "CCa" ~ "CCA",
    siglas == "U.P.N." ~ "UPN",
    TRUE ~ "OTROS"
  )
}

mun <- mun |> mutate(prov = as.integer(codigo_provincia))

votos_prov <- mun |>
  mutate(partido = agrupa_siglas(siglas)) |>
  group_by(prov, partido) |>
  summarise(votos = sum(votos), .groups = "drop")

censo_23j <- mun |>
  distinct(
    prov,
    codigo_municipio,
    codigo_distrito,
    censo_escrutinio,
    votos_candidaturas,
    votos_blancos,
    votos_nulos
  ) |>
  group_by(prov) |>
  summarise(
    across(
      c(censo_escrutinio, votos_candidaturas, votos_blancos, votos_nulos),
      sum
    ),
    .groups = "drop"
  ) |>
  mutate(
    ABST = censo_escrutinio - votos_candidaturas - votos_blancos - votos_nulos
  )

stopifnot(nrow(censo_23j) == 52)
cat("\n== 23-J (CER) ==\n")
cat(
  "censo:",
  format(sum(censo_23j$censo_escrutinio), big.mark = " "),
  "  abstención:",
  round(100 * sum(censo_23j$ABST) / sum(censo_23j$censo_escrutinio), 1),
  "%\n"
)

# Blancos y nulos a OTROS, como en las encuestas
recuerdo_prov <- votos_prov |>
  left_join(
    censo_23j |> select(prov, votos_blancos, votos_nulos),
    by = "prov"
  ) |>
  mutate(
    votos = ifelse(
      partido == "OTROS",
      votos + votos_blancos + votos_nulos,
      votos
    )
  ) |>
  select(prov, rec = partido, N = votos) |>
  bind_rows(censo_23j |> transmute(prov, rec = "ABST", N = ABST))

# ---------------------------------------------------------------------------
# 3. Universo y totales consistentes
# ---------------------------------------------------------------------------
# Universo: el electorado de hoy = censo del 23-J (CER) + quienes tienen
# 18-20 años, que no pudieron votar. El Censo de 2024 cuenta residentes, no
# electores, así que de él solo se toma la estructura dentro de cada provincia.

no_podia <- pob_prov |>
  filter(edad == "18-20") |>
  group_by(prov) |>
  summarise(NO_PODIA = sum(N), .groups = "drop")

total_prov <- censo_23j |>
  select(prov, censo_escrutinio) |>
  left_join(no_podia, by = "prov") |>
  mutate(total = censo_escrutinio + NO_PODIA)

cat(
  "\nelectorado actual:",
  format(round(sum(total_prov$total)), big.mark = " "),
  "\n"
)

# Se reescalan por separado los dos bloques de edad. Con los ceros
# estructurales (18-20 <-> NO_PODIA), el bloque de 18-20 de T1 tiene que
# sumar exactamente lo mismo que NO_PODIA en T2, y el de 21+ lo mismo que el
# censo del 23-J. Reescalando la pirámide entera no cuadran y el IPF no
# puede ajustar T1 (error de ~10.000 personas por celda).
t_pob_prov <- pob_prov |>
  mutate(bloque = ifelse(edad == "18-20", "NO_PODIA", "censo")) |>
  group_by(prov, bloque) |>
  mutate(N = N / sum(N)) |>
  ungroup() |>
  left_join(
    total_prov |> select(prov, censo_escrutinio, NO_PODIA),
    by = "prov"
  ) |>
  mutate(N = N * ifelse(bloque == "NO_PODIA", NO_PODIA, censo_escrutinio)) |>
  select(prov, sexo, edad, edu, N)

t_rec_prov <- recuerdo_prov |>
  bind_rows(no_podia |> transmute(prov, rec = "NO_PODIA", N = NO_PODIA))

cat("\n== los dos targets suman lo mismo por provincia? ==\n")
dif <- t_pob_prov |>
  group_by(prov) |>
  summarise(pob = sum(N)) |>
  left_join(
    t_rec_prov |> group_by(prov) |> summarise(rec = sum(N)),
    by = "prov"
  ) |>
  mutate(dif = pob - rec)
cat("diferencia máxima:", signif(max(abs(dif$dif)), 3), "\n")

# Comparación con los targets por CCAA del post de octubre
ccaa_cod <- read_csv(
  file.path(dir_pob, "ccaa_codigos.csv"),
  col_types = cols(.default = "c", cod_cis = "i")
)
tg_ccaa <- readRDS(file.path(dir_pob, "targets.rds"))
cat(
  "\nelectorado del post de octubre:",
  format(round(sum(tg_ccaa$total_ccaa$total)), big.mark = " "),
  "\n"
)

# Factores con el orden fijo: Ipfp empareja por posición, no por nombre
prov_lv <- sort(unique(t_pob_prov$prov))
como_factor <- function(d) {
  d |>
    mutate(
      prov = factor(prov, levels = prov_lv),
      across(any_of("sexo"), ~ factor(.x, levels = c("H", "M"))),
      across(any_of("edu"), ~ factor(.x, levels = edu_lv)),
      across(any_of("edad"), ~ factor(.x, levels = edad_lv)),
      across(any_of("rec"), ~ factor(.x, levels = rec_lv))
    )
}

tg <- list(
  pob_prov = como_factor(t_pob_prov),
  rec_prov = como_factor(t_rec_prov),
  total_prov = como_factor(total_prov),
  fp_superior = fp_sup
)
saveRDS(tg, file.path(dir_datos, "targets_prov.rds"))
cat("\nguardado targets_prov.rds\n")

# ---------------------------------------------------------------------------
# 4. Semilla y raking
# ---------------------------------------------------------------------------
# Solo el CIS: tiene provincia en todas las entrevistas y estudios con la
# codificación exacta. La semilla solo aporta la asociación entre variables
# dentro de cada provincia; los totales los fijan los targets.

enc <- readRDS(file.path(dir_datos, "encuestas.rds"))
prov_lv <- levels(tg$pob_prov$prov)
edad_lv <- levels(tg$pob_prov$edad)
edu_lv <- levels(tg$pob_prov$edu)

# Provincia -> CCAA, para los ceros de los partidos regionales
prov_ccaa <- enc |>
  filter(casa == "CIS") |>
  distinct(prov, ccaa) |>
  mutate(prov = factor(prov, levels = prov_lv)) |>
  arrange(prov)
stopifnot(nrow(prov_ccaa) == 52, !anyDuplicated(prov_ccaa$prov))

sem_dat <- enc |>
  filter(
    casa == "CIS",
    estudio != excluye,
    !is.na(prov),
    !is.na(sexo),
    !is.na(edad),
    !is.na(edu),
    !is.na(rec)
  ) |>
  mutate(prov = factor(prov, levels = prov_lv))

cat("\n== semilla (sin:", excluye, ", alpha:", alpha, ") ==\n")
cat("entrevistas del CIS con todas las variables:", nrow(sem_dat), "\n")

semilla_encuesta <- xtabs(~ prov + sexo + edad + edu + rec, sem_dat)
dims <- dim(semilla_encuesta)
dn <- dimnames(semilla_encuesta)
cat(
  "celdas:",
  length(semilla_encuesta),
  "  vacías:",
  sum(semilla_encuesta == 0),
  sprintf("(%.0f%%)\n", 100 * mean(semilla_encuesta == 0))
)

# Con 28.000 celdas y 11.000 entrevistas casi todo está vacío: se suaviza
# mezclando con la independencia, como en el post de octubre
semilla <- alpha *
  array(1, dims, dn) /
  length(semilla_encuesta) +
  (1 - alpha) * semilla_encuesta / sum(semilla_encuesta)

# Ceros estructurales: 18-20 años <-> no podía votar
i18 <- which(dn$edad == "18-20")
inp <- which(dn$rec == "NO_PODIA")
semilla[,, i18, , -inp] <- 0
semilla[,, -i18, , inp] <- 0

# Partidos regionales fuera de su comunidad
ccaa_de_prov <- as.character(prov_ccaa$ccaa[match(dn$prov, prov_ccaa$prov)])
for (p in intersect(names(territorio), dn$rec)) {
  semilla[!ccaa_de_prov %in% territorio[[p]], , , , p] <- 0
}
cat("ceros estructurales:", sum(semilla == 0), "celdas\n")

# Targets: 1,2,3,4 -> provincia x sexo x edad x estudios; 1,5 -> provincia x recuerdo
t1 <- xtabs(N ~ prov + sexo + edad + edu, tg$pob_prov)
t2 <- xtabs(N ~ prov + rec, tg$rec_prov)

# Ipfp empareja por posición: los nombres tienen que coincidir en orden
stopifnot(
  identical(dimnames(t1)$prov, dn$prov),
  identical(dimnames(t1)$sexo, dn$sexo),
  identical(dimnames(t1)$edad, dn$edad),
  identical(dimnames(t1)$edu, dn$edu),
  identical(dimnames(t2)$prov, dn$prov),
  identical(dimnames(t2)$rec, dn$rec)
)

# Un target positivo donde la semilla es cero estructural no se puede ajustar
t2_cero <- apply(semilla, c(1, 5), sum) == 0 & t2 > 0
if (any(t2_cero)) {
  cat("\n== votos donde el partido no se presentaba (se pasan a OTROS) ==\n")
  print(which(t2_cero, arr.ind = TRUE))
}

# ---------------------------------------------------------------------------
# 5. IPF
# ---------------------------------------------------------------------------
# En proporciones y reescalado al final: mipfp considera inconsistentes
# targets que difieren en 1e-08 por redondeo.

total <- sum(t1)
fit <- Ipfp(
  seed = semilla / sum(semilla),
  target.list = list(c(1, 2, 3, 4), c(1, 5)),
  target.data = list(t1 / sum(t1), t2 / sum(t2)),
  iter = 1000,
  tol = 1e-10,
  print = FALSE
)
tabla <- fit$x.hat * total

cat("\n== convergencia ==\n")
cat(
  "iteraciones:",
  length(fit$evol.stp.crit),
  "  criterio final:",
  signif(tail(fit$evol.stp.crit, 1), 3),
  "\n"
)

# ---------------------------------------------------------------------------
# 6. ¿Cuadra?
# ---------------------------------------------------------------------------

err1 <- max(abs(apply(tabla, 1:4, sum) - t1))
err2 <- max(abs(apply(tabla, c(1, 5), sum) - t2))
cat(
  "error máximo en T1:",
  signif(err1, 3),
  "  en T2:",
  signif(err2, 3),
  "personas\n"
)

cat("\n== recuerdo nacional: objetivo vs tabla (miles) ==\n")
print(
  data.frame(
    rec = rec_lv,
    objetivo = round(colSums(t2)[rec_lv] / 1000),
    tabla = round(apply(tabla, 5, sum)[rec_lv] / 1000)
  ),
  row.names = FALSE
)

# Comparación con la tabla por CCAA del post de octubre, agregando provincias
tabla_oct <- readRDS("2026/10/mrp-raking/datos/tabla_postestratificacion.rds")
ccaa_prov <- apply(tabla, c(1, 5), sum) |>
  as.data.frame.table(responseName = "N") |>
  left_join(prov_ccaa, by = "prov") |>
  group_by(ccaa, rec) |>
  summarise(N = sum(N), .groups = "drop")
ccaa_oct <- apply(tabla_oct, c(1, 5), sum) |>
  as.data.frame.table(responseName = "N_oct")
comp <- ccaa_prov |>
  mutate(ccaa = as.character(ccaa), rec = as.character(rec)) |>
  left_join(
    ccaa_oct |> mutate(ccaa = as.character(ccaa), rec = as.character(rec)),
    by = c("ccaa", "rec")
  )
cat("\n== CCAA x recuerdo: esta tabla vs la del post de octubre ==\n")
cat(
  "correlación:",
  round(cor(comp$N, comp$N_oct), 5),
  "  diferencia relativa máxima en celdas > 10.000:",
  round(100 * max(abs(comp$N / comp$N_oct - 1)[comp$N_oct > 1e4]), 1),
  "%\n"
)

celdas <- as.data.frame.table(tabla, responseName = "N") |> filter(N > 1e-6)
cat("\nceldas con población:", nrow(celdas), "de", length(tabla), "\n")
cat("total de la tabla:", format(round(sum(tabla)), big.mark = " "), "\n")

saveRDS(tabla, file.path(dir_datos, "tabla_post_prov.rds"))
saveRDS(prov_ccaa, file.path(dir_datos, "prov_ccaa.rds"))
cat("guardadas tabla_post_prov.rds y prov_ccaa.rds\n")
