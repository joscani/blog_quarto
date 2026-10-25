# Targets por provincia para el raking:
#   T1: provincia x sexo x edad x estudios  (Censo anual de Población, INE)
#   T2: provincia x recuerdo del 23-J       (resultados por municipio, solo CER)
# A diferencia del post de octubre, el Censo da la conjunta de sexo, edad y
# estudios, así que no hace falta juntar ECP y EPA.

library(ineapir)
library(dplyr)
library(tidyr)
library(readr)

dir_datos <- "2026/11/mrp-multiencuesta/datos"
dir_pob   <- "2026/10/mrp-raking/datos"   # mun_23j.rds y ccaa_codigos.csv
source("2026/11/mrp-multiencuesta/00_partidos.R")   # partidos_lv, rec_lv

edad_lv <- c("18-20", "21-24", "25-34", "35-44", "45-54", "55-64", "65+")
edu_lv  <- c("baja", "media", "alta")

# ---------------------------------------------------------------------------
# 1. Censo: tabla 66600, provincia a provincia
# ---------------------------------------------------------------------------
# La API no devuelve la tabla entera ("restricciones de volumen"), así que se
# pide filtrada por provincia (variable 115). La 67573 es idéntica.
# Último periodo: 1 de enero de 2024.

f_censo <- file.path(dir_datos, "censo_66600.rds")
if (!file.exists(f_censo)) {
  vv <- get_metadata_table_varval(idTable = 66600)
  provs <- vv %>% filter(Fk_Variable == 115)
  censo_raw <- bind_rows(lapply(provs$Id, function(id) {
    get_data_table(idTable = 66600, nlast = 1, unnest = TRUE, metanames = TRUE,
                   metacodes = TRUE, tip = "AM", filter = list("115" = as.character(id)))
  }))
  saveRDS(censo_raw, f_censo)
}
censo_raw <- readRDS(f_censo)
cat("== Censo 66600 ==\n")
cat("filas:", nrow(censo_raw), " periodo:", unique(substr(censo_raw$Fecha, 1, 10)), "\n")
cat("provincias:", length(unique(censo_raw$Comunidad.Autónoma.y.Provincia.Codigo)), "\n")
print(unique(censo_raw$Nivel.de.estudios))

# Estudios a los tres niveles del post. La FP de grado superior es "alta",
# como en la CNED-2014 (y como la EPA en el post de octubre).
recod_edu_censo <- function(x) {
  case_when(
    x %in% c("Sin estudios", "Educación primaria",
             "Primera etapa de Educación Secundaria y similar") ~ "baja",
    grepl("^Segunda etapa", x) ~ "media",
    grepl("^Enseñanzas de formación profesional|^Grados universitarios|^Másteres|^Doctorado", x) ~ "alta",
    TRUE ~ NA_character_   # "Total"
  )
}

censo <- censo_raw %>%
  transmute(
    prov = as.integer(Comunidad.Autónoma.y.Provincia.Codigo),
    sexo_txt = Sexo, edad_txt = Edad, nivel = Nivel.de.estudios, valor = Valor
  ) %>%
  filter(sexo_txt %in% c("Hombres", "Mujeres"),
         grepl("^[0-9]+ años?$", edad_txt)) %>%   # fuera "15 y más" y "90 y más"
  mutate(edad_n = as.integer(sub(" .*", "", edad_txt)),
         sexo = ifelse(sexo_txt == "Hombres", "H", "M"),
         edu = recod_edu_censo(nivel),
         fp_superior = grepl("^Enseñanzas de formación profesional", nivel)) %>%
  filter(edad_n >= 18, !is.na(edu))

# "90 y más años" es un tramo abierto: se añade aparte
censo_90 <- censo_raw %>%
  transmute(prov = as.integer(Comunidad.Autónoma.y.Provincia.Codigo),
            sexo_txt = Sexo, edad_txt = Edad, nivel = Nivel.de.estudios, valor = Valor) %>%
  filter(sexo_txt %in% c("Hombres", "Mujeres"), edad_txt == "90 y más años") %>%
  mutate(edad_n = 90L, sexo = ifelse(sexo_txt == "Hombres", "H", "M"),
         edu = recod_edu_censo(nivel),
         fp_superior = grepl("^Enseñanzas de formación profesional", nivel)) %>%
  filter(!is.na(edu))

# Comprobación: si las edades simples llegan hasta 89, "90 y más" no se solapa
cat("edad máxima en años simples:", max(censo$edad_n), "\n")
censo <- bind_rows(censo, censo_90) %>%
  mutate(edad = cut(edad_n, breaks = c(18, 21, 25, 35, 45, 55, 65, Inf),
                    labels = edad_lv, right = FALSE))
stopifnot(!anyNA(censo$edad), !anyNA(censo$valor))

pob_prov <- censo %>%
  group_by(prov, sexo, edad, edu) %>%
  summarise(N = sum(valor), .groups = "drop")

cat("\npoblación 18+ (Censo 2024):", format(sum(pob_prov$N), big.mark = " "), "\n")
cat("celdas provincia x sexo x edad x estudios:", nrow(pob_prov),
    " vacías:", sum(pob_prov$N == 0), "\n")

cat("\n== estudios en población (%) ==\n")
print(pob_prov %>% group_by(edu) %>% summarise(N = sum(N)) %>%
        mutate(pct = round(100 * N / sum(N), 1)))

# Para repartir después las categorías ambiguas de 40dB y More in Common:
# qué parte de "alta" es FP de grado superior en cada celda
fp_sup <- censo %>%
  filter(edu == "alta") %>%
  group_by(prov, sexo, edad) %>%
  summarise(N_alta = sum(valor), N_fp_superior = sum(valor[fp_superior]), .groups = "drop")

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
    TRUE ~ "OTROS"
  )
}

mun <- mun %>% mutate(prov = as.integer(codigo_provincia))

votos_prov <- mun %>%
  mutate(partido = agrupa_siglas(siglas)) %>%
  group_by(prov, partido) %>%
  summarise(votos = sum(votos), .groups = "drop")

censo_23j <- mun %>%
  distinct(prov, codigo_municipio, codigo_distrito,
           censo_escrutinio, votos_candidaturas, votos_blancos, votos_nulos) %>%
  group_by(prov) %>%
  summarise(across(c(censo_escrutinio, votos_candidaturas, votos_blancos, votos_nulos), sum),
            .groups = "drop") %>%
  mutate(ABST = censo_escrutinio - votos_candidaturas - votos_blancos - votos_nulos)

stopifnot(nrow(censo_23j) == 52)
cat("\n== 23-J (CER) ==\n")
cat("censo:", format(sum(censo_23j$censo_escrutinio), big.mark = " "),
    " abstención:", round(100 * sum(censo_23j$ABST) / sum(censo_23j$censo_escrutinio), 1), "%\n")

# Blancos y nulos a OTROS, como en las encuestas
recuerdo_prov <- votos_prov %>%
  left_join(censo_23j %>% select(prov, votos_blancos, votos_nulos), by = "prov") %>%
  mutate(votos = ifelse(partido == "OTROS", votos + votos_blancos + votos_nulos, votos)) %>%
  select(prov, rec = partido, N = votos) %>%
  bind_rows(censo_23j %>% transmute(prov, rec = "ABST", N = ABST))

# ---------------------------------------------------------------------------
# 3. Universo y totales consistentes
# ---------------------------------------------------------------------------
# Universo: el electorado de hoy = censo del 23-J (CER) + quienes tienen
# 18-20 años, que no pudieron votar. El Censo de 2024 cuenta residentes, no
# electores, así que de él solo se toma la estructura dentro de cada provincia.

no_podia <- pob_prov %>% filter(edad == "18-20") %>%
  group_by(prov) %>% summarise(NO_PODIA = sum(N), .groups = "drop")

total_prov <- censo_23j %>% select(prov, censo_escrutinio) %>%
  left_join(no_podia, by = "prov") %>%
  mutate(total = censo_escrutinio + NO_PODIA)

cat("\nelectorado actual:", format(round(sum(total_prov$total)), big.mark = " "), "\n")

# Se reescalan por separado los dos bloques de edad. Con los ceros
# estructurales (18-20 <-> NO_PODIA), el bloque de 18-20 de T1 tiene que
# sumar exactamente lo mismo que NO_PODIA en T2, y el de 21+ lo mismo que el
# censo del 23-J. Reescalando la pirámide entera no cuadran y el IPF no
# puede ajustar T1 (error de ~10.000 personas por celda).
t_pob_prov <- pob_prov %>%
  mutate(bloque = ifelse(edad == "18-20", "NO_PODIA", "censo")) %>%
  group_by(prov, bloque) %>% mutate(N = N / sum(N)) %>% ungroup() %>%
  left_join(total_prov %>% select(prov, censo_escrutinio, NO_PODIA), by = "prov") %>%
  mutate(N = N * ifelse(bloque == "NO_PODIA", NO_PODIA, censo_escrutinio)) %>%
  select(prov, sexo, edad, edu, N)

t_rec_prov <- recuerdo_prov %>%
  bind_rows(no_podia %>% transmute(prov, rec = "NO_PODIA", N = NO_PODIA))

cat("\n== los dos targets suman lo mismo por provincia? ==\n")
dif <- t_pob_prov %>% group_by(prov) %>% summarise(pob = sum(N)) %>%
  left_join(t_rec_prov %>% group_by(prov) %>% summarise(rec = sum(N)), by = "prov") %>%
  mutate(dif = pob - rec)
cat("diferencia máxima:", signif(max(abs(dif$dif)), 3), "\n")

# Comparación con los targets por CCAA del post de octubre
ccaa_cod <- read_csv(file.path(dir_pob, "ccaa_codigos.csv"),
                     col_types = cols(.default = "c", cod_cis = "i"))
tg_ccaa <- readRDS(file.path(dir_pob, "targets.rds"))
cat("\nelectorado del post de octubre:", format(round(sum(tg_ccaa$total_ccaa$total)), big.mark = " "), "\n")

# Factores con el orden fijo: Ipfp empareja por posición, no por nombre
prov_lv <- sort(unique(t_pob_prov$prov))
como_factor <- function(d) {
  d %>% mutate(prov = factor(prov, levels = prov_lv),
               across(any_of("sexo"), ~factor(.x, levels = c("H", "M"))),
               across(any_of("edu"), ~factor(.x, levels = edu_lv)),
               across(any_of("edad"), ~factor(.x, levels = edad_lv)),
               across(any_of("rec"), ~factor(.x, levels = rec_lv)))
}

saveRDS(list(pob_prov = como_factor(t_pob_prov),
             rec_prov = como_factor(t_rec_prov),
             total_prov = como_factor(total_prov),
             fp_superior = fp_sup),
        file.path(dir_datos, "targets_prov.rds"))
cat("\nguardado targets_prov.rds\n")
