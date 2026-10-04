# Armoniza las tres fuentes a un vocabulario común y construye los targets
# del raking. Imprime los objetos intermedios: la idea es poder seguirlo.

library(haven)
library(dplyr)
library(tidyr)
library(readr)

dir_datos <- "2026/10/mrp-raking/datos"

# ---------------------------------------------------------------------------
# 1. Tabla puente de CCAA
# ---------------------------------------------------------------------------
# CIS, INE y Ministerio del Interior usan tres codificaciones distintas.
# CIS e INE solo se diferencian en que intercambian las dos Castillas;
# Interior cambia bastante más. Unir por código sin mirar es un desastre.

ccaa_cod <- read_csv(file.path(dir_datos, "ccaa_codigos.csv"),
                     col_types = cols(.default = "c", cod_cis = "i"))
print(ccaa_cod, n = 19)

ccaa_lv <- ccaa_cod$ccaa

# ---------------------------------------------------------------------------
# 2. Encuesta del CIS: recodificación
# ---------------------------------------------------------------------------

cis_raw <- read_sav(file.path(dir_datos, "3577.sav"))

# Tramos de edad. El 18-20 va aparte porque son quienes no tenían edad de
# votar el 23-J (comprobado: las edades 18, 19 y 20 son todas "no tenía edad").
tramo_edad <- function(edad) {
  cut(edad,
      breaks = c(18, 21, 25, 35, 45, 55, 65, Inf),
      labels = c("18-20", "21-24", "25-34", "35-44", "45-54", "55-64", "65+"),
      right = FALSE)
}
edad_lv <- c("18-20", "21-24", "25-34", "35-44", "45-54", "55-64", "65+")

# Nivel de estudios: las 19 categorías del CIS a los 3 sectores de la EPA.
# FP de Grado Superior va en "alta" porque en la CNED es educación superior,
# que es como la clasifica la EPA.
recod_edu <- function(nivel) {
  nivel <- as.character(nivel)
  case_when(
    grepl("^Menos de 5|^Educación Primaria|^FP Básica|^Educación secundaria", nivel) ~ "baja",
    grepl("^FP de Grado Medio|^Bachillerato", nivel) ~ "media",
    grepl("^FP de Grado Superior|^Arquitectura|^Diplomatura|^Grado|^Licenciatura|^Máster|^Doctorado|^Títulos propios", nivel) ~ "alta",
    TRUE ~ NA_character_
  )
}
edu_lv <- c("baja", "media", "alta")

# Recuerdo de voto. Las candidaturas pequeñas y regionales van a "otros":
# el objetivo del post es la mecánica, no el detalle de cada partido.
recod_partido <- function(x) {
  x <- as.character(x)
  case_when(
    x == "PP" ~ "PP",
    x == "PSOE" ~ "PSOE",
    x == "VOX" ~ "VOX",
    x == "Sumar" ~ "SUMAR",
    x %in% c("N.C.", "N.R.", "N.P.") ~ NA_character_,
    TRUE ~ "OTROS"   # ERC, Junts, Bildu, PNV, BNG, CCa, PACMA, UPN, blanco, nulo
  )
}
rec_lv <- c("PP", "PSOE", "VOX", "SUMAR", "OTROS", "ABST", "NO_PODIA")

# La intención de voto, que es lo que se quiere estimar. Se mide sobre voto
# válido, que es la base en la que el CIS publica su estimación: quedan fuera
# la abstención declarada, el voto nulo, los indecisos y los que no contestan.
# Repartir a los indecisos es la "cocina", y es otro asunto.
voto_lv <- c("PP", "PSOE", "VOX", "SUMAR", "OTROS")
recod_voto <- function(x) {
  x <- as.character(x)
  case_when(
    x == "PP" ~ "PP",
    x == "PSOE" ~ "PSOE",
    x == "VOX" ~ "VOX",
    x == "Sumar" ~ "SUMAR",
    x %in% c("No sabe todavía", "N.C.", "N.R.", "N.P.",
             "No votaría", "Voto nulo") ~ NA_character_,
    TRUE ~ "OTROS"   # resto de candidaturas y voto en blanco
  )
}

cis <- cis_raw %>%
  transmute(
    cod_cis = as.integer(CCAA),
    voto = factor(recod_voto(as_factor(INTENCIONG)), levels = voto_lv),
    sexo = factor(as_factor(SEXO), levels = c("Hombre", "Mujer"),
                  labels = c("H", "M")),
    edad_n = as.numeric(EDAD),
    edad = tramo_edad(edad_n),
    edu = factor(recod_edu(as_factor(NIVELESTENTREV)), levels = edu_lv),
    participacion = as.character(as_factor(PARTICIPACIONG)),
    partido_rec = recod_partido(as_factor(RECUVOTOG)),
    intencion = recod_partido(as_factor(INTENCIONG)),
    ideol = ifelse(as.numeric(ESCIDEOL) %in% 1:10, as.numeric(ESCIDEOL), NA),
    peso = as.numeric(PESO)
  ) %>%
  left_join(ccaa_cod %>% select(ccaa, cod_cis), by = "cod_cis") %>%
  mutate(ccaa = factor(ccaa, levels = ccaa_lv))

# El recuerdo combina dos variables del CIS: participación y, para quien
# votó, el partido. Quien no tenía edad en 2023 es una categoría propia.
cis <- cis %>%
  mutate(rec = case_when(
    edad == "18-20" ~ "NO_PODIA",
    participacion == "No votó" ~ "ABST",
    participacion == "Votó" ~ partido_rec,
    TRUE ~ NA_character_     # no tenía derecho, N.R./N.C. de participación
  ),
  rec = factor(rec, levels = rec_lv))

cat("\n== recuerdo recodificado ==\n")
print(table(cis$rec, useNA = "ifany"))

cat("\n== cuántos casos sin recuerdo y por qué ==\n")
print(cis %>% filter(is.na(rec)) %>% count(participacion, sin_partido = is.na(partido_rec)))

# ---------------------------------------------------------------------------
# 3. Diagnóstico: lo torcida que está la muestra
# ---------------------------------------------------------------------------

cat("\n== estudios: CIS (bruto y ponderado) ==\n")
print(cis %>% filter(!is.na(edu)) %>%
        summarise(across(everything(), ~NULL)) %>% bind_cols(
          cis %>% filter(!is.na(edu)) %>%
            group_by(edu) %>%
            summarise(bruto = n() / sum(!is.na(cis$edu)),
                      ponderado = sum(peso) / sum(cis$peso[!is.na(cis$edu)]),
                      .groups = "drop")))

saveRDS(cis, file.path(dir_datos, "cis_recodificado.rds"))
cat("\nguardado cis_recodificado.rds\n")
