# Construye los cuatro targets del raking y los deja con totales
# consistentes, que es requisito del IPF.

library(dplyr)
library(tidyr)
library(readr)

dir_datos <- "2026/10/mrp-raking/datos"

ccaa_cod <- read_csv(file.path(dir_datos, "ccaa_codigos.csv"),
                     col_types = cols(.default = "c", cod_cis = "i"))
ccaa_lv <- ccaa_cod$ccaa
edad_lv <- c("18-20", "21-24", "25-34", "35-44", "45-54", "55-64", "65+")
edu_lv  <- c("baja", "media", "alta")
rec_lv  <- c("PP", "PSOE", "VOX", "SUMAR", "OTROS", "ABST", "NO_PODIA")

# ---------------------------------------------------------------------------
# A. Resultados del 23-J por CCAA  ->  marginal de recuerdo
# ---------------------------------------------------------------------------
# Se usan los resultados por municipio, no por provincia. Los provinciales
# incluyen el voto CERA (residentes en el extranjero: 2,3 millones de censo y
# un 8,9 % de participación), que el CIS no entrevista. Con ellos la
# abstención del objetivo sube casi cuatro puntos. Los municipales son solo CER.

mun <- readRDS(file.path(dir_datos, "mun_23j.rds"))

res_ccaa <- mun %>%
  left_join(ccaa_cod %>% select(ccaa, cod_mir), by = c("codigo_ccaa" = "cod_mir"))

stopifnot(!any(is.na(res_ccaa$ccaa)))

# El PSOE y Sumar concurren con siglas propias en varias comunidades.
agrupa_siglas <- function(siglas) {
  case_when(
    siglas == "PP" ~ "PP",
    grepl("^PSOE$|^PSC$|PSdeG|^PSE-EE|^PSIB|^PSN-PSOE|^PSOE-", siglas) ~ "PSOE",
    siglas == "VOX" ~ "VOX",
    grepl("SUMAR", siglas) ~ "SUMAR",   # incluye MÉS PER MALLORCA-...-SUMAR
    TRUE ~ "OTROS"
  )
}

votos_ccaa <- res_ccaa %>%
  mutate(partido = agrupa_siglas(siglas)) %>%
  group_by(ccaa, partido) %>%
  summarise(votos = sum(votos), .groups = "drop")

# Censo y participación: una fila por comunidad
censo_ccaa <- res_ccaa %>%
  distinct(ccaa, codigo_provincia, codigo_municipio, codigo_distrito,
           censo_escrutinio, votos_candidaturas, votos_blancos, votos_nulos) %>%
  group_by(ccaa) %>%
  summarise(across(c(censo_escrutinio, votos_candidaturas, votos_blancos, votos_nulos), sum),
            .groups = "drop") %>%
  mutate(votantes = votos_candidaturas + votos_blancos + votos_nulos,
         ABST = censo_escrutinio - votantes)

cat("== comprobación nacional ==\n")
cat("censo 23-J (CER):", format(sum(censo_ccaa$censo_escrutinio), big.mark = " "), "\n")
cat("abstención:", round(100 * sum(censo_ccaa$ABST) / sum(censo_ccaa$censo_escrutinio), 1), "%\n")
print(votos_ccaa %>% group_by(partido) %>% summarise(votos = sum(votos)) %>% arrange(desc(votos)))

# Los blancos y nulos se suman a OTROS: en el CIS también están ahí.
recuerdo_ccaa <- votos_ccaa %>%
  left_join(censo_ccaa %>% select(ccaa, votos_blancos, votos_nulos, ABST), by = "ccaa") %>%
  mutate(votos = ifelse(partido == "OTROS", votos + votos_blancos + votos_nulos, votos)) %>%
  select(ccaa, partido, votos) %>%
  bind_rows(censo_ccaa %>% transmute(ccaa, partido = "ABST", votos = ABST))

# ---------------------------------------------------------------------------
# B. Población por sexo, edad y CCAA (ECP, años simples)
# ---------------------------------------------------------------------------

pob <- readRDS(file.path(dir_datos, "ine_pob_sexo_edad_ccaa.rds"))

pob_sec <- pob %>%
  transmute(
    cod_ine = `Comunidades.y.ciudades.autónomas.Codigo`,
    sexo_txt = Sexo,
    edad_txt = `Edad.simple`,
    valor = Valor
  ) %>%
  filter(sexo_txt %in% c("Hombres", "Mujeres"),
         cod_ine != "00", !is.na(cod_ine), cod_ine != "",
         grepl("^[0-9]+ (año|años)$|^100 y más", edad_txt)) %>%
  mutate(edad_n = as.integer(sub(" .*", "", edad_txt)),
         sexo = ifelse(sexo_txt == "Hombres", "H", "M")) %>%
  filter(edad_n >= 18) %>%
  left_join(ccaa_cod %>% select(ccaa, cod_ine), by = "cod_ine") %>%
  mutate(edad = cut(edad_n, breaks = c(18, 21, 25, 35, 45, 55, 65, Inf),
                    labels = edad_lv, right = FALSE))

stopifnot(!any(is.na(pob_sec$ccaa)), !any(is.na(pob_sec$edad)))

pob_sexo_edad_ccaa <- pob_sec %>%
  group_by(ccaa, sexo, edad) %>%
  summarise(N = sum(valor), .groups = "drop")

cat("\n== población 18+ por tramo (miles) ==\n")
print(pob_sexo_edad_ccaa %>% group_by(edad) %>% summarise(N = round(sum(N) / 1000)))

# Los de 18 a 20 años son quienes no pudieron votar el 23-J
pob_18_20 <- pob_sec %>% filter(edad_n <= 20) %>%
  group_by(ccaa) %>% summarise(NO_PODIA = sum(valor), .groups = "drop")

cat("\nnuevos electores (18-20):", format(round(sum(pob_18_20$NO_PODIA)), big.mark = " "), "\n")

# ---------------------------------------------------------------------------
# C. Nivel de estudios por sexo y CCAA (EPA)
# ---------------------------------------------------------------------------

edu_raw <- readRDS(file.path(dir_datos, "epa_edu_sexo_ccaa.rds"))
col_form <- grep("ormaci", names(edu_raw), value = TRUE)[1]

# Categorías de la EPA vigente (CNED-2014). Cuidado: las tablas con los
# mismos nombres pero ids 66xxx son la serie antigua, que acaba en 2013.
recod_edu_epa <- function(x) {
  case_when(
    grepl("^Analfabetos|^Estudios primarios incompletos|^Educación primaria|^Primera etapa", x) ~ "baja",
    grepl("^Segunda etapa", x) ~ "media",
    grepl("^Educación superior", x) ~ "alta",
    TRUE ~ NA_character_
  )
}

edu_sexo_ccaa <- edu_raw %>%
  transmute(
    cod_ine = `Comunidades.y.Ciudades.Autónomas.Codigo`,
    sexo_txt = Sexo,
    nivel = .data[[col_form]],
    valor = Valor
  ) %>%
  filter(sexo_txt %in% c("Hombres", "Mujeres"),
         cod_ine != "00", !is.na(cod_ine), cod_ine != "") %>%
  mutate(edu = recod_edu_epa(nivel),
         sexo = ifelse(sexo_txt == "Hombres", "H", "M")) %>%
  filter(!is.na(edu)) %>%
  left_join(ccaa_cod %>% select(ccaa, cod_ine), by = "cod_ine") %>%
  group_by(ccaa, sexo, edu) %>%
  # La EPA suprime 10 celdas de 304, todas en las dos categorías residuales
  # ("FP con título de secundaria" y un Doctorado de Ceuta). Los niveles
  # gruesos están completos, así que se ignoran.
  summarise(N = sum(valor, na.rm = TRUE), .groups = "drop")

stopifnot(!any(is.na(edu_sexo_ccaa$ccaa)), all(edu_sexo_ccaa$N > 0))

cat("\n== estudios en población (EPA, %) ==\n")
print(edu_sexo_ccaa %>% group_by(edu) %>% summarise(N = sum(N)) %>%
        mutate(pct = round(100 * N / sum(N), 1)))

# ---------------------------------------------------------------------------
# D. Totales consistentes
# ---------------------------------------------------------------------------
# Los tres universos no coinciden: la ECP son residentes, la EPA es población
# de 16 y más, y el censo electoral son españoles de 18 y más. El IPF exige
# un gran total común, así que se fija el universo del post -- el electorado
# de hoy: censo del 23-J más quienes han cumplido 18 desde entonces -- y se
# reescala cada target dentro de cada comunidad.

total_ccaa <- censo_ccaa %>%
  select(ccaa, censo_escrutinio) %>%
  left_join(pob_18_20, by = "ccaa") %>%
  mutate(total = censo_escrutinio + NO_PODIA)

cat("\n== universo del post ==\n")
cat("electorado actual:", format(round(sum(total_ccaa$total)), big.mark = " "), "\n")

reescala <- function(d, total_ccaa) {
  d %>% group_by(ccaa) %>% mutate(N = N / sum(N)) %>% ungroup() %>%
    left_join(total_ccaa %>% select(ccaa, total), by = "ccaa") %>%
    mutate(N = N * total) %>% select(-total)
}

t_sexo_edad_ccaa <- reescala(pob_sexo_edad_ccaa, total_ccaa)
t_edu_sexo_ccaa  <- reescala(edu_sexo_ccaa, total_ccaa)

t_rec_ccaa <- recuerdo_ccaa %>%
  bind_rows(pob_18_20 %>% transmute(ccaa, partido = "NO_PODIA", votos = NO_PODIA)) %>%
  rename(rec = partido, N = votos) %>%
  mutate(rec = factor(rec, levels = rec_lv))

cat("\n== los tres targets suman lo mismo por CCAA? ==\n")
comp <- t_sexo_edad_ccaa %>% group_by(ccaa) %>% summarise(sexo_edad = sum(N)) %>%
  left_join(t_edu_sexo_ccaa %>% group_by(ccaa) %>% summarise(edu_sexo = sum(N)), by = "ccaa") %>%
  left_join(t_rec_ccaa %>% group_by(ccaa) %>% summarise(recuerdo = sum(N)), by = "ccaa")
print(comp %>% mutate(across(-ccaa, ~round(.x / 1000))), n = 19)

# Importante: los niveles tienen que quedar como factores con el MISMO orden
# que en la semilla. Ipfp empareja los targets con el array por posición, no
# por nombre, así que un factor convertido a texto (que xtabs ordena
# alfabéticamente) permuta la tabla sin avisar.
como_factor <- function(d) {
  d %>% mutate(ccaa = factor(ccaa, levels = ccaa_lv),
               across(any_of("edu"), ~factor(.x, levels = edu_lv)),
               across(any_of("edad"), ~factor(.x, levels = edad_lv)),
               across(any_of("rec"), ~factor(.x, levels = rec_lv)))
}

saveRDS(list(sexo_edad_ccaa = como_factor(t_sexo_edad_ccaa),
             edu_sexo_ccaa = como_factor(t_edu_sexo_ccaa),
             rec_ccaa = como_factor(t_rec_ccaa),
             total_ccaa = como_factor(total_ccaa)),
        file.path(dir_datos, "targets.rds"))
cat("\nguardado targets.rds\n")
