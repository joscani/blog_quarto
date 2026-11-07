# Ejemplo de juguete: MRP ampliado con variables que el censo no tiene.
#
# Idea (2026/11/mrp-multiencuesta/PARTICIPACION.md, "La idea general"): el
# censo dice cuánta gente hay en cada celda (sexo x edad x recuerdo). La
# ideología y la seguridad de ir a votar no están en el censo, pero un modelo
# dice cómo se reparten dentro de cada celda. Cadena:
#   1. ideología ~ celda                      (ordinal: izq, centro, der)
#   2. seguro    ~ celda + ideología          (logit: "10, seguro que voto")
#   3. voto      ~ celda + ideología + seguro (multinomial)
# Cada modelo usa la ideología y la seguridad que declara el entrevistado, así
# que se ajustan por separado. Las predicciones solo se encadenan al
# postestratificar.
#
# Comparamos con el MRP clásico (voto ~ celda), en dos cosas:
#   a. el voto si votara todo el mundo;
#   b. escenarios de participación: los seguros votan todos y de los dudosos
#      vota una fracción r.
#
# Datos: CIS 3582 (octubre de 2026), de la armonización del proyecto
# multiencuesta. Población: la tabla de postestratificación de ese proyecto,
# agregada a sexo x edad x recuerdo.
#
#   Rscript 2026/11/mrp-ampliado-juguete/juguete.R

library(dplyr)
library(tidyr)
library(brms)

dir_juguete <- here::here("2026/11/mrp-ampliado-juguete")
dir_modelos <- file.path(dir_juguete, "modelos")
dir.create(dir_modelos, showWarnings = FALSE)
dir_multi <- here::here("2026/11/mrp-multiencuesta/datos")

# ---- 1. Datos ----------------------------------------------------------------

cis <- readRDS(file.path(dir_multi, "encuestas.rds")) |>
  filter(estudio == "CIS 3582")
cat("entrevistas del CIS 3582:", nrow(cis), "\n")

# El voto en blanco va a OTROS en la armonización (recod_cis_partido no lo
# trata como no válido). Se recupera del fichero original del CIS, que tiene
# las mismas filas en el mismo orden (comprobado con edad y sexo). La
# simpatía nunca da "En blanco": basta con la intención directa.
crudo <- haven::read_sav(here::here("descargas/MD3582/3582.sav"))
stopifnot(
  nrow(crudo) == nrow(cis),
  all(as.numeric(crudo$EDAD) == cis$edad_n)
)
cis$blanco <- as.character(haven::as_factor(crudo$INTENCIONG)) == "En blanco"
cat("votos en blanco:", sum(cis$blanco), "\n")

# recuerdo y voto en pocas categorías
rec_lv <- c("PP", "PSOE", "VOX", "SUMAR", "OTROS", "ABST", "NO_PODIA")
voto_lv <- c("PP", "PSOE", "VOX", "SUMAR", "OTROS", "BLANCO")

cis <- cis |>
  mutate(
    rec = case_when(
      rec %in% c("PP", "PSOE", "VOX", "SUMAR", "ABST", "NO_PODIA") ~ as.character(rec),
      is.na(rec) ~ NA_character_,
      TRUE ~ "OTROS"
    ),
    rec = factor(rec, levels = rec_lv),
    # voto + simpatía; Podemos va con Sumar (el recuerdo es de Sumar en el 23-J)
    voto = case_when(
      blanco ~ "BLANCO",
      voto_sim %in% c("PP", "PSOE", "VOX") ~ as.character(voto_sim),
      voto_sim %in% c("SUMAR", "PODEMOS") ~ "SUMAR",
      is.na(voto_sim) ~ NA_character_,
      TRUE ~ "OTROS"
    ),
    voto = factor(voto, levels = voto_lv),
    # el CIS pregunta de 1 a 10: izquierda 1-4, centro 5-6, derecha 7-10
    ideol = case_when(
      ideol01 < 0.4 ~ "izq",
      ideol01 < 0.6 ~ "centro",
      !is.na(ideol01) ~ "der"
    ),
    ideol = factor(ideol, levels = c("izq", "centro", "der")),
    seguro = vota_seguro
  ) |>
  filter(!is.na(sexo), !is.na(edad), !is.na(rec))

cat("\nideología (1-10 del CIS en tres tramos):\n")
print(table(round(9 * cis$ideol01 + 1), cis$ideol, useNA = "ifany"))

cat("\n¿dice partido? por seguridad de ir a votar:\n")
print(xtabs(~ seguro + is.na(voto), cis, addNA = TRUE))

# ---- 2. Población: cuánta gente hay en cada celda ------------------------------

tabla <- readRDS(file.path(dir_multi, "tabla_post_prov.rds"))
poblacion <- as.data.frame(tabla, responseName = "N") |>
  mutate(
    rec = case_when(
      rec %in% c("PP", "PSOE", "VOX", "SUMAR", "ABST", "NO_PODIA") ~ as.character(rec),
      TRUE ~ "OTROS"
    ),
    rec = factor(rec, levels = rec_lv)
  ) |>
  group_by(sexo, edad, rec) |>
  summarise(N = sum(N), .groups = "drop")
cat("\nceldas de población:", nrow(poblacion), " personas:", round(sum(poblacion$N)), "\n")
print(xtabs(round(N / 1000) ~ edad + rec, poblacion))

# ---- 3. Modelos ------------------------------------------------------------------

opciones <- list(
  chains = 4,
  cores = 4,
  iter = 1000,
  seed = 2026,
  backend = "cmdstanr",
  refresh = 100,
  silent = 0,
  file_refit = "on_change"
)
ajusta <- function(formula, datos, familia, nombre) {
  do.call(
    brm,
    c(
      list(
        formula = formula,
        data = datos,
        family = familia,
        file = file.path(dir_modelos, nombre)
      ),
      opciones
    )
  )
}

# MRP clásico: voto según la celda, solo quien dice partido
m_clasico <- ajusta(
  voto ~ sexo + (1 | edad) + (1 | rec),
  filter(cis, !is.na(voto)),
  categorical(),
  "m_clasico"
)

# Cadena del MRP ampliado
m_ideol <- ajusta(
  ideol ~ sexo + (1 | edad) + (1 | rec),
  # cumulative() pide la respuesta como factor ordenado; en los otros modelos
  # la ideología es predictor y va sin ordenar (un efecto por tramo)
  filter(cis, !is.na(ideol)) |> mutate(ideol = ordered(ideol)),
  cumulative(),
  "m_ideol"
)
m_seguro <- ajusta(
  seguro ~ sexo + ideol + (1 | edad) + (1 | rec),
  filter(cis, !is.na(ideol), !is.na(seguro)),
  bernoulli(),
  "m_seguro"
)
m_voto <- ajusta(
  voto ~ sexo + ideol + seguro + (1 | edad) + (1 | rec),
  filter(cis, !is.na(ideol), !is.na(seguro), !is.na(voto)),
  categorical(),
  "m_voto"
)

# ---- 4. Predicciones por celda (medias de la posterior) -----------------------

# 4a. MRP clásico: P(voto | celda)
pred_clasico <- fitted(m_clasico, newdata = poblacion, summary = TRUE)[, "Estimate", ]
colnames(pred_clasico) <- voto_lv
celdas_clasico <- bind_cols(poblacion, as_tibble(pred_clasico))

# 4b. P(ideología | celda)
pred_ideol <- fitted(m_ideol, newdata = poblacion, summary = TRUE)[, "Estimate", ]
colnames(pred_ideol) <- c("izq", "centro", "der")
paso1 <- bind_cols(poblacion, as_tibble(pred_ideol)) |>
  pivot_longer(c(izq, centro, der), names_to = "ideol", values_to = "p_ideol")

# 4c. P(seguro | celda, ideología)
paso1$ideol <- factor(paso1$ideol, levels = c("izq", "centro", "der"))
paso1$p_seguro <- fitted(m_seguro, newdata = paso1, summary = TRUE)[, "Estimate"]
paso2 <- bind_rows(
  mutate(paso1, seguro = 1, p_nivel = p_seguro),
  mutate(paso1, seguro = 0, p_nivel = 1 - p_seguro)
)

# 4d. P(voto | celda, ideología, seguro)
pred_voto <- fitted(m_voto, newdata = paso2, summary = TRUE)[, "Estimate", ]
colnames(pred_voto) <- voto_lv
celdas_ampliado <- bind_cols(paso2, as_tibble(pred_voto)) |>
  mutate(personas = N * p_ideol * p_nivel)

cat("\nuna celda partida en ideología x seguridad (mujeres 25-34, abstuvieron el 23-J):\n")
print(
  celdas_ampliado |>
    filter(sexo == "M", edad == "25-34", rec == "ABST") |>
    select(ideol, seguro, personas, all_of(voto_lv)) |>
    mutate(across(all_of(voto_lv), \(p) round(100 * p, 1)), personas = round(personas)) |>
    arrange(ideol, desc(seguro)) |>
    as.data.frame(),
  row.names = FALSE
)

# ---- 5. Voto si votara todo el mundo -------------------------------------------

reparto <- function(celdas, peso) {
  votos <- sapply(voto_lv, function(k) sum(celdas[[peso]] * celdas[[k]]))
  round(100 * votos / sum(votos), 1)
}
cat("\n== % de voto si votara todo el mundo ==\n")
print(rbind(
  clasico = reparto(celdas_clasico, "N"),
  ampliado = reparto(celdas_ampliado, "personas")
))

# ---- 6. Escenarios de participación --------------------------------------------

# Los seguros votan todos; de los dudosos vota una fracción r. En el clásico
# la celda tiene un solo reparto de voto: sus dudosos votan como sus seguros.
p_seguro_celda <- celdas_ampliado |>
  filter(seguro == 1) |>
  group_by(sexo, edad, rec) |>
  summarise(p_seguro = sum(p_ideol * p_nivel), .groups = "drop")
celdas_clasico <- celdas_clasico |> left_join(p_seguro_celda, by = c("sexo", "edad", "rec"))

escenario <- function(r) {
  cl <- celdas_clasico |> mutate(votantes = N * (p_seguro + r * (1 - p_seguro)))
  am <- celdas_ampliado |> mutate(votantes = personas * if_else(seguro == 1, 1, r))
  tibble(
    r = r,
    participacion = round(100 * sum(am$votantes) / sum(poblacion$N), 1),
    metodo = c("clasico", "ampliado"),
    bind_rows(reparto(cl, "votantes"), reparto(am, "votantes"))
  )
}
cat("\n== escenarios: % de voto según la fracción r de dudosos que vota ==\n")
print(
  bind_rows(lapply(c(0, 0.25, 0.5, 0.75, 1), escenario)) |>
    arrange(metodo, r) |>
    as.data.frame(),
  row.names = FALSE
)
