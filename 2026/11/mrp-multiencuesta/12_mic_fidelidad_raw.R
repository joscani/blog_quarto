#' ---
#' title: "Abstención 2023 y fidelidad de voto: More in Common frente a CIS y 40dB"
#' author: "José Luis Cañadas Reche"
#' date: "2026-10-26"
#' format:
#'   html:
#'     embed-resources: true
#'     toc: true
#'     code-fold: show
#' execute:
#'   message: false
#'   warning: false
#' ---
#'
#' Este fichero es a la vez un script de R y un informe. Se puede ejecutar con
#' `Rscript 12_mic_fidelidad_raw.R` o convertir en HTML con
#' `quarto render 12_mic_fidelidad_raw.R` (formato `knitr::spin`: las líneas
#' `#'` son texto y `#+` abre un chunk).
#'
#' Lee directamente los `.sav` originales de More in Common, del CIS y de
#' 40dB, sin ningún paso intermedio. Cada etiqueta de recuerdo y de intención
#' se clasifica de forma explícita, y la clasificación se muestra para poder
#' revisarla.
#'
#' ## Qué mido
#'
#' Clases de cada respuesta:
#'
#' - **Recuerdo del 23J**: PSOE, PP, OTRO_PARTIDO, BLANCO_NULO, ABST o NA (no
#'   recuerda, no contesta, sin edad o sin derecho a voto).
#' - **Intención de voto directa**: PSOE, PP, OTRO_PARTIDO, BLANCO_NULO o
#'   SIN_PARTIDO (no sabe, no contesta, no votaría).
#'
#' Métricas, en %, en bruto y con el peso de cada casa. "Es un partido"
#' significa PSOE, PP u OTRO_PARTIDO: el blanco y el nulo no cuentan.
#'
#' | Métrica | Definición |
#' |---|---|
#' | `recuerdo_PSOE` | P(recuerdo = PSOE \| recuerdo es un partido) |
#' | `fid_PSOE` | P(intención = PSOE \| recuerdo = PSOE, intención es un partido) |
#' | `fid_PP` | P(intención = PP \| recuerdo = PP, intención es un partido) |
#' | `abst_PSOE` | P(intención = PSOE \| recuerdo = ABST, intención es un partido) |
#'
#' Los intervalos son del 90 % (Jeffreys) y solo en bruto: con pesos haría
#' falta el efecto de diseño.
#'
#' More in Common no pregunta si se votó el 23J. `participacion_2023_raw`
#' responde "Votó", "No pudo", "No suele votar", "Sin edad" o "No recuerda".
#' Aquí ABST = "No pudo" + "No suele votar". La última sección compara los dos
#' grupos.

#+ paquetes
library(haven)
library(tidyverse)
library(knitr)

options(OutDec = ",") # decimales con coma en el texto y las tablas (no en el CSV)

#' ## Ficheros
#'
#' Rutas relativas a la raíz del proyecto (`here::here()`). Si los datos están
#' en otro sitio, basta con cambiar esta tabla.

#+ ficheros
ficheros <- tribble(
  ~casa            , ~estudio         , ~fichero                                                                               ,
  "More in Common" , "MIC jun"        , "descargas/microdatos-2026-07/microdatos-julio-2026.sav"                               ,
  "More in Common" , "MIC sep"        , "descargas/microdatos-2026-09/microdatos-septiembre-2026.sav"                          ,
  "CIS"            , "CIS 3567"       , "descargas/MD3567/3567.sav"                                                            ,
  "CIS"            , "CIS 3571"       , "2026/10/mrp-raking/datos/MD3571/3571.sav"                                             ,
  "CIS"            , "CIS 3577"       , "descargas/MD3577/3577.sav"                                                            ,
  "CIS"            , "CIS 3582"       , "descargas/MD3582/3582.sav"                                                            ,
  "40dB"           , "40dB jul"       , "descargas/92_Julio_2026/03_Datos_julio_2026.sav"                                      ,
  "40dB"           , "40dB sep"       , "2026/10/mrp-raking/datos/2026_09_barometro_descargables/03_Datos_septiembre_2026.sav" ,
  "40dB"           , "40dB oct"       , "2026/10/mrp-raking/datos/2026_10_barometro_descargables/03_Datos_octubre_2026.sav"    ,
  "40dB"           , "40dB flash oct" , "descargas/2026_10_08_barometro_flash_descargables/03_Datos_octubre_2026_flash.sav"
) |>
  mutate(fichero = here::here(fichero))

f_salida <- here::here("2026/11/mrp-multiencuesta/mic_fidelidad_raw.csv")
partidos <- c("PSOE", "PP", "OTRO_PARTIDO")

#' ## Clasificación de etiquetas
#'
#' Respuestas que no son partido, de las tres casas juntas. Lo que no esté
#' aquí y no sea PSOE o PP es OTRO_PARTIDO.

#+ clasificacion
blanco_nulo <- c(
  "Blanco", "Nulo", # More in Common
  "En blanco", "Voto nulo", # CIS
  "Votaría en blanco", "Votaría nulo", "Voté en blanco", "Voté nulo" # 40dB
)
sin_partido <- c(
  "NS", "NC", "No votaría", # More in Common
  "No sabe todavía", "N.C.", "N.R.", # CIS
  "No lo sé", "Prefiero no contestar" # 40dB
)

clasifica <- function(x) {
  case_when(
    is.na(x) ~ NA,
    x %in% blanco_nulo ~ "BLANCO_NULO",
    x %in% sin_partido ~ "SIN_PARTIDO",
    str_detect(x, "^PSOE\\b") ~ "PSOE",
    str_detect(x, "^PP\\b") ~ "PP",
    .default = "OTRO_PARTIDO"
  )
}

# El recuerdo de quien no da partido (no sabe, no contesta) es NA
sin_na <- function(x) if_else(x == "SIN_PARTIDO", NA, x)

#' ## Lectura
#'
#' Una función por casa. Todas devuelven las etiquetas originales
#' (`part_txt`, `rec_txt`, `int_txt`), las clases (`rec`, `int`) y el peso.

#+ lectura
lee_mic <- function(f) {
  read_sav(f) |>
    as_factor() |>
    transmute(
      part_txt = as.character(participacion_2023_raw),
      rec_txt = as.character(voto_2023_raw),
      int_txt = as.character(intencion_voto_raw),
      rec = case_when(
        part_txt %in% c("No pudo", "No suele votar") ~ "ABST",
        part_txt == "Votó" ~ sin_na(clasifica(rec_txt)),
        .default = NA # Sin edad, No recuerda
      ),
      int = clasifica(int_txt),
      peso = as.numeric(peso),
      prob_voto = as.numeric(as.character(prob_voto_declarada))
    )
}

lee_cis <- function(f) {
  read_sav(f) |>
    transmute(
      part_txt = as.character(as_factor(PARTICIPACIONG)),
      rec_txt = as.character(as_factor(RECUVOTOG)),
      int_txt = as.character(as_factor(INTENCIONG)),
      rec = case_when(
        part_txt == "No votó" ~ "ABST",
        part_txt == "Votó" ~ sin_na(clasifica(rec_txt)),
        .default = NA # sin edad, sin derecho a voto, N.C., N.R.
      ),
      int = clasifica(int_txt),
      peso = as.numeric(PESO)
    )
}

# 40dB cambia el número de pregunta entre oleadas: se busca por etiqueta
var_por_etiqueta <- function(d, patron) {
  etiquetas <- map_chr(d, \(x) attr(x, "label") %||% "")
  names(d)[str_detect(etiquetas, patron)][1]
}

lee_40db <- function(f) {
  d <- read_sav(f)
  v_int <- var_por_etiqueta(d, "^Intención de voto elecciones generales - Y")
  v_rec <- var_por_etiqueta(d, "^Recuerdo de voto")
  d |>
    transmute(
      part_txt = NA_character_,
      rec_txt = as.character(as_factor(.data[[v_rec]])),
      int_txt = as.character(as_factor(.data[[v_int]])),
      rec = case_when(
        rec_txt == "No voté" ~ "ABST",
        rec_txt == "No tenía edad para votar" ~ NA,
        .default = sin_na(clasifica(rec_txt))
      ),
      int = clasifica(int_txt),
      peso = as.numeric(ponde)
    )
}

lee <- function(casa, fichero) {
  switch(
    casa,
    "More in Common" = lee_mic(fichero),
    "CIS" = lee_cis(fichero),
    "40dB" = lee_40db(fichero)
  )
}

encuestas <- ficheros |>
  mutate(datos = map2(casa, fichero, lee)) |>
  select(-fichero) |>
  unnest(datos)

count(encuestas, casa, estudio, name = "entrevistas") |> kable()

#' ## Revisión de la clasificación
#'
#' Recuerdo: etiqueta original y clase asignada.

#+ revision-recuerdo
encuestas |>
  count(casa, part_txt, rec_txt, rec) |>
  arrange(casa, rec, desc(n)) |>
  kable()

#' Intención: etiquetas que no van a OTRO_PARTIDO.

#+ revision-intencion
encuestas |>
  filter(int != "OTRO_PARTIDO") |>
  count(casa, int_txt, int) |>
  arrange(casa, int, desc(n)) |>
  kable()

#' ## Métricas
#'
#' Cada métrica es un subconjunto de filas y un `exito` lógico. Se apilan en
#' formato largo y se resumen todas con el mismo `summarise()`.

#+ metricas
subconjuntos <- function(d) {
  bind_rows(
    recuerdo_PSOE = d |>
      filter(rec %in% partidos) |>
      mutate(exito = rec == "PSOE"),
    fid_PSOE = d |>
      filter(rec %in% "PSOE", int %in% partidos) |>
      mutate(exito = int == "PSOE"),
    fid_PP = d |>
      filter(rec %in% "PP", int %in% partidos) |>
      mutate(exito = int == "PP"),
    abst_PSOE = d |>
      filter(rec %in% "ABST", int %in% partidos) |>
      mutate(exito = int == "PSOE"),
    .id = "metrica"
  )
}

resume <- function(d, ...) {
  d |>
    summarise(
      n = n(),
      bruto = 100 * mean(exito),
      ic90_inf = 100 * qbeta(0.05, sum(exito) + 0.5, n - sum(exito) + 0.5),
      ic90_sup = 100 * qbeta(0.95, sum(exito) + 0.5, n - sum(exito) + 0.5),
      con_peso = 100 * weighted.mean(exito, peso),
      .by = c(...)
    ) |>
    mutate(across(bruto:con_peso, \(x) round(x, 1)))
}

largo <- subconjuntos(encuestas)

resultados <- bind_rows(
  largo |> mutate(estudio = "todas") |> resume(metrica, casa, estudio),
  largo |> resume(metrica, casa, estudio)
) |>
  arrange(metrica, casa, estudio != "todas", estudio)

# el CSV con punto decimal: OutDec = "," hace que write_csv() avise
withr::with_options(list(OutDec = "."), write_csv(resultados, f_salida))

# atajo para citar cifras en el texto
cifra <- function(m, c, col = "con_peso") {
  resultados |>
    filter(metrica == m, casa == c, estudio == "todas") |>
    pull({{ col }})
}

#' Totales por casa, todas las oleadas juntas:

#+ totales
resultados |>
  filter(estudio == "todas") |>
  select(metrica, casa, n, bruto, ic90_inf, ic90_sup, con_peso) |>
  kable()

#' Por oleada:

#+ por-oleada
resultados |>
  filter(estudio != "todas") |>
  kable()

#' ## Peso de la abstención de 2023
#'
#' Porcentaje de abstencionistas de 2023 entre quienes tienen recuerdo
#' (partido, blanco o nulo, o abstención).

#+ peso-abstencion
abst <- encuestas |>
  filter(!is.na(rec)) |>
  mutate(exito = rec == "ABST") |>
  resume(casa) |>
  select(casa, n, bruto, con_peso)
kable(abst)

#' Con su peso, More in Common pasa la abstención de 2023 del
#' `r abst$bruto[abst$casa == "More in Common"]` % al
#' `r abst$con_peso[abst$casa == "More in Common"]` % de la muestra. El CIS
#' la deja en el `r abst$con_peso[abst$casa == "CIS"]` % y 40dB en el
#' `r abst$con_peso[abst$casa == "40dB"]` %. De esos abstencionistas, el
#' `r cifra("abst_PSOE", "More in Common")` % declara voto al PSOE en More in
#' Common, frente al `r cifra("abst_PSOE", "CIS")` % del CIS y el
#' `r cifra("abst_PSOE", "40dB")` % de 40dB (con peso).
#'
#' ## More in Common: "No pudo" frente a "No suele votar"
#'
#' Si "No pudo" son votantes que el 23J no pudieron votar, deberían parecerse
#' más a "Votó" que a "No suele votar". Cifras en bruto.

#+ no-pudo
perfil <- encuestas |>
  filter(casa == "More in Common", prob_voto %in% 0:10) |>
  summarise(
    n = n(),
    prob_voto_media = mean(prob_voto),
    pct_prob_10 = 100 * mean(prob_voto == 10),
    pct_no_votaria = 100 * mean(int_txt %in% "No votaría"),
    pct_PSOE_con_partido = 100 * mean(int[int %in% partidos] == "PSOE"),
    .by = part_txt
  ) |>
  mutate(across(where(is.double), \(x) round(x, 1))) |>
  arrange(desc(prob_voto_media))
kable(perfil)

#' "No pudo" declara una probabilidad media de votar de
#' `r perfil$prob_voto_media[perfil$part_txt == "No pudo"]` sobre 10, y solo
#' el `r perfil$pct_no_votaria[perfil$part_txt == "No pudo"]` % dice que no
#' votaría. En "No suele votar" esas cifras son
#' `r perfil$prob_voto_media[perfil$part_txt == "No suele votar"]` y
#' `r perfil$pct_no_votaria[perfil$part_txt == "No suele votar"]` %. Si el
#' peso calibra la abstención real del 23J y "No pudo" cuenta como
#' abstención, ese grupo pesa más de lo que le tocaría como votante.
