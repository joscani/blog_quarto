# Junta las 9 encuestas (CIS, 40dB y More in Common) en un formato común.
# Una fila por entrevista. Imprime los intermedios para poder seguirlo.

library(haven)
library(dplyr)
library(tidyr)
library(readr)

dir_datos <- "2026/11/mrp-multiencuesta/datos"
dir_pob <- "2026/10/mrp-raking/datos" # tabla puente de CCAA y microdatos del CIS 3571

source("2026/11/mrp-multiencuesta/00_partidos.R") # partidos_lv, rec_lv, territorio

# ---------------------------------------------------------------------------
# 1. Las encuestas
# ---------------------------------------------------------------------------
# Fecha: el mes de campo. En More in Common es el mes de la portada del
# codebook, que no coincide con el nombre del fichero.

encuestas <- tribble(
  ~casa             , ~estudio         , ~mes      , ~fichero                                                                               ,
  "CIS"             , "CIS 3567"       , "2026-06" , "descargas/MD3567/3567.sav"                                                            , # campo 1-4 jun 2026
  "CIS"             , "CIS 3571"       , "2026-07" , "2026/10/mrp-raking/datos/MD3571/3571.sav"                                             ,
  "CIS"             , "CIS 3577"       , "2026-09" , "descargas/MD3577/3577.sav"                                                            ,
  "CIS"             , "CIS 3582"       , "2026-10" , "descargas/MD3582/3582.sav"                                                            ,
  "40dB"            , "40dB jul"       , "2026-06" , "descargas/92_Julio_2026/03_Datos_julio_2026.sav"                                      , # campo 26-29 jun 2026
  "40dB"            , "40dB sep"       , "2026-09" , "2026/10/mrp-raking/datos/2026_09_barometro_descargables/03_Datos_septiembre_2026.sav" ,
  "40dB"            , "40dB oct"       , "2026-10" , "2026/10/mrp-raking/datos/2026_10_barometro_descargables/03_Datos_octubre_2026.sav"    ,
  "40dB"            , "40dB flash oct" , "2026-10" , "descargas/2026_10_08_barometro_flash_descargables/03_Datos_octubre_2026_flash.sav"    ,
  # MIC abr (2026-04, fichero microdatos-mayo-2026.sav) quitado el 2026-10-22: muy antiguo
  "More in Common"  , "MIC jun"        , "2026-06" , "descargas/microdatos-2026-07/microdatos-julio-2026.sav"                               ,
  "More in Common"  , "MIC sep"        , "2026-09" , "descargas/microdatos-2026-09/microdatos-septiembre-2026.sav"                          ,
  "Ateneo del Dato" , "Ateneo jun"     , "2026-06" , "descargas/atdt-dat-expresidentes-corrupcion.csv" # campo 9-12 jun 2026
)
print(encuestas |> select(-fichero))

# ---------------------------------------------------------------------------
# 2. Vocabulario común
# ---------------------------------------------------------------------------

ccaa_cod <- read_csv(
  file.path(dir_pob, "ccaa_codigos.csv"),
  col_types = cols(.default = "c", cod_cis = "i")
)
ccaa_lv <- ccaa_cod$ccaa

# Provincia (código INE) -> CCAA. CIS y 40dB usan los códigos del INE
# (comprobado con las etiquetas). La tabla sale del propio CIS 3582.
cis_3582 <- read_sav("descargas/MD3582/3582.sav")
prov_ccaa <- cis_3582 |>
  distinct(prov = as.integer(PROV), cod_cis = as.integer(CCAA)) |>
  left_join(ccaa_cod |> select(ccaa, cod_cis), by = "cod_cis") |>
  select(prov, ccaa) |>
  arrange(prov)
stopifnot(nrow(prov_ccaa) == 52, !anyNA(prov_ccaa$ccaa))

# Las uniprovinciales: ahí la provincia se sabe aunque la encuesta no la dé
uniprov <- prov_ccaa |> group_by(ccaa) |> filter(n() == 1) |> ungroup()
cat("\n== CCAA uniprovinciales ==\n")
print(uniprov, n = Inf)

edad_lv <- c("18-20", "21-24", "25-34", "35-44", "45-54", "55-64", "65+")
tramo_edad <- function(edad) {
  cut(
    edad,
    breaks = c(18, 21, 25, 35, 45, 55, 65, Inf),
    labels = edad_lv,
    right = FALSE
  )
}
edu_lv <- c("baja", "media", "alta")

# Etiquetas de partido de 40dB y More in Common -> partidos_lv. Las del CIS
# usan recod_cis_partido() de 00_partidos.R.
recod_partido_otras <- function(x, no_validos) {
  x <- as.character(x)
  out <- case_when(
    x %in% no_validos ~ NA_character_,
    grepl("^PP\\b", x) ~ "PP",
    grepl("^PSOE\\b", x) ~ "PSOE",
    grepl("^Vox$", x) ~ "VOX",
    grepl("^Sumar", x) ~ "SUMAR",
    grepl("Podemos", x) ~ "PODEMOS",
    grepl("^ERC\\b", x) ~ "ERC",
    grepl("Junts", x) ~ "JUNTS",
    grepl("PNV", x) ~ "PNV",
    grepl("Bildu", x) ~ "BILDU",
    grepl("^BNG\\b", x) ~ "BNG",
    grepl("Adelante Andaluc", x) ~ "AA",
    x %in% c("Coalición Canaria", "CC") ~ "CCA",
    # 40dB: "UPN (Unión del Pueblo Navarro)"; MIC y Ateneo: "UPN"
    grepl("^UPN", x) ~ "UPN",
    is.na(x) ~ NA_character_,
    TRUE ~ "OTROS"
  )
  out
}

# Respuestas que no son voto válido
no_validos_40db_int <- c(
  "Votaría nulo",
  "No votaría",
  "No lo sé",
  "Prefiero no contestar"
)
no_validos_40db_rec <- c(
  "No voté",
  "No tenía edad para votar",
  "No lo sé",
  "Prefiero no contestar"
)
no_validos_mic_int <- c("Nulo", "No votaría", "NC", "NS")
no_validos_atdt_int <- c(
  "Nulo",
  "No votaría",
  "No lo sé",
  "Prefiero no contestar"
)
no_validos_atdt_rec <- c(
  "No votó",
  "No tenía edad para votar",
  "No lo sé",
  "Prefiero no contestar"
)

# ---------------------------------------------------------------------------
# 3. Una función por casa
# ---------------------------------------------------------------------------
# Salida común: sexo, edad, edu (y si es ambigua), ccaa, prov, voto,
# participación y partido del 23-J, prob. de votar, ideología en 0-1, peso.

lee_cis <- function(f) {
  d <- read_sav(f)
  recod_edu_cis <- function(nivel) {
    nivel <- as.character(nivel)
    case_when(
      grepl(
        "^Menos de 5|^Educación Primaria|^FP Básica|^Educación secundaria",
        nivel
      ) ~ "baja",
      grepl("^FP de Grado Medio|^Bachillerato", nivel) ~ "media",
      grepl(
        "^FP de Grado Superior|^Arquitectura|^Diplomatura|^Grado|^Licenciatura|^Máster|^Doctorado|^Títulos propios",
        nivel
      ) ~ "alta",
      TRUE ~ NA_character_
    )
  }
  d |>
    transmute(
      sexo = as.character(as_factor(SEXO)),
      edad_n = as.numeric(EDAD),
      edu = recod_edu_cis(as_factor(NIVELESTENTREV)),
      edu_ambigua = FALSE,
      prov = as.integer(PROV),
      ccaa_txt = NA_character_,
      voto = recod_cis_partido(
        as_factor(INTENCIONG),
        no_validos = c(
          "No sabe todavía",
          "N.C.",
          "N.R.",
          "N.P.",
          "No votaría",
          "Voto nulo"
        )
      ),
      # Voto + simpatía. VOTOSIMG lo da hecho, pero mete a Aliança Catalana
      # y otros pequeños en "Otro partido". Se rehace con SIMPATIA (pregunta
      # con filtro) y la misma regla que VOTOSIMG: la simpatía solo rellena a
      # quien dice "No sabe todavía", "N.C." o "No votaría".
      voto_sim = ifelse(
        as.character(as_factor(INTENCIONG)) %in%
          c("No sabe todavía", "N.C.", "No votaría"),
        recod_cis_partido(
          as_factor(SIMPATIA),
          no_validos = c("N.S.", "N.C.", "N.R.", "N.P.", "Ninguno")
        ),
        recod_cis_partido(
          as_factor(INTENCIONG),
          no_validos = c(
            "No sabe todavía",
            "N.C.",
            "N.R.",
            "N.P.",
            "No votaría",
            "Voto nulo"
          )
        )
      ),
      participacion = case_when(
        as_factor(PARTICIPACIONG) == "Votó" ~ "voto",
        as_factor(PARTICIPACIONG) == "No votó" ~ "abst",
        as_factor(PARTICIPACIONG) == "No tenía edad para votar" ~ "sin_edad",
        TRUE ~ NA_character_
      ),
      partido_rec = recod_cis_partido(
        as_factor(RECUVOTOG),
        no_validos = c("N.C.", "N.R.", "N.P."),
        # AA no se presentó en las generales del 23-J: el recuerdo de AA va a OTROS
        aa = FALSE
      ),
      prob_voto = ifelse(
        as.numeric(PROBVOTO) %in% 0:10,
        as.numeric(PROBVOTO),
        NA
      ),
      ideol01 = ifelse(
        as.numeric(ESCIDEOL) %in% 1:10,
        (as.numeric(ESCIDEOL) - 1) / 9,
        NA
      ),
      peso = as.numeric(PESO)
    )
}

lee_40db <- function(f) {
  d <- read_sav(f)
  # El número de cada pregunta cambia entre oleadas: se buscan por etiqueta
  etq <- sapply(d, function(x) attr(x, "label") %||% "")
  busca <- function(patron) {
    v <- names(etq)[grepl(patron, etq)]
    stopifnot(length(v) >= 1)
    v[1] # la primera: en el flash, la pregunta general va antes que la del escenario
  }
  v_prob <- busca("probabilidad de que acud")
  v_int <- busca("^Intención de voto elecciones generales - Y")
  v_rec <- busca("^Recuerdo de voto")
  v_ideo <- busca("izquierdas")
  v_sim <- busca("^Simpatía en elecciones generales - En todo")
  cat(
    basename(f),
    ": prob =",
    v_prob,
    " intención =",
    v_int,
    " simpatía =",
    v_sim,
    " recuerdo =",
    v_rec,
    " ideología =",
    v_ideo,
    "\n"
  )

  # Voto + simpatía con la regla del CIS: 40dB pregunta la simpatía a todos,
  # pero solo se usa para quien no da partido en la intención directa
  int_txt <- as.character(as_factor(d[[v_int]]))
  sin_partido <- int_txt %in%
    c("No lo sé", "No votaría", "Prefiero no contestar")
  voto_dir <- recod_partido_otras(int_txt, no_validos_40db_int)
  voto_simp <- recod_partido_otras(
    as_factor(d[[v_sim]]),
    c("Ninguno", "No lo sé", "Prefiero no contestar")
  )

  num_0_10 <- function(x) {
    n <- suppressWarnings(as.numeric(sub(
      "^(\\d+).*",
      "\\1",
      as.character(as_factor(x))
    )))
    ifelse(n %in% 0:10, n, NA)
  }
  rec_txt <- as.character(as_factor(d[[v_rec]]))
  edu_r <- as.character(as_factor(d$educacion_r))

  d |>
    transmute(
      sexo = as.character(as_factor(sexo)),
      edad_n = as.numeric(edad),
      edu = case_when(
        edu_r == "Segundo grado, primer ciclo y menos" ~ "baja",
        edu_r == "Segundo grado, segundo ciclo" ~ "media",
        grepl("^Tercer grado", edu_r) ~ "alta",
        TRUE ~ NA_character_
      ),
      # "Segundo grado, segundo ciclo" junta Bachillerato con la FP I y II
      # antiguas: parte es FP superior, que el CIS y la EPA cuentan como alta
      edu_ambigua = edu_r == "Segundo grado, segundo ciclo",
      prov = as.integer(prov),
      ccaa_txt = NA_character_,
      voto = voto_dir,
      voto_sim = ifelse(sin_partido, voto_simp, voto_dir),
      participacion = case_when(
        rec_txt == "No voté" ~ "abst",
        rec_txt == "No tenía edad para votar" ~ "sin_edad",
        rec_txt %in%
          c("No lo sé", "Prefiero no contestar") |
          is.na(rec_txt) ~ NA_character_,
        TRUE ~ "voto"
      ),
      partido_rec = recod_partido_otras(rec_txt, no_validos_40db_rec),
      prob_voto = num_0_10(d[[v_prob]]),
      ideol01 = num_0_10(d[[v_ideo]]) / 10,
      peso = as.numeric(ponde)
    )
}

lee_mic <- function(f) {
  d <- read_sav(f)
  niv <- as.character(as_factor(d$nivel_estudios))
  part <- as.character(as_factor(d$participacion_2023_raw))
  d |>
    transmute(
      sexo = as.character(as_factor(genero)),
      edad_n = as.numeric(edad),
      edu = case_when(
        niv %in% c("Primaria", "Secundaria/ESO") ~ "baja",
        niv %in% c("Bachillerato", "FP") ~ "media",
        niv == "Universitaria" ~ "alta",
        TRUE ~ NA_character_
      ),
      # "FP" no separa grado medio (media) y superior (alta)
      edu_ambigua = niv %in% "FP",
      prov = NA_integer_,
      ccaa_txt = as.character(as_factor(comunidad_autonoma)),
      voto = recod_partido_otras(
        as_factor(intencion_voto_raw),
        no_validos_mic_int
      ),
      # More in Common lo da hecho, con la misma regla que el CIS (B5 + B5a)
      voto_sim = recod_partido_otras(
        as_factor(intencion_voto_simpatia),
        no_validos_mic_int
      ),
      participacion = case_when(
        part == "Votó" ~ "voto",
        part %in% c("No pudo", "No suele votar") ~ "abst",
        part == "Sin edad" ~ "sin_edad",
        TRUE ~ NA_character_ # "No recuerda"
      ),
      partido_rec = recod_partido_otras(as_factor(voto_2023_raw), character(0)),
      prob_voto = ifelse(
        as.numeric(prob_voto_declarada) %in% 0:10,
        as.numeric(prob_voto_declarada),
        NA
      ),
      ideol01 = ifelse(
        as.numeric(ideologia) %in% 0:10,
        as.numeric(ideologia) / 10,
        NA
      ),
      peso = as.numeric(peso)
    )
}

lee_atdt <- function(f) {
  d <- read_csv(f, col_types = cols(.default = "c", ideol = "i", ponde = "d"))
  # Sin edad exacta: solo tramos, con 18-24 junto. Quien "no tenía edad" el
  # 23-J tiene 18-20; el resto de 18-24 va a 21-24.
  edad_txt <- case_when(
    d$edad_r == "18-24" & d$recuerdo == "No tenía edad para votar" ~ "18-20",
    d$edad_r == "18-24" ~ "21-24",
    TRUE ~ d$edad_r
  )
  sin_partido <- d$intencion %in%
    c("No lo sé", "No votaría", "Prefiero no contestar")
  voto_dir <- recod_partido_otras(d$intencion, no_validos_atdt_int)
  voto_simp <- recod_partido_otras(
    d$simpatia,
    c("Ninguno", "No lo sé", "Prefiero no contestar")
  )
  d |>
    transmute(
      sexo = sexo,
      edad_n = NA_real_,
      edad_txt = edad_txt,
      edu = case_when(
        educacion_r %in% c("Primaria", "Secundaria 1er ciclo") ~ "baja",
        educacion_r == "Secundaria 2do ciclo" ~ "media",
        educacion_r == "Superior" ~ "alta",
        TRUE ~ NA_character_
      ),
      # Como en 40dB: "Secundaria 2do ciclo" no separa Bachillerato de FP
      edu_ambigua = educacion_r == "Secundaria 2do ciclo",
      prov = NA_integer_,
      ccaa_txt = ccaa,
      voto = voto_dir,
      voto_sim = ifelse(sin_partido, voto_simp, voto_dir),
      participacion = case_when(
        recuerdo == "No votó" ~ "abst",
        recuerdo == "No tenía edad para votar" ~ "sin_edad",
        recuerdo %in%
          c("No lo sé", "Prefiero no contestar") |
          is.na(recuerdo) ~ NA_character_,
        TRUE ~ "voto"
      ),
      partido_rec = recod_partido_otras(recuerdo, no_validos_atdt_rec),
      prob_voto = NA_real_, # no la pregunta: no entra en el modelo de participación
      ideol01 = ifelse(ideol %in% 0:10, ideol / 10, NA), # 11 = NS/NC (deducido, sin codebook)
      peso = ponde
    )
}

lectores <- list(
  CIS = lee_cis,
  `40dB` = lee_40db,
  `More in Common` = lee_mic,
  `Ateneo del Dato` = lee_atdt
)

cat("\n== variables de 40dB localizadas por etiqueta ==\n")
todas <- bind_rows(lapply(seq_len(nrow(encuestas)), function(i) {
  e <- encuestas[i, ]
  lectores[[e$casa]](e$fichero) |>
    mutate(casa = e$casa, estudio = e$estudio, mes = e$mes, .before = 1)
}))

# ---------------------------------------------------------------------------
# 4. CCAA, provincia, edad y recuerdo
# ---------------------------------------------------------------------------

# Nombres de CCAA de More in Common -> los de la tabla puente
ccaa_mic <- c(
  "Andalucía" = "Andalucía",
  "Aragón" = "Aragón",
  "Principado de Asturias" = "Asturias",
  "Islas Baleares" = "Baleares",
  "Islas Canarias" = "Canarias",
  "Cantabria" = "Cantabria",
  "Castilla-La Mancha" = "Castilla-La Mancha",
  "Castilla y León" = "Castilla y León",
  "Cataluña" = "Cataluña",
  "Comunidad Valenciana" = "Comunitat Valenciana",
  "Extremadura" = "Extremadura",
  "Galicia" = "Galicia",
  "Comunidad de Madrid" = "Madrid",
  "Región de Murcia" = "Murcia",
  "Comunidad Foral de Navarra" = "Navarra",
  "País Vasco" = "País Vasco",
  "La Rioja" = "La Rioja",
  "Ciudad Autónoma de Ceuta" = "Ceuta",
  "Ciudad Autónoma de Melilla" = "Melilla"
)
# Ateneo usa los nombres cortos; solo cambian estos dos
ccaa_atdt <- c(
  "Illes Balears" = "Baleares",
  "Comunidad Valenciana" = "Comunitat Valenciana",
  setNames(ccaa_lv, ccaa_lv)
)
ccaa_txt_lv <- c(ccaa_mic, ccaa_atdt[!names(ccaa_atdt) %in% names(ccaa_mic)])
stopifnot(all(na.omit(unique(todas$ccaa_txt)) %in% names(ccaa_txt_lv)))

todas <- todas |>
  left_join(prov_ccaa, by = "prov") |>
  mutate(ccaa = coalesce(ccaa, unname(ccaa_txt_lv[ccaa_txt]))) |>
  # More in Common y Ateneo: la provincia se conoce en las uniprovinciales
  left_join(uniprov |> rename(prov_uni = prov), by = "ccaa") |>
  mutate(prov = coalesce(prov, prov_uni)) |>
  select(-prov_uni, -ccaa_txt) |>
  mutate(
    ccaa = factor(ccaa, levels = ccaa_lv),
    sexo = factor(sexo, levels = c("Hombre", "Mujer"), labels = c("H", "M")),
    # Ateneo trae el tramo ya hecho (edad_txt); el resto, la edad en años
    edad = factor(
      coalesce(as.character(tramo_edad(edad_n)), edad_txt),
      levels = edad_lv
    ),
    edu = factor(edu, levels = edu_lv)
  )

# Partidos regionales fuera de su comunidad -> OTROS (como en el post del CIS).
# Aliança Catalana no existía en 2023: en el recuerdo va con OTROS.
fuera <- function(p, ccaa) !is.na(p) & !se_presenta(p, ccaa)
cat("\n== respuestas fuera de territorio (van a OTROS) ==\n")
print(
  todas |>
    group_by(estudio) |>
    summarise(
      intencion = sum(fuera(voto, ccaa)),
      voto_sim = sum(fuera(voto_sim, ccaa)),
      recuerdo = sum(fuera(partido_rec, ccaa))
    )
)

todas <- todas |>
  mutate(
    voto = ifelse(fuera(voto, ccaa), "OTROS", voto),
    voto_sim = ifelse(fuera(voto_sim, ccaa), "OTROS", voto_sim),
    voto_sim = factor(voto_sim, levels = partidos_lv),
    partido_rec = ifelse(fuera(partido_rec, ccaa), "OTROS", partido_rec),
    partido_rec = ifelse(partido_rec == "AC", "OTROS", partido_rec),
    # El recuerdo es del 23-J, donde Podemos iba dentro de Sumar: se pliega
    # para que rec_lv no cambie.
    partido_rec = ifelse(partido_rec == "PODEMOS", "SUMAR", partido_rec),
    voto = factor(voto, levels = partidos_lv),
    # Recuerdo: quien tiene 18-20 años no pudo votar el 23-J (regla del post del CIS)
    rec = case_when(
      edad == "18-20" ~ "NO_PODIA",
      participacion == "abst" ~ "ABST",
      participacion == "voto" ~ partido_rec,
      TRUE ~ NA_character_ # sin edad con 21+, no recuerda, no contesta
    ),
    rec = factor(rec, levels = rec_lv),
    vota_seguro = as.integer(prob_voto == 10)
  ) |>
  select(
    casa,
    estudio,
    mes,
    sexo,
    edad_n,
    edad,
    edu,
    edu_ambigua,
    ccaa,
    prov,
    voto,
    voto_sim,
    rec,
    prob_voto,
    vota_seguro,
    ideol01,
    peso
  )

# ---------------------------------------------------------------------------
# 5. Comprobaciones
# ---------------------------------------------------------------------------

cat(
  "\n== entrevistas por encuesta, y cuántas sirven para el modelo de voto ==\n"
)
print(
  todas |>
    group_by(casa, estudio, mes) |>
    summarise(
      n = n(),
      con_voto = sum(!is.na(voto)),
      con_voto_sim = sum(!is.na(voto_sim)),
      con_rec = sum(!is.na(rec)),
      sin_prov = sum(is.na(prov)),
      edu_ambigua = sum(edu_ambigua, na.rm = TRUE),
      .groups = "drop"
    ),
  n = Inf
)

cat("\n== faltantes por variable ==\n")
print(
  todas |>
    summarise(across(
      c(sexo, edad, edu, ccaa, prov, voto, rec, prob_voto, ideol01, peso),
      ~ sum(is.na(.x))
    ))
)

cat("\n== intención de voto sobre voto válido, bruta (%) ==\n")
print(
  todas |>
    filter(!is.na(voto)) |>
    count(estudio, voto) |>
    group_by(estudio) |>
    mutate(pct = round(100 * n / sum(n), 1)) |>
    select(-n) |>
    pivot_wider(names_from = estudio, values_from = pct) |>
    as.data.frame(),
  row.names = FALSE
)

cat("\n== voto + simpatía sobre voto válido, bruto (%) ==\n")
print(
  todas |>
    filter(!is.na(voto_sim)) |>
    count(estudio, voto_sim) |>
    group_by(estudio) |>
    mutate(pct = round(100 * n / sum(n), 1)) |>
    select(-n) |>
    pivot_wider(names_from = estudio, values_from = pct) |>
    as.data.frame(),
  row.names = FALSE
)

cat("\n== recuerdo 23-J (% del electorado de la muestra) ==\n")
print(
  todas |>
    filter(!is.na(rec)) |>
    count(estudio, rec) |>
    group_by(estudio) |>
    mutate(pct = round(100 * n / sum(n), 1)) |>
    select(-n) |>
    pivot_wider(names_from = estudio, values_from = pct) |>
    as.data.frame(),
  row.names = FALSE
)

saveRDS(todas, file.path(dir_datos, "encuestas.rds"))
cat("\nguardado encuestas.rds:", nrow(todas), "entrevistas\n")
