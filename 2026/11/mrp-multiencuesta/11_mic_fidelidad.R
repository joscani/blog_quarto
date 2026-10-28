# ============================================================================
# FIDELIDAD DE VOTO Y RECUERDO 23J: MORE IN COMMON FRENTE A CIS Y 40dB
#
# Replica la nota de PLAN.md ("Efecto casa de More in Common a favor del
# PSOE", 2026-10-21): con el MRP, More in Common da más PSOE que CIS y 40dB
# aunque su recuerdo PSOE no es el más inflado. La diferencia está en la
# transición recuerdo 2023 -> intención: fidelidad PSOE -> PSOE y paso de la
# abstención de 2023 al PSOE.
#
# Entrada: datos/encuestas.rds (01_armoniza.R). Recuerdo y voto directo en
# bruto, sin simpatía: en More in Common, voto_2023_raw +
# participacion_2023_raw e intencion_voto_raw (no las _final, que ya están
# imputadas por su modelo).
#
# Definiciones (proporciones en %):
#   recuerdo_PSOE = P(rec = PSOE | rec es un partido)
#   fid_PSOE      = P(voto = PSOE | rec = PSOE, voto es un partido)
#   fid_PP        = P(voto = PP   | rec = PP,   voto es un partido)
#   abst_PSOE     = P(voto = PSOE | rec = ABST, voto es un partido)
#   sin_partido   = P(voto sin partido | rec = PSOE): indecisos, blanco, nulo
#                   y no votaría entre los que recuerdan PSOE
# "voto es un partido" excluye no sabe, no contesta, blanco, nulo y no
# votaría. Cada cifra sale en bruto (sin pesos) y con el peso de la casa.
#
# Salidas (en esta carpeta):
#   mic_fidelidad_casa.csv        por casa, todos los meses
#   mic_fidelidad_septiembre.csv  por casa, solo 2026-09 (único mes con las
#                                 tres casas: CIS 3577, 40dB sep, MIC sep)
#   mic_fidelidad_estudio.csv     por estudio
#   mic_fidelidad_educacion.csv   PSOE -> PSOE y ABST -> PSOE por nivel de
#                                 estudios, y estandarizadas a la población
#
# More in Common no pondera por estudios. Para ver si eso explica la
# fidelidad alta, la tasa de cada casa se estandariza con la mezcla de
# estudios de la población que recuerda PSOE (o abstención), sacada de
# datos/tabla_post_prov.rds:
#   estandarizada = sum_e P(voto = PSOE | rec, e) * P(e | rec) en la población
# Ojo: la codificación de estudios puede no coincidir entre casas (40dB y
# More in Common tienen niveles marcados en edu_ambigua), así que la
# comparación nivel a nivel entre casas es aproximada.
#
# Uso (desde la raíz del proyecto):
#   Rscript 2026/11/mrp-multiencuesta/11_mic_fidelidad.R
# ============================================================================

suppressMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
})

dir_mrp <- here::here("2026/11/mrp-multiencuesta")
partidos_rec <- c(
  "PP", "PSOE", "VOX", "SUMAR", "ERC", "JUNTS", "BILDU", "PNV", "BNG", "CCA",
  "OTROS"
)

d <- readRDS(file.path(dir_mrp, "datos/encuestas.rds")) |>
  filter(casa %in% c("CIS", "40dB", "More in Common")) |>
  mutate(
    rec = as.character(rec),
    voto = as.character(voto),
    con_partido = !is.na(voto)
  )

# proporción ponderada (w = 1 para bruto), en %
pct <- function(x, w) 100 * sum(w[x]) / sum(w)

resumen <- function(df, w) {
  df <- mutate(df, w = {{ w }})
  tibble(
    n = nrow(df),
    n_rec_PSOE = sum(df$rec %in% "PSOE"),
    recuerdo_PSOE = with(
      filter(df, rec %in% partidos_rec),
      pct(rec == "PSOE", w)
    ),
    fid_PSOE = with(
      filter(df, rec %in% "PSOE", con_partido),
      pct(voto == "PSOE", w)
    ),
    fid_PP = with(filter(df, rec %in% "PP", con_partido), pct(voto == "PP", w)),
    abst_PSOE = with(
      filter(df, rec %in% "ABST", con_partido),
      pct(voto == "PSOE", w)
    ),
    sin_partido = with(filter(df, rec %in% "PSOE"), pct(!con_partido, w))
  )
}

por <- function(df, ...) {
  bind_rows(
    df |> group_by(...) |> group_modify(\(g, k) resumen(g, 1)) |>
      mutate(pesos = "bruto"),
    df |> group_by(...) |> group_modify(\(g, k) resumen(g, peso)) |>
      mutate(pesos = "peso de la casa")
  ) |>
    ungroup() |>
    relocate(pesos, .after = last_col()) |>
    mutate(across(recuerdo_PSOE:sin_partido, \(x) round(x, 1)))
}

t_casa <- por(d, casa)
t_sep <- por(filter(d, mes == "2026-09"), casa)
t_est <- por(d, casa, estudio)

write_csv(t_casa, file.path(dir_mrp, "mic_fidelidad_casa.csv"))
write_csv(t_sep, file.path(dir_mrp, "mic_fidelidad_septiembre.csv"))
write_csv(t_est, file.path(dir_mrp, "mic_fidelidad_estudio.csv"))

options(width = 150)
cat("\n== Por casa, todos los meses\n")
print(as.data.frame(t_casa))
cat("\n== Solo 2026-09\n")
print(as.data.frame(t_sep))
cat("\n== Por estudio\n")
print(as.data.frame(t_est))

# ---- estudios: ¿explica la falta de peso por estudios la fidelidad? ---------

pob_edu <- as.data.frame(
  readRDS(file.path(dir_mrp, "datos/tabla_post_prov.rds")),
  stringsAsFactors = FALSE
) |>
  filter(rec %in% c("PSOE", "ABST")) |>
  group_by(rec, edu) |>
  summarise(N = sum(Freq), .groups = "drop") |>
  group_by(rec) |>
  mutate(pob = 100 * N / sum(N)) |>
  ungroup() |>
  select(rec, edu, pob)

d_edu <- d |>
  filter(!is.na(edu), rec %in% c("PSOE", "ABST")) |>
  mutate(edu = as.character(edu))

# composición de estudios en la muestra (con peso de la casa), por recuerdo
comp_edu <- d_edu |>
  group_by(rec, casa, edu) |>
  summarise(w = sum(peso), .groups = "drop") |>
  group_by(rec, casa) |>
  mutate(muestra = 100 * w / sum(w)) |>
  ungroup() |>
  select(-w)

# tasa rec -> PSOE por estudios, entre quienes declaran partido
tasa_edu <- d_edu |>
  filter(con_partido) |>
  group_by(rec, casa, edu) |>
  summarise(
    n = n(),
    bruto = pct(voto == "PSOE", rep(1, n())),
    peso = pct(voto == "PSOE", peso),
    .groups = "drop"
  )

t_edu <- tasa_edu |>
  left_join(comp_edu, by = c("rec", "casa", "edu")) |>
  left_join(pob_edu, by = c("rec", "edu"))

t_estand <- t_edu |>
  group_by(rec, casa) |>
  summarise(
    estandarizada_bruto = sum(bruto * pob) / 100,
    estandarizada_peso = sum(peso * pob) / 100,
    .groups = "drop"
  ) |>
  left_join(
    d_edu |>
      filter(con_partido) |>
      group_by(rec, casa) |>
      summarise(observada_peso = pct(voto == "PSOE", peso), .groups = "drop"),
    by = c("rec", "casa")
  )

write_csv(
  bind_rows(
    t_edu,
    t_estand |> mutate(edu = "total estandarizado")
  ) |>
    mutate(across(where(is.double), \(x) round(x, 1))),
  file.path(dir_mrp, "mic_fidelidad_educacion.csv")
)

cat("\n== Rec -> PSOE por estudios (%): tasa, % en la muestra y en la población\n")
print(as.data.frame(mutate(t_edu, across(where(is.double), \(x) round(x, 1)))))
cat("\n== Rec -> PSOE: observada frente a estandarizada a la población (%)\n")
print(as.data.frame(mutate(t_estand, across(where(is.double), \(x) round(x, 1)))))

# ---- comprobación directa en el .sav de More in Common ----------------------
# Mismo cálculo con los nombres de variable de su fichero, para que lo puedan
# repetir sin nuestra armonización. Recuerdo PSOE = voto_2023_raw == "PSOE"
# entre los que recuerdan partido (sin blanco ni nulo); abstención 2023 =
# participacion_2023_raw "No suele votar" (la pregunta no dice "no votó").

f_mic <- here::here(
  "descargas/microdatos-2026-09/microdatos-septiembre-2026.sav"
)
if (file.exists(f_mic)) {
  m <- haven::read_sav(f_mic) |>
    mutate(
      part = as.character(haven::as_factor(participacion_2023_raw)),
      rec = as.character(haven::as_factor(voto_2023_raw)),
      int = as.character(haven::as_factor(intencion_voto_raw)),
      int = ifelse(grepl("^Sumar", int), "Sumar", int),
      int_partido = !int %in% c("Blanco", "Nulo", "No votaría", "NC", "NS")
    )
  no_partido_rec <- c("Blanco", "Nulo")
  sav <- bind_rows(lapply(c("bruto", "peso"), \(p) {
    m2 <- mutate(m, w = if (p == "bruto") 1 else as.numeric(peso))
    tibble(
      pesos = p,
      recuerdo_PSOE = with(
        filter(m2, !is.na(rec), !rec %in% no_partido_rec),
        pct(rec == "PSOE", w)
      ),
      fid_PSOE = with(
        filter(m2, rec %in% "PSOE", int_partido),
        pct(int == "PSOE", w)
      ),
      fid_PP = with(filter(m2, rec %in% "PP", int_partido), pct(int == "PP", w)),
      abst_PSOE = with(
        filter(m2, part == "No suele votar", int_partido),
        pct(int == "PSOE", w)
      )
    )
  })) |>
    mutate(across(recuerdo_PSOE:abst_PSOE, \(x) round(x, 1)))
  cat("\n== More in Common sep, directo del .sav\n")
  print(as.data.frame(sav))
}
