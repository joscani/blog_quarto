# Los comentarios de X al mapa del ganador, comprobados todos igual:
# bruto, ponderada sin recuerdo, ponderada con recuerdo y MRP (media, intervalo
# del 90 % y probabilidad de ser el primero) para un grupo de provincia(s) y
# edades. Lo usa el post (IV) vía iv-funciones.R.
#
# Las ponderadas se hacen con toda la comunidad (ponderaciones.R, como en
# 23_bruto_vs_mrp_edad.R y 24_mapa_ganador_bruto_mrp.R) y después se filtra el
# grupo. El % es siempre de voto válido entre quien da partido (voto + simpatía,
# seguro o no).
#
# Salida: completo/datos/comentarios.rds
#
#   Rscript 2026/11/mrp-multiencuesta/completo/35_comentarios.R   (~2 min)

library(dplyr)
library(tidyr)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo/datos")
source(here::here("2026/11/mrp-multiencuesta/completo/ponderaciones.R"))

# ---- Los casos -------------------------------------------------------------------
# provincias = NULL: toda la comunidad

casos <- list(
  list(id = "ourense", leccion = "A", comunidad = "Galicia", provincias = "Ourense",
       edades = c("18-20", "21-24", "25-34"), etiqueta = "Ourense, 18-34"),
  list(id = "badajoz", leccion = "B", comunidad = "Extremadura", provincias = "Badajoz",
       edades = c("18-20", "21-24", "25-34"), etiqueta = "Badajoz, 18-34"),
  list(id = "navarra", leccion = "B", comunidad = "Navarra", provincias = NULL,
       edades = c("18-20", "21-24", "25-34"), etiqueta = "Navarra, 18-34"),
  list(id = "navarra_total", leccion = "D", comunidad = "Navarra", provincias = NULL,
       edades = c("18-20", "21-24", "25-34", "35-44", "45-54", "55-64", "65+"),
       etiqueta = "Navarra, todas las edades"),
  list(id = "cataluna_25_34", leccion = "C", comunidad = "Cataluña", provincias = NULL,
       edades = "25-34", etiqueta = "Cataluña, 25-34"),
  list(id = "cataluna_18_20", leccion = "D", comunidad = "Cataluña", provincias = NULL,
       edades = "18-20", etiqueta = "Cataluña, 18-20"),
  list(id = "pais_vasco_18_20", leccion = "D", comunidad = "País Vasco", provincias = NULL,
       edades = "18-20", etiqueta = "País Vasco, 18-20"),
  list(id = "galicia_18_20", leccion = "?", comunidad = "Galicia", provincias = NULL,
       edades = "18-20", etiqueta = "Galicia, 18-20"),
  list(id = "galicia_menos_45", leccion = "?", comunidad = "Galicia", provincias = NULL,
       edades = c("18-20", "21-24", "25-34", "35-44"), etiqueta = "Galicia, menos de 45")
)

encuestas_todas <- readRDS(file.path(dir_completo, "encuestas_completo.rds"))
# VARIANTE=semilla_mn_gp (16 nov 2026): lee tabla_post y posterior_celdas_80 de esa
# variante y escribe con el sufijo _semilla_mn_gp (no toca lo publicado)
variante <- Sys.getenv("VARIANTE", "")
stopifnot(variante %in% c("", "semilla_mn", "semilla_mn_gp"))
suf <- if (variante == "") "" else paste0("_", variante)
celdas_todas <- readRDS(file.path(dir_completo, paste0("tabla_post", suf, ".rds")))
posterior <- readRDS(file.path(dir_completo, paste0("posterior_celdas_80", suf, ".rds")))
partidos <- setdiff(names(posterior), c("draw", "celda", "part"))

# % de voto por partido a partir de pesos (1 = bruto)
pct_ponderado <- function(d) {
  d |>
    summarise(votos = sum(peso), .by = partido) |>
    mutate(pct = 100 * votos / sum(votos)) |>
    select(partido, pct)
}

calcula <- function(caso) {
  enc_ccaa <- encuestas_todas |> filter(as.character(ccaa) == caso$comunidad)
  cel_ccaa <- celdas_todas |> filter(ccaa == caso$comunidad)
  provincias <- caso$provincias %||% unique(cel_ccaa$provincia)
  cods <- unique(cel_ccaa$cod[cel_ccaa$provincia %in% provincias])
  stopifnot(length(cods) == length(provincias))
  # toda la comunidad: también las entrevistas sin provincia (More in Common y
  # Ateneo fuera de las uniprovinciales). Las ponderadas no pueden usarlas: el
  # raking lleva la provincia
  en_grupo <- function(d) {
    d <- d |> filter(as.character(edad) %in% caso$edades)
    if (is.null(caso$provincias)) d else filter(d, as.integer(as.character(prov)) %in% cods)
  }

  # 1. bruto
  bruto_d <- enc_ccaa |> filter(!is.na(voto_sim)) |> en_grupo() |>
    mutate(partido = as.character(voto_sim), peso = 1)
  # 2 y 3. ponderadas con toda la comunidad, y luego el grupo
  sin_d <- pondera_sin_rec(enc_ccaa, cel_ccaa) |> filter(!is.na(voto_sim)) |> en_grupo() |>
    mutate(partido = as.character(voto_sim), peso = peso_rk)
  con_d <- pondera_con_rec(enc_ccaa, cel_ccaa) |> filter(!is.na(voto_sim)) |> en_grupo() |>
    mutate(partido = as.character(voto_sim), peso = peso_rk)

  # 4. MRP: votos = N * part * p, sumados en cada draw
  cel_grupo <- cel_ccaa |> filter(cod %in% cods, as.character(edad) %in% caso$edades)
  draws <- posterior |>
    filter(celda %in% cel_grupo$celda) |>
    left_join(select(cel_grupo, celda, N), by = "celda") |>
    summarise(across(all_of(partidos), \(p) sum(N * part * p)), .by = draw) |>
    pivot_longer(all_of(partidos), names_to = "partido", values_to = "votos") |>
    mutate(pct = 100 * votos / sum(votos), primero = votos == max(votos), .by = draw)
  stopifnot(n_distinct(draws$draw) == 1000,
            all(abs(summarise(draws, s = sum(pct), .by = draw)$s - 100) < 1e-6))
  mrp <- draws |>
    # los cuantiles antes que la media: summarise sobrescribe pct al momento
    summarise(q05 = quantile(pct, 0.05), q95 = quantile(pct, 0.95),
              p_primero = 100 * mean(primero), pct = mean(pct), .by = partido)
  stopifnot(all(mrp$q05 < mrp$pct | mrp$pct == 0), all(mrp$pct < mrp$q95 | mrp$pct == 0))

  fuentes <- bind_rows(
    pct_ponderado(bruto_d) |> mutate(fuente = "bruto"),
    pct_ponderado(sin_d) |> mutate(fuente = "sin_rec"),
    pct_ponderado(con_d) |> mutate(fuente = "con_rec"),
    mrp |> mutate(fuente = "mrp")
  ) |>
    mutate(id = caso$id, leccion = caso$leccion, etiqueta = caso$etiqueta)
  stopifnot(all(abs(summarise(fuentes, s = sum(pct), .by = fuente)$s - 100) < 1e-6))

  entrevistas <- tibble(id = caso$id, fuente = c("bruto", "sin_rec", "con_rec"),
                        n = c(nrow(bruto_d), nrow(sin_d), nrow(con_d)),
                        n_casa_cis = c(sum(bruto_d$casa == "CIS"), NA, NA))
  list(fuentes = fuentes, entrevistas = entrevistas)
}

res <- lapply(casos, calcula)
fuentes <- bind_rows(lapply(res, `[[`, "fuentes"))
entrevistas <- bind_rows(lapply(res, `[[`, "entrevistas"))

# Comprobación independiente del bruto: Navarra 18-34, contado a mano
nav <- encuestas_todas |>
  filter(as.character(ccaa) == "Navarra", edad %in% c("18-20", "21-24", "25-34"), !is.na(voto_sim))
stopifnot(abs(100 * mean(nav$voto_sim == "VOX") -
                fuentes$pct[fuentes$id == "navarra" & fuentes$fuente == "bruto" & fuentes$partido == "VOX"]) < 1e-9)

# En consola: los 3 primeros del MRP de cada caso, en las cuatro fuentes
top <- fuentes |>
  filter(fuente == "mrp", !partido %in% c("OTROS", "BLANCO")) |>
  slice_max(pct, n = 3, by = id) |>
  select(id, partido) |>
  bind_rows(tibble(id = c("galicia_18_20", "galicia_menos_45", "ourense"), partido = "BNG")) |>
  distinct()
resumen <- fuentes |>
  semi_join(top, by = c("id", "partido")) |>
  select(id, partido, fuente, pct) |>
  pivot_wider(names_from = fuente, values_from = pct)
print(resumen |> mutate(across(where(is.double), \(x) round(x, 1))), n = Inf)
print(entrevistas |> filter(fuente == "bruto"))

saveRDS(list(casos = casos, fuentes = fuentes, entrevistas = entrevistas),
        file.path(dir_completo, paste0("comentarios", suf, ".rds")))
cat("guardado comentarios.rds\n")
