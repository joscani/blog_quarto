# Voto de los jóvenes (18-34) en una comunidad y sus provincias: MRP con
# intervalo y probabilidad de ser primero, y la encuesta en bruto.
# Para contestar comentarios de X (Navarra, Cataluña, Badajoz...).
#
#   Rscript 2026/11/mrp-multiencuesta/completo/32_jovenes_provincia.R "Extremadura"
#
# Salida: completo/datos/jovenes_<comunidad>.rds (mrp y bruto)

library(dplyr)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo/datos")
args <- commandArgs(trailingOnly = TRUE)
comunidad <- if (length(args) > 0) args[[1]] else "Navarra"
slug <- comunidad |> iconv(to = "ASCII//TRANSLIT") |> tolower() |>
  stringr::str_remove_all("[^a-z ]") |> stringr::str_replace_all(" +", "_")
jovenes <- c("18-20", "21-24", "25-34")

celdas <- readRDS(file.path(dir_completo, "tabla_post.rds")) |>
  filter(ccaa == comunidad, edad %in% jovenes)
stopifnot(nrow(celdas) > 0)
posterior <- readRDS(file.path(dir_completo, "posterior_celdas_80.rds")) |>
  filter(celda %in% celdas$celda) |>
  left_join(celdas |> select(celda, N, provincia), by = "celda")
partidos <- setdiff(names(posterior), c("draw", "celda", "part", "N", "provincia"))

# % de voto válido en cada draw, para la comunidad y para cada provincia
pct_draws <- function(d) {
  d |>
    group_by(zona, draw) |>
    summarise(across(all_of(partidos), \(p) sum(N * part * p)), .groups = "drop") |>
    tidyr::pivot_longer(all_of(partidos), names_to = "partido", values_to = "votos") |>
    mutate(pct = 100 * votos / sum(votos), .by = c(zona, draw))
}
# en las uniprovinciales la provincia es la comunidad: no se repite
draws <- pct_draws(posterior |> mutate(zona = comunidad))
if (n_distinct(celdas$provincia) > 1) {
  draws <- bind_rows(draws, pct_draws(posterior |> mutate(zona = provincia)))
}
stopifnot(all(abs(summarise(draws, s = sum(pct), .by = c(zona, draw))$s - 100) < 1e-6))

mrp <- draws |>
  mutate(primero = pct == max(pct), .by = c(zona, draw)) |>
  summarise(media = mean(pct), q05 = quantile(pct, 0.05), q95 = quantile(pct, 0.95),
            p_primero = 100 * mean(primero), .by = c(zona, partido)) |>
  arrange(zona, desc(media))
print(mrp |> filter(media > 5) |> mutate(across(where(is.numeric), \(x) round(x, 1))), n = Inf)

# Bruto: entrevistas de 18-34 con voto + simpatía, sin ponderar
encuestas <- readRDS(file.path(dir_completo, "encuestas_completo.rds")) |>
  filter(as.character(ccaa) == comunidad, edad %in% jovenes, !is.na(voto_sim))
bruto <- encuestas |>
  count(partido = as.character(voto_sim)) |>
  mutate(pct = 100 * n / sum(n)) |>
  arrange(desc(n))
cat("\nBruto, 18-34 en", comunidad, ": n =", nrow(encuestas), "\n")
print(bruto |> mutate(pct = round(pct, 1)))
print(encuestas |> count(casa))

saveRDS(list(mrp = mrp, bruto = bruto, n_casa = count(encuestas, casa)),
        file.path(dir_completo, paste0("jovenes_", slug, ".rds")))
