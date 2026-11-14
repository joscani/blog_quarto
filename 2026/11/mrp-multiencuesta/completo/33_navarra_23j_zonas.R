# Navarra en el 23-J por partido judicial: VOX y Bildu de Pamplona hacia el
# norte frente a la Ribera. El MRP solo llega a provincia; esto enseña lo que
# esconde una media provincial.
#
# Entrada: 2026/10/mrp-raking/datos/mun_23j.rds (infoelectoral, por municipio)
# Salida:  completo/datos/navarra_23j_zonas.rds
#
#   Rscript 2026/11/mrp-multiencuesta/completo/33_navarra_23j_zonas.R

library(dplyr)

mun_23j <- readRDS(here::here("2026/10/mrp-raking/datos/mun_23j.rds")) |>
  filter(codigo_provincia == "31")

# En el fichero no hay comarcas para Navarra (código 000): partido judicial
# (comprobado por municipio el 14 nov: el de Aoiz incluye Burlada y Sangüesa; el
# de Pamplona, la Sakana y Puente la Reina)
partido_judicial <- c("019" = "Aoiz (Pirineo, Sangüesa y Burlada)", "097" = "Estella",
                      "201" = "Pamplona (ciudad, norte, Sakana y Puente la Reina)",
                      "227" = "Tafalla", "232" = "Tudela (Ribera)")

# Una fila por municipio: votos válidos = candidaturas + blancos
municipios <- mun_23j |>
  group_by(codigo_municipio, municipio, codigo_partido_judicial) |>
  summarise(validos = sum(votos) + first(votos_blancos),
            vox = sum(votos[siglas == "VOX"]),
            bildu = sum(votos[siglas == "EH Bildu"]),
            .groups = "drop")
stopifnot(all(municipios$validos > 0),
          all(municipios$codigo_partido_judicial %in% names(partido_judicial)))

zonas <- municipios |>
  mutate(zona = partido_judicial[codigo_partido_judicial]) |>
  group_by(zona) |>
  summarise(municipios = n(), votos_validos = sum(validos),
            vox_pct = 100 * sum(vox) / sum(validos),
            bildu_pct = 100 * sum(bildu) / sum(validos))
navarra <- municipios |>
  summarise(zona = "Navarra", municipios = n(), votos_validos = sum(validos),
            vox_pct = 100 * sum(vox) / sum(validos),
            bildu_pct = 100 * sum(bildu) / sum(validos))
zonas <- bind_rows(zonas, navarra)
print(zonas |> mutate(across(ends_with("pct"), \(x) round(x, 1))))

ejemplos <- municipios |>
  filter(municipio %in% c("Pamplona/Iruña", "Baztan", "Bera", "Leitza", "Etxarri Aranatz",
                          "Altsasu/Alsasua", "Tudela")) |>
  transmute(municipio, votos_validos = validos,
            vox_pct = 100 * vox / validos, bildu_pct = 100 * bildu / validos)
stopifnot(nrow(ejemplos) == 7)
print(ejemplos |> mutate(across(ends_with("pct"), \(x) round(x, 1))))

saveRDS(list(zonas = zonas, ejemplos = ejemplos),
        here::here("2026/11/mrp-multiencuesta/completo/datos/navarra_23j_zonas.rds"))
