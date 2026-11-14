# Mapa del ganador por provincia y tramo de edad, con su seguridad
#
# Mismo ganador que el mapa de 20_preguntas.R (3.7): el partido con mayor %
# medio en los 1.000 draws. Lo nuevo: la intensidad del color dice en qué % de
# los draws gana ese partido (contar). Pálido = el ganador cambia de un draw a
# otro.
#
# Salida: completo/mapa_ganador_seguridad.png
#
#   Rscript 2026/11/mrp-multiencuesta/completo/28_mapa_seguridad_ganador.R   (~30 s)

library(tidyverse)
library(sf)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo/datos")

colores <- c(PP = "#1D84CE", PSOE = "#E30613", VOX = "#5AC035", SUMAR = "#E5007D",
             PODEMOS = "#6B2E68", ERC = "#FFB232", JUNTS = "#00C3B2", BILDU = "#00A39A",
             PNV = "#008135", BNG = "#ADCFEF", CCA = "#FFD700", AA = "#2E8B57",
             UPN = "#0A3A80", "UPN+PP" = "#0A3A80", OTROS = "grey60", BLANCO = "grey90")

celdas <- readRDS(file.path(dir_completo, "tabla_post.rds")) |>
  mutate(tramo = case_when(edad %in% c("18-20", "21-24", "25-34") ~ "18-34",
                           edad == "65+" ~ "65+",
                           .default = "35-64"))
posterior <- readRDS(file.path(dir_completo, "posterior_celdas_80.rds"))
provincias_sf <- readRDS(file.path(dir_completo, "provincias_sf.rds"))
partidos <- setdiff(names(posterior), c("draw", "celda", "part"))

# % de voto por draw, provincia y tramo (como votos_por(c("cod", "tramo")))
voto <- posterior |>
  inner_join(select(celdas, celda, N, cod, tramo), by = "celda") |>
  summarise(across(all_of(partidos), ~ sum(N * part * .x)), .by = c(draw, cod, tramo)) |>
  pivot_longer(all_of(partidos), names_to = "partido", values_to = "votos") |>
  # Navarra (31): UPN y PP sumados
  mutate(partido = if_else(cod == 31 & partido %in% c("UPN", "PP"), "UPN+PP", partido)) |>
  summarise(votos = sum(votos), .by = c(draw, cod, tramo, partido)) |>
  mutate(pct = 100 * votos / sum(votos), .by = c(draw, cod, tramo))

# Ganador del mapa publicado: mayor % medio
ganador <- voto |>
  summarise(pct = mean(pct), .by = c(cod, tramo, partido)) |>
  slice_max(pct, n = 1, by = c(cod, tramo))

# Contar: en qué % de draws es el primero ese partido
p_gana <- voto |>
  mutate(primero = pct == max(pct), .by = c(draw, cod, tramo)) |>
  semi_join(ganador, by = c("cod", "tramo", "partido")) |>
  summarise(p_gana = mean(primero), .by = c(cod, tramo))

seguridad <- ganador |>
  left_join(p_gana, by = c("cod", "tramo")) |>
  mutate(nivel = cut(p_gana, c(0, 0.6, 0.8, 0.95, 1.0001), right = FALSE,
                     labels = c("Menos del 60 %", "60-80 %", "80-95 %", "95 % o más")))

# Comprobaciones: 156 recuadros, ninguno sin dato
stopifnot(nrow(seguridad) == 52 * 3, !anyNA(seguridad$nivel))

# Cuántos recuadros en cada nivel
seguridad |> count(nivel)
seguridad |> count(tramo, nivel) |> pivot_wider(names_from = nivel, values_from = n, values_fill = 0)

g <- provincias_sf |>
  inner_join(seguridad, by = c("codigo_provincia" = "cod"), relationship = "one-to-many") |>
  ggplot() +
  geom_sf(aes(fill = partido, alpha = nivel), colour = "white", linewidth = 0.1) +
  facet_wrap(~tramo) +
  scale_fill_manual(values = colores) +
  scale_alpha_manual(values = c(0.2, 0.45, 0.75, 1), drop = FALSE) +
  guides(fill = guide_legend(order = 1, override.aes = list(alpha = 1)),
         alpha = guide_legend(order = 2, override.aes = list(fill = "grey30"))) +
  labs(fill = NULL, alpha = "Gana en el … de las simulaciones",
       title = "Partido más votado por provincia: ¿con qué seguridad?",
       subtitle = "Color pálido: el ganador cambia de unas simulaciones a otras. Versión corregida (14 nov): el recuerdo de voto por edad sale de la comunidad, no de la provincia.",
       caption = "MRP multiencuesta, octubre de 2026, participación del 80 %. En Navarra, UPN y PP sumados. Ganador: mayor % medio. Seguridad: % de las 1.000 simulaciones (draws de la posterior) en que ese partido es el primero.") +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(colour = "grey30", size = 10),
        plot.caption = element_text(colour = "grey50", hjust = 0, size = 8),
        strip.text = element_text(face = "bold", size = 12),
        legend.position = "bottom", legend.box = "vertical",
        plot.margin = margin(10, 10, 10, 10))

ggsave(here::here("2026/11/mrp-multiencuesta/completo/mapa_ganador_seguridad.png"), g,
       width = 11, height = 4.6, dpi = 200, bg = "white")
