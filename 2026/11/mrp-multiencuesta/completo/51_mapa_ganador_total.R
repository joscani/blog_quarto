# Ganador sin separar por edad, con su seguridad, por provincia y por comunidad
# (16 nov 2026). % de voto válido, participación del 80 %, media de 1.000 draws.
# Seguridad: proporción de draws en que ese partido es el primero (color más
# pálido cuanto menos segura). En Navarra, UPN y PP sumados.
#
# VARIANTE=semilla_mn_gp: el modelo nuevo; sin variable: lo publicado.
# Salidas: completo/mapa_ganador_total_80[_variante].png (solo el ganador)
#          completo/mapa_ganador_segundo_80[_variante].png (primero y segundo, 2 x 2)
#          completo/datos/ganador_segundo_80[_variante].rds
#
#   VARIANTE=semilla_mn_gp Rscript 2026/11/mrp-multiencuesta/completo/51_mapa_ganador_total.R

pdf(NULL) # que no escriba Rplots.pdf en la raíz del repo
suppressPackageStartupMessages({
  library(tidyverse)
  library(sf)
})
variante <- Sys.getenv("VARIANTE", "")
stopifnot(variante %in% c("", "semilla_mn", "semilla_mn_gp"))
suf <- if (variante == "") "" else paste0("_", variante)
# PART=70 (17 nov 2026): posterior_celdas_70 (participación del 23-J, 70,4 %);
# por defecto, 80. Las salidas llevan el código de la participación.
part_cod <- Sys.getenv("PART", "80")
stopifnot(part_cod %in% c("70", "75", "80"))
part_txt <- c("70" = "70,4 %", "75" = "75 %", "80" = "80 %")[[part_cod]]
dir_completo <- here::here("2026/11/mrp-multiencuesta/completo/datos")
dir_salida <- here::here("2026/11/mrp-multiencuesta/completo")

colores <- c(PP = "#1D84CE", PSOE = "#E30613", VOX = "#5AC035", SUMAR = "#E5007D",
             PODEMOS = "#6B2E68", ERC = "#FFB232", JUNTS = "#00C3B2", BILDU = "#00A39A",
             PNV = "#008135", BNG = "#ADCFEF", CCA = "#FFD700", AA = "#2E8B57",
             "UPN+PP" = "#0A3A80", OTROS = "grey60", BLANCO = "grey90")

celdas <- readRDS(file.path(dir_completo, paste0("tabla_post", suf, ".rds")))
posterior <- readRDS(file.path(dir_completo, paste0("posterior_celdas_", part_cod, suf, ".rds")))
partidos <- setdiff(names(posterior), c("draw", "celda", "part"))
stopifnot(nrow(posterior) == 1000 * nrow(celdas))

ganador <- function(nivel) {
  v <- posterior |>
    inner_join(celdas |> mutate(nav = ccaa == "Navarra") |> select(celda, N, nav, unidad = all_of(nivel)), by = "celda") |>
    summarise(across(all_of(partidos), ~ sum(N * part * .x)), navarra = all(nav), .by = c(draw, unidad)) |>
    pivot_longer(all_of(partidos), names_to = "partido", values_to = "votos") |>
    mutate(partido = if_else(navarra & partido %in% c("UPN", "PP"), "UPN+PP", partido)) |>
    summarise(votos = sum(votos), .by = c(draw, unidad, partido)) |>
    mutate(pct = 100 * votos / sum(votos), primero = votos == max(votos), .by = c(draw, unidad))
  stopifnot(all(abs(summarise(v, s = sum(pct), .by = c(draw, unidad))$s - 100) < 1e-9))
  v |>
    mutate(puesto = rank(-pct, ties.method = "first"), .by = c(draw, unidad)) |>
    summarise(pct = mean(pct), p1 = mean(primero), p2 = mean(puesto == 2), .by = c(unidad, partido)) |>
    arrange(unidad, desc(pct)) |>
    mutate(posicion = row_number(), .by = unidad) |>
    filter(posicion <= 2) |>
    mutate(posicion = factor(posicion, 1:2, c("Primer partido", "Segundo partido")),
           p_puesto = if_else(posicion == "Primer partido", p1, p2)) |>
    mutate(nivel = cut(p_puesto, c(0, 0.6, 0.8, 0.95, 1.0001), right = FALSE,
                       labels = c("Menos del 60 %", "60-80 %", "80-95 %", "95 % o más")))
}
gs_prov <- ganador("cod")
gs_ccaa <- ganador("ccaa")
stopifnot(nrow(gs_prov) == 52 * 2, nrow(gs_ccaa) == 19 * 2, !anyNA(gs_prov$nivel), !anyNA(gs_ccaa$nivel))
g_prov <- filter(gs_prov, posicion == "Primer partido")
g_ccaa <- filter(gs_ccaa, posicion == "Primer partido")

prov_sf <- readRDS(file.path(dir_completo, "provincias_sf.rds"))
ccaa_sf <- prov_sf |>
  inner_join(distinct(celdas, cod, ccaa), by = c("codigo_provincia" = "cod")) |>
  group_by(ccaa) |> summarise(.groups = "drop")
geo <- bind_rows(
  prov_sf |> inner_join(g_prov, by = c("codigo_provincia" = "unidad")) |> mutate(mapa = "Por provincia"),
  ccaa_sf |> inner_join(g_ccaa, by = c("ccaa" = "unidad")) |> mutate(mapa = "Por comunidad")
) |>
  mutate(mapa = factor(mapa, c("Por provincia", "Por comunidad")), texto = sprintf("%s\n%.0f", partido, pct))
stopifnot(nrow(geo) == 52 + 19)
pt <- suppressWarnings(st_point_on_surface(geo))

pie <- paste0("MRP multiencuesta (CIS, 40dB, More in Common y Ateneo del Dato), octubre de 2026. Participación del ", part_txt, ".\n",
              "% de voto válido, media de 1.000 simulaciones. Seguridad: % de simulaciones en que ese partido es el primero. En Navarra, UPN y PP sumados.",
              if (variante == "semilla_mn_gp") " Semilla multinivel y predictor de grupo." else "",
              "\nmuestrear-no-es-pecado.es")
g <- ggplot(geo) +
  geom_sf(aes(fill = partido, alpha = nivel), colour = "white", linewidth = 0.15) +
  geom_sf_text(data = pt, aes(label = texto), size = 2.2, lineheight = 0.85, fontface = "bold") +
  facet_wrap(~mapa, ncol = 2) +
  scale_fill_manual(values = colores) +
  scale_alpha_manual(values = c(0.2, 0.45, 0.75, 1), drop = FALSE) +
  guides(fill = guide_legend(order = 1, override.aes = list(alpha = 1)),
         alpha = guide_legend(order = 2, override.aes = list(fill = "grey30"))) +
  labs(fill = NULL, alpha = "Gana en el … de las simulaciones",
       title = "Partido más votado, todas las edades: ¿con qué seguridad?",
       subtitle = paste0("Si votase el ", part_txt, ". Color pálido: el ganador cambia de unas simulaciones a otras."), caption = pie) +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), plot.subtitle = element_text(colour = "grey30", size = 10),
        plot.caption = element_text(colour = "grey50", hjust = 0, size = 8),
        strip.text = element_text(face = "bold", size = 12),
        legend.position = "bottom", legend.box = "vertical", plot.margin = margin(10, 10, 10, 10))
f_png <- file.path(dir_salida, paste0("mapa_ganador_total_", part_cod, suf, ".png"))
suppressWarnings(ggsave(f_png, g, width = 14, height = 7.5, dpi = 250, bg = "white"))

# ---- primero y segundo, 2 x 2 (filas: puesto; columnas: provincia / comunidad) ----
geo2 <- bind_rows(
  prov_sf |> inner_join(gs_prov, by = c("codigo_provincia" = "unidad"), relationship = "one-to-many") |> mutate(mapa = "Por provincia"),
  ccaa_sf |> inner_join(gs_ccaa, by = c("ccaa" = "unidad"), relationship = "one-to-many") |> mutate(mapa = "Por comunidad")
) |>
  mutate(mapa = factor(mapa, c("Por provincia", "Por comunidad")), texto = sprintf("%s\n%.0f", partido, pct))
stopifnot(nrow(geo2) == (52 + 19) * 2)
pt2 <- suppressWarnings(st_point_on_surface(geo2))
g2 <- ggplot(geo2) +
  geom_sf(aes(fill = partido, alpha = nivel), colour = "white", linewidth = 0.15) +
  geom_sf_text(data = pt2, aes(label = texto), size = 2, lineheight = 0.85, fontface = "bold") +
  facet_grid(rows = vars(posicion), cols = vars(mapa), switch = "y") +
  scale_fill_manual(values = colores) +
  scale_alpha_manual(values = c(0.2, 0.45, 0.75, 1), drop = FALSE) +
  guides(fill = guide_legend(order = 1, override.aes = list(alpha = 1)),
         alpha = guide_legend(order = 2, override.aes = list(fill = "grey30"))) +
  labs(fill = NULL, alpha = "Ocupa ese puesto en el … de las simulaciones",
       title = "Primer y segundo partido, todas las edades: ¿con qué seguridad?",
       subtitle = paste0("Si votase el ", part_txt, ". Color pálido: el puesto cambia de unas simulaciones a otras."),
       caption = sub("Seguridad: % de simulaciones en que ese partido es el primero.",
                     "Seguridad: % de simulaciones en que ese partido ocupa ese puesto.", pie, fixed = TRUE)) +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), plot.subtitle = element_text(colour = "grey30", size = 10),
        plot.caption = element_text(colour = "grey50", hjust = 0, size = 8),
        strip.text = element_text(face = "bold", size = 12), strip.text.y.left = element_text(angle = 90),
        legend.position = "bottom", legend.box = "vertical", plot.margin = margin(10, 10, 10, 10))
f_png2 <- file.path(dir_salida, paste0("mapa_ganador_segundo_", part_cod, suf, ".png"))
suppressWarnings(ggsave(f_png2, g2, width = 14, height = 13, dpi = 250, bg = "white"))
saveRDS(list(provincia = gs_prov, ccaa = gs_ccaa), file.path(dir_completo, paste0("ganador_segundo_", part_cod, suf, ".rds")))
cat("guardado:", f_png2, "\n")

cat("provincias por ganador y seguridad:\n"); print(count(g_prov, partido, nivel) |> pivot_wider(names_from = nivel, values_from = n, values_fill = 0))
cat("\ncomunidades:\n"); print(as.data.frame(g_ccaa |> arrange(p1) |> mutate(across(where(is.double), ~ round(.x, 2)))), row.names = FALSE)
cat("guardado:", f_png, "\n")
