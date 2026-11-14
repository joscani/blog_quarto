# De la posterior a las respuestas: ¿cómo vota la sociedad española?
#
# Dos tablas (las crea 21_posterior_celdas.R):
#
#   celdas     La tabla de postestratificación: quiénes son y cuántos.
#              Una fila por celda (provincia x sexo x edad x estudios x
#              recuerdo del 23-J): 12.300 celdas, con N = personas de 18 y más.
#
#   posterior  Lo que el modelo dice de cada celda: 1.000 versiones plausibles
#              (draws). Una fila por draw y celda (12,3 millones de filas):
#                part        probabilidad de votar;
#                PP ... BLANCO   reparto del voto entre los que votan (suma 1).
#
# Escenario: participación del 80 % (calibrada en cada draw), método seg, casa
# media, octubre de 2026. Todas las cifras son "si votase el 80 %".
#
# Toda respuesta sale de cuatro operaciones:
#   1. de probabilidades a personas:  votos = N * part * p
#   2. agregar:  sumar los votos de las celdas del grupo, en cada draw
#   3. resumir:  media e intervalo de los 1.000 draws
#   4. contar:   la probabilidad de algo = proporción de draws en que ocurre
#
# Límites del modelo:
#   - No hay interacciones sexo x edad ni edad x estudios. Si la brecha de sexo
#     cambia con la edad es por la composición de cada grupo, no porque el
#     modelo la haya estimado.
#   - No hay ideología ni tamaño de municipio.
#   - Aliança Catalana y la CUP van en OTROS.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/20_preguntas.R   (~2 min)

library(tidyverse)
library(sf)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo/datos")

colores <- c(PP = "#1D84CE", PSOE = "#E30613", VOX = "#5AC035", SUMAR = "#E5007D",
             PODEMOS = "#6B2E68", ERC = "#FFB232", JUNTS = "#00C3B2", BILDU = "#00A39A",
             PNV = "#008135", BNG = "#ADCFEF", CCA = "#FFD700", AA = "#2E8B57",
             UPN = "#0A3A80", OTROS = "grey60", BLANCO = "grey90")

# =============================================================================
# PARTE 1. EL MÉTODO, PASO A PASO
# =============================================================================

# ---- 1.1 Las dos tablas --------------------------------------------------------

celdas <- readRDS(file.path(dir_completo, "tabla_post.rds"))
posterior <- readRDS(file.path(dir_completo, "posterior_celdas_80.rds"))

celdas
posterior

partidos <- setdiff(names(posterior), c("draw", "celda", "part"))
partidos

sum(celdas$N) # población de 18 y más (unos 36,7 millones)

# ---- 1.2 Una celda -------------------------------------------------------------
# Hombres de 25 a 34 años, estudios medios, Madrid, que no votaron en 2023.

ejemplo <- celdas |>
  filter(provincia == "Madrid", sexo == "Hombres", edad == "25-34",
         edu == "media", rec == "ABST")
ejemplo

# Su posterior: 1.000 filas, una por draw
post_ejemplo <- posterior |>
  filter(celda == ejemplo$celda) |>
  select(draw, part, PP, PSOE, VOX, SUMAR)
post_ejemplo

# Cada fila es una versión plausible de la celda. Resumirlas es la incertidumbre.
post_ejemplo |>
  summarise(across(-draw, list(media = mean,
                               q05 = \(x) quantile(x, 0.05),
                               q95 = \(x) quantile(x, 0.95))))

# Para dibujar los cuatro partidos juntos, de ancho (un partido por columna)
# a largo (una fila por draw y partido)
post_ejemplo_largo <- post_ejemplo |>
  pivot_longer(c(PP, PSOE, VOX, SUMAR), names_to = "partido", values_to = "p")
post_ejemplo_largo

etiquetas <- post_ejemplo_largo |>
  summarise(p = mean(p), .by = partido)

ggplot(post_ejemplo_largo, aes(p, fill = partido)) +
  geom_histogram(binwidth = 0.005, boundary = 0, position = "identity", alpha = 0.7) +
  geom_text(data = etiquetas, aes(p, y = 0, label = partido, colour = partido),
            vjust = -0.5, nudge_y = 230, fontface = "bold") +
  scale_fill_manual(values = colores, aesthetics = c("fill", "colour"), guide = "none") +
  scale_x_continuous(labels = scales::percent) +
  labs(x = "Proporción de voto", y = "draws",
       title = "La posterior de una celda: 1.000 valores plausibles por partido",
       subtitle = "Hombres, 25-34, estudios medios, Madrid, abstención en 2023 (N = 54.876)") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"))

# ---- 1.3 De probabilidades a personas -----------------------------------------
# votantes = N * part;  votos a VOX = N * part * VOX

post_ejemplo |>
  mutate(
    N = ejemplo$N,
    votantes = N * part,
    votos_vox = votantes * VOX
  ) |>
  select(draw, N, part, votantes, VOX, votos_vox)

# ---- 1.4 Agregar: un grupo son muchas celdas -----------------------------------
# Hombres de 25 a 34 años en toda España: elegir sus celdas y sumar en cada draw.

celdas_grupo <- celdas |> filter(sexo == "Hombres", edad == "25-34")
nrow(celdas_grupo) # celdas que forman el grupo

grupo <- posterior |>
  inner_join(select(celdas_grupo, celda, N), by = "celda") |>
  summarise(
    votantes = sum(N * part),
    vox = sum(N * part * VOX),
    .by = draw
  ) |>
  mutate(pct_vox = 100 * vox / votantes)
grupo # una fila por draw: 1.000 valores plausibles del % de VOX en el grupo

# ---- 1.5 Resumir los draws -----------------------------------------------------

grupo |>
  summarise(media = mean(pct_vox),
            q05 = quantile(pct_vox, 0.05),
            q95 = quantile(pct_vox, 0.95))

# ---- 1.6 Contar ----------------------------------------------------------------
# ¿Qué probabilidad hay de que más del 30 % de ese grupo vote a VOX?

mean(grupo$pct_vox > 30)

# =============================================================================
# PARTE 2. LAS MISMAS CUATRO OPERACIONES, EN FUNCIONES
# =============================================================================

# votos_por: votos de cada partido por draw y grupo.
#   por:    variables que definen el grupo (character(0) = toda la población)
#   celdas: las celdas que entran (filtra antes para una comunidad, etc.)
# Suma en formato ancho (un partido por columna) y pasa a largo al final:
# así son 15 columnas de 12,3 millones de filas y no 184 millones de filas.
votos_por <- function(por, celdas_sel = celdas) {
  posterior |>
    inner_join(select(celdas_sel, celda, N, all_of(por)), by = "celda") |>
    summarise(across(all_of(partidos), ~ sum(N * part * .x)),
              .by = all_of(c("draw", por))) |>
    pivot_longer(all_of(partidos), names_to = "partido", values_to = "votos") |>
    mutate(pct = 100 * votos / sum(votos), .by = all_of(c("draw", por)))
}

# resume: media e intervalo de credibilidad del 90 % (percentiles 5 y 95) de una
# columna sobre los draws. Es punto a punto: un intervalo por grupo, por separado.
resume <- function(datos, por, x = "pct") {
  datos |>
    summarise(media = mean(.data[[x]]),
              q05 = quantile(.data[[x]], 0.05),
              q95 = quantile(.data[[x]], 0.95),
              .by = all_of(por)) |>
    arrange(across(all_of(por))) # .by no ordena: deja los grupos como aparecen
}

# participacion_por: % que vota en cada grupo y draw
participacion_por <- function(por, celdas_sel = celdas) {
  posterior |>
    select(draw, celda, part) |>
    inner_join(select(celdas_sel, celda, N, all_of(por)), by = "celda") |>
    summarise(part = 100 * sum(N * part) / sum(N), .by = all_of(c("draw", por)))
}

# tabla ancha para leer en consola
ancho <- function(datos, filas, columnas, valor = "media", digitos = 1) {
  datos |>
    select(all_of(c(filas, columnas, valor))) |>
    mutate(across(all_of(valor), \(x) round(x, digitos))) |>
    pivot_wider(names_from = all_of(columnas), values_from = all_of(valor))
}

# comprobación: la función da lo mismo que el cálculo a mano de 1.4
votos_por(c("sexo", "edad")) |>
  filter(sexo == "Hombres", edad == "25-34", partido == "VOX") |>
  resume("partido")

# ---- Aspecto de los gráficos ---------------------------------------------------

principales <- c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS")

tema <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title = element_text(face = "bold"),
        plot.title.position = "plot",
        plot.caption = element_text(colour = "grey50", hjust = 0),
        legend.position = "top",
        strip.text = element_text(face = "bold", size = 12))
fuente <- "MRP multiencuesta, octubre de 2026, participación del 80 %. Media e intervalo de credibilidad del 90 % (percentiles 5 y 95 de los draws)."
fuente_media <- "MRP multiencuesta, octubre de 2026, participación del 80 %. Media de los 1.000 draws."

# Gráfico de efectos con la posterior: % de voto de cada grupo real
# (postestratificado), no "con lo demás constante". Si los jóvenes tienen más
# estudios, eso entra en la curva de la edad.
efecto <- function(x, partidos_sel, color = NULL, celdas_sel = celdas, ylim = NULL,
                   escalas = "free_y") {
  d <- votos_por(c(x, color), celdas_sel) |>
    resume(c(x, color, "partido")) |>
    filter(partido %in% partidos_sel) |>
    mutate(partido = factor(partido, partidos_sel))
  if (is.null(ylim)) ylim <- c(0, max(d$q95))

  # Sin `color`: un solo panel, todos los partidos en el mismo eje Y y el
  # nombre de cada uno al final de su línea
  if (is.null(color)) {
    final <- d |> filter(.data[[x]] == last(levels(.data[[x]])))
    return(
      ggplot(d, aes(.data[[x]], media, ymin = q05, ymax = q95,
                    colour = partido, fill = partido, group = partido)) +
        geom_ribbon(alpha = 0.2, colour = NA) +
        geom_line(linewidth = 1) +
        geom_point(size = 2) +
        ggrepel::geom_text_repel(data = final, aes(label = partido), hjust = 0,
                                 nudge_x = 0.15, direction = "y", fontface = "bold",
                                 size = 4, segment.colour = NA, box.padding = 0.1) +
        scale_colour_manual(values = colores, aesthetics = c("colour", "fill"), guide = "none") +
        scale_y_continuous(breaks = seq(0, 100, by = 5)) +
        coord_cartesian(ylim = ylim, clip = "off") +
        labs(x = NULL, y = "% de voto", caption = fuente) +
        tema +
        theme(plot.margin = margin(5.5, 60, 5.5, 5.5))
    )
  }

  # Con `color`: un panel por partido y una línea por grupo.
  # escalas = "free_y": cada panel con su eje Y; "fixed": todos con `ylim`.
  ggplot(d, aes(.data[[x]], media, ymin = q05, ymax = q95,
                colour = .data[[color]], fill = .data[[color]], group = .data[[color]])) +
    geom_ribbon(alpha = 0.2, colour = NA) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    scale_colour_manual(values = c("#D95F02", "#1B9E77", "#7570B3"),
                        aesthetics = c("colour", "fill")) +
    facet_wrap(~partido, scales = escalas) +
    (if (escalas == "fixed") coord_cartesian(ylim = ylim)) +
    scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
    labs(x = NULL, y = "% de voto", colour = NULL, fill = NULL, caption = fuente) +
    tema
}

provincias_sf <- readRDS(file.path(dir_completo, "provincias_sf.rds"))
mapa <- function(datos) {
  provincias_sf |>
    left_join(datos, by = c("codigo_provincia" = "cod")) |>
    ggplot() +
    theme_void(base_size = 12) +
    theme(plot.title = element_text(face = "bold"),
          plot.caption = element_text(colour = "grey50", hjust = 0),
          strip.text = element_text(face = "bold", size = 12),
          plot.margin = margin(10, 10, 10, 10))
}

# Grupos derivados: se añaden como columnas de `celdas`
bloque_de <- function(partido) {
  case_when(
    partido %in% c("PP", "VOX", "UPN") ~ "Derecha",
    partido %in% c("PSOE", "SUMAR", "PODEMOS", "ERC", "BILDU", "BNG", "AA") ~ "Izquierda",
    partido == "BLANCO" ~ "Blanco",
    .default = "Otros" # JUNTS, PNV, CCA y OTROS (SALF y AC van en OTROS)
  )
}
celdas <- celdas |>
  mutate(tramo = case_when(
    edad %in% c("18-20", "21-24", "25-34") ~ "18-34",
    edad == "65+" ~ "65+",
    .default = "35-64"
  ))

# =============================================================================
# PARTE 3. PREGUNTAS
# =============================================================================

# ---- 3.1 Generaciones: ¿vota distinto cada edad? -------------------------------

voto_edad <- votos_por("edad")

voto_edad |>
  resume(c("edad", "partido")) |>
  filter(partido %in% principales) |>
  ancho("edad", "partido")

# Partido más votado en cada edad, y en qué % de draws lo es (contar)
voto_edad |>
  mutate(primero = pct == max(pct), .by = c(draw, edad)) |>
  summarise(p_primero = mean(primero), pct = mean(pct), .by = c(edad, partido)) |>
  filter(p_primero > 0.05) |>
  arrange(edad, desc(p_primero))

efecto("edad", c("PP", "PSOE", "VOX", "SUMAR"), ylim = c(0, 35)) +
  labs(title = "Voto por edad")

# ¿El bipartidismo es cosa de mayores? % de PP + PSOE por edad, en cada draw
voto_edad |>
  filter(partido %in% c("PP", "PSOE")) |>
  summarise(pct = sum(pct), .by = c(draw, edad)) |>
  resume("edad")

# ---- 3.2 Sexo ------------------------------------------------------------------

# Brecha por partido: % entre hombres - % entre mujeres, en cada draw
votos_por("sexo") |>
  select(draw, sexo, partido, pct) |>
  pivot_wider(names_from = sexo, values_from = pct) |>
  mutate(brecha = Hombres - Mujeres) |>
  resume("partido", "brecha") |>
  arrange(desc(media))

# La pregunta inversa: ¿quién es el votante de cada partido?
# Mismos votos que arriba, pero se divide por el total del partido (en cada
# draw), no por el total del grupo. Ejemplo: de cada 100 votantes de VOX,
# cuántos son hombres de 18 a 34 años.
composicion_sexo_tramo <- votos_por(c("sexo", "tramo")) |>
  mutate(pct_partido = 100 * votos / sum(votos), .by = c(draw, partido)) |>
  resume(c("sexo", "tramo", "partido"), "pct_partido")

composicion_sexo_tramo |>
  filter(partido %in% c("PP", "PSOE", "VOX", "SUMAR")) |>
  ancho(c("sexo", "tramo"), "partido")

# Para comparar: qué peso tiene cada grupo entre todos los votantes
peso_votantes <- votos_por(c("sexo", "tramo")) |>
  summarise(votos = sum(votos), .by = c(draw, sexo, tramo)) |>
  mutate(pct_votantes = 100 * votos / sum(votos), .by = draw) |>
  resume(c("sexo", "tramo"), "pct_votantes")
peso_votantes

# Gráfico: barra = % de los votantes del partido; marca gris = % de todos los
# votantes. Barra más larga que la marca: el grupo pesa más en ese partido.
composicion_sexo_tramo |>
  filter(partido %in% c("PP", "PSOE", "VOX", "SUMAR")) |>
  mutate(partido = factor(partido, c("PP", "PSOE", "VOX", "SUMAR")),
         grupo = fct_rev(paste(sexo, tramo))) |>
  left_join(peso_votantes |> select(sexo, tramo, todos = media), by = c("sexo", "tramo")) |>
  ggplot(aes(y = grupo)) +
  geom_col(aes(x = media, fill = partido), width = 0.7) +
  geom_errorbar(aes(xmin = todos, xmax = todos), width = 0.9, colour = "grey20",
                linewidth = 1.2, orientation = "y") +
  facet_wrap(~partido, nrow = 1) +
  scale_fill_manual(values = colores, guide = "none") +
  labs(x = "% de los votantes del partido", y = NULL,
       title = "¿Quién vota a cada partido?",
       subtitle = "Barra: % de los votantes del partido. Marca: % de todos los votantes.",
       caption = paste("MRP multiencuesta, octubre de 2026, participación del 80 %.",
                       "Media de los draws; los intervalos del 90 % miden menos de 2,5 puntos.")) +
  tema +
  theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line())

# SUMAR (con Frente Amplio, Más Madrid, Compromís, Comuns...) por edad y sexo
efecto("edad", "SUMAR", color = "sexo") + labs(title = "SUMAR por edad y sexo")
efecto("edad", c("PP", "PSOE", "VOX", "SUMAR"), color = "sexo",
       ylim = c(0, 40), escalas = "fixed") +
  labs(title = "Voto por edad y sexo")

# ---- 3.3 Estudios --------------------------------------------------------------

efecto("edad", principales, color = "edu") + labs(title = "Voto por edad y estudios")

# La pregunta inversa: ¿qué estudios tiene el votante de cada partido?
# Mismos votos, pero se divide por el total del partido, no por el del grupo.
votos_por("edu") |>
  mutate(pct_partido = 100 * votos / sum(votos), .by = c(draw, partido)) |>
  resume(c("edu", "partido"), "pct_partido") |>
  filter(partido %in% principales) |>
  ancho("partido", "edu")

# ¿Es educativa la brecha de los jóvenes? Solo menores de 35, por estudios
efecto("edu", c("PP", "PSOE", "VOX", "SUMAR"),
       celdas_sel = filter(celdas, tramo == "18-34"), ylim = c(0, 35)) +
  labs(title = "Menores de 35: voto por nivel de estudios", x = "Estudios")

# ---- 3.4 Desde el 23-J: ¿de dónde vienen y a dónde van? ------------------------
# El recuerdo es una variable más de la celda: se agrega igual.

votos_por("rec") |>
  resume(c("rec", "partido")) |>
  filter(rec %in% c("PP", "PSOE", "VOX", "SUMAR", "ABST", "NO_PODIA"),
         partido %in% c(principales, "BLANCO")) |>
  ancho("rec", "partido")

# Y se cruza con lo demás: los que no votaron en 2023, por edad
votos_por(c("rec", "edad"), filter(celdas, rec == "ABST")) |>
  resume(c("edad", "partido")) |>
  filter(partido %in% principales) |>
  ancho("edad", "partido")

# ¿De dónde viene el voto de cada partido? La pregunta inversa con el recuerdo:
# de cada 100 votantes del partido, cuántos votaron a cada cosa en 2023.
origen <- votos_por("rec") |>
  mutate(pct_partido = 100 * votos / sum(votos), .by = c(draw, partido)) |>
  resume(c("rec", "partido"), "pct_partido")

origen |>
  filter(partido %in% c("PP", "PSOE", "VOX", "SUMAR")) |>
  ancho("rec", "partido")

origen |>
  filter(partido %in% c("PP", "PSOE", "VOX", "SUMAR")) |>
  mutate(
    partido = factor(partido, c("SUMAR", "VOX", "PSOE", "PP")),
    origen = case_when(
      rec %in% c("PP", "PSOE", "VOX", "SUMAR") ~ as.character(rec),
      rec == "ABST" ~ "Abstención",
      rec == "NO_PODIA" ~ "No podía votar",
      .default = "Otros"
    ),
    origen = factor(origen, c("PP", "PSOE", "VOX", "SUMAR", "Otros", "Abstención",
                              "No podía votar"))
  ) |>
  summarise(media = sum(media), .by = c(partido, origen)) |>
  ggplot(aes(media, partido, fill = origen)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.3) +
  scale_fill_manual(values = c(colores[c("PP", "PSOE", "VOX", "SUMAR")],
                               Otros = "grey80", `Abstención` = "grey45",
                               `No podía votar` = "grey20")) +
  scale_x_continuous(labels = \(x) paste0(x, " %")) +
  labs(x = "% de los votantes del partido", y = NULL, fill = "Votó en 2023",
       title = "¿De dónde viene el voto de cada partido?", caption = fuente_media) +
  tema +
  theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line())

# ¿Repiten más los mayores? De cada 100 votantes de un partido en 2023,
# cuántos lo vuelven a votar, por edad. (18-20 no aparece: no podían votar)
fidelidad <- votos_por(c("rec", "edad"),
                       filter(celdas, rec %in% c("PP", "PSOE", "VOX", "SUMAR"))) |>
  filter(partido == as.character(rec)) |>
  mutate(partido = factor(partido, c("PP", "PSOE", "VOX", "SUMAR"))) |>
  resume(c("partido", "edad"))

ggplot(fidelidad, aes(edad, media, ymin = q05, ymax = q95,
                      colour = partido, fill = partido, group = partido)) +
  geom_ribbon(alpha = 0.2, colour = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_colour_manual(values = colores, aesthetics = c("colour", "fill")) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(x = NULL, y = "% que repite partido", colour = NULL, fill = NULL,
       title = "¿Repiten más los mayores?",
       subtitle = "De cada 100 votantes de cada partido en 2023, cuántos lo vuelven a votar",
       caption = fuente) +
  tema

# Y la fuga del PP a VOX, por edad
votos_por("edad", filter(celdas, rec == "PP")) |>
  filter(partido == "VOX") |>
  resume("edad")

# ---- 3.5 ¿Quién vota? ----------------------------------------------------------
# La participación total es el 80 % por construcción; lo que cambia es quién.

participacion_por("edad") |> resume("edad", "part")
participacion_por(c("edu", "sexo")) |> resume(c("edu", "sexo"), "part") |>
  ancho("edu", "sexo", "media")

# ---- 3.6 Territorio: nacionalismos por edad ------------------------------------

efecto("edad", c("PSOE", "ERC", "JUNTS", "VOX", "PP"),
       celdas_sel = filter(celdas, ccaa == "Cataluña")) +
  labs(title = "Cataluña: voto por edad")

efecto("edad", c("BILDU", "PNV", "PSOE", "PP"),
       celdas_sel = filter(celdas, ccaa == "País Vasco")) +
  labs(title = "País Vasco: voto por edad")

# ---- 3.7 Mapas -----------------------------------------------------------------

# Partido más votado por provincia en tres tramos de edad
voto_prov_tramo <- votos_por(c("cod", "tramo")) # se usa en varios mapas

ganador <- voto_prov_tramo |>
  summarise(pct = mean(pct), .by = c(cod, tramo, partido)) |>
  slice_max(pct, n = 1, by = c(cod, tramo))
ganador |> count(tramo, partido)

mapa(ganador) +
  geom_sf(aes(fill = partido), colour = "white", linewidth = 0.1) +
  facet_wrap(~tramo) +
  scale_fill_manual(values = colores) +
  labs(fill = NULL, title = "Partido más votado por provincia",
       caption = "MRP multiencuesta, octubre de 2026, participación del 80 %. Partido con mayor % medio en los 1.000 draws.") +
  theme(legend.position = "bottom")

# ¿Dónde no sabemos quién gana? Contar: en cada draw, quién es el primero.
# Color: el partido que gana en más draws. Transparencia: en qué % de draws gana.
ganador_p <- voto_prov_tramo |>
  mutate(primero = pct == max(pct), .by = c(draw, cod, tramo)) |>
  summarise(p_primero = mean(primero), .by = c(cod, tramo, partido)) |>
  slice_max(p_primero, n = 1, by = c(cod, tramo), with_ties = FALSE)

# Las provincias más reñidas
ganador_p |>
  filter(p_primero < 0.6) |>
  left_join(distinct(celdas, cod, provincia), by = "cod") |>
  arrange(tramo, p_primero) |>
  select(tramo, provincia, partido, p_primero)

mapa(ganador_p) +
  geom_sf(aes(fill = partido, alpha = p_primero), colour = "white", linewidth = 0.1) +
  facet_wrap(~tramo) +
  scale_fill_manual(values = colores) +
  scale_alpha_continuous(range = c(0.15, 1), limits = c(0.3, 1),
                         breaks = c(0.4, 0.6, 0.8, 1), labels = scales::percent) +
  labs(fill = NULL, alpha = "% de draws en que gana",
       title = "¿Dónde no sabemos quién gana?",
       subtitle = "Color pálido: el ganador cambia de un draw a otro",
       caption = "MRP multiencuesta, octubre de 2026, participación del 80 %. 1.000 draws.") +
  theme(legend.position = "bottom")

# Bipartidismo por provincia: % de PP + PSOE entre los menores de 35 y los de 65+
bipartidismo <- voto_prov_tramo |>
  filter(tramo != "35-64", partido %in% c("PP", "PSOE")) |>
  summarise(pct = sum(pct), .by = c(draw, cod, tramo)) |>
  resume(c("cod", "tramo"))

mapa(bipartidismo) +
  geom_sf(aes(fill = media), colour = "white", linewidth = 0.1) +
  facet_wrap(~tramo) +
  scale_fill_viridis_c(limits = c(0, 80)) +
  labs(fill = "%", title = "¿El bipartidismo es cosa de mayores?",
       subtitle = "% de voto a PP + PSOE", caption = fuente_media) +
  theme(legend.position = "bottom")

# Brecha generacional: % de derecha entre 18-34 menos entre 65+
brecha_gen <- votos_por(c("cod", "tramo"), filter(celdas, tramo != "35-64")) |>
  mutate(bloque = bloque_de(partido)) |>
  summarise(pct = sum(pct), .by = c(draw, cod, tramo, bloque)) |>
  filter(bloque == "Derecha") |>
  select(draw, cod, tramo, pct) |>
  pivot_wider(names_from = tramo, values_from = pct) |>
  mutate(brecha = `18-34` - `65+`) |>
  resume("cod", "brecha")

# Etiqueta en las dos provincias extremas
brecha_gen <- brecha_gen |>
  left_join(distinct(celdas, cod, provincia), by = "cod") |>
  mutate(etiqueta = if_else(media %in% range(media),
                            paste0(provincia, "\n", sprintf("%+.0f", media)), NA))

# Leyenda en palabras: quién vota más a la derecha y por cuántos puntos
quien_mas <- function(x) {
  case_when(x < 0 ~ paste0("Mayores +", abs(x)),
            x > 0 ~ paste0("Jóvenes +", x),
            .default = "Igual")
}

mapa(brecha_gen) +
  geom_sf(aes(fill = media), colour = "white", linewidth = 0.1) +
  geom_sf_text(aes(label = etiqueta), size = 3.2, fontface = "bold", lineheight = 0.9,
               na.rm = TRUE,
               # sf avisa de que el punto interior se calcula en grados; para
               # colocar una etiqueta da igual
               fun.geometry = \(g) suppressWarnings(sf::st_point_on_surface(g))) +
  scale_fill_gradient2(low = "#E66101", mid = "grey97", high = "#5E3C99",
                       limits = c(-20, 20), breaks = c(-20, -10, 0, 10, 20),
                       labels = quien_mas) +
  guides(fill = guide_colourbar(barwidth = unit(12, "cm"), barheight = unit(0.4, "cm"),
                                title.position = "top", title.hjust = 0.5)) +
  labs(fill = "Puntos de diferencia en el % de voto a la derecha (PP + VOX + UPN)",
       title = "¿Votan más a la derecha los jóvenes o los mayores?",
       subtitle = "Menores de 35 frente a 65 y más, por provincia",
       caption = fuente_media) +
  theme(legend.position = "bottom")

# Fragmentación: número efectivo de partidos, 1 / sum(p^2), sin blanco
fragmentacion <- votos_por("cod") |>
  filter(partido != "BLANCO") |>
  mutate(p = votos / sum(votos), .by = c(draw, cod)) |>
  summarise(nep = 1 / sum(p^2), .by = c(draw, cod)) |>
  resume("cod", "nep")

mapa(fragmentacion) +
  geom_sf(aes(fill = media), colour = "white", linewidth = 0.1) +
  scale_fill_viridis_c() +
  labs(fill = "NEP", title = "Fragmentación del voto",
       subtitle = "Número efectivo de partidos (Laakso-Taagepera)", caption = fuente_media)

# Fugas desde el 23-J: de cada 100 votantes de un partido en 2023, cuántos van a otro
flujo_prov <- function(de, a) {
  votos_por("cod", filter(celdas, rec == de)) |>
    filter(partido == a) |>
    resume("cod")
}
mapa(flujo_prov("PP", "VOX")) +
  geom_sf(aes(fill = media), colour = "white", linewidth = 0.1) +
  scale_fill_viridis_c() +
  labs(fill = "%", title = "De cada 100 votantes del PP en 2023, cuántos votan VOX",
       caption = fuente_media)

# ---- 3.8 ¿Y si solo votasen los jóvenes? ¿Y solo las mujeres? -----------------
# D'Hondt por provincia con el umbral del 3 % (sobre voto válido con blanco),
# como 12_escenarios_voto_seg.R, con los votos de un solo grupo de edad.

escanos <- celdas |> distinct(cod, escanos)

reparto <- function(votos) {
  votos |>
    filter(pct >= 3, !partido %in% c("OTROS", "BLANCO")) |>
    left_join(escanos, by = "cod") |>
    cross_join(tibble(divisor = 1:37)) |>
    filter(divisor <= escanos) |>
    mutate(orden = rank(-votos / divisor, ties.method = "first"), .by = c(draw, cod)) |>
    filter(orden <= escanos) |>
    count(draw, partido, name = "escanos")
}

congresos <- bind_rows(
  Todos = reparto(votos_por("cod")),
  `18-34` = reparto(votos_por("cod", filter(celdas, tramo == "18-34"))),
  `65+` = reparto(votos_por("cod", filter(celdas, tramo == "65+"))),
  Mujeres = reparto(votos_por("cod", filter(celdas, sexo == "Mujeres"))),
  Hombres = reparto(votos_por("cod", filter(celdas, sexo == "Hombres"))),
  .id = "electorado"
) |>
  complete(electorado, draw, partido, fill = list(escanos = 0L))

# comprobación: 350 escaños en cada draw
congresos |> summarise(total = sum(escanos), .by = c(electorado, draw)) |> count(total)

congresos |>
  resume(c("electorado", "partido"), "escanos") |>
  ancho("partido", "electorado", digitos = 0) |>
  arrange(desc(Todos))

# Contar: probabilidad de mayoría absoluta de PP + VOX (176) en cada electorado
congresos |>
  filter(partido %in% c("PP", "VOX")) |>
  summarise(escanos = sum(escanos), .by = c(electorado, draw)) |>
  summarise(media = mean(escanos), p_mayoria = mean(escanos >= 176), .by = electorado)

# ---- 3.9 El votante típico de cada partido -------------------------------------
# Perfiles sexo x edad x estudios x recuerdo (sin provincia): los tres que más
# votos dan a cada partido, como % de los votantes del partido.

perfiles <- votos_por(c("sexo", "edad", "edu", "rec")) |>
  filter(partido %in% c("PP", "PSOE", "VOX", "SUMAR")) |>
  mutate(pct_partido = 100 * votos / sum(votos), .by = c(draw, partido)) |>
  resume(c("partido", "sexo", "edad", "edu", "rec"), "pct_partido")

perfiles |>
  slice_max(media, n = 3, by = partido) |>
  mutate(across(media:q95, \(x) round(x, 1)))

# ---- 3.10 Lo que la encuesta sola no puede responder ---------------------------
# ¿Cómo votan los jóvenes (18-34) de cada provincia? Comparamos la encuesta en
# bruto (% de VOX entre los entrevistados jóvenes de la provincia) con el MRP.
#   - Provincias con pocas entrevistas: el bruto es ruido; el multinivel (M)
#     toma información de provincias y grupos parecidos.
#   - Provincias con muchas: el bruto es estable pero sesgado; la
#     postestratificación (P) corrige la composición de la muestra.
# Ojo: el MRP también promedia las casas (casa media) y fija la participación
# en el 80 %; eso también mueve la cifra, sobre todo donde hay muchos datos.

encuestas <- readRDS(file.path(dir_completo, "encuestas_completo.rds"))

bruto_jov <- encuestas |>
  filter(edad %in% c("18-20", "21-24", "25-34"), !is.na(voto_sim), !is.na(prov)) |>
  summarise(n = n(), bruto = 100 * mean(voto_sim == "VOX"), .by = prov) |>
  rename(cod = prov)

jovenes_prov <- votos_por("cod", filter(celdas, tramo == "18-34")) |>
  filter(partido == "VOX") |>
  resume("cod") |>
  left_join(bruto_jov, by = "cod") |>
  left_join(distinct(celdas, cod, provincia), by = "cod")

# cuántas provincias tienen pocas entrevistas de jóvenes
jovenes_prov |> summarise(menos_de_10 = sum(n < 10), menos_de_20 = sum(n < 20))

jovenes_prov |>
  arrange(n) |>
  select(provincia, n, bruto, mrp = media, q05, q95) |>
  mutate(across(where(is.double), \(x) round(x, 1))) |>
  print(n = 52)

# Gráfico de pesas: una fila por provincia, de menos a más entrevistas.
# Gris = encuesta en bruto; verde = MRP con su intervalo; la línea los une.
jovenes_prov |>
  mutate(
    etiqueta = paste0(provincia, " (", n, ")"),
    etiqueta = fct_reorder(etiqueta, n, .desc = TRUE), # pocas entrevistas arriba
    grupo = factor(if_else(n < 50, "Menos de 50\nentrevistas:\nmanda la M", "50 o más\nentrevistas:\nmanda la P"),
                   c("Menos de 50\nentrevistas:\nmanda la M", "50 o más\nentrevistas:\nmanda la P"))
  ) |>
  ggplot(aes(y = etiqueta)) +
  # detrás: el intervalo del MRP como franja ancha y transparente
  geom_linerange(aes(xmin = q05, xmax = q95), colour = colores[["VOX"]], alpha = 0.25,
                 linewidth = 4.5) +
  # encima: la línea del bruto al MRP y los dos puntos
  geom_segment(aes(x = bruto, xend = media), colour = "grey55", linewidth = 0.5) +
  geom_point(aes(x = bruto), colour = "grey40", size = 2.2) +
  geom_point(aes(x = media), colour = colores[["VOX"]], size = 2.6) +
  facet_grid(grupo ~ ., scales = "free_y", space = "free_y") +
  scale_x_continuous(labels = \(x) paste0(x, " %"), breaks = seq(0, 60, 10)) +
  labs(x = "% de voto a VOX entre los jóvenes de 18 a 34 años", y = NULL,
       title = "¿Cómo votan los jóvenes de cada provincia?",
       subtitle = paste("Gris: la encuesta en bruto. Verde: el MRP. Franja: su intervalo de credibilidad del 90 %.",
                        "Entre paréntesis, entrevistas a jóvenes.", sep = "\n"),
       caption = paste("Encuestas: CIS, 40dB, More in Common y Ateneo con provincia.",
                       "MRP: casa media, participación del 80 %, octubre de 2026.")) +
  tema +
  theme(panel.grid.major.x = element_line(), panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 9),
        strip.text.y = element_text(angle = 0, hjust = 0))

# ---- 3.11 Galicia: ¿gana el BNG entre los menores de 45? ----------------------
# Pregunta de un lector en X tras el mapa de 3.7. Mismo escenario: generales,
# participación del 80 %. Ojo: en autonómicas el BNG saca unas tres veces más.

celdas <- celdas |>
  mutate(tramo45 = if_else(edad %in% c("18-20", "21-24", "25-34", "35-44"),
                           "Menos de 45", "45 y más"))
galicia <- filter(celdas, ccaa == "Galicia")

# Las edades sueltas, los tramos del mapa y el corte en 45, en una sola tabla
voto_galicia <- bind_rows(
  votos_por("edad", galicia) |> mutate(grupo = as.character(edad)),
  votos_por("tramo", galicia) |> rename(grupo = tramo),
  votos_por("tramo45", galicia) |> rename(grupo = tramo45)
) |>
  select(draw, grupo, partido, votos, pct)

voto_galicia |>
  resume(c("grupo", "partido")) |>
  filter(partido %in% c("BNG", "PP", "PSOE", "VOX")) |>
  ancho("grupo", "partido")

# Contar: puesto del BNG en cada draw (1 = el más votado)
voto_galicia |>
  mutate(puesto = rank(-pct, ties.method = "min"), .by = c(draw, grupo)) |>
  filter(partido == "BNG") |>
  summarise(p_primero = mean(puesto == 1), p_segundo = mean(puesto == 2), .by = grupo)

# Lo mismo por provincia, menores de 45
votos_por(c("provincia", "tramo45"), galicia) |>
  filter(tramo45 == "Menos de 45") |>
  mutate(puesto = rank(-pct, ties.method = "min"), .by = c(draw, provincia)) |>
  filter(partido == "BNG") |>
  summarise(pct = mean(pct), p_primero = mean(puesto == 1), p_segundo = mean(puesto == 2),
            .by = provincia)

# El BNG en Galicia: generales frente a autonómicas (% y votos, Público)
tribble(
  ~generales,  ~pct_gen, ~votos_gen, ~autonomicas, ~pct_aut, ~votos_aut,
  "jun 2016",  2.92,     44902,      "sep 2016",   8.44,     118982,
  "nov 2019",  8.18,     120456,     "jul 2020",   24.00,    311340,
  "jul 2023",  9.55,     152327,     "feb 2024",   31.86,    467074
) |>
  mutate(ratio_votos = round(votos_aut / votos_gen, 1))

# ---- 3.12 ¿Qué aporta el MRP? Bruto, ponderada y MRP ---------------------------
# Tres maneras de estimar el voto de un grupo con las mismas entrevistas:
#   Bruto:     % entre los entrevistados que dan partido (voto + simpatía),
#              sin pesos.
#   Ponderada: lo que hace una casa que pondera por sexo, edad, estudios y
#              provincia (raking), pero no por recuerdo de voto.
#   MRP:       el modelo (la M toma información de grupos y provincias
#              parecidos) y la postestratificación con recuerdo (la P).
# Los tres sobre voto válido (partidos, otros y blanco).

source(here::here("2026/11/mrp-multiencuesta/completo/ponderaciones.R"))

# Las entrevistas con las mismas variables de grupo que `celdas`
encuestas <- encuestas |>
  mutate(tramo = case_when(edad %in% c("18-20", "21-24", "25-34") ~ "18-34",
                           edad == "65+" ~ "65+",
                           .default = "35-64"),
         cod = prov)

# Bruto: contar respuestas por partido en cada grupo
bruto_por <- function(por, enc) {
  enc |>
    filter(!is.na(voto_sim)) |>
    mutate(partido = as.character(voto_sim)) |>
    count(across(all_of(c(por, "partido")))) |>
    mutate(n_grupo = sum(n), pct = 100 * n / sum(n), .by = all_of(por)) |>
    select(-n)
}

# Ponderada: sumar pesos en vez de contar. Se pondera toda la muestra de la
# comunidad y el % se calcula entre quien da partido.
ponderada_por <- function(por, enc, celdas_sel) {
  pondera_sin_rec(enc, celdas_sel) |>
    filter(!is.na(voto_sim)) |>
    mutate(partido = as.character(voto_sim)) |>
    summarise(votos = sum(peso_rk), .by = all_of(c(por, "partido"))) |>
    mutate(pct = 100 * votos / sum(votos), .by = all_of(por)) |>
    select(-votos)
}

# Las tres juntas, para una comunidad. MRP: media de los draws.
tres_fuentes <- function(por, comunidad) {
  enc <- filter(encuestas, ccaa == comunidad)
  cel <- filter(celdas, ccaa == comunidad)
  mrp <- votos_por(por, cel) |>
    summarise(pct = mean(pct), .by = all_of(c(por, "partido")))
  bind_rows(Bruto = bruto_por(por, enc),
            Ponderada = ponderada_por(por, enc, cel),
            MRP = mrp,
            .id = "fuente") |>
    mutate(fuente = factor(fuente, c("Bruto", "Ponderada", "MRP")))
}

# Galicia y Andalucía por tramo de edad
tabla_fuentes <- function(datos, partidos_sel) {
  datos |>
    filter(partido %in% partidos_sel) |>
    mutate(pct = round(pct)) |>
    select(tramo, fuente, partido, pct) |>
    pivot_wider(names_from = partido, values_from = pct) |>
    arrange(tramo, fuente)
}
tres_fuentes("tramo", "Galicia") |> tabla_fuentes(c("BNG", "PP", "PSOE", "VOX"))
tres_fuentes("tramo", "Andalucía") |> tabla_fuentes(c("PP", "PSOE", "VOX", "SUMAR"))

# ¿Cuánto mueven los pesos? Peso relativo mínimo y máximo, y efecto de diseño
# de Kish (1 + var/media²): con 2, la muestra efectiva es la mitad.
pondera_sin_rec(filter(encuestas, ccaa == "Andalucía"), filter(celdas, ccaa == "Andalucía")) |>
  mutate(w = peso_rk / mean(peso_rk)) |>
  summarise(n = n(), min = min(w), max = max(w), kish = 1 + var(w))

# Mapa: partido más votado por provincia y tramo, con las tres fuentes en filas.
# En cada provincia, el primero y el segundo; en bruto, también las entrevistas.
mapa_tres_fuentes <- function(comunidad) {
  ganador <- tres_fuentes(c("cod", "tramo"), comunidad) |>
    filter(!is.na(cod)) |> # el bruto incluye entrevistas sin provincia
    arrange(desc(pct)) |>
    summarise(primero = first(partido), pct1 = first(pct),
              segundo = nth(partido, 2), pct2 = nth(pct, 2),
              n_grupo = first(n_grupo),
              .by = c(fuente, cod, tramo)) |>
    mutate(etiqueta = paste0(primero, " ", round(pct1), "\n", segundo, " ", round(pct2),
                             if_else(is.na(n_grupo), "", paste0("\nn = ", n_grupo))))

  provincias_sf |>
    inner_join(ganador, by = c("codigo_provincia" = "cod"), relationship = "one-to-many") |>
    ggplot() +
    geom_sf(aes(fill = primero), colour = "white", linewidth = 0.4) +
    geom_sf_text(aes(label = etiqueta), size = 2.4, lineheight = 0.9, colour = "grey10",
                 fun.geometry = \(g) suppressWarnings(sf::st_point_on_surface(g))) +
    facet_grid(fuente ~ tramo, switch = "y") +
    scale_fill_manual(values = colores) +
    labs(fill = NULL,
         title = paste0(comunidad, ": partido más votado por provincia y edad"),
         subtitle = "Bruto: sin ponderar. Ponderada: por sexo, edad, estudios y provincia, sin recuerdo. MRP: modelo y postestratificación con recuerdo.",
         caption = paste("Encuestas: CIS, 40dB, More in Common y Ateneo del Dato, jun-oct 2026. Voto + simpatía, % de voto válido.",
                         "MRP: media de 1.000 draws, octubre de 2026, participación del 80 %, casa media.", sep = "\n")) +
    theme_void(base_size = 12) +
    theme(plot.title = element_text(face = "bold"),
          plot.subtitle = element_text(colour = "grey30", size = 9),
          plot.caption = element_text(colour = "grey50", hjust = 0, size = 8),
          strip.text = element_text(face = "bold", size = 11, margin = margin(4, 4, 4, 4)),
          strip.text.y.left = element_text(angle = 90),
          legend.position = "bottom")
}

mapa_tres_fuentes("Galicia")
mapa_tres_fuentes("Andalucía")
