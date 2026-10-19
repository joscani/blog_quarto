# ============================================================================
# ESCENARIOS DEL 29N CON EL MAPA DE COALICIONES DE LA IZQUIERDA
#
# Coaliciones registradas el 16 oct 2026 (coaliciones_izquierda_29N_provincias.csv):
#   - Cataluña: Podemos no se presenta (acuerdo con ERC, que va con sus siglas)
#   - Canarias y C. Valenciana: Sumar y Podemos en lista conjunta
#   - Andalucía: Adelante Andalucía compite fuera del Frente Amplio (familia
#     AA propia; share andaluza logit-normal centrada en el 9,62% de las
#     autonómicas del 17 may 2026, con patrón provincial de
#     aa_autonomicas2026_provincias.csv y shock heredado de los grandes)
#   - resto: listas separadas
#
# Pasos:
#   1. actualiza las encuestas y reconstruye los CSV de entrada
#   2. reajusta los tres modelos (unidas, separadas y bloque; seed fija)
#   3. simula unidas, separadas y mapa, los tres con el modelo de 7
#      categorías y la misma semilla: solo cambia la regla provincial (unida
#      en las 52, separadas en las 52, o la registrada), así que los draws
#      están emparejados. En el mapa, la transferencia catalana del voto de
#      Podemos (ERC/SUMAR/fuera) y la share andaluza de AA se sortean draw a
#      draw de priors explícitos (Dirichlet y logit-normal)
#   4. guarda los resúmenes en CSV (los lee el post):
#        escenarios_29N_escanos.csv     mediana, media e IC90 por partido
#        escenarios_29N_mayorias.csv    P(mayorías) por escenario
#        escenarios_29N_bloques.csv     escaños de derecha e investidura
#        escenarios_29N_transicion.csv  media de escaños y diferencia draw a draw
#        escenarios_29N_provincias.csv  mediana de voto y escaños por provincia
#        escenarios_29N_voto.csv        voto nacional, mediana e IC90 (%)
#   5. pinta y guarda los gráficos del post (PNG, en esta carpeta):
#        voto_29N.png                   estimación de voto por escenario
#        escanos_29N.png                distribución posterior de escaños
#        escanos_mapa_29N.png           íd. solo con el escenario mapa
#        escanos_andalucia_29N.png      escaños mediana en Andalucía (mapa)
#        mayorias_29N.png               P(mayoría) por escenario
#
# Uso (desde la raíz del proyecto, para que se active renv):
#   Rscript 2026/10/frente_amplio_y_adelante_andalucia/escenarios_mapa_29N.R
#   Rscript 2026/10/frente_amplio_y_adelante_andalucia/escenarios_mapa_29N.R --sin-actualizar
# ============================================================================

library(purrr)

args <- commandArgs(trailingOnly = TRUE)
actualizar <- !"--sin-actualizar" %in% args
n_draws <- 6000
seed_sim <- 2911

log_paso <- function(...) {
  cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")
}

# ---- 1. encuestas -------------------------------------------------------------

log_paso("1. preparar_todo(actualizar = ", actualizar, ")")
source(here::here("2026/10/preparar_meta.R"))
preparar_todo(actualizar = actualizar)

# ---- 2. modelos ---------------------------------------------------------------

source(here::here("2026/10/simular_escanos.R"))
log_paso("2a. ajustar_grande(refit = TRUE)")
invisible(ajustar_grande(refit = TRUE))
log_paso("2b. ajustar_bloque(refit = TRUE)")
invisible(ajustar_bloque(refit = TRUE))
log_paso("2c. ajustar_grande_sep(refit = TRUE)")
invisible(ajustar_grande_sep(refit = TRUE))

ult <- readr::read_csv(f_grande_sep, show_col_types = FALSE)
log_paso(
  "encuestas en el modelo separado: ",
  nrow(distinct(ult, encuestadora, fecha)),
  ", última: ",
  format(max(ult$fecha))
)

# ---- 3. escenarios ------------------------------------------------------------

# Los tres escenarios van por la rama "mapa" (modelo de 7 categorías) y solo
# cambia la regla provincial: unidas = "unida" en las 52, separadas =
# "separadas" en las 52, mapa = la regla registrada de cada provincia. Así
# comparten posterior, capa de error, shocks y secuencia aleatoria: el draw k
# es el mismo mundo en los tres y la diferencia draw a draw es el efecto
# puro del mapa de listas
mapa_reglas <- readr::read_csv(
  f_mapa_listas,
  col_types = readr::cols(.default = readr::col_character())
)
f_regla_unica <- function(regla) {
  f <- tempfile(fileext = ".csv")
  readr::write_csv(
    tibble(codigo_provincia = mapa_reglas$codigo_provincia, regla = regla),
    f
  )
  f
}
escenarios <- list(
  unidas = list(listas = "mapa", f_mapa = f_regla_unica("unida")),
  separadas = list(listas = "mapa", f_mapa = f_regla_unica("separadas")),
  mapa = list(listas = "mapa")
)

sims <- imap(escenarios, \(e, nombre) {
  log_paso("3. simular: ", nombre)
  set.seed(seed_sim)
  do.call(simular, c(list(n_draws = n_draws, provincial = TRUE), e))
})

# ---- 4. resúmenes -------------------------------------------------------------

# los tres escenarios llevan las 15 familias; las que no se presentan en
# ninguna provincia del escenario (PODEMOS en unidas, AA fuera del mapa)
# tienen voto 0 en todos los draws y se quitan de los resúmenes
presentes <- function(s) {
  fams <- dimnames(s$prov_voto)[[3]]
  fams[apply(s$prov_voto, 3, max) > 0]
}

escanos <- imap_dfr(sims, \(s, nombre) {
  medias <- colMeans(s$draws[, -1])
  resumen_escanos(s, width = 0.9) |>
    filter(partido %in% presentes(s)) |>
    transmute(
      escenario = nombre,
      partido,
      mediana = escanos,
      media = unname(medias[partido]),
      q05 = .lower,
      q95 = .upper
    )
})

# bloques por escenario (definición de bloques(): derecha = PP + VOX;
# izquierda = bloque de investidura de 2023 + PODEMOS + AA)
bloques_esc <- imap_dfr(sims, \(s, nombre) {
  bloques(s$draws) |>
    select(derecha, izquierda) |>
    pivot_longer(everything(), names_to = "bloque", values_to = "escanos") |>
    group_by(bloque) |>
    summarise(
      mediana = median(escanos),
      media = mean(escanos),
      q05 = quantile(escanos, 0.05),
      q95 = quantile(escanos, 0.95),
      p_176 = mean(escanos >= 176),
      .groups = "drop"
    ) |>
    mutate(escenario = nombre, .before = 1)
})

# transición draw a draw: los escenarios comparten draws, así que la
# diferencia de escaños draw a draw (y su media) es el efecto de las listas
transicion <- map_dfr(names(sims$unidas$draws)[-1], \(f) {
  u <- sims$unidas$draws[[f]]
  sp <- sims$separadas$draws[[f]]
  m <- sims$mapa$draws[[f]]
  tibble(
    partido = f,
    media_unidas = mean(u),
    media_separadas = mean(sp),
    media_mapa = mean(m),
    dif_mapa_unidas = mean(m - u),
    dif_separadas_unidas = mean(sp - u)
  )
})
stopifnot(
  abs(sum(transicion$dif_mapa_unidas)) < 1e-8,
  abs(sum(transicion$dif_separadas_unidas)) < 1e-8
)

mayorias <- imap_dfr(sims, \(s, nombre) {
  # prob_mayorias() llama "escenario" a la mayoría: se renombra para no pisarla
  prob_mayorias(s) |>
    rename(mayoria = escenario) |>
    mutate(escenario = nombre, .before = 1)
})

provincias <- imap_dfr(sims, \(s, nombre) {
  regla <- if (is.null(s$regla_prov)) {
    rep(s$listas, nrow(s$prov_ref))
  } else {
    unname(s$regla_prov)
  }
  fams <- dimnames(s$prov_voto)[[3]]
  map_dfr(seq_len(nrow(s$prov_ref)), \(i) {
    tibble(
      escenario = nombre,
      codigo_provincia = s$prov_ref$codigo_provincia[i],
      provincia = s$prov_ref$provincia[i],
      regla = regla[i],
      partido = fams,
      voto_mediana = apply(s$prov_voto[, i, , drop = FALSE], 3, median),
      escanos_mediana = apply(s$prov_escanos[, i, , drop = FALSE], 3, median)
    )
  })
})

dir_out <- here::here("2026/10/frente_amplio_y_adelante_andalucia")
readr::write_csv(escanos, file.path(dir_out, "escenarios_29N_escanos.csv"))
readr::write_csv(mayorias, file.path(dir_out, "escenarios_29N_mayorias.csv"))
readr::write_csv(
  provincias,
  file.path(dir_out, "escenarios_29N_provincias.csv")
)
readr::write_csv(bloques_esc, file.path(dir_out, "escenarios_29N_bloques.csv"))
readr::write_csv(
  transicion,
  file.path(dir_out, "escenarios_29N_transicion.csv")
)

log_paso("4. guardado en ", dir_out)

# ---- 5. gráficos -------------------------------------------------------------

library(ggplot2)
library(ggdist)

# paleta del post de escaños de generales; PODEMOS y AA con sus morados
# (el de AA, el mismo lila que el post andaluz dio a Adelante A.)
colores_esc <- c(
  "PP" = "#005999",
  "PSOE" = "#FF0126",
  "VOX" = "#51962A",
  "SUMAR" = "#A00B85",
  "PODEMOS" = "#6F2C91",
  "AA" = "#8C66F1",
  "SALF" = "#00C3B6",
  "ERC" = "#F5A21D",
  "Junts" = "#00E0D1",
  "EH Bildu" = "#00A56D",
  "PNV" = "#008141",
  "BNG" = "#B5C9E4",
  "CC" = "#D44A20",
  "UPN" = "#1B4A9B",
  "RESTO" = "grey70"
)

# voto nacional por draw: prov_voto ponderado por los votos válidos de la
# ref del 23J (la misma aproximación de válidos del re-centrado del shock).
# Así el voto del escenario refleja las reglas (AA ya con su masa movida)
validos_p <- ref13 |>
  group_by(codigo_provincia) |>
  summarise(v = sum(votos), .groups = "drop") |>
  slice(match(prov_ref$codigo_provincia, codigo_provincia)) |>
  pull(v)
wv_p <- validos_p / sum(validos_p)

todo <- imap_dfr(sims, \(s, nombre) {
  fams <- dimnames(s$prov_voto)[[3]]
  voto <- sapply(fams, \(f) 100 * as.numeric(s$prov_voto[,, f] %*% wv_p))
  voto_l <- as_tibble(voto) |>
    mutate(.draw = row_number()) |>
    pivot_longer(-.draw, names_to = "partido", values_to = "voto")
  esc_l <- s$draws |>
    pivot_longer(-.draw, names_to = "partido", values_to = "escanos")
  left_join(voto_l, esc_l, by = c(".draw", "partido")) |>
    filter(partido %in% presentes(s)) |>
    mutate(escenario = nombre)
})

# voto nacional por partido y escenario (mediana e IC90, en %)
voto_nac <- todo |>
  group_by(escenario, partido) |>
  summarise(
    mediana = median(voto),
    q05 = quantile(voto, 0.05),
    q95 = quantile(voto, 0.95),
    .groups = "drop"
  )
readr::write_csv(voto_nac, file.path(dir_out, "escenarios_29N_voto.csv"))

# partidos ordenados por su mediana de escaños en el mapa (arriba = grande)
niveles <- todo |>
  filter(escenario == "mapa", partido != "RESTO") |>
  group_by(partido) |>
  summarise(m = median(escanos), .groups = "drop") |>
  arrange(desc(m)) |>
  pull(partido)
niveles <- union(niveles, setdiff(unique(as.character(todo$partido)), niveles))
todo <- todo |> mutate(partido = factor(partido, levels = rev(niveles)))

p_voto <- todo |>
  filter(partido != "RESTO") |>
  ggplot(aes(x = voto, y = partido, fill = partido)) +
  stat_halfeye(
    .width = c(0.5, 0.9),
    normalize = "groups",
    height = 0.6,
    alpha = 0.8
  ) +
  facet_wrap(~escenario, nrow = 1) +
  scale_fill_manual(values = colores_esc, guide = "none") +
  labs(
    title = "29-N: estimación de voto por escenario",
    subtitle = "Mediana e IC50/IC90; el escenario mapa lleva ya las reglas de coalición",
    x = "% del voto",
    y = NULL
  ) +
  theme_minimal(base_size = 12)

# etiqueta: media de escaños redondeada por resto mayor (suma 350 en cada
# escenario; las medianas por partido no suman 350)
etiquetas_esc <- imap_dfr(sims, \(s, nombre) {
  escanos_resto_mayor(s) |>
    filter(partido %in% presentes(s), partido != "RESTO") |>
    mutate(escenario = nombre)
}) |>
  mutate(partido = factor(partido, levels = levels(todo$partido)))
stopifnot(all(
  tapply(etiquetas_esc$escanos, etiquetas_esc$escenario, sum) == 350
))

# histograma alineado al entero (ver post de escaños de generales);
# normalize = "groups" para que cada partido ocupe su fila.
hist_escanos <- list(
  stat_halfeye(
    density = "histogram",
    breaks = ggdist::breaks_fixed(width = 1),
    align = ggdist::align_center(at = 0),
    normalize = "groups",
    height = 0.6,
    .width = c(0.5, 0.9),
    alpha = 0.8
  ),
  scale_fill_manual(values = colores_esc, guide = "none")
)

p_esc <- todo |>
  filter(partido != "RESTO") |>
  ggplot(aes(x = escanos, y = partido, fill = partido)) +
  hist_escanos +
  geom_label(
    data = etiquetas_esc,
    aes(x = escanos, y = partido, label = escanos),
    inherit.aes = FALSE,
    size = 3,
    fill = "white"
  ) +
  facet_wrap(~escenario, nrow = 1) +
  labs(
    title = "29-N: distribución posterior de escaños",
    subtitle = "La cifra etiquetada es la media de escaños, redondeada para que sumen 350",
    x = "Escaños",
    y = NULL
  ) +
  theme_minimal(base_size = 12)

# versión solo con el escenario mapa: es el que ya refleja cómo van a concurrir
p_esc_mapa <- todo |>
  filter(partido != "RESTO", escenario == "mapa") |>
  ggplot(aes(x = escanos, y = partido, fill = partido)) +
  hist_escanos +
  geom_label(
    data = filter(etiquetas_esc, escenario == "mapa"),
    aes(x = escanos, y = partido, label = escanos),
    inherit.aes = FALSE,
    size = 3,
    fill = "white"
  ) +
  # en el mapa, la familia SUMAR es el Frente Amplio
  scale_y_discrete(labels = c(SUMAR = "Frente Amplio")) +
  labs(
    title = "29-N: distribución posterior de escaños (escenario mapa)",
    subtitle = "La cifra etiquetada es la media de escaños, redondeada para que sumen 350",
    x = "Escaños",
    y = NULL
  ) +
  theme_minimal(base_size = 12)

# el detalle andaluz del escenario mapa: mediana de escaños por provincia
p_and <- provincias |>
  filter(
    escenario == "mapa",
    regla == "con_aa",
    partido %in% c("PSOE", "SUMAR", "PODEMOS", "AA")
  ) |>
  ggplot(aes(x = partido, y = reorder(provincia, escanos_mediana))) +
  geom_tile(aes(fill = escanos_mediana), color = "white", linewidth = 1) +
  geom_text(aes(label = escanos_mediana), size = 4, fontface = "bold") +
  scale_fill_gradient(low = "grey95", high = "#A00B85", guide = "none") +
  labs(
    title = "29-N, escenario mapa: escaños mediana en Andalucía",
    subtitle = paste(
      "AA se estima con las autonómicas del 17 may 2026 (9,62 % en Andalucía),",
      "no con encuestas generales",
      sep = "\n"
    ),
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12)

p_may <- mayorias |>
  mutate(
    mayoria = factor(
      mayoria,
      levels = c("PP solo >= 176", "PP + VOX >= 176", "izquierda >= 176"),
      labels = c("PP solo ≥ 176", "PP + VOX ≥ 176", "Investidura ≥ 176")
    )
  ) |>
  ggplot(aes(x = mayoria, y = prob, fill = escenario)) +
  geom_col(position = "dodge", width = 0.8) +
  geom_text(
    aes(label = paste0(round(100 * prob, 1), "%")),
    position = position_dodge(width = 0.8),
    vjust = -0.3,
    size = 3
  ) +
  scale_y_continuous(limits = c(0, 1.05), labels = \(x) paste0(100 * x, "%")) +
  labs(title = "29-N: probabilidad de cada mayoría", x = NULL, y = NULL) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(dir_out, "voto_29N.png"),
  p_voto,
  width = 13,
  height = 5.5,
  dpi = 200
)
ggsave(
  file.path(dir_out, "escanos_29N.png"),
  p_esc,
  width = 13,
  height = 5.5,
  dpi = 200
)
ggsave(
  file.path(dir_out, "escanos_mapa_29N.png"),
  p_esc_mapa,
  width = 7.5,
  height = 5.5,
  dpi = 200
)
ggsave(
  file.path(dir_out, "escanos_andalucia_29N.png"),
  p_and,
  width = 6.5,
  height = 4.5,
  dpi = 200
)
ggsave(
  file.path(dir_out, "mayorias_29N.png"),
  p_may,
  width = 8,
  height = 4,
  dpi = 200
)
log_paso("5. gráficos en ", dir_out)
print(
  escanos |>
    select(escenario, partido, mediana) |>
    tidyr::pivot_wider(names_from = escenario, values_from = mediana),
  n = Inf
)
print(mayorias, n = Inf)
log_paso("terminado")
