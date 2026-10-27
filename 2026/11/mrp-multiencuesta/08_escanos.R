# Escaños del Congreso con D'Hondt sobre el posterior MRP por celda
# (datos/post_celdas_casa_media.rds, de 07_post_celdas.R): casa media, último
# mes, si votara todo el mundo.
#
# Elecciones al Congreso: D'Hondt por provincia con el mapa de escaños del
# 29-N (RD 806/2026: Madrid 37 -> 38, Cádiz 9 -> 8), como prov_ref en
# simular_escanos.R. Detalles:
#   - OTROS no compite (como RESTO en 2026/10/simular_escanos.R: agrupa a
#     SALF, UPN y resto, que en la realidad van en listas distintas;
#     su voto se trata como no representado). El umbral legal del 3% provincial
#     (LOREG art. 163) se aplica sobre el voto válido ANTES de quitar OTROS.
#   - El voto válido del modelo no lleva blancos (~0,5%): misma aproximación
#     que simular_escanos.R.
#   - Incertidumbre: solo la del posterior MRP. La capa de error de encuestas
#     del post del 29-N sería una extensión.
#   - Pendiente: versión con participación (peso pv de mrp_part.stan por celda)
#     cuando esté el ajuste nuts de las 4 casas en 05b_separado.R.
#
# Salidas (en el post y en datos/):
#   escanos_mrp_resumen.csv     mediana, media e IC90 por partido
#   escanos_mrp_mayorias.csv    P(mayorías)
#   escanos_mrp_provincias.csv  voto y escaños por provincia y partido
#   escanos_mrp.png             distribución posterior de escaños
#   datos/estim_escanos_mrp.rds arrays draw x provincia x partido
#
#   Rscript 2026/11/mrp-multiencuesta/08_escanos.R              # entrada preferente: _part
#   Rscript 2026/11/mrp-multiencuesta/08_escanos.R brms         # entrada base (brms), salidas _brms
#   Rscript 2026/11/mrp-multiencuesta/08_escanos.R rtmb         # entrada RTMB (_rtmb_part o _rtmb), salidas _rtmb

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(ggdist)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

f_post_part <- file.path(dir_datos, "post_celdas_casa_media_part.rds")
f_post_base <- file.path(dir_datos, "post_celdas_casa_media.rds")
f_post_brms <- file.path(dir_datos, "post_celdas_casa_media_brms.rds")
etiq <- commandArgs(trailingOnly = TRUE)[1]
suf <- if (!is.na(etiq) && etiq != "") paste0("_", etiq) else ""
f_post_rtmb <- file.path(dir_datos, "post_celdas_casa_media_rtmb_part.rds")
f_post_rtmb_base <- file.path(dir_datos, "post_celdas_casa_media_rtmb.rds")
if (!is.na(etiq) && etiq == "brms") {
  x <- readRDS(f_post_brms)
} else if (!is.na(etiq) && etiq == "rtmb") {
  x <- readRDS(if (file.exists(f_post_rtmb)) f_post_rtmb else f_post_rtmb_base)
} else if (file.exists(f_post_part)) {
  # versión con participación: el peso de cada celda (pv) varía por draw
  x <- readRDS(f_post_part)
} else if (file.exists(f_post_base)) {
  x <- readRDS(f_post_base)
} else {
  stop("No existe post_celdas_casa_media*.rds: ejecuta antes 07_post_celdas.R")
}
p_celda <- x$p # draws x celda x partido
post <- x$post
voto_lv <- x$voto_lv
nd <- dim(p_celda)[1]

if (!is.null(x$pv)) {
  # participación: el peso de cada celda (pv) varía por draw
  for (d in seq_len(nd)) {
    p_celda[d, , ] <- p_celda[d, , ] * x$pv[d, ]
  }
  cat("ponderando celdas por la participación de mrp_part.stan\n")
}

umbral <- 0.03

# ---- mapa de escaños por provincia (el del 29-N) ------------------------------

ref13 <- read_csv(
  here::here("2026/10", "ref_provincial_23J_13.csv"),
  show_col_types = FALSE
)
prov_ref <- ref13 |>
  distinct(codigo_provincia, provincia, ccaa, diputados) |>
  arrange(codigo_provincia)
escanos_29n <- c("28" = 38, "11" = 8)
prov_ref <- prov_ref |>
  mutate(
    diputados = coalesce(
      unname(escanos_29n[as.character(codigo_provincia)]),
      diputados
    )
  )
stopifnot(sum(prov_ref$diputados) == 350)

# votos por provincia y draw: sum(N * p) sobre las celdas de cada provincia
# (p_celda es draws x celda x partido, luego p[,,j] %*% A es draw x provincia)
# Normaliza a entero: ref13 usa códigos INE con cero a la izquierda ("01"),
# post$prov no lleva relleno ("1")
prov_i <- match(
  as.integer(as.character(post$prov)),
  as.integer(prov_ref$codigo_provincia)
)
stopifnot(!anyNA(prov_i))
A <- matrix(0, nrow(post), nrow(prov_ref))
A[cbind(seq_len(nrow(post)), prov_i)] <- post$N
votos <- array(
  0,
  c(nd, nrow(prov_ref), length(voto_lv)),
  list(draw = NULL, provincia = NULL, partido = voto_lv)
)
for (j in seq_along(voto_lv)) {
  votos[,, j] <- p_celda[,, j] %*% A
}

# ---- D'Hondt -------------------------------------------------------------------

# copiada de 2026/10/simular_escanos.R
dhondt <- function(votos, escanos) {
  result <- setNames(numeric(length(votos)), names(votos))
  for (i in seq_len(escanos)) {
    ganador <- which.max(votos / (result + 1))
    result[ganador] <- result[ganador] + 1
  }
  result
}

partidos_dh <- setdiff(voto_lv, "OTROS")
esc <- array(
  0L,
  c(nd, nrow(prov_ref), length(partidos_dh)),
  list(draw = NULL, provincia = prov_ref$provincia, partido = partidos_dh)
)
for (d in seq_len(nd)) {
  for (i in seq_len(nrow(prov_ref))) {
    v <- votos[d, i, ]
    names(v) <- voto_lv
    v <- v / sum(v)
    v[v < umbral] <- 0
    v["OTROS"] <- 0
    esc[d, i, ] <- dhondt(v, prov_ref$diputados[i])[partidos_dh]
  }
}
totales <- apply(esc, c(1, 3), sum)
stopifnot(all(rowSums(totales) == 350))

# ---- resúmenes -----------------------------------------------------------------

# etiquetas del post (los nombres del MRP: JUNTS -> Junts, BILDU -> EH Bildu, CCA -> CC)
etq <- c(
  PP = "PP",
  PSOE = "PSOE",
  VOX = "VOX",
  SUMAR = "SUMAR",
  PODEMOS = "PODEMOS",
  ERC = "ERC",
  JUNTS = "Junts",
  BILDU = "EH Bildu",
  PNV = "PNV",
  BNG = "BNG",
  CCA = "CC",
  UPN = "UPN"
)
stopifnot(setequal(names(etq), partidos_dh))

# etiqueta de escaños: las medias suman 350 en cada draw; se redondean por
# resto mayor (las medianas por partido no suman 350), como en el post del 29-N
medias <- colMeans(totales)
base <- floor(medias)
extra <- rank(-(medias - base), ties.method = "first") <=
  round(sum(medias)) - sum(base)
resumen <- tibble(
  partido = factor(
    etq[partidos_dh],
    levels = etq[partidos_dh][order(medias, decreasing = TRUE)]
  ),
  mediana = apply(totales, 2, median),
  media = unname(medias),
  q05 = apply(totales, 2, quantile, 0.05),
  q95 = apply(totales, 2, quantile, 0.95),
  escanos = unname(base + extra),
  # P(más escaños): los empates reparten el crédito entre los empatados
  p_mas_esc = sapply(seq_along(partidos_dh), function(j) {
    m <- apply(totales, 1, max)
    empate <- rowSums(totales == m)
    mean((totales[, j] == m) / empate)
  })
)

# mayorías: derecha = PP + VOX; investidura = bloque del 23-J + CC, como
# bloques() en simular_escanos.R (SALF está dentro de OTROS y no se asigna)
bloques_d <- cbind(
  derecha = totales[, "PP"] + totales[, "VOX"],
  izquierda = totales[, "PSOE"] +
    totales[, "SUMAR"] +
    totales[, "PODEMOS"] +
    totales[, "ERC"] +
    totales[, "JUNTS"] +
    totales[, "BILDU"] +
    totales[, "PNV"] +
    totales[, "BNG"] +
    totales[, "CCA"]
)
mayorias <- tibble(
  mayoria = c("PP solo >= 176", "PP + VOX >= 176", "Investidura >= 176"),
  prob = c(
    mean(totales[, "PP"] >= 176),
    mean(bloques_d[, "derecha"] >= 176),
    mean(bloques_d[, "izquierda"] >= 176)
  )
)

# por provincia: voto mediana (%), escaños mediana, P(gana la provincia, por
# voto) y P(al menos un escaño). Los empates reparten el crédito
provincias <- bind_rows(lapply(seq_len(nrow(prov_ref)), function(i) {
  # shares sobre el voto válido (OTROS en el denominador, pero no compite)
  v <- 100 * votos[, i, partidos_dh] / rowSums(votos[, i, ]) # nd x 11
  m <- apply(v, 1, max)
  empate <- rowSums(v == m)
  tibble(
    codigo_provincia = prov_ref$codigo_provincia[i],
    provincia = prov_ref$provincia[i],
    ccaa = prov_ref$ccaa[i],
    diputados = prov_ref$diputados[i],
    partido = factor(etq[partidos_dh], levels = levels(resumen$partido)),
    voto_mediana = apply(v, 2, median),
    escanos_mediana = apply(esc[, i, ], 2, median),
    p_gana = sapply(seq_along(partidos_dh), function(j) {
      mean((v[, j] == m) / empate)
    }),
    p_escano = colMeans(esc[, i, ] > 0)
  )
}))

cat(
  "\n== escaños Congreso, MRP casa media,",
  x$mes,
  if (is.null(x$pv)) {
    ", si votara todo el mundo ==\n"
  } else {
    ", con participación ==\n"
  }
)
print(as.data.frame(resumen), row.names = FALSE)
cat("\n== probabilidad de cada mayoría ==\n")
print(as.data.frame(mayorias), row.names = FALSE)

readr::write_csv(
  resumen,
  file.path(dir_proy, paste0("escanos_mrp", suf, "_resumen.csv"))
)
readr::write_csv(
  mayorias,
  file.path(dir_proy, paste0("escanos_mrp", suf, "_mayorias.csv"))
)
readr::write_csv(
  provincias,
  file.path(dir_proy, paste0("escanos_mrp", suf, "_provincias.csv"))
)
saveRDS(
  list(
    totales = totales,
    esc = esc,
    votos = votos,
    prov_ref = prov_ref,
    resumen = resumen,
    mayorias = mayorias,
    umbral = umbral,
    mes = x$mes
  ),
  file.path(dir_datos, paste0("estim_escanos_mrp", suf, ".rds"))
)

# ---- gráfico -------------------------------------------------------------------

# paleta del post del 29-N (escenarios_mapa_29N.R)
colores <- c(
  "PP" = "#005999",
  "PSOE" = "#FF0126",
  "VOX" = "#51962A",
  "SUMAR" = "#A00B85",
  "PODEMOS" = "#6F2C91",
  "ERC" = "#F5A21D",
  "Junts" = "#00E0D1",
  "EH Bildu" = "#00A56D",
  "PNV" = "#008141",
  "BNG" = "#B5C9E4",
  "CC" = "#D44A20"
)

todo <- as_tibble(totales) |>
  mutate(.draw = row_number()) |>
  pivot_longer(-.draw, names_to = "partido", values_to = "escanos") |>
  mutate(
    partido = etq[partido],
    partido = factor(partido, levels = levels(resumen$partido))
  )

p_esc <- todo |>
  ggplot(aes(x = escanos, y = partido, fill = partido)) +
  stat_halfeye(
    density = "histogram",
    breaks = breaks_fixed(width = 1),
    align = align_center(at = 0),
    normalize = "groups",
    height = 0.6,
    .width = c(0.5, 0.9),
    alpha = 0.8
  ) +
  geom_label(
    data = resumen,
    aes(x = escanos, y = partido, label = escanos),
    inherit.aes = FALSE,
    size = 3,
    fill = "white"
  ) +
  scale_fill_manual(values = colores, guide = "none") +
  labs(
    title = "Escaños del Congreso: posterior MRP",
    subtitle = paste0(
      "Casa media, ",
      x$mes,
      if (is.null(x$pv)) ", si votara todo el mundo" else ", con participación",
      ". La cifra etiquetada es la media (suma 350)."
    ),
    x = "Escaños",
    y = NULL
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(dir_proy, paste0("escanos_mrp", suf, ".png")),
  p_esc,
  width = 8,
  height = 5.5,
  dpi = 200
)
cat(
  "\nguardados escanos_mrp",
  suf,
  "_*.csv, escanos_mrp",
  suf,
  ".png y estim_escanos_mrp",
  suf,
  ".rds\n",
  sep = ""
)
