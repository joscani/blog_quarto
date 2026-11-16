# Funciones del post (IV): Y ahora, a jugar con el MRP.
#
# El post (iv-a-jugar.qmd) solo llama a estas funciones. Leen salidas ya
# calculadas por los scripts de completo/ (ver el apéndice del post):
#   21_posterior_celdas.R  tabla_post.rds y posterior_celdas_80.rds
#   31_compara_semillas.R  compara_semillas.rds
#   32_jovenes_provincia.R jovenes_<comunidad>.rds
#   33_navarra_23j_zonas.R navarra_23j_zonas.rds
#   34_vox_18_20_ccaa.R    vox_18_20_ccaa.rds

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(gt)
})

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_completo <- file.path(dir_proy, "completo", "datos")
# Modelo del post (16 nov 2026): semilla multinivel + predictor de grupo. Los
# ficheros publicados el 14 nov (sin sufijo) siguen en el disco; "" vuelve a ellos.
sufijo_modelo <- "_semilla_mn_gp"

# ---- Formato -------------------------------------------------------------------

num <- function(x, d = 0) {
  format(round(x, d), big.mark = ".", decimal.mark = ",", nsmall = d, trim = TRUE)
}
pct <- function(x, d = 1) paste(num(x, d), "%")

# gt con coma decimal y punto de miles
gt_es <- function(d, ...) {
  gt(d, ...) |> opt_table_font(size = px(14))
}

# ---- 1. El cambio de semilla ---------------------------------------------------

semillas <- function() readRDS(file.path(dir_completo, paste0("compara_semillas", sufijo_modelo, ".rds")))

tabla_semillas <- function() {
  semillas()$resumen |>
    gt_es() |>
    cols_label(agregado = "", grupos = "Grupos",
               dif_mediana = "Mediana", dif_p95 = "p95", dif_max = "Máximo",
               cambia_primero = "Cambia el primero") |>
    tab_spanner("Diferencia, puntos (valor absoluto)", c(dif_mediana, dif_p95, dif_max)) |>
    fmt_number(c(dif_mediana, dif_p95, dif_max), decimals = 2, dec_mark = ",", sep_mark = ".") |>
    tab_source_note("Mismo posterior (participación del 80 %, casa media); solo cambia la N de cada celda. % de voto válido, media de 1.000 draws.")
}

# % de un grupo y partido con la tabla vieja y la nueva
semilla_grupo <- function(grupo, partidos) {
  r <- semillas()$res |>
    filter(agregado == "Provincia x edad (3 tramos)", .data$grupo == !!grupo, partido %in% partidos)
  # con el modelo nuevo: tres tablas (provincia, comunidad, multinivel), misma posterior
  if ("pct_ccaa" %in% names(r)) {
    r <- r |> transmute(grupo, partido, semilla_provincia = pct_vieja, semilla_comunidad = pct_ccaa,
                        semilla_multinivel = pct_nueva)
  }
  r |> mutate(across(where(is.double), \(x) round(x, 1)))
}

# ---- 2. Jóvenes de una comunidad -----------------------------------------------

jovenes <- function(slug) readRDS(file.path(dir_completo, paste0("jovenes_", slug, ".rds")))

tabla_jovenes <- function(slug, zonas = NULL, n_partidos = 4) {
  j <- jovenes(slug)
  d <- j$mrp
  if (!is.null(zonas)) d <- filter(d, zona %in% zonas)
  d |>
    group_by(zona) |>
    slice_max(media, n = n_partidos) |>
    ungroup() |>
    gt_es(groupname_col = "zona") |>
    cols_label(partido = "", media = "MRP (%)", q05 = "p5", q95 = "p95",
               p_primero = "P(primero), %") |>
    fmt_number(c(media, q05, q95), decimals = 1, dec_mark = ",") |>
    fmt_number(p_primero, decimals = 0) |>
    tab_source_note("18-34 años. % de voto válido, participación del 80 %, casa media. Intervalo del 90 % y % de draws en que el partido es el más votado.")
}

# ---- 3. Navarra en el 23-J -----------------------------------------------------

navarra_23j <- function() readRDS(file.path(dir_completo, "navarra_23j_zonas.rds"))

tabla_navarra_zonas <- function() {
  navarra_23j()$zonas |>
    gt_es() |>
    cols_label(zona = "Partido judicial", municipios = "Municipios",
               votos_validos = "Votos válidos", vox_pct = "VOX (%)", bildu_pct = "Bildu (%)") |>
    fmt_number(votos_validos, decimals = 0, sep_mark = ".") |>
    fmt_number(c(vox_pct, bildu_pct), decimals = 1, dec_mark = ",") |>
    tab_source_note("Generales del 23-J de 2023, por municipio (infoelectoral). Votos válidos = candidaturas + blancos.")
}

tabla_navarra_municipios <- function() {
  navarra_23j()$ejemplos |>
    arrange(vox_pct) |>
    gt_es() |>
    cols_label(municipio = "Municipio", votos_validos = "Votos válidos",
               vox_pct = "VOX (%)", bildu_pct = "Bildu (%)") |>
    fmt_number(votos_validos, decimals = 0, sep_mark = ".") |>
    fmt_number(c(vox_pct, bildu_pct), decimals = 1, dec_mark = ",") |>
    tab_source_note("Generales del 23-J de 2023 (infoelectoral). Votos válidos = candidaturas + blancos.")
}

# ---- 4. VOX en 18-20 por comunidad ---------------------------------------------

vox_18_20 <- function() readRDS(file.path(dir_completo, paste0("vox_18_20_ccaa", sufijo_modelo, ".rds")))

tabla_vox_18_20 <- function(n_min = 20) {
  vox_18_20() |>
    filter(tramo == "18-20", n >= n_min) |>
    arrange(z) |>
    select(ccaa, n, bruto, bruto_q05, bruto_q95, mrp, q05, q95) |>
    gt_es() |>
    cols_label(ccaa = "", n = "Entrevistas", bruto = "%", bruto_q05 = "p5", bruto_q95 = "p95",
               mrp = "%", q05 = "p5", q95 = "p95") |>
    tab_spanner("Bruto", c(bruto, bruto_q05, bruto_q95)) |>
    tab_spanner("MRP", c(mrp, q05, q95)) |>
    fmt_number(c(bruto, bruto_q05, bruto_q95, mrp, q05, q95), decimals = 1, dec_mark = ",") |>
    tab_source_note("VOX entre los de 18 a 20 años, % de voto válido. Bruto: entrevistas con voto + simpatía, intervalo Beta(1 + votos, 1 + resto). MRP: participación del 80 %, casa media, intervalo del 90 %. Comunidades con 20 entrevistas o más, ordenadas por cuánto se separa el bruto del MRP.")
}

# ---- 5. Un comentario de X, siempre con la misma tabla ---------------------------
# Filas: bruto, ponderada sin recuerdo, ponderada con recuerdo y MRP (con su
# intervalo del 90 %). Columnas: entrevistas y los partidos en disputa.

comentarios <- function() readRDS(file.path(dir_completo, paste0("comentarios", sufijo_modelo, ".rds")))

nombres_fuente <- c(bruto = "Bruto", sin_rec = "Ponderada sin recuerdo",
                    con_rec = "Ponderada con recuerdo", mrp = "MRP")

# partidos: por defecto, los 3 primeros del MRP (sin OTROS ni BLANCO)
tabla_comentario <- function(caso, partidos = NULL) {
  com <- comentarios()
  f <- com$fuentes |> filter(id == caso)
  stopifnot(nrow(f) > 0)
  if (is.null(partidos)) {
    partidos <- f |>
      filter(fuente == "mrp", !partido %in% c("OTROS", "BLANCO")) |>
      slice_max(pct, n = 3) |>
      pull(partido)
  }
  mrp <- f |> filter(fuente == "mrp")
  celdas_txt <- f |>
    filter(partido %in% partidos) |>
    mutate(txt = if_else(fuente == "mrp",
                         paste0(num(pct, 1), " (", num(q05, 0), "-", num(q95, 0), ")"),
                         num(pct, 1))) |>
    select(fuente, partido, txt) |>
    pivot_wider(names_from = partido, values_from = txt, values_fill = "0,0") |>
    left_join(com$entrevistas |> filter(id == caso) |> select(fuente, n), by = "fuente") |>
    mutate(n = if_else(is.na(n), "—", num(n)),
           fuente = factor(nombres_fuente[fuente], nombres_fuente)) |>
    arrange(fuente) |>
    select(fuente, n, all_of(partidos))
  primero <- mrp |>
    filter(p_primero >= 1) |>
    arrange(desc(p_primero)) |>
    with(paste0(partido, " ", num(p_primero), " %", collapse = ", "))
  celdas_txt |>
    gt_es(rowname_col = "fuente") |>
    cols_label(n = "Entrevistas") |>
    cols_align("right", -fuente) |>
    tab_style(cell_text(align = "left"), cells_stub()) |>
    tab_style(cell_text(weight = "bold"), list(cells_body(rows = fuente == "MRP"),
                                              cells_stub(rows = fuente == "MRP"))) |>
    tab_header(title = f$etiqueta[1], subtitle = "% de voto válido") |>
    tab_source_note(paste0("MRP: media e intervalo del 90 % (participación del 80 %, casa media). ",
                           "Primero en los draws del MRP: ", primero, ".")) |>
    tab_source_note("Bruto: entrevistas con voto + simpatía, sin ponderar. Ponderadas: raking de la comunidad por sexo, edad, estudios y provincia (y recuerdo del 23-J); no entran las entrevistas sin provincia.")
}
