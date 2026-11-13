# ============================================================================
# SEGUIMIENTO 29-N · mando único del tracker
#
#   1. pipeline de datos y ajustes (actualizar_full_bayesian.R); todas las
#      salidas van a 2026/10/full-bayesian/vivo/, los fits a la release
#      full-bayesian-29n-vivo
#   2. fila nueva (o reemplazo de la de la misma semana) en
#      vivo/historico_pronostico.csv
#   3. render de seguimiento.qmd
#
# El post de octubre (full-bayesian.qmd) queda congelado: nadie lo renderiza
# ni sobrescribe sus datos.
#
# Uso (desde la raíz del proyecto, para que se active renv):
#   Rscript 2026/10/full-bayesian/actualizar_seguimiento.R
#     [--forzar] [--con-backtest] [--sin-publicar]  (pasan al pipeline)
# ============================================================================

log <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")
suppressPackageStartupMessages(library(tidyverse))
dir_fb <- here::here("2026/10/full-bayesian")
dir_vivo <- file.path(dir_fb, "vivo")

# ---- 1. datos y ajustes ------------------------------------------------------

log("1. pipeline de datos y ajustes")
st <- system2(
  "Rscript",
  c(
    file.path(dir_fb, "actualizar_full_bayesian.R"),
    commandArgs(trailingOnly = TRUE)
  )
)
if (st != 0) {
  stop("Falló actualizar_full_bayesian.R (status ", st, ")")
}

# ---- 2. histórico ------------------------------------------------------------

leer <- \(f) read_csv(file.path(dir_vivo, f), show_col_types = FALSE)
esc <- leer("conjunto_escanos.csv")
may <- leer("conjunto_mayorias.csv")
meta <- map_dfr(
  c(
    "encuestas_meta_grande_sep.csv",
    "encuestas_meta_bloque.csv",
    "encuestas_meta_agregadas.csv"
  ),
  \(f) read_csv(file.path(here::here("2026/10"), f), show_col_types = FALSE)
)
enc_hasta <- max(meta$fecha)

f_hist <- file.path(dir_vivo, "historico_pronostico.csv")
hist <- if (file.exists(f_hist)) {
  read_csv(f_hist, show_col_types = FALSE)
} else {
  tibble()
}
# semilla del histórico: el estado publicado del post (fit del 26-10,
# encuestas hasta el 25-10), de los CSV congelados de la raíz, para que la
# gráfica semana a semana empiece en octubre y no en la primera actualización
if (!file.exists(f_hist)) {
  esc_pub <- read_csv(
    file.path(dir_fb, "conjunto_escanos.csv"),
    show_col_types = FALSE
  )
  may_pub <- read_csv(
    file.path(dir_fb, "conjunto_mayorias.csv"),
    show_col_types = FALSE
  )
  semilla <- esc_pub |>
    transmute(
      fecha_actualizacion = as.Date("2026-10-26"),
      encuestas_hasta = as.Date("2026-10-25"),
      partido,
      mediana,
      q05,
      q95,
      p_pp_vox_176 = may_pub$prob[may_pub$escenario == "PP + VOX >= 176"],
      p_izquierda_176 = may_pub$prob[may_pub$escenario == "izquierda >= 176"]
    )
} else {
  semilla <- NULL
}
fila <- esc |>
  transmute(
    fecha_actualizacion = Sys.Date(),
    encuestas_hasta = enc_hasta,
    partido,
    mediana,
    q05,
    q95,
    p_pp_vox_176 = may$prob[may$escenario == "PP + VOX >= 176"],
    p_izquierda_176 = may$prob[may$escenario == "izquierda >= 176"]
  )
if (!is.null(semilla)) {
  fila <- bind_rows(semilla, fila)
}
# una fila por semana de campo: si se re-renderiza sin encuesta nueva, se
# reemplaza en vez de duplicar
if (nrow(hist) > 0) {
  hist <- filter(hist, encuestas_hasta != !!enc_hasta)
}
hist <- hist |>
  bind_rows(fila) |>
  arrange(encuestas_hasta, fecha_actualizacion)
write_csv(hist, f_hist)
log(
  "2. histórico:",
  nrow(hist),
  "filas; encuestas hasta",
  as.character(enc_hasta)
)

# escaños medios por provincia y partido (sí van a git, a diferencia de
# conjunto_sim.rds): la media es aditiva, así que el cambio nacional de cada
# partido es la suma exacta de sus cambios provinciales. Empieza el 9 nov
# 2026, con el ajuste del 3 nov reconstruido con las entradas de git
sim <- readRDS(file.path(dir_vivo, "conjunto_sim.rds"))
f_hprov <- file.path(dir_vivo, "historico_provincias.csv")
fila_prov <- as_tibble(
  apply(sim$prov_escanos, c(2, 3), mean),
  rownames = "provincia"
) |>
  pivot_longer(-provincia, names_to = "partido", values_to = "media") |>
  filter(media > 0) |>
  mutate(
    fecha_actualizacion = Sys.Date(),
    encuestas_hasta = enc_hasta,
    .before = 1
  )
hprov <- if (file.exists(f_hprov)) {
  read_csv(f_hprov, show_col_types = FALSE) |>
    filter(encuestas_hasta != !!enc_hasta)
} else {
  tibble()
}
hprov <- bind_rows(hprov, fila_prov) |>
  arrange(encuestas_hasta, provincia, partido)
write_csv(hprov, f_hprov)
log("2b. histórico provincial:", nrow(hprov), "filas")

# ---- 3. render ---------------------------------------------------------------

log("3. render del tracker")
# La fecha del post es la de la actualización: así sale el primero en blog.qmd
f_qmd <- file.path(dir_fb, "seguimiento.qmd")
qmd <- readLines(f_qmd)
qmd <- sub("^date: .*$", paste0("date: '", Sys.Date(), "'"), qmd)
writeLines(qmd, f_qmd)
log("3. fecha del post:", format(Sys.Date()))

st <- system2("quarto", c("render", f_qmd))
if (st != 0) {
  stop("Falló el render del tracker (status ", st, ")")
}

# Los listados no se regeneran al renderizar un post suelto
for (lst in c("blog.qmd", "archive.qmd", "2026.qmd")) {
  st <- system2("quarto", c("render", here::here(lst)))
  if (st != 0) stop("Falló el render de ", lst, " (status ", st, ")")
}

log("Listo. Recuerda el commit y el push: los CSV de vivo/ van en git y el")
log("post enlaza a los scripts en blob/master.")
