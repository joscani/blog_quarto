# ============================================================================
# FULL BAYESIAN 29-N · actualizar el pipeline de datos y ajustes
#
# El mando de datos y ajustes de la cadena (los posts se encargan de su
# propio render; ver actualizar_seguimiento.R, que es el mando único y
# llama a este). GUIA.md tiene los detalles:
#
#   1. baja las encuestas nuevas (scrape incremental) y reconstruye los CSV
#   2. compara la huella (md5) de los ficheros de entrada con la del ÚLTIMO
#      AJUSTE (vivo/huella_entradas.csv): si coinciden, no hay nada que
#      re-ajustar (salvo --forzar). Hasta el 7 nov 2026 se comparaba con la
#      de antes de bajar, y fallaba si otro proceso ya había bajado las
#      encuestas (scripts/actualizar.sh de las slides, u otro equipo vía git)
#   3. si han cambiado: 12_ajustar_conjunto.R (el modelo) + 16 (ablación)
#      + 17 (Omega). El tracker solo necesita el modelo de Stan: los brms
#      de la comparación con octubre (05) y sus reajustes solo corren con
#      --con-comparar
#   4. (opcional) 13_backtest_conjunto.R: solo con --con-backtest; una
#      encuesta nueva de 2026 no lo afecta (usa solo historia)
#   5. sube los ajustes a la release de GitHub (--clobber)
#
# Todas las salidas del pipeline van a 2026/10/full-bayesian/vivo/ (los
# datos vivos del tracker). Las copias en la raíz del folder quedaron
# congeladas con el post de octubre y nadie las sobrescribe.
#
# Lo que NO hace: 06_scrapear_autonomicas.R (historia cerrada), los
# backtests congelados, los posts anteriores (2026/10/actualizar_pipeline.R).
#
# Uso (desde la raíz del proyecto, para que se active renv):
#   Rscript 2026/10/full-bayesian/actualizar_full_bayesian.R
#     --forzar        reajusta aunque los datos no hayan cambiado
#     --con-backtest  añade el backtest del 13 (~40 min)
#     --sin-publicar  no sube los fits a la release
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)
forzar <- "--forzar" %in% args
con_backtest <- "--con-backtest" %in% args
con_comparar <- "--con-comparar" %in% args
con_publicar <- !"--sin-publicar" %in% args

log <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# los ficheros de los que depende el modelo conjunto (sin los congelados)
inputs <- c(
  sep = here::here("2026/10/encuestas_meta_grande_sep.csv"),
  bloque = here::here("2026/10/encuestas_meta_bloque.csv"),
  agregadas = here::here("2026/10/encuestas_meta_agregadas.csv"),
  autonomicas = here::here("2026/10/full-bayesian/errores_autonomicas.csv")
)
hash <- function() vapply(unname(inputs), \(f) unname(tools::md5sum(f)), "")
dir_fb <- here::here("2026/10/full-bayesian")
# la release del post congelado (full-bayesian-29n) queda como archivo: aquí
# van los fits vivos, que se sobrescriben con cada actualización
release_tag <- "full-bayesian-29n-vivo"

run <- function(script, ...) {
  log("·", basename(script), paste0(list(...), collapse = " "))
  st <- system2("Rscript", c(script, ...))
  if (st != 0) stop("Falló ", script, " (status ", st, ")")
}

# ---- 1. datos ---------------------------------------------------------------

log("1. bajar encuestas nuevas y reconstruir los CSV")
source(here::here("2026/10/preparar_meta.R"))
preparar_todo(actualizar = TRUE)

# huella (md5) de las entradas con las que se hizo el último ajuste correcto.
# Se compara con ella y no con la de antes de bajar: si otro proceso ya bajó
# las encuestas (scripts/actualizar.sh de las slides, u otro equipo vía git),
# la descarga de aquí no cambia nada, pero el ajuste sigue sin ellas. Si falta
# el fichero, se reajusta
f_huella <- file.path(dir_fb, "vivo", "huella_entradas.csv")
previa <- if (file.exists(f_huella)) {
  h <- read.csv(f_huella, colClasses = "character")
  h$md5[match(names(inputs), h$entrada)]
} else {
  rep(NA_character_, length(inputs))
}
cambiado <- names(inputs)[is.na(previa) | previa != hash()]
if (length(cambiado) == 0 && !forzar) {
  log(
    "Las entradas son las del último ajuste: nada que re-ajustar. ",
    "(Si ha bajado una encuesta nueva, ha quedado excluida por las reglas ",
    "de inclusión; ver GUIA.md, sección 4. --forzar para reajustar igualmente.)"
  )
  quit(status = 0)
}
log("Cambios en:", paste(cambiado, collapse = ", "))

# ---- 2. modelo conjunto y derivados ------------------------------------------

log("2. modelo conjunto (12) y derivados (16, 17)")
run(file.path(dir_fb, "12_ajustar_conjunto.R"), "principal")
run(file.path(dir_fb, "16_ablacion_error.R"))
run(file.path(dir_fb, "17_sensibilidad_omega.R"))
# solo tras un ajuste correcto: si algo falla antes, la próxima vez reajusta
write.csv(
  data.frame(entrada = names(inputs), md5 = hash()),
  f_huella,
  row.names = FALSE
)
log("Huella de las entradas guardada en vivo/huella_entradas.csv")

# ---- 3. comparación con el pipeline de octubre (05), solo a petición ----------
# El tracker solo necesita el modelo de Stan. 05 simula el escenario mapa con
# los brms del pipeline de octubre (mod_meta_grande_sep.rds y el modelo 2 del
# bloque dentro de simular()); refit y 05 solo corren con --con-comparar.

if (con_comparar) {
  log("3. comparación con octubre: refit de brms y 05")
  source(here::here("2026/10/ajustar_meta.R"))
  if ("sep" %in% cambiado || forzar) {
    invisible(ajustar_grande_sep(refit = TRUE))
  }
  if ("bloque" %in% cambiado || forzar) {
    invisible(ajustar_bloque(refit = TRUE))
  }
  run(file.path(dir_fb, "05_comparar.R"))
} else {
  log("3. comparación con octubre: se salta (usa --con-comparar)")
}

# ---- 4. backtest (opcional) ---------------------------------------------------

if (con_backtest) {
  log("4. backtest (26j, 28a, 10n, 23j; ~40 min)")
  run(
    file.path(dir_fb, "13_backtest_conjunto.R"),
    "principal",
    "26j,28a,10n,23j"
  )
}

# ---- 5. release ---------------------------------------------------------------

if (con_publicar) {
  files <- c(
    file.path(dir_fb, "vivo", "fit_conjunto.rds"),
    file.path(dir_fb, "vivo", "conjunto_sim.rds")
  )
  if (con_backtest) {
    files <- c(
      files,
      file.path(
        dir_fb,
        "vivo",
        list.files(
          file.path(dir_fb, "vivo"),
          "^fit_conjunto_bt_.*principal_26j_28a_10n_23j\\.rds$"
        )
      )
    )
  }
  log("5. subiendo a la release:", paste(basename(files), collapse = ", "))
  st <- system2(
    "gh",
    c("release", "upload", release_tag, shQuote(files), "--clobber")
  )
  if (st != 0) stop("Falló gh release upload (status ", st, ")")
}

log(
  "Listo. Recuerda el commit y el push: los enlaces del post apuntan a blob/master."
)
