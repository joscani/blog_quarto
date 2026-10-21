# ============================================================================
# FULL BAYESIAN 29-N · actualizar el post de arriba a abajo
#
# El mando único de la cadena del post (ver GUIA.md para los detalles):
#
#   1. baja las encuestas nuevas (scrape incremental) y reconstruye los CSV
#   2. compara la huella de los ficheros de entrada con la de antes de bajar:
#      si no han cambiado, no hay nada que re-ajustar (salvo --forzar)
#   3. si han cambiado:
#      - 12_ajustar_conjunto.R (el modelo) + 16 (ablación) + 17 (Omega)
#      - reajusta los modelos brms cuyas entradas hayan cambiado y
#        regenera la comparación con octubre (05)
#   4. (opcional) 13_backtest_conjunto.R: solo con --con-backtest; una
#      encuesta nueva de 2026 no lo afecta (usa solo historia)
#   5. renderiza el .qmd
#   6. sube los ajustes a la release de GitHub (--clobber)
#
# Lo que NO hace: 06_scrapear_autonomicas.R (historia cerrada), los
# backtests congelados, los posts anteriores (2026/10/actualizar_pipeline.R).
#
# Uso (desde la raíz del proyecto, para que se active renv):
#   Rscript 2026/10/full-bayesian/actualizar_full_bayesian.R
#     --forzar        reajusta aunque los datos no hayan cambiado
#     --con-backtest  añade el backtest del 13 (~40 min)
#     --sin-render    no renderiza el .qmd al final
#     --sin-publicar  no sube los fits a la release
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)
forzar <- "--forzar" %in% args
con_backtest <- "--con-backtest" %in% args
con_render <- !"--sin-render" %in% args
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
release_tag <- "full-bayesian-29n"

run <- function(script, ...) {
  log("·", basename(script), paste0(list(...), collapse = " "))
  st <- system2("Rscript", c(script, ...))
  if (st != 0) stop("Falló ", script, " (status ", st, ")")
}

# ---- 1. datos ---------------------------------------------------------------

antes <- hash()
log("1. bajar encuestas nuevas y reconstruir los CSV")
source(here::here("2026/10/preparar_meta.R"))
preparar_todo(actualizar = TRUE)

cambiado <- names(inputs)[antes != hash()]
if (length(cambiado) == 0 && !forzar) {
  log(
    "Los datos de entrada no han cambiado: nada que re-ajustar. ",
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

# ---- 3. comparación con el pipeline de octubre (05) ---------------------------
# 05 simula el escenario mapa, que lee mod_meta_grande_sep.rds (y el modelo 2
# del bloque dentro de simular()); solo se refitan los que hayan cambiado.

if (any(c("sep", "bloque") %in% cambiado) || forzar) {
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
  log(
    "3. comparación con octubre: sin cambios en sus entradas, se queda como está"
  )
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

# ---- 5. render ----------------------------------------------------------------

if (con_render) {
  log("5. render del post")
  st <- system2("quarto", c("render", file.path(dir_fb, "full-bayesian.qmd")))
  if (st != 0) stop("Falló el render del .qmd (status ", st, ")")
}

# ---- 6. release ---------------------------------------------------------------

if (con_publicar) {
  files <- c(
    file.path(dir_fb, "fit_conjunto.rds"),
    file.path(dir_fb, "conjunto_sim.rds")
  )
  if (con_backtest) {
    files <- c(
      files,
      file.path(
        dir_fb,
        list.files(
          dir_fb,
          "^fit_conjunto_bt_.*principal_26j_28a_10n_23j\\.rds$"
        )
      )
    )
  }
  log("6. subiendo a la release:", paste(basename(files), collapse = ", "))
  st <- system2(
    "gh",
    c("release", "upload", release_tag, shQuote(files), "--clobber")
  )
  if (st != 0) stop("Falló gh release upload (status ", st, ")")
}

log(
  "Listo. Recuerda el commit y el push: los enlaces del post apuntan a blob/master."
)
