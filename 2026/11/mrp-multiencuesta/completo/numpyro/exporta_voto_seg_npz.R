# Exporta completo/datos/datos_voto_seg.rds (10_datos_voto_seg.R) a .npz para
# completo/numpyro/mrp_voto_seg_np.py. Índices 1-based; se pasan a 0-based en
# Python.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/numpyro/exporta_voto_seg_npz.R
#
# Salida: completo/datos/voto_seg_numpyro.npz

suppressMessages(library(reticulate))
use_virtualenv("~/.virtualenvs/r-mrp-numpyro", required = TRUE)

dir_completo <- here::here("2026/11/mrp-multiencuesta/completo/datos")
x <- readRDS(file.path(dir_completo, "datos_voto_seg.rds"))
d <- x$datos

np <- import("numpy")
f_npz <- file.path(dir_completo, "voto_seg_numpyro.npz")
escalares <- c(
  "s_alpha", "s_b_mujer", "s_b_dud", "s_sd_rec", "s_sd_main", "s_sd_int",
  "s_sd_dud", "n_casa", "n_mes", "n_edad", "n_edu", "n_rec", "n_ccaa",
  "n_prov", "j_aa"
)
vectores <- c(
  "mujer", "casa", "mes", "edad", "edu", "rec", "ccaa", "sin_aa",
  "fila1", "prov1", "seg1", "fila2", "ini", "len", "ent_prov", "ent_seg", "ent_w"
)
args <- c(
  list(f_npz, y = d$y, off = d$off),
  lapply(d[escalares], function(v) np$array(v)),
  lapply(d[vectores], function(v) np$array(as.vector(v))),
  list(voto_lv = x$lv$voto, casa_lv = x$lv$casa)
)
do.call(np$savez_compressed, args)
cat("Escrito", f_npz, "\n")
