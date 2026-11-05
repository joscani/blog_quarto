# Lo que 12_escenarios_voto_seg.R y 16_vox_salto_nacional.R necesitan de datos/,
# en un fichero pequeño que va en la release. Evita bajar o regenerar
# post_celdas_casa_media_part.rds (~700 MB), del que solo se usa la tabla de celdas.
#
# Entradas: datos/post_celdas_casa_media_part.rds, datos/datos_stan.rds.
# Salida: completo/datos/entradas_escenarios.rds (post, lv, part_obj, mes_post).
#
#   Rscript 2026/11/mrp-multiencuesta/completo/11b_entradas_escenarios.R

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

post <- readRDS(file.path(dir_datos, "post_celdas_casa_media_part.rds"))$post
ds <- readRDS(file.path(dir_datos, "datos_stan.rds"))
post <- post[, c("sexo", "edad", "edu", "rec", "ccaa", "prov", "N")]

f <- file.path(dir_proy, "completo", "datos", "entradas_escenarios.rds")
saveRDS(list(post = post, lv = ds$lv, part_obj = ds$datos$part_obj, mes_post = ds$datos$mes_post), f)
cat("Escrito", f, " celdas:", nrow(post), " tamaño (KB):", round(file.size(f) / 1024), "\n")
