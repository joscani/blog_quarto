# Datos por celda para la figura de pooling del post (II):
#   - celdas del CIS agregadas a celda de población (prov x sexo x edad x
#     edu x rec): n, cruda del partido;
#   - media e IC90 del posterior de Stan (post_celdas_casa_media_part.rds,
#     casa media, era AA);
#   - celdas de población sin datos del CIS (la cruda no existe).
# Salida: datos/datos_pooling.rds. Necesita el post_celdas, que se regenera
# con 07_post_celdas.R en minutos (no hace falta reajustar).
#
#   Rscript 2026/11/mrp-multiencuesta/07b_pooling_datos.R

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

celdas <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
partidos <- colnames(celdas$y)
elegidos <- c("PP", "PSOE", "VOX", "SUMAR", "BILDU")

# CIS agregado a celda de población: las celdas del modelo van a nivel de
# estudio (máscara de AA), la de la población no distingue estudio
cis <- celdas[celdas$casa == "CIS", ]
Y <- cis$y
key <- cis[, c("prov", "sexo", "edad", "edu", "rec")]
agg <- aggregate(Y, by = key, FUN = sum)
n_voto <- rowSums(agg[, partidos])
colnames(agg)[match(partidos, colnames(agg))] <- paste0("y_", partidos)

post <- readRDS(file.path(dir_datos, "post_celdas_casa_media_part.rds"))
p <- post$p
meta <- post$post
idx_pob <- paste(meta$prov, meta$sexo, meta$edad, meta$edu, meta$rec, sep = "|")
idx_cis <- paste(agg$prov, agg$sexo, agg$edad, agg$edu, agg$rec, sep = "|")
stopifnot(!any(duplicated(idx_pob)))

i <- match(idx_cis, idx_pob)
if (anyNA(i)) {
  cat(
    "celdas CIS sin celda de población (se ignoran):",
    sum(is.na(i)),
    "\n"
  )
  keep <- !is.na(i)
  agg <- agg[keep, ]
  n_voto <- n_voto[keep]
  i <- i[keep]
  idx_cis <- idx_cis[keep]
}

res <- lapply(elegidos, function(pp) {
  k <- match(pp, partidos)
  data.frame(
    partido = pp,
    agg[, c("prov", "sexo", "edad", "edu", "rec")],
    n = n_voto,
    y = agg[[paste0("y_", pp)]],
    cruda = agg[[paste0("y_", pp)]] / n_voto,
    media = colMeans(p[, i, k]),
    q05 = apply(p[, i, k], 2, quantile, 0.05),
    q95 = apply(p[, i, k], 2, quantile, 0.95)
  )
})
res <- do.call(rbind, res)

# celdas de población sin datos del CIS: el multinivel predice, la cruda no existe
sin_datos <- meta[!(idx_pob %in% idx_cis), c("prov", "sexo", "edad", "edu", "rec", "N")]
vacias <- lapply(elegidos, function(pp) {
  k <- match(pp, partidos)
  data.frame(
    partido = pp,
    sin_datos[, c("prov", "sexo", "edad", "edu", "rec")],
    n = 0L, y = NA, cruda = NA,
    media = colMeans(p[, which(!(idx_pob %in% idx_cis)), k]),
    q05 = NA_real_, q95 = NA_real_
  )
})
res <- rbind(res, do.call(rbind, vacias))
saveRDS(res, file.path(dir_datos, "datos_pooling.rds"))
cat("guardado datos/datos_pooling.rds:", nrow(res), "filas\n")
cat(
  "celdas CIS con datos:", sum(res$n > 0) / length(elegidos),
  " vacías:", sum(res$n == 0) / length(elegidos),
  "\n"
)
