# Escenarios de participación con dos métodos (PARTICIPACION.md):
#   logit: lo de ahora. Se desplaza el logit de P(seguro) de todas las celdas
#          hasta la participación del escenario; los que entran votan como la
#          media de su celda.
#   alfa:  los seguros votan siempre y de los no seguros vota una fracción α,
#          calibrada a la participación del escenario. Cada grupo vota con su
#          reparto: el de la celda, separado en seguros y no seguros con el Δ
#          de 15_delta_seguro.R.
#
# Entradas: datos/post_celdas_casa_media_part.rds (07_post_celdas.R, stan con
# participación), datos/fit_separado_nuts_part.rds (delta de calibración, con
# el mismo thinning que 07), DIR_SALIDA/delta_seguro.rds (15_delta_seguro.R).
#
# Separar el voto de una celda: con v el reparto del modelo de voto, m la
# proporción de seguros entre los que dan partido y E = exp(Δ),
#   v_n = v_s * E / sum(v_s * E)       (no seguros: los seguros inclinados por Δ)
#   m * v_s + (1 - m) * v_n = v        (la mezcla devuelve el reparto original)
# Se resuelve por punto fijo: v_s = v / (m + (1 - m) * E / Z), Z = sum(v_s * E).
#
# Salidas, en DIR_SALIDA (fuera de datos/ para no tocar el post):
#   escenarios_voto.csv       % voto válido nacional por método, escenario y partido
#   escenarios_escanos.csv    escaños (media, mediana, IC90)
#   escenarios_mayorias.csv   P(PP + VOX >= 176) y P(investidura >= 176)
#   escenarios_entrantes.csv  a quién votan los que entran entre la base y el 80 %
#   escenarios_part.png       escaños de PP + VOX según la participación
#   escenarios_part.rds       arrays de votos y escaños por draw
#
#   DIR_SALIDA=/ruta Rscript 2026/11/mrp-multiencuesta/16_escenarios_part.R
#   DIR_SALIDA=/ruta VAR_VOTO=voto_sim Rscript ...   # no seguros con simpatía (_sim)

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(cmdstanr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_salida <- Sys.getenv("DIR_SALIDA")
stopifnot(nzchar(dir_salida), dir.exists(dir_salida))

x <- readRDS(file.path(dir_datos, "post_celdas_casa_media_part.rds"))
# ds separa el voto de la celda (Δ con voto, la variable del modelo de voto,
# para que la mezcla cuadre); ds_n da el voto de los no seguros. Con
# VAR_VOTO=voto_sim, ds_n es el Δ con voto + simpatía: los no seguros incluyen
# a los indecisos que la simpatía recupera. Salidas con sufijo _sim.
var_voto <- Sys.getenv("VAR_VOTO", "voto")
# VAR_VOTO=imp: además, los indecisos imputados por ideología
# (15b_imputa_ideologia.R). Salidas con sufijo _imp.
stopifnot(var_voto %in% c("voto", "voto_sim", "imp"))
suf <- c(voto = "", voto_sim = "_sim", imp = "_imp")[[var_voto]]
ds <- readRDS(file.path(dir_salida, "delta_seguro.rds"))
ds_n <- readRDS(file.path(dir_salida, paste0("delta_seguro", suf, ".rds")))
stopifnot(
  identical(ds$grupos, ds_n$grupos),
  identical(ds$rec_lv, ds_n$rec_lv),
  identical(ds$edad_lv, ds_n$edad_lv)
)
cat("voto de los no seguros con:", var_voto, "\n")
part_obj <- readRDS(file.path(dir_datos, "datos_stan.rds"))$datos$part_obj

p_celda <- x$p # draw x celda x partido
post <- x$post
voto_lv <- x$voto_lv
nd <- dim(p_celda)[1]
P <- nrow(post)
J <- length(voto_lv)
stopifnot(dim(ds$delta)[1] == nd)

escenarios <- sort(c(0.65, part_obj, 0.75, 0.80))
metodos <- c("logit", "alfa")

# ---- P(seguro) sin calibrar, por celda y draw --------------------------------

# pv = plogis(eta_t + delta): se quita el delta de calibración de cada draw,
# con el mismo thinning que 07_post_celdas.R
ap <- readRDS(file.path(dir_datos, "fit_separado_nuts_part.rds"))
dl <- as.numeric(ap$fit$draws("delta", format = "draws_matrix"))
dl <- dl[round(seq(1, length(dl), length.out = nd))]
p_seg <- plogis(qlogis(x$pv) - dl)
N <- post$N
cat(
  "% seguros, casa media (mediana de los draws):",
  round(100 * median(p_seg %*% N / sum(N)), 1),
  "\n"
)
# comprobación: con el delta se recupera la participación calibrada
stopifnot(abs(median(x$pv %*% N / sum(N)) - part_obj) < 0.005)

# ---- índices de celda para Δ ---------------------------------------------------

i_rec <- match(as.character(post$rec), ds$rec_lv)
i_edad <- match(as.character(post$edad), ds$edad_lv)
stopifnot(!anyNA(i_rec), !anyNA(i_edad))
g_partido <- match(
  if_else(voto_lv %in% ds$grupos, voto_lv, "RESTO"),
  ds$grupos
)
cat("grupo de cada partido:\n")
print(setNames(ds$grupos[g_partido], voto_lv))

separa <- function(v, m, E, tol = 1e-10, max_it = 200) {
  Z <- rep(1, nrow(v))
  for (it in seq_len(max_it)) {
    vs <- v / (m + (1 - m) * E / Z)
    Z_nuevo <- rowSums(vs * E)
    if (max(abs(Z_nuevo - Z)) < tol) break
    Z <- Z_nuevo
  }
  vs <- vs / rowSums(vs)
  vn <- vs * E / rowSums(vs * E)
  list(vs = vs, vn = vn, it = it)
}

# ---- mapa de provincias (el del 29-N, como en 08_escanos.R) --------------------

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
prov_i <- match(
  as.integer(as.character(post$prov)),
  as.integer(prov_ref$codigo_provincia)
)
stopifnot(!anyNA(prov_i))
A <- matrix(0, P, nrow(prov_ref))
A[cbind(seq_len(P), prov_i)] <- 1

# ---- votos por escenario, método y draw ----------------------------------------

ne <- length(escenarios)
votos <- array(
  0,
  c(length(metodos), ne, nd, nrow(prov_ref), J),
  list(metodo = metodos, escenario = NULL, draw = NULL, provincia = NULL, partido = voto_lv)
)
entrantes <- array(0, c(length(metodos), nd, J), list(metodo = metodos, draw = NULL, partido = voto_lv))
i_base <- which(escenarios == part_obj)
i_alto <- ne
alfa <- matrix(NA, nd, ne)
max_it <- 0

for (d in seq_len(nd)) {
  v <- p_celda[d, , ]
  ps <- p_seg[d, ]
  m <- plogis(qlogis(ps) + ds$c_mezcla)
  Dg <- sapply(seq_along(ds$grupos), function(g) {
    ds$delta[d, , , g][cbind(i_rec, i_edad)]
  })
  sp <- separa(v, m, exp(Dg[, g_partido]))
  max_it <- max(max_it, sp$it)
  if (var_voto != "voto") {
    En <- exp(sapply(seq_along(ds_n$grupos), function(g) {
      ds_n$delta[d, , , g][cbind(i_rec, i_edad)]
    })[, g_partido])
    sp$vn <- sp$vs * En / rowSums(sp$vs * En)
  }

  w_logit <- matrix(NA, P, ne)
  w_s <- matrix(NA, P, ne) # peso de los seguros (alfa)
  w_n <- matrix(NA, P, ne) # peso de los no seguros (alfa)
  for (e in seq_len(ne)) {
    obj <- escenarios[e]
    dlt <- uniroot(
      function(z) sum(N * plogis(qlogis(ps) + z)) / sum(N) - obj,
      c(-10, 10),
      tol = 1e-10
    )$root
    w_logit[, e] <- N * plogis(qlogis(ps) + dlt)
    a <- (obj * sum(N) - sum(N * ps)) / sum(N * (1 - ps))
    alfa[d, e] <- a
    w_s[, e] <- N * ps
    w_n[, e] <- N * (1 - ps) * a
    votos["logit", e, d, , ] <- crossprod(A, w_logit[, e] * v)
    votos["alfa", e, d, , ] <- crossprod(A, w_s[, e] * sp$vs + w_n[, e] * sp$vn)
  }
  ent_l <- colSums((w_logit[, i_alto] - w_logit[, i_base]) * v)
  ent_a <- colSums((w_n[, i_alto] - w_n[, i_base]) * sp$vn)
  entrantes["logit", d, ] <- ent_l / sum(ent_l)
  entrantes["alfa", d, ] <- ent_a / sum(ent_a)
}
cat("iteraciones máximas del punto fijo:", max_it, "\n")
cat("α por escenario (mediana):\n")
print(setNames(round(apply(alfa, 2, median), 3), round(100 * escenarios, 1)))
stopifnot(all(alfa > 0))

# comprobación: en el escenario base, el método logit reproduce el post (II)
base_post <- t(apply(p_celda, 1, function(q) colSums(q * colMeans(x$pv) * N)))
stopifnot(
  max(abs(
    colMeans(base_post / rowSums(base_post)) -
      colMeans(apply(votos["logit", i_base, , , ], c(1, 3), sum) /
        rowSums(apply(votos["logit", i_base, , , ], c(1, 3), sum)))
  )) < 0.005
)

# ---- D'Hondt (como 08_escanos.R) -----------------------------------------------

dhondt <- function(votos, escanos) {
  result <- setNames(numeric(length(votos)), names(votos))
  for (i in seq_len(escanos)) {
    ganador <- which.max(votos / (result + 1))
    result[ganador] <- result[ganador] + 1
  }
  result
}
umbral <- 0.03
partidos_dh <- setdiff(voto_lv, "OTROS")
totales <- array(
  0L,
  c(length(metodos), ne, nd, length(partidos_dh)),
  list(metodo = metodos, escenario = NULL, draw = NULL, partido = partidos_dh)
)
for (k in seq_along(metodos)) {
  for (e in seq_len(ne)) {
    for (d in seq_len(nd)) {
      for (i in seq_len(nrow(prov_ref))) {
        vp <- votos[k, e, d, i, ]
        names(vp) <- voto_lv
        vp <- vp / sum(vp)
        vp[vp < umbral] <- 0
        vp["OTROS"] <- 0
        totales[k, e, d, ] <- totales[k, e, d, ] +
          dhondt(vp, prov_ref$diputados[i])[partidos_dh]
      }
    }
  }
}
stopifnot(all(apply(totales, 1:3, sum) == 350))

# ---- resúmenes ------------------------------------------------------------------

etq_esc <- paste0(format(round(100 * escenarios, 1), decimal.mark = ","), " %")
izq <- intersect(
  c("PSOE", "SUMAR", "PODEMOS", "ERC", "JUNTS", "BILDU", "PNV", "BNG", "CCA", "AA"),
  partidos_dh
)

res_voto <- list()
res_esc <- list()
res_may <- list()
for (k in seq_along(metodos)) {
  for (e in seq_len(ne)) {
    nac <- apply(votos[k, e, , , ], c(1, 3), sum)
    nac <- 100 * nac / rowSums(nac)
    res_voto[[length(res_voto) + 1]] <- tibble(
      metodo = metodos[k],
      participacion = etq_esc[e],
      partido = voto_lv,
      mediana = apply(nac, 2, median),
      q05 = apply(nac, 2, quantile, 0.05),
      q95 = apply(nac, 2, quantile, 0.95)
    )
    tt <- totales[k, e, , ]
    res_esc[[length(res_esc) + 1]] <- tibble(
      metodo = metodos[k],
      participacion = etq_esc[e],
      partido = partidos_dh,
      media = colMeans(tt),
      mediana = apply(tt, 2, median),
      q05 = apply(tt, 2, quantile, 0.05),
      q95 = apply(tt, 2, quantile, 0.95)
    )
    der <- tt[, "PP"] + tt[, "VOX"]
    res_may[[length(res_may) + 1]] <- tibble(
      metodo = metodos[k],
      participacion = etq_esc[e],
      part = escenarios[e],
      pp_vox_media = mean(der),
      pp_vox_q05 = quantile(der, 0.05),
      pp_vox_q95 = quantile(der, 0.95),
      p_pp_vox_176 = mean(der >= 176),
      p_investidura_176 = mean(rowSums(tt[, izq, drop = FALSE]) >= 176)
    )
  }
}
res_voto <- bind_rows(res_voto)
res_esc <- bind_rows(res_esc)
res_may <- bind_rows(res_may)
res_ent <- bind_rows(lapply(metodos, function(k) {
  tibble(
    metodo = k,
    partido = voto_lv,
    mediana = 100 * apply(entrantes[k, , ], 2, median),
    q05 = 100 * apply(entrantes[k, , ], 2, quantile, 0.05),
    q95 = 100 * apply(entrantes[k, , ], 2, quantile, 0.95)
  )
}))

cat("\n== % voto válido nacional, PP, PSOE, VOX, SUMAR, PODEMOS (mediana) ==\n")
print(
  res_voto |>
    filter(partido %in% c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS")) |>
    select(metodo, participacion, partido, mediana) |>
    mutate(mediana = round(mediana, 1)) |>
    pivot_wider(names_from = partido, values_from = mediana) |>
    as.data.frame(),
  row.names = FALSE
)
cat("\n== mayorías ==\n")
print(
  res_may |> mutate(across(where(is.numeric), \(z) round(z, 3))) |> as.data.frame(),
  row.names = FALSE
)
cat(
  "\n== a quién votan los que entran entre",
  etq_esc[i_base],
  "y",
  etq_esc[i_alto],
  "(% , mediana) ==\n"
)
print(
  res_ent |>
    select(metodo, partido, mediana) |>
    mutate(mediana = round(mediana, 1)) |>
    pivot_wider(names_from = metodo, values_from = mediana) |>
    as.data.frame(),
  row.names = FALSE
)

write_csv(res_voto, file.path(dir_salida, paste0("escenarios_voto", suf, ".csv")))
write_csv(res_esc, file.path(dir_salida, paste0("escenarios_escanos", suf, ".csv")))
write_csv(res_may, file.path(dir_salida, paste0("escenarios_mayorias", suf, ".csv")))
write_csv(res_ent, file.path(dir_salida, paste0("escenarios_entrantes", suf, ".csv")))
saveRDS(
  list(
    votos = votos,
    totales = totales,
    entrantes = entrantes,
    alfa = alfa,
    escenarios = escenarios
  ),
  file.path(dir_salida, paste0("escenarios_part", suf, ".rds"))
)

g <- ggplot(
  res_may,
  aes(
    100 * part,
    pp_vox_media,
    ymin = pp_vox_q05,
    ymax = pp_vox_q95,
    colour = metodo
  )
) +
  geom_hline(yintercept = 176, linetype = 2, colour = "grey50") +
  geom_pointrange(position = position_dodge(width = 0.8)) +
  geom_line(position = position_dodge(width = 0.8)) +
  scale_colour_manual(
    values = c(logit = "#7f7f7f", alfa = "#1f77b4"),
    labels = c(
      logit = "logit (los que entran votan como su celda)",
      alfa = "α (los que entran votan como los no seguros)"
    )
  ) +
  labs(
    x = "Participación (% de residentes)",
    y = "Escaños de PP + VOX (media e IC90)",
    colour = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom", legend.direction = "vertical")
ggsave(file.path(dir_salida, paste0("escenarios_part", suf, ".png")), g, width = 7, height = 5, dpi = 150)
cat("\nSalidas en", dir_salida, "\n")
