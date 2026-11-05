# Escenarios de participación con el modelo de voto seguro / dudoso
# (completo/stan/mrp_voto_seg.stan; con FUENTE=numpyro, su port a NumPyro) y la
# participación del (II) (stan/mrp_part.stan). Casa media, mes 2026-10, celdas del (II).
#
# En cada celda q y draw:
#   p_seg = softmax(eta_q),  p_dud = softmax(eta_q + d_q)     (modelo de voto)
#   pi    = P("10" | q), sin calibrar;  pi_cal = plogis(logit(pi) + delta_e)
# con delta_e tal que la participación de la población es la del escenario e.
# Dos métodos:
#   seg:   votan los seguros (N * pi) con p_seg y, si pi_cal > pi, los dudosos
#          que entran (N * (pi_cal - pi)) con p_dud. Si pi_cal < pi, votan
#          N * pi_cal seguros. Los que entran votan como los dudosos de su celda.
#   celda: votan N * pi_cal con el reparto medio de la celda,
#          m * p_seg + (1 - m) * p_dud, m = P(seguro | da partido) =
#          plogis(logit(pi) + c_mezcla). Es lo que daría un modelo sin
#          seguro / dudoso: los que entran votan como la media de su celda.
#
# Escaños: D'Hondt por provincia con el mapa del 29-N, como 08_escanos.R. El
# umbral del 3 % se aplica sobre el voto válido CON blancos (LOREG: el voto en
# blanco es válido); después BLANCO y OTROS no compiten.
#
# Entradas: completo/datos/fit_voto_seg.rds (con FUENTE=numpyro,
# completo/datos/voto_seg_numpyro_draws_nuts.npz), datos/fit_separado_nuts_part.rds,
# completo/datos/entradas_escenarios.rds (celdas y niveles, de 11b_entradas_escenarios.R),
# completo/datos/datos_voto_seg.rds (c_mezcla).
# Salidas en completo/datos/escenarios_seg/ (con FUENTE=numpyro,
# escenarios_seg_numpyro/): escenarios_{voto,escanos,mayorias,entrantes}.csv,
# escenarios_part.png, escenarios.rds.
#
#   Rscript 2026/11/mrp-multiencuesta/completo/12_escenarios_voto_seg.R
#   FUENTE=numpyro Rscript ...

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(posterior)

fuente <- Sys.getenv("FUENTE", "stan")
stopifnot(fuente %in% c("stan", "numpyro"))
dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_completo <- file.path(dir_proy, "completo", "datos")
dir_salida <- file.path(dir_completo, if (fuente == "stan") "escenarios_seg" else "escenarios_seg_numpyro")
dir.create(dir_salida, showWarnings = FALSE)
source(file.path(dir_proy, "00_partidos.R")) # territorio

nd <- 1000L
set.seed(2026)

# ---- 1. Celdas, niveles, escenarios -------------------------------------------

entradas <- readRDS(file.path(dir_completo, "entradas_escenarios.rds"))
post <- entradas$post
lv <- entradas$lv
part_obj <- entradas$part_obj
mes_post <- entradas$mes_post
x_voto <- readRDS(file.path(dir_completo, "datos_voto_seg.rds"))
c_mezcla <- x_voto$c_mezcla
# los índices de las celdas valen para los draws solo si los niveles son los del ajuste
for (k in names(lv)) stopifnot(identical(as.character(lv[[k]]), as.character(x_voto$lv[[k]])))
P <- nrow(post)
N <- post$N
mujer <- as.numeric(post$sexo == "M")
ix <- function(x, niveles) {
  i <- match(as.character(x), niveles)
  stopifnot(!anyNA(i))
  i
}
i_edad <- ix(post$edad, lv$edad)
i_edu <- ix(post$edu, lv$edu)
i_rec <- ix(post$rec, lv$rec)
i_ccaa <- ix(post$ccaa, lv$ccaa)
i_prov <- ix(post$prov, lv$prov)
escenarios <- sort(c(0.65, part_obj, 0.75, 0.80))
metodos <- c("seg", "celda")
ne <- length(escenarios)
i_base <- which(escenarios == part_obj)

# ---- 2. Draws del voto, adelgazados a nd ----------------------------------------

vars_voto <- c("alpha", "b_mujer", "b_dud", "a_casa", "a_mes", "a_edad", "a_edu", "a_rec",
               "a_ccaa", "a_prov", "a_casa_rec", "a_dud_edad", "a_dud_rec")
if (fuente == "stan") {
  dv <- as_draws_rvars(readRDS(file.path(dir_completo, "fit_voto_seg.rds"))$fit$draws(vars_voto))
  g <- function(k) unname(draws_of(dv[[k]])) # draw x fila x partido, cadenas seguidas
  voto_lv <- as.character(x_voto$lv$voto)
} else {
  reticulate::use_virtualenv("~/.virtualenvs/r-mrp-numpyro", required = TRUE)
  np <- reticulate::import("numpy", convert = FALSE)
  z <- np$load(file.path(dir_completo, "voto_seg_numpyro_draws_nuts.npz"))
  g <- function(k) reticulate::py_to_r(z$`__getitem__`(k))
  voto_lv <- as.character(g("voto_lv"))
  stopifnot(identical(voto_lv, as.character(x_voto$lv$voto)))
}
J <- length(voto_lv)
n_tot <- dim(g("alpha"))[1]
sel <- round(seq(1, n_tot, length.out = nd))
V <- lapply(vars_voto, function(k) {
  a <- g(k)
  if (length(dim(a)) == 2) a[sel, , drop = FALSE] else a[sel, , , drop = FALSE]
})
names(V) <- vars_voto
stopifnot(dim(V$alpha)[2] == J)
n_casa <- dim(V$a_casa)[2]
n_rec <- length(lv$rec)
stopifnot(dim(V$a_casa_rec)[2] == n_casa * n_rec)

# máscara de territorio (log: 0 o -Inf), como el offset de 10_datos_voto_seg.R
territorio_mod <- territorio[names(territorio) != "AC"]
log_mask <- sapply(voto_lv, function(p) {
  if (p %in% names(territorio_mod)) {
    ifelse(post$ccaa %in% territorio_mod[[p]], 0, -Inf)
  } else {
    rep(0, P)
  }
})

softmax_filas <- function(e) {
  e <- e - apply(e, 1, max)
  p <- exp(e)
  p / rowSums(p)
}

# ---- 3. Draws de la participación (Stan, (II)), adelgazados a nd ---------------

ap <- readRDS(file.path(dir_datos, "fit_separado_nuts_part.rds"))$fit
dp <- as_draws_matrix(ap$draws(c("alpha_t", "b_mujer_t", "a_mes_t", "a_edad_t",
                                 "a_edu_t", "a_rec_t", "a_ccaa_t", "a_prov_t", "delta")))
dp <- dp[round(seq(1, nrow(dp), length.out = nd)), ]
dp <- matrix(as.numeric(dp), nrow(dp), dimnames = list(NULL, colnames(dp))) # matriz simple
col_t <- function(v, i) unname(dp[, paste0(v, "[", i, "]"), drop = FALSE])

# ---- 4. Mapa de provincias (como 08_escanos.R) ---------------------------------

ref13 <- read_csv(here::here("2026/10", "ref_provincial_23J_13.csv"), show_col_types = FALSE)
prov_ref <- ref13 |>
  distinct(codigo_provincia, provincia, ccaa, diputados) |>
  arrange(codigo_provincia)
escanos_29n <- c("28" = 38, "11" = 8)
prov_ref <- prov_ref |>
  mutate(diputados = coalesce(unname(escanos_29n[as.character(codigo_provincia)]), diputados))
stopifnot(sum(prov_ref$diputados) == 350)
prov_i <- match(as.integer(as.character(post$prov)), as.integer(prov_ref$codigo_provincia))
stopifnot(!anyNA(prov_i))
A <- matrix(0, P, nrow(prov_ref))
A[cbind(seq_len(P), prov_i)] <- 1

# ---- 5. Votos por método, escenario y draw -------------------------------------

votos <- array(0, c(length(metodos), ne, nd, nrow(prov_ref), J),
               list(metodo = metodos, escenario = NULL, draw = NULL, provincia = NULL,
                    partido = voto_lv))
entrantes <- array(0, c(length(metodos), nd, J), list(metodo = metodos, draw = NULL, partido = voto_lv))
seguros_pct <- numeric(nd)
dif_delta <- numeric(nd)
t0 <- Sys.time()
for (d in seq_len(nd)) {
  # voto: casa media = media de (a_casa + a_casa_rec) sobre las casas
  casa_media <- Reduce(`+`, lapply(seq_len(n_casa), function(h) {
    matrix(V$a_casa[d, h, ], n_rec, J, byrow = TRUE) +
      V$a_casa_rec[d, (h - 1) * n_rec + seq_len(n_rec), ]
  })) / n_casa
  fila <- function(a, i) a[d, i, , drop = TRUE]
  eta <- matrix(V$alpha[d, ], P, J, byrow = TRUE) +
    outer(mujer, V$b_mujer[d, ]) +
    matrix(V$a_mes[d, mes_post, ], P, J, byrow = TRUE) +
    fila(V$a_edad, i_edad) + fila(V$a_edu, i_edu) + fila(V$a_rec, i_rec) +
    fila(V$a_ccaa, i_ccaa) + fila(V$a_prov, i_prov) + casa_media[i_rec, ] + log_mask
  dd <- matrix(V$b_dud[d, ], P, J, byrow = TRUE) + fila(V$a_dud_edad, i_edad) +
    fila(V$a_dud_rec, i_rec)
  p_s <- softmax_filas(eta)
  p_d <- softmax_filas(eta + dd)

  # participación: casa media (a_casa_t suma 0), sin calibrar
  eta_t <- dp[d, "alpha_t"] + mujer * dp[d, "b_mujer_t"] +
    dp[d, paste0("a_mes_t[", mes_post, "]")] +
    col_t("a_edad_t", i_edad)[d, ] + col_t("a_edu_t", i_edu)[d, ] +
    col_t("a_rec_t", i_rec)[d, ] + col_t("a_ccaa_t", i_ccaa)[d, ] +
    col_t("a_prov_t", i_prov)[d, ]
  eta_t <- as.numeric(eta_t)
  pi <- plogis(eta_t)
  seguros_pct[d] <- sum(N * pi) / sum(N)
  m <- plogis(eta_t + c_mezcla)
  p_mix <- m * p_s + (1 - m) * p_d

  w_d_e <- matrix(0, P, ne)
  pcal_e <- matrix(0, P, ne)
  for (e in seq_len(ne)) {
    dl <- uniroot(function(x) sum(N * plogis(eta_t + x)) / sum(N) - escenarios[e],
                  c(-10, 10), tol = 1e-10)$root
    if (e == i_base) dif_delta[d] <- dl - dp[d, "delta"]
    pcal <- plogis(eta_t + dl)
    w_s <- N * pmin(pi, pcal)
    w_d <- N * pmax(pcal - pi, 0)
    w_d_e[, e] <- w_d
    pcal_e[, e] <- pcal
    votos["seg", e, d, , ] <- crossprod(A, w_s * p_s + w_d * p_d)
    votos["celda", e, d, , ] <- crossprod(A, N * pcal * p_mix)
  }
  ent_s <- colSums((w_d_e[, ne] - w_d_e[, i_base]) * p_d)
  ent_c <- colSums(N * (pcal_e[, ne] - pcal_e[, i_base]) * p_mix)
  entrantes["seg", d, ] <- ent_s / sum(ent_s)
  entrantes["celda", d, ] <- ent_c / sum(ent_c)
  if (d %% 100 == 0) cat("draw", d, "de", nd, "-", format(Sys.time(), "%H:%M:%S"), "\n")
}
cat("minutos en votos:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "\n")
cat("% seguros en la población (mediana):", round(100 * median(seguros_pct), 1), "\n")
# comprobación: el delta del escenario base es el de mrp_part.stan
cat("|delta base - delta de Stan| máx:", signif(max(abs(dif_delta)), 3), "\n")
stopifnot(max(abs(dif_delta)) < 1e-4)

# ---- 6. D'Hondt -----------------------------------------------------------------

dhondt <- function(v, escanos) {
  r <- setNames(numeric(length(v)), names(v))
  for (i in seq_len(escanos)) {
    k <- which.max(v / (r + 1))
    r[k] <- r[k] + 1
  }
  r
}
umbral <- 0.03
partidos_dh <- setdiff(voto_lv, c("OTROS", "BLANCO"))
totales <- array(0L, c(length(metodos), ne, nd, length(partidos_dh)),
                 list(metodo = metodos, escenario = NULL, draw = NULL, partido = partidos_dh))
for (k in seq_along(metodos)) {
  for (e in seq_len(ne)) {
    for (d in seq_len(nd)) {
      for (i in seq_len(nrow(prov_ref))) {
        v <- votos[k, e, d, i, ]
        v <- v / sum(v) # voto válido con blancos
        v[v < umbral] <- 0
        v[c("OTROS", "BLANCO")] <- 0
        totales[k, e, d, ] <- totales[k, e, d, ] + dhondt(v, prov_ref$diputados[i])[partidos_dh]
      }
    }
  }
}
stopifnot(all(apply(totales, 1:3, sum) == 350))

# ---- 7. Resúmenes ---------------------------------------------------------------

etq_esc <- paste0(format(round(100 * escenarios, 1), decimal.mark = ","), " %")
izq <- intersect(c("PSOE", "SUMAR", "PODEMOS", "ERC", "JUNTS", "BILDU", "PNV", "BNG",
                   "CCA", "AA"), partidos_dh)
res_voto <- res_esc <- res_may <- list()
for (k in seq_along(metodos)) {
  for (e in seq_len(ne)) {
    nac <- apply(votos[k, e, , , ], c(1, 3), sum)
    nac <- 100 * nac / rowSums(nac)
    res_voto[[length(res_voto) + 1]] <- tibble(
      metodo = metodos[k], participacion = etq_esc[e], partido = voto_lv,
      mediana = apply(nac, 2, median), q05 = apply(nac, 2, quantile, 0.05),
      q95 = apply(nac, 2, quantile, 0.95)
    )
    tt <- totales[k, e, , ]
    res_esc[[length(res_esc) + 1]] <- tibble(
      metodo = metodos[k], participacion = etq_esc[e], partido = partidos_dh,
      media = colMeans(tt), mediana = apply(tt, 2, median),
      q05 = apply(tt, 2, quantile, 0.05), q95 = apply(tt, 2, quantile, 0.95)
    )
    der <- tt[, "PP"] + tt[, "VOX"]
    res_may[[length(res_may) + 1]] <- tibble(
      metodo = metodos[k], participacion = etq_esc[e], part = escenarios[e],
      pp_vox_media = mean(der), pp_vox_q05 = quantile(der, 0.05),
      pp_vox_q95 = quantile(der, 0.95), p_pp_vox_176 = mean(der >= 176),
      p_investidura_176 = mean(rowSums(tt[, izq, drop = FALSE]) >= 176)
    )
  }
}
res_voto <- bind_rows(res_voto)
res_esc <- bind_rows(res_esc)
res_may <- bind_rows(res_may)
res_ent <- bind_rows(lapply(metodos, function(k) {
  tibble(metodo = k, partido = voto_lv,
         mediana = 100 * apply(entrantes[k, , ], 2, median),
         q05 = 100 * apply(entrantes[k, , ], 2, quantile, 0.05),
         q95 = 100 * apply(entrantes[k, , ], 2, quantile, 0.95))
}))

cat("\n== % voto válido nacional (con blancos), mediana ==\n")
print(res_voto |>
  filter(partido %in% c("PP", "PSOE", "VOX", "SUMAR", "PODEMOS", "BLANCO")) |>
  select(metodo, participacion, partido, mediana) |>
  mutate(mediana = round(mediana, 1)) |>
  pivot_wider(names_from = partido, values_from = mediana) |>
  as.data.frame(), row.names = FALSE)
cat("\n== escaños de PP + VOX y mayorías ==\n")
print(res_may |> mutate(across(where(is.numeric), \(x) round(x, 3))) |> as.data.frame(),
      row.names = FALSE)
cat("\n== a quién votan los que entran entre", etq_esc[i_base], "y", etq_esc[ne],
    "(%, mediana) ==\n")
print(res_ent |> select(metodo, partido, mediana) |> mutate(mediana = round(mediana, 1)) |>
  pivot_wider(names_from = metodo, values_from = mediana) |> as.data.frame(), row.names = FALSE)

write_csv(res_voto, file.path(dir_salida, "escenarios_voto.csv"))
write_csv(res_esc, file.path(dir_salida, "escenarios_escanos.csv"))
write_csv(res_may, file.path(dir_salida, "escenarios_mayorias.csv"))
write_csv(res_ent, file.path(dir_salida, "escenarios_entrantes.csv"))
saveRDS(list(votos = votos, totales = totales, entrantes = entrantes,
             escenarios = escenarios, prov_ref = prov_ref),
        file.path(dir_salida, "escenarios.rds"))

gg <- ggplot(res_may, aes(100 * part, pp_vox_media, ymin = pp_vox_q05, ymax = pp_vox_q95,
                          colour = metodo)) +
  geom_hline(yintercept = 176, linetype = 2, colour = "grey50") +
  geom_pointrange(position = position_dodge(width = 0.8)) +
  geom_line(position = position_dodge(width = 0.8)) +
  scale_colour_manual(
    values = c(celda = "#7f7f7f", seg = "#1f77b4"),
    labels = c(celda = "los que entran votan como la media de su celda",
               seg = "los que entran votan como los dudosos de su celda")
  ) +
  labs(x = "Participación (% de residentes)", y = "Escaños de PP + VOX (media e IC90)",
       colour = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom", legend.direction = "vertical")
ggsave(file.path(dir_salida, "escenarios_part.png"), gg, width = 7, height = 5, dpi = 150)
cat("\nSalidas en", dir_salida, "\n")
