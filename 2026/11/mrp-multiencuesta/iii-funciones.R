# Funciones del post (III): MRP con varias encuestas, lo que no está en el censo.
#
# El post (iii-participacion.qmd) solo llama a estas funciones. Aquí están las
# cuentas: leer las encuestas y los fits, el modelo de participación por celda,
# la celda de ejemplo, los escenarios, el anexo provincial, y el mapa y el voto
# por sexo y edad (estos dos ya no salen en el post, pero se quedan aquí).
#
# Las entradas las generan los scripts de completo/ (ver el apéndice del post):
#   10_datos_voto_seg.R, 11_ajusta_voto_seg.R, 12_escenarios_voto_seg.R,
#   17_votos_grupos.R y 18_seg_dud_provincia.R.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(gt)
  library(ggplot2)
  library(posterior)
  library(sf)
})

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_completo <- file.path(dir_proy, "completo", "datos")
dir_esc <- file.path(dir_completo, "escenarios_seg")
source(file.path(dir_proy, "00_partidos.R")) # territorio, mascara_territorio()

# ---- Formato -------------------------------------------------------------------

num <- function(x, d = 0) {
  format(round(x, d), big.mark = ".", decimal.mark = ",", nsmall = d)
}
pct <- function(x, d = 1) paste(num(x, d), "%")

# % dentro de cada fila de las dos primeras dimensiones: draw x grupo x partido
a_pct <- function(a) 100 * a / array(apply(a, c(1, 2), sum), dim(a))

# ---- Encuestas -----------------------------------------------------------------

leer_encuestas <- function() {
  readRDS(file.path(dir_completo, "encuestas_completo.rds")) |>
    filter(!is.na(sexo), !is.na(edad), !is.na(edu), !is.na(rec), !is.na(ccaa))
}

# entrevistas con voto + simpatía y seguridad de voto; seguro = dice 10
muestra_seg <- function(enc) {
  enc |>
    filter(!is.na(voto_sim), !is.na(prob_voto)) |>
    mutate(
      seg = if_else(prob_voto == 10, "seguro", "no_seguro"),
      voto = as.character(voto_sim)
    )
}

tabla_simpatia <- function(enc) {
  n <- enc |>
    group_by(casa) |>
    summarise(
      entrevistas = n(),
      con_voto = sum(!is.na(voto)),
      con_voto_sim = sum(!is.na(voto_sim))
    ) |>
    arrange(desc(entrevistas))
  bind_rows(n, n |> summarise(casa = "Total", across(where(is.numeric), sum))) |>
    mutate(pct_ganado = 100 * (con_voto_sim - con_voto) / con_voto) |>
    gt() |>
    fmt_number(c(entrevistas, con_voto, con_voto_sim), decimals = 0, sep_mark = ".", dec_mark = ",") |>
    fmt_number(pct_ganado, decimals = 1, sep_mark = ".", dec_mark = ",") |>
    cols_label(
      casa = "Casa",
      entrevistas = "Entrevistas",
      con_voto = "Con voto",
      con_voto_sim = "Con voto + simpatía",
      pct_ganado = "% ganado"
    )
}

# % de voto en bruto de seguros y no seguros
voto_seguros <- function(muestra) {
  principales <- c("PP", "PSOE", "VOX", "SUMAR", "OTROS", "BLANCO")
  muestra |>
    mutate(voto = if_else(voto %in% principales, voto, "resto")) |>
    count(seg, voto) |>
    mutate(pct = 100 * n / sum(n), .by = seg) |>
    select(-n) |>
    pivot_wider(names_from = seg, values_from = pct) |>
    select(voto, seguro, no_seguro) |>
    mutate(voto = factor(voto, c(principales, "resto"))) |>
    arrange(voto)
}

tabla_seguros <- function(t_voto) {
  t_voto |>
    gt() |>
    fmt_number(-voto, decimals = 1, sep_mark = ".", dec_mark = ",") |>
    cols_label(voto = "Voto + simpatía", seguro = "Seguros (%)", no_seguro = "No seguros (%)") |>
    tab_caption("Voto en bruto de seguros y no seguros. 'resto': regionales y Podemos.")
}

# ---- Paso 1: participación por celda --------------------------------------------

# Devuelve la tabla de postestratificación con pi (dice "10"), pi_cal_base y
# pi_cal_80 (participación de la celda en cada escenario) por celda.
participacion_celdas <- function() {
  entradas <- readRDS(file.path(dir_completo, "entradas_escenarios.rds"))
  post <- entradas$post |>
    mutate(across(c(sexo, edad, edu, rec, ccaa, prov), as.character))
  lv <- entradas$lv

  # 200 draws del modelo de participación, como rvar
  fit_part <- readRDS(file.path(dir_datos, "fit_separado_nuts_part.rds"))$fit
  dm <- as_draws_matrix(fit_part$draws(c(
    "alpha_t", "b_mujer_t", "a_mes_t", "a_edad_t",
    "a_edu_t", "a_rec_t", "a_ccaa_t", "a_prov_t"
  )))
  pt <- as_draws_rvars(dm[round(seq(1, nrow(dm), length.out = 200)), ])

  # eta_part de cada celda: la suma de sus efectos
  eta_t <- pt$alpha_t +
    pt$b_mujer_t * (post$sexo == "M") +
    pt$a_mes_t[entradas$mes_post] +
    pt$a_edad_t[match(post$edad, lv$edad)] +
    pt$a_edu_t[match(post$edu, lv$edu)] +
    pt$a_rec_t[match(post$rec, lv$rec)] +
    pt$a_ccaa_t[match(post$ccaa, lv$ccaa)] +
    pt$a_prov_t[match(post$prov, lv$prov)]
  eta_t_draws <- draws_of(eta_t) # draws x celdas

  # delta: lo que se suma a todos los logit para que la participación total sea T
  delta_para <- function(T) {
    delta <- numeric(nrow(eta_t_draws))
    for (d in seq_along(delta)) {
      participacion <- function(z) weighted.mean(plogis(eta_t_draws[d, ] + z), post$N)
      delta[d] <- uniroot(function(z) participacion(z) - T, c(-10, 10), tol = 1e-10)$root
    }
    delta
  }

  # eta_t_draws + delta suma a cada fila (draw) su delta
  delta_base <- delta_para(entradas$part_obj)
  post <- post |>
    mutate(
      pi = colMeans(plogis(eta_t_draws)),
      pi_cal_base = colMeans(plogis(eta_t_draws + delta_base)),
      pi_cal_80 = colMeans(plogis(eta_t_draws + delta_para(0.80)))
    )
  list(post = post, part_base = entradas$part_obj, mes = entradas$mes_post,
       delta_base = delta_base,
       seguros_pob = mean(colSums(t(plogis(eta_t_draws)) * post$N) / sum(post$N)))
}

# ---- Pasos 2 y 3: la celda de ejemplo --------------------------------------------

# P(partido | celda, S = 1) y P(partido | celda, S = 0), como rvar, con la casa media
voto_celda <- function(celda, mes) {
  lv <- readRDS(file.path(dir_completo, "datos_voto_seg.rds"))$lv
  fit_voto <- readRDS(file.path(dir_completo, "fit_voto_seg.rds"))$fit
  v <- as_draws_rvars(fit_voto$draws(c(
    "alpha", "b_mujer", "b_dud", "a_casa", "a_mes", "a_edad", "a_edu", "a_rec",
    "a_ccaa", "a_prov", "a_casa_rec", "a_dud_edad", "a_dud_rec"
  )))
  fila <- function(efecto, i) efecto[i, , drop = TRUE] # un logit por partido

  ie <- match(celda$edad, lv$edad)
  ir <- match(celda$rec, lv$rec)

  # casa media; en casa x recuerdo las filas van por casa (14 recuerdos cada una)
  n_casa <- length(lv$casa)
  n_rec <- length(lv$rec)
  casa_media <- Reduce(`+`, lapply(seq_len(n_casa), function(h) {
    fila(v$a_casa, h) + fila(v$a_casa_rec, (h - 1) * n_rec + ir)
  })) / n_casa

  eta <- v$alpha +
    (celda$sexo == "M") * v$b_mujer +
    fila(v$a_mes, mes) +
    fila(v$a_edad, ie) +
    fila(v$a_edu, match(celda$edu, lv$edu)) +
    fila(v$a_rec, ir) +
    fila(v$a_ccaa, match(celda$ccaa, lv$ccaa)) +
    fila(v$a_prov, match(celda$prov, lv$prov)) +
    casa_media
  d <- v$b_dud + fila(v$a_dud_edad, ie) + fila(v$a_dud_rec, ir)

  # d se suma cuando S = 0 (no seguro)
  # partidos que no se presentan en la CCAA: probabilidad 0
  log_mascara <- log(mascara_territorio(celda$ccaa, lv$voto))
  softmax <- function(e) exp(e) / rvar_sum(exp(e))
  p_seguro <- softmax(eta + log_mascara)
  p_no_seguro <- softmax(eta + d + log_mascara)
  names(p_seguro) <- names(p_no_seguro) <- lv$voto
  list(seguro = p_seguro, no_seguro = p_no_seguro)
}

# la celda partida en dos: personas, voto de cada parte y voto de la celda
ejemplo_celda <- function(celda, mes, partidos = c("PP", "PSOE", "VOX", "SUMAR", "BLANCO")) {
  p <- voto_celda(celda, mes)
  voto_seg <- median(p$seguro)[partidos]
  voto_noseg <- median(p$no_seguro)[partidos]
  personas <- c(
    seguros = celda$N * celda$pi,
    no_seguros = celda$N * (celda$pi_cal_base - celda$pi)
  )
  voto <- (personas["seguros"] * voto_seg + personas["no_seguros"] * voto_noseg) / sum(personas)
  list(personas = personas, voto_seg = voto_seg, voto_noseg = voto_noseg, voto_celda = voto,
       p = p, N = celda$N)
}

# Gráfico de la celda partida en dos: arriba, cómo se reparte la gente de la celda
# (seguros que votan, no seguros que votan, no votan); abajo, el voto de seguros y
# no seguros por partido (mediana e intervalo del 90 % de la posterior).
grafico_celda <- function(ej, part_base) {
  col_grupo <- c("Seguros" = "#1f77b4", "No seguros que votan" = "#ff7f0e",
                 "No votan" = "grey80")
  # 1. la gente de la celda
  gente <- tibble(
    grupo = factor(names(col_grupo), rev(names(col_grupo))),
    personas = c(ej$personas[["seguros"]], ej$personas[["no_seguros"]],
                 ej$N - sum(ej$personas))
  ) |>
    mutate(etiqueta = paste0(num(personas), "\n(", pct(100 * personas / ej$N), ")"))
  g1 <- ggplot(gente, aes(x = personas, y = "", fill = grupo)) +
    geom_col(width = 0.6, colour = "white") +
    geom_text(aes(label = etiqueta), position = position_stack(vjust = 0.5), size = 3.2,
              lineheight = 0.9) +
    scale_fill_manual(values = col_grupo, breaks = names(col_grupo)) +
    scale_x_continuous(labels = \(x) num(x), expand = expansion(mult = c(0, 0.01))) +
    labs(x = paste0("personas de la celda (", num(ej$N), ")"), y = NULL, fill = NULL,
         title = paste0("1. Quién vota (participación del ", pct(100 * part_base),
                        " en España)")) +
    theme_minimal() +
    theme(legend.position = "top", panel.grid.major.y = element_blank())

  # 2. el voto de cada parte: partidos que se presentan en la celda
  resume <- function(p, grupo) {
    d <- 100 * matrix(draws_of(p), ndraws(p)) # draws x partido
    q <- apply(d, 2, quantile, c(0.05, 0.5, 0.95))
    tibble(partido = names(p), grupo = grupo,
           mediana = q[2, ], q05 = q[1, ], q95 = q[3, ])
  }
  voto <- bind_rows(resume(ej$p$seguro, "Seguros"),
                    resume(ej$p$no_seguro, "No seguros que votan")) |>
    filter(mediana > 0)
  # las medianas son las de tabla_celda()
  chk <- voto |> filter(partido %in% names(ej$voto_seg))
  stopifnot(
    isTRUE(all.equal(chk$mediana[chk$grupo == "Seguros"], 100 * unname(ej$voto_seg[chk$partido[chk$grupo == "Seguros"]]))),
    isTRUE(all.equal(chk$mediana[chk$grupo != "Seguros"], 100 * unname(ej$voto_noseg[chk$partido[chk$grupo != "Seguros"]])))
  )
  orden <- voto |> filter(grupo == "Seguros") |> arrange(mediana) |> pull(partido)
  voto <- mutate(voto, partido = factor(partido, orden),
                 grupo = factor(grupo, names(col_grupo)[1:2]))
  g2 <- ggplot(voto, aes(x = mediana, y = partido, colour = grupo)) +
    geom_pointrange(aes(xmin = q05, xmax = q95), position = position_dodge(width = 0.6),
                    size = 0.3) +
    scale_colour_manual(values = col_grupo[1:2]) +
    scale_x_continuous(labels = \(x) paste(num(x), "%")) +
    labs(x = "voto (%)", y = NULL, colour = NULL,
         title = "2. A quién votan",
         caption = paste("Estimaciones del modelo, no votos observados. Punto: mediana de la",
                         "posterior; línea: intervalo del 90 %. Casa media.")) +
    theme_minimal() +
    theme(legend.position = "top")

  patchwork::wrap_plots(g1, g2, ncol = 1, heights = c(1, 3))
}

tabla_celda <- function(ej, part_base) {
  partidos <- names(ej$voto_seg)
  tibble(
    grupo = c("Seguros", "No seguros que votan", "La celda (suma)"),
    votantes = c(ej$personas, sum(ej$personas))
  ) |>
    bind_cols(as_tibble(100 * rbind(ej$voto_seg, ej$voto_noseg, ej$voto_celda))) |>
    gt() |>
    fmt_number(votantes, decimals = 0, sep_mark = ".", dec_mark = ",") |>
    fmt_number(all_of(partidos), decimals = 1, sep_mark = ".", dec_mark = ",") |>
    tab_spanner("Voto (%)", all_of(partidos)) |>
    cols_label(grupo = "", votantes = "Votantes") |>
    tab_style(cell_text(weight = "bold"), cells_body(rows = 3)) |>
    tab_caption(paste0(
      "La celda partida en dos, participación del ", pct(100 * part_base),
      " en España. Voto: mediana de la posterior, casa media."
    ))
}

# ---- Escenarios de participación (de completo/12_escenarios_voto_seg.R) ----------

leer_escenarios <- function(part_base) {
  mayorias <- read.csv(file.path(dir_esc, "escenarios_mayorias.csv")) |>
    filter(metodo == "seg")
  entrantes <- read.csv(file.path(dir_esc, "escenarios_entrantes.csv")) |>
    filter(metodo == "seg") |>
    select(-metodo) |>
    arrange(desc(mediana))
  # voto nacional con la participación base
  voto_base <- read.csv(file.path(dir_esc, "escenarios_voto.csv")) |>
    filter(metodo == "seg") |>
    mutate(part = as.numeric(sub(" %", "", sub(",", ".", participacion))) / 100) |>
    filter(abs(part - part_base) < 0.0005)
  # lo mismo si los que entran votan como la media de su celda (sin seguro / dudoso)
  mayorias_celda <- read.csv(file.path(dir_esc, "escenarios_mayorias.csv")) |>
    filter(metodo == "celda")
  list(mayorias = mayorias, entrantes = entrantes, voto_base = voto_base,
       mayorias_celda = mayorias_celda)
}

grafico_escenarios <- function(mayorias) {
  ggplot(mayorias, aes(100 * part, pp_vox_media)) +
    geom_hline(yintercept = 176, linetype = 2, colour = "grey50") +
    geom_pointrange(aes(ymin = pp_vox_q05, ymax = pp_vox_q95)) +
    geom_line() +
    scale_x_continuous(breaks = 100 * mayorias$part, labels = \(x) pct(x)) +
    labs(
      x = "participación",
      y = "escaños de PP + VOX",
      caption = "Media e intervalo del 90 % (1.000 draws). Línea: mayoría absoluta (176)."
    ) +
    theme_minimal()
}

tabla_entrantes <- function(entrantes) {
  entrantes |>
    gt() |>
    fmt_number(-partido, decimals = 1, sep_mark = ".", dec_mark = ",") |>
    cols_label(partido = "Partido", mediana = "Mediana (%)", q05 = "p5 (%)", q95 = "p95 (%)")
}

# ---- Mapa: seguros menos no seguros por provincia (de completo/18_seg_dud_provincia.R)

# brecha (puntos) = % de voto entre seguros - % entre no seguros que votan, mediana
brechas_provincia <- function(partidos = c("PP", "PSOE", "VOX", "SUMAR", "BLANCO")) {
  sd <- readRDS(file.path(dir_completo, "seg_dud_prov.rds"))
  brecha <- a_pct(sd$seguros) - a_pct(sd$dudosos) # draw x provincia x partido
  nivel <- a_pct(sd$seguros + sd$dudosos)
  tibble(
    codigo_provincia = rep(as.integer(sd$prov_ref$codigo_provincia), times = length(partidos)),
    provincia = rep(sd$prov_ref$provincia, times = length(partidos)),
    partido = rep(partidos, each = nrow(sd$prov_ref)),
    brecha = as.vector(apply(brecha[, , partidos], c(2, 3), median)),
    voto = as.vector(apply(nivel[, , partidos], c(2, 3), median))
  )
}

mapa_brechas <- function(br, partidos = c("PP", "PSOE", "VOX", "SUMAR")) {
  br <- br |>
    filter(partido %in% partidos) |>
    mutate(partido = factor(partido, partidos))
  lim <- max(abs(br$brecha))
  readRDS(file.path(dir_completo, "provincias_sf.rds")) |>
    inner_join(br, by = "codigo_provincia") |>
    ggplot() +
    geom_sf(aes(fill = brecha), colour = "white", linewidth = 0.1) +
    facet_wrap(~partido) +
    scale_fill_gradient2(low = "#2166ac", mid = "grey95", high = "#b2182b", limits = c(-lim, lim)) +
    labs(
      fill = "seguros − no seguros\n(puntos)",
      caption = "Mediana de la posterior (200 draws). Escenario base, casa media."
    ) +
    theme_void()
}

# cambio de voto en cada provincia (puntos) al pasar de la participación base al 80 %
cambio_participacion <- function(part_base, partidos = c("PP", "PSOE", "VOX", "SUMAR")) {
  esc <- readRDS(file.path(dir_esc, "escenarios.rds"))
  v_base <- a_pct(esc$votos["seg", which(esc$escenarios == part_base), , , ])
  v_80 <- a_pct(esc$votos["seg", which(esc$escenarios == 0.80), , , ])
  tibble(
    provincia = rep(esc$prov_ref$provincia, times = length(partidos)),
    partido = rep(partidos, each = nrow(esc$prov_ref)),
    cambio = as.vector(apply(v_80[, , partidos] - v_base[, , partidos], c(2, 3), median))
  )
}

# ---- Voto por sexo y edad (de completo/17_votos_grupos.R) -------------------------

voto_sexo_edad <- function(partidos = c("PP", "PSOE", "VOX", "SUMAR")) {
  vg <- readRDS(file.path(dir_completo, "votos_grupos.rds"))
  grupo <- paste(if_else(vg$grupos_psee$sexo == "M", "Mujeres", "Hombres"), vg$grupos_psee$edad)
  niveles <- sort(unique(grupo))

  # votos de cada grupo de sexo x edad: draw x grupo x partido
  votos <- array(
    0,
    c(dim(vg$votos_psee)[1], length(niveles), dim(vg$votos_psee)[3]),
    list(NULL, niveles, dimnames(vg$votos_psee)[[3]])
  )
  for (l in niveles) {
    votos[, l, ] <- apply(vg$votos_psee[, grupo == l, ], c(1, 3), sum)
  }
  a <- a_pct(votos)[, , partidos]

  tibble(
    grupo = rep(niveles, times = length(partidos)),
    partido = factor(rep(partidos, each = length(niveles)), partidos),
    mediana = as.vector(apply(a, c(2, 3), median)),
    q05 = as.vector(apply(a, c(2, 3), quantile, 0.05)),
    q95 = as.vector(apply(a, c(2, 3), quantile, 0.95))
  ) |>
    separate(grupo, c("sexo", "edad"), sep = " ")
}

grafico_sexo_edad <- function(vse) {
  ggplot(vse, aes(edad, mediana, colour = sexo, group = sexo)) +
    geom_pointrange(aes(ymin = q05, ymax = q95), position = position_dodge(width = 0.4)) +
    geom_line(position = position_dodge(width = 0.4)) +
    facet_wrap(~partido) +
    scale_colour_manual(values = c(Hombres = "#1f77b4", Mujeres = "#d62728")) +
    labs(
      x = "edad",
      y = "% de voto",
      colour = NULL,
      caption = "Mediana e intervalo del 90 % (200 draws). España, escenario base, casa media."
    ) +
    theme_minimal() +
    theme(legend.position = "top")
}

# ---- Anexo: resultados provinciales (de completo/12_escenarios_voto_seg.R) -------
#
# Usa los draws de votos por provincia que guardó 12_escenarios_voto_seg.R
# (escenarios.rds, método "seg"). Los escaños por provincia no se guardaron, así
# que se repite D'Hondt draw a draw con la misma regla (umbral del 3 % del voto
# válido con blancos; OTROS y BLANCO no compiten) y se comprueba que la suma de
# las provincias da exactamente los totales nacionales guardados.

# D'Hondt para muchos draws a la vez. v: draws x partidos (proporciones ya con
# umbral). Devuelve draws x partidos con los escaños.
dhondt_draws <- function(v, escanos) {
  J <- ncol(v)
  q <- v[, rep(seq_len(J), escanos), drop = FALSE] /
    rep(rep(seq_len(escanos), each = J), each = nrow(v)) # draws x (J * escanos)
  n <- ncol(q)
  umbral_q <- apply(q, 1, \(x) sort(x, partial = n - escanos + 1)[n - escanos + 1])
  gana <- q >= umbral_q
  # suma por partido: las columnas van partido 1..J, repetido 'escanos' veces
  matrix(rowsum(t(gana) * 1, rep(seq_len(J), escanos)), nrow = nrow(v), byrow = TRUE,
         dimnames = list(NULL, colnames(v)))
}

resultados_provinciales <- function(part = c(0.704, 0.80)) {
  x <- readRDS(file.path(dir_esc, "escenarios.rds"))
  ie <- match(round(part, 3), round(x$escenarios, 3))
  stopifnot(!anyNA(ie))
  etq <- paste0(num(100 * part, 1), " %")
  v <- x$votos["seg", ie, , , , drop = TRUE] # escenario x draw x provincia x partido
  ne <- dim(v)[1]
  nd <- dim(v)[2]
  partidos <- dimnames(v)$partido
  provs <- x$prov_ref

  # % de voto válido (con blancos), por draw
  pct_v <- 100 * v / c(apply(v, 1:3, sum))

  # escaños por provincia y draw (un bucle por provincia: cada una tiene su número
  # de escaños; los draws y escenarios van vectorizados)
  partidos_dh <- dimnames(x$totales)$partido
  esc <- array(0, c(ne, nd, nrow(provs), length(partidos_dh)),
               list(NULL, NULL, NULL, partidos_dh))
  for (i in seq_len(nrow(provs))) {
    p <- matrix(pct_v[, , i, ] / 100, ne * nd, length(partidos), dimnames = list(NULL, partidos))
    p[p < 0.03] <- 0
    p[, c("OTROS", "BLANCO")] <- 0
    esc[, , i, ] <- dhondt_draws(p[, partidos_dh], provs$diputados[i])
  }
  # comprobación: suma de provincias = totales nacionales guardados, draw a draw
  stopifnot(identical(unname(apply(esc, c(1, 2, 4), sum)),
                      unname(x$totales["seg", ie, , , drop = TRUE] * 1)))

  # resumen por provincia y partido de un array draw x provincia x partido
  resume <- function(a, escenario, media = FALSE) {
    q <- apply(a, c(2, 3), quantile, c(0.05, 0.5, 0.95))
    out <- expand_grid(partido = dimnames(a)[[3]], provincia = provs$provincia) |>
      select(provincia, partido) |>
      mutate(escenario = escenario,
             p5 = c(q[1, , ]), mediana = c(q[2, , ]), p95 = c(q[3, , ]))
    if (media) out$media <- c(apply(a, c(2, 3), mean))
    out
  }
  dimnames(pct_v)[[3]] <- dimnames(esc)[[3]] <- provs$provincia
  etq_dif <- paste0("Diferencia (", etq[2], " − ", etq[1], ")")
  voto <- bind_rows(
    resume(pct_v[1, , , ], etq[1]),
    resume(pct_v[2, , , ], etq[2]),
    resume(pct_v[2, , , ] - pct_v[1, , , ], etq_dif) # diferencia draw a draw
  )
  escanos <- bind_rows(
    resume(esc[1, , , ], etq[1], TRUE),
    resume(esc[2, , , ], etq[2], TRUE),
    resume(esc[2, , , ] - esc[1, , , ], etq_dif, TRUE)
  )
  # fuera las filas de partidos que no se presentan en la provincia (voto 0 en
  # todos los draws de los dos escenarios)
  presente <- apply(v > 0, c(3, 4), any)
  dimnames(presente) <- list(provs$provincia, partidos)
  esta <- \(d) presente[cbind(d$provincia, d$partido)]
  orden <- \(d) d |>
    filter(esta(d)) |>
    mutate(provincia = factor(provincia, provs$provincia),
           partido = factor(partido, partidos),
           escenario = factor(escenario, c(etq, etq_dif))) |>
    arrange(provincia, partido, escenario)
  list(voto = orden(voto) |> select(provincia, partido, escenario, mediana, p5, p95),
       escanos = orden(escanos) |>
         select(provincia, partido, escenario, media, mediana, p5, p95),
       v = v, pct_v = pct_v, esc = esc, etq = etq, prov_ref = provs)
}

# lee las tablas que prepara completo/19_anexo_tablas.R
leer_anexo <- function() readRDS(file.path(dir_completo, "anexo_tablas.rds"))

# DT con filtros por columna, 15 filas por página y botones CSV / Excel. Los
# botones exportan todas las filas que pasan los filtros (no solo la página).
# etiquetas: nombre visible de cada columna; digitos: decimales por columna numérica.
tabla_dt <- function(d, etiquetas, digitos, nombre) {
  base <- c(provincia = "Provincia", partido = "Partido", edad = "Edad",
            escenario = "Escenario")
  etiquetas <- c(base, etiquetas)
  stopifnot(all(names(d) %in% names(etiquetas)), all(names(digitos) %in% names(d)))
  exporta <- list(modifier = list(page = "all", search = "applied"))
  dt <- DT::datatable(
    d,
    rownames = FALSE,
    colnames = unname(etiquetas[names(d)]),
    filter = "top",
    extensions = "Buttons",
    options = list(
      dom = "Bltip",
      pageLength = 15,
      lengthMenu = c(15, 50, 200),
      autoWidth = TRUE,
      scrollX = TRUE,
      buttons = list(
        list(extend = "csv", text = "CSV", filename = nombre, exportOptions = exporta),
        list(extend = "excel", text = "Excel", filename = nombre, exportOptions = exporta)
      ),
      language = list(
        lengthMenu = "Mostrar _MENU_ filas",
        info = "Filas _START_ a _END_ de _TOTAL_",
        infoFiltered = "(filtradas de _MAX_)",
        infoEmpty = "Sin filas",
        paginate = list(previous = "Anterior", `next` = "Siguiente"),
        zeroRecords = "No hay filas con ese filtro",
        thousands = "."
      )
    )
  )
  for (k in unique(digitos)) {
    cols <- match(names(digitos)[digitos == k], names(d))
    dt <- DT::formatRound(dt, cols, digits = k, mark = ".", dec.mark = ",")
  }
  dt
}
