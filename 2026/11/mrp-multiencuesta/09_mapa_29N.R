# Escaños del Congreso con el mapa de coaliciones del 29-N sobre el posterior
# MRP (datos/estim_escanos_mrp.rds, de 08_escanos.R). Los draws van emparejados
# con el baseline (mismo array de votos), así que la diferencia draw a draw es
# el efecto puro del mapa, como escenarios_29N_transicion.csv.
#
# Reglas (coaliciones_izquierda_29N_provincias.csv, con una nueva):
#   - unida (C. Valenciana y Canarias): la lista conjunta absorbe todo el voto
#     de PODEMOS: SUMAR += PODEMOS.
#   - sin_podemos (Cataluña): PODEMOS no se presenta; su voto se reparte draw a
#     draw con Dirichlet(ERC 10, SUMAR 4.5, fuera 0.75), los transfer_cat_alpha
#     del 29-N. Un solo sorteo por draw, compartido por las 4 provincias
#     catalanas. Solo entran ERC y SUMAR; la parte "fuera" sale del voto
#     válido (renormalización, como en el 29-N).
#   - con_aa (Andalucía): entra AA con share provincial = share andaluza
#     logit-normal centrada en el 9,62% de las autonómicas del 17-may-2026
#     (sd 0.2) x patrón provincial de aa_autonomicas2026_provincias.csv. La
#     masa sale primero de OTROS (su casa en el MRP, que no compite) y después
#     de PODEMOS, SUMAR y PSOE con Dirichlet(PODEMOS 4, SUMAR 4, PSOE 1); si
#     el pool no llega, el sobrante cae en PSOE (como aplicar_regla_mapa del
#     29-N, con aa_caniba_alpha).
#   - sin_fa (Teruel, nueva respecto al 29-N, donde era "separadas"): FA no se
#     presenta; su voto (en el MRP, dentro de SUMAR: CHA-FA Aragón) va a PSOE
#     con Dirichlet(PSOE 9, fuera 1), draw a draw. La parte "fuera" sale del
#     voto válido.
#
# El orden es el del 29-N: regla -> umbral del 3% -> OTROS fuera -> D'Hondt.
#
# Comprobaciones:
#   - el pipeline con todas las provincias "separadas" reproduce el baseline
#     de 08_escanos.R exactamente (stopifnot contra el esc guardado)
#   - las shares suman 1 después de cada regla
#
# Salidas (en esta carpeta y en datos/):
#   escanos_mapa_mrp.csv         mediana, media e IC90 por partido y escenario
#   escanos_mapa_transicion.csv  media por escenario y diferencia draw a draw
#   escanos_mapa_mayorias.csv    P(mayorías) por escenario
#   escanos_mapa_provincias.csv  voto y escaños mediana por provincia y partido
#   escanos_mapa.png             distribución posterior de escaños por escenario
#   escanos_mapa_detalle.png     escaños mediana en las provincias con regla
#   escanos_mapa_mayorias.png    probabilidades de mayoría
#
#   Rscript 2026/11/mrp-multiencuesta/09_mapa_29N.R

library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(ggplot2)
library(ggdist)
library(gtools)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

# ---- constantes de las reglas (las del 29-N) -----------------------------------

reglas_mapa <- c("separadas", "unida", "sin_podemos", "con_aa", "sin_fa")
aa_pct_and <- 0.0962
aa_sd_logit <- 0.2
aa_caniba_alpha <- c(PODEMOS = 4, SUMAR = 4, PSOE = 1)
transfer_cat_alpha <- c(ERC = 10, SUMAR = 4.5, fuera = 0.75)
fa_alpha <- c(PSOE = 9, fuera = 1)

# patrón provincial de AA: multiplicador peso_p = pct_AA(p) / pct_AA(Andalucía)
# (con pesos de válidos), de forma que el agregado andaluz vuelve a aa_and
patron_aa <- read_csv(
  here::here("2026/10/aa_autonomicas2026_provincias.csv"),
  show_col_types = FALSE
)
peso_aa <- patron_aa$votos_aa /
  sum(patron_aa$votos_aa) /
  (patron_aa$votos_validos / sum(patron_aa$votos_validos))
names(peso_aa) <- as.character(as.integer(patron_aa$codigo_provincia))

leer_reglas <- function(codigos) {
  reglas <- read_csv(
    here::here("2026/10/coaliciones_izquierda_29N_provincias.csv"),
    col_types = cols(.default = col_character())
  )
  # Teruel: CHA-FA Aragón no concurre (regla nueva; en el 29-N era "separadas")
  reglas$regla[reglas$codigo_provincia == "44"] <- "sin_fa"
  stopifnot(
    all(reglas$regla %in% reglas_mapa),
    !anyDuplicated(reglas$codigo_provincia),
    setequal(as.integer(reglas$codigo_provincia), as.integer(codigos))
  )
  reglas$regla[match(as.integer(codigos), as.integer(reglas$codigo_provincia))]
}

# copiada de 08_escanos.R (y de 2026/10/simular_escanos.R)
dhondt <- function(votos, escanos) {
  result <- setNames(numeric(length(votos)), names(votos))
  for (i in seq_len(escanos)) {
    ganador <- which.max(votos / (result + 1))
    result[ganador] <- result[ganador] + 1
  }
  result
}

# ---- simulación ----------------------------------------------------------------
# estim: la lista que guarda 08_escanos.R en datos/estim_escanos_mrp.rds
# (votos, esc, totales, prov_ref, umbral, ...). Devuelve totales y voto
# provincial de los dos escenarios y escribe los CSV y los gráficos en
# dir_salida.

simular_mapa <- function(
  estim,
  dir_salida,
  seed = 2026,
  cat_mult = 1,
  aa_mult = 1,
  aa_pct = aa_pct_and,
  aa_sd = aa_sd_logit
) {
  votos <- estim$votos # draws x provincia x partido (voto_lv), en votos
  voto_lv <- dimnames(votos)[[3]]
  prov_ref <- estim$prov_ref
  umbral <- estim$umbral
  nd <- dim(votos)[1]
  n_prov <- dim(votos)[2]
  stopifnot(
    "PODEMOS" %in% voto_lv,
    "OTROS" %in% voto_lv,
    !("AA" %in% voto_lv)
  )
  partidos_dh <- setdiff(voto_lv, "OTROS")
  partidos_mapa <- c(partidos_dh, "AA")

  regla_prov <- leer_reglas(prov_ref$codigo_provincia)
  g_aa <- peso_aa[as.character(as.integer(prov_ref$codigo_provincia))]
  g_aa[is.na(g_aa)] <- 0
  stopifnot(all(g_aa[regla_prov == "con_aa"] > 0))

  # sorteos draw a draw (los mismos draws para los dos escenarios)
  set.seed(seed)
  # aa_pct y aa_sd sensibilizan el share andaluz de AA (el 9,62 % de las
  # autonómicas del 17-may-2026 y su incertidumbre en logit)
  aa_and <- plogis(qlogis(aa_pct) + rnorm(nd, 0, aa_sd))
  # rdirichlet no pone nombres a las columnas: hacen falta para indexar por partido
  # cat_mult sensibiliza el Dirichlet de Cataluña (0.5 = más disperso, 2 = más
  # concentrado en ERC/SUMAR; 1 = los transfer_cat_alpha del 29-N)
  pi_cat <- rdirichlet(nd, transfer_cat_alpha * cat_mult)
  colnames(pi_cat) <- names(transfer_cat_alpha)
  # aa_mult sensibiliza el Dirichlet de Adelante Andalucía (a quién quita el
  # voto que AA necesita si OTROS no llega: PODEMOS/SUMAR/PSOE; 1 = el del 29-N)
  pi_caniba <- rdirichlet(nd, aa_caniba_alpha * aa_mult)
  colnames(pi_caniba) <- names(aa_caniba_alpha)
  pi_fa <- rdirichlet(nd, fa_alpha)
  colnames(pi_fa) <- names(fa_alpha)
  aa_prov <- aa_and * matrix(g_aa, nd, length(g_aa), byrow = TRUE)

  aplicar_regla <- function(v, regla, d, i) {
    # v: shares con nombres voto_lv (suman 1); devuelve shares (suman 1)
    if (regla == "unida") {
      v["SUMAR"] <- v["SUMAR"] + v["PODEMOS"]
      v["PODEMOS"] <- 0
    } else if (regla == "sin_podemos") {
      pod <- unname(v["PODEMOS"])
      v[c("ERC", "SUMAR")] <- v[c("ERC", "SUMAR")] +
        pod * pi_cat[d, c("ERC", "SUMAR")]
      v["PODEMOS"] <- 0
      v <- v / sum(v) # la parte "fuera" sale del voto válido
    } else if (regla == "con_aa") {
      v["AA"] <- aa_prov[d, i]
      de_otros <- min(v["OTROS"], aa_prov[d, i])
      v["OTROS"] <- v["OTROS"] - de_otros
      quede <- aa_prov[d, i] - de_otros
      w <- pi_caniba[d, ] / sum(pi_caniba[d, ])
      quita <- pmin(quede * w, v[names(w)])
      v[names(w)] <- v[names(w)] - quita
      sobra <- quede - sum(quita)
      if (sobra > 0) {
        v["PSOE"] <- v["PSOE"] - sobra
      }
      v <- pmax(v, 0)
      v <- v / sum(v)
    } else if (regla == "sin_fa") {
      fa <- unname(v["SUMAR"])
      v["PSOE"] <- v["PSOE"] + fa * pi_fa[d, "PSOE"]
      v["SUMAR"] <- 0
      v <- v / sum(v) # la parte "fuera" sale del voto válido
    }
    v
  }

  escanos_de <- function(reglas_prov, con_aa_col = TRUE, guardar_voto = FALSE) {
    partidos <- if (con_aa_col) partidos_mapa else partidos_dh
    esc <- array(
      0L,
      c(nd, n_prov, length(partidos)),
      list(draw = NULL, provincia = prov_ref$provincia, partido = partidos)
    )
    pv <- if (guardar_voto) {
      array(
        NA_real_,
        c(nd, n_prov, length(partidos)),
        list(draw = NULL, provincia = prov_ref$provincia, partido = partidos)
      )
    } else {
      NULL
    }
    for (d in seq_len(nd)) {
      for (i in seq_len(n_prov)) {
        v <- votos[d, i, ]
        v <- v / sum(v)
        v <- aplicar_regla(v, reglas_prov[i], d, i)
        if (!"AA" %in% names(v)) {
          v <- c(v, setNames(0, "AA"))
        }
        stopifnot(abs(sum(v) - 1) < 1e-8)
        if (guardar_voto) {
          pv[d, i, ] <- v[partidos]
        }
        v[v < umbral] <- 0
        v["OTROS"] <- 0
        esc[d, i, ] <- dhondt(v, prov_ref$diputados[i])[partidos]
      }
    }
    list(esc = esc, voto = pv)
  }

  # comprobación: sin reglas, el pipeline reproduce el baseline de 08
  esc_check <- escanos_de(rep("separadas", n_prov), con_aa_col = FALSE)$esc
  stopifnot(identical(esc_check, estim$esc))

  mapa <- escanos_de(regla_prov, con_aa_col = TRUE, guardar_voto = TRUE)
  tot_sin <- estim$totales # nd x partidos_dh
  tot_mapa <- apply(mapa$esc, c(1, 3), sum)
  stopifnot(all(rowSums(tot_sin) == 350), all(rowSums(tot_mapa) == 350))

  # voto provincial (%), con y sin reglas; el denominador lleva OTROS
  shares_sin <- sweep(
    votos[,, partidos_dh, drop = FALSE],
    1:2,
    rowSums(votos, dims = 2),
    "/"
  ) *
    100
  shares_mapa <- mapa$voto * 100

  # ---- resúmenes -----------------------------------------------------------------

  etq_mapa <- c(
    PP = "PP",
    PSOE = "PSOE",
    VOX = "VOX",
    SUMAR = "SUMAR",
    PODEMOS = "PODEMOS",
    ERC = "ERC",
    JUNTS = "Junts",
    BILDU = "EH Bildu",
    PNV = "PNV",
    BNG = "BNG",
    CCA = "CC",
    AA = "AA"
  )
  colores <- c(
    "PP" = "#005999",
    "PSOE" = "#FF0126",
    "VOX" = "#51962A",
    "SUMAR" = "#A00B85",
    "PODEMOS" = "#6F2C91",
    "AA" = "#8C66F1",
    "ERC" = "#F5A21D",
    "Junts" = "#00E0D1",
    "EH Bildu" = "#00A56D",
    "PNV" = "#008141",
    "BNG" = "#B5C9E4",
    "CC" = "#D44A20"
  )

  # etiqueta de escaños: media redondeada por resto mayor (suma 350); las
  # medianas por partido no suman 350, como en el post del 29-N
  resumen_de <- function(tot, escenario) {
    presentes <- colnames(tot)[apply(tot, 2, max) > 0]
    medias <- colMeans(tot)
    base <- floor(medias)
    extra <- rank(-(medias - base), ties.method = "first") <=
      round(sum(medias)) - sum(base)
    tibble(
      escenario = escenario,
      partido = etq_mapa[presentes],
      mediana = apply(tot[, presentes, drop = FALSE], 2, median),
      media = unname(medias[presentes]),
      q05 = apply(tot[, presentes, drop = FALSE], 2, quantile, 0.05),
      q95 = apply(tot[, presentes, drop = FALSE], 2, quantile, 0.95),
      escanos = unname(base[presentes] + extra[presentes])
    )
  }
  resumen <- bind_rows(
    resumen_de(tot_sin, "sin_mapa"),
    resumen_de(tot_mapa, "mapa")
  )

  # nivel de los partidos: mediana de escaños en el mapa (arriba = grande)
  niveles <- resumen |>
    filter(escenario == "mapa") |>
    group_by(partido) |>
    summarise(m = median(mediana), .groups = "drop") |>
    arrange(desc(m)) |>
    pull(partido)
  niveles <- union(niveles, unique(resumen$partido))
  resumen <- resumen |>
    mutate(
      partido = factor(partido, levels = niveles),
      escenario = factor(
        escenario,
        c("sin_mapa", "mapa"),
        c("sin mapa (baseline)", "mapa 29-N")
      )
    )

  # transición draw a draw: la diferencia de medias es el efecto puro del mapa
  comunes <- intersect(colnames(tot_sin), colnames(tot_mapa))
  transicion <- tibble(
    partido = etq_mapa[c(comunes, "AA")],
    media_sin_mapa = c(colMeans(tot_sin[, comunes]), 0),
    media_mapa = colMeans(tot_mapa)[c(comunes, "AA")],
    dif_mapa = media_mapa - media_sin_mapa
  )
  stopifnot(abs(sum(transicion$dif_mapa)) < 1e-8)

  bloques_de <- function(tot) {
    cbind(
      derecha = tot[, "PP"] + tot[, "VOX"],
      izquierda = tot[, "PSOE"] +
        tot[, "SUMAR"] +
        tot[, "PODEMOS"] +
        tot[, "ERC"] +
        tot[, "JUNTS"] +
        tot[, "BILDU"] +
        tot[, "PNV"] +
        tot[, "BNG"] +
        tot[, "CCA"] +
        (if ("AA" %in% colnames(tot)) tot[, "AA"] else 0)
    )
  }
  bloques_d <- bloques_de(tot_sin)
  bloques_m <- bloques_de(tot_mapa)
  mayorias <- bind_rows(
    tibble(
      escenario = "sin_mapa",
      mayoria = c("PP solo >= 176", "PP + VOX >= 176", "izquierda >= 176"),
      prob = c(
        mean(tot_sin[, "PP"] >= 176),
        mean(bloques_d[, "derecha"] >= 176),
        mean(bloques_d[, "izquierda"] >= 176)
      )
    ),
    tibble(
      escenario = "mapa",
      mayoria = c("PP solo >= 176", "PP + VOX >= 176", "izquierda >= 176"),
      prob = c(
        mean(tot_mapa[, "PP"] >= 176),
        mean(bloques_m[, "derecha"] >= 176),
        mean(bloques_m[, "izquierda"] >= 176)
      )
    )
  )

  provincias <- bind_rows(
    lapply(
      list(
        sin_mapa = list(shares = shares_sin, esc = estim$esc),
        mapa = list(shares = shares_mapa, esc = mapa$esc)
      ),
      function(e) {
        fams <- dimnames(e$shares)[[3]]
        map_dfr(seq_len(n_prov), function(i) {
          tibble(
            codigo_provincia = prov_ref$codigo_provincia[i],
            provincia = prov_ref$provincia[i],
            ccaa = prov_ref$ccaa[i],
            partido = fams,
            voto_mediana = apply(e$shares[, i, , drop = FALSE], 3, median),
            escanos_mediana = apply(e$esc[, i, , drop = FALSE], 3, median)
          )
        })
      }
    ),
    .id = "escenario"
  ) |>
    mutate(
      regla = ifelse(
        escenario == "mapa",
        regla_prov[match(codigo_provincia, prov_ref$codigo_provincia)],
        NA
      ),
      .after = provincia
    )

  cat("\n== escaños Congreso, MRP casa media,", estim$mes, "==\n")
  print(as.data.frame(
    resumen |>
      select(escenario, partido, mediana, media, q05, q95, escanos) |>
      mutate(across(where(is.numeric), ~ round(.x, 1))) |>
      pivot_wider(
        names_from = escenario,
        values_from = c(mediana, media, q05, q95, escanos)
      ),
    row.names = FALSE
  ))
  cat("\n== diferencia draw a draw (efecto del mapa, media de escaños) ==\n")
  print(
    as.data.frame(
      transicion |> mutate(across(where(is.numeric), ~ round(.x, 2)))
    ),
    row.names = FALSE
  )
  cat("\n== probabilidad de cada mayoría ==\n")
  print(as.data.frame(mayorias), row.names = FALSE)

  write_csv(resumen, file.path(dir_salida, "escanos_mapa_mrp.csv"))
  write_csv(transicion, file.path(dir_salida, "escanos_mapa_transicion.csv"))
  write_csv(mayorias, file.path(dir_salida, "escanos_mapa_mayorias.csv"))
  write_csv(provincias, file.path(dir_salida, "escanos_mapa_provincias.csv"))

  # ---- gráficos ------------------------------------------------------------------

  todo <- bind_rows(
    as_tibble(tot_sin) |> mutate(.draw = row_number(), escenario = "sin_mapa"),
    as_tibble(tot_mapa) |> mutate(.draw = row_number(), escenario = "mapa")
  ) |>
    pivot_longer(
      -c(.draw, escenario),
      names_to = "partido",
      values_to = "escanos"
    ) |>
    filter(partido %in% unique(resumen$partido)) |>
    mutate(
      partido = etq_mapa[partido],
      partido = factor(partido, levels = levels(resumen$partido)),
      escenario = factor(
        escenario,
        c("sin_mapa", "mapa"),
        c("sin mapa (baseline)", "mapa 29-N")
      )
    )

  hist_escanos <- list(
    stat_halfeye(
      density = "histogram",
      breaks = breaks_fixed(width = 1),
      align = align_center(at = 0),
      normalize = "groups",
      height = 0.6,
      .width = c(0.5, 0.9),
      alpha = 0.8
    ),
    scale_fill_manual(values = colores, guide = "none")
  )

  p_esc <- todo |>
    ggplot(aes(x = escanos, y = partido, fill = partido)) +
    hist_escanos +
    geom_label(
      data = resumen,
      aes(x = escanos, y = partido, label = escanos),
      inherit.aes = FALSE,
      size = 3,
      fill = "white"
    ) +
    facet_wrap(~escenario, nrow = 1) +
    labs(
      title = "Escaños del Congreso: baseline MRP frente al mapa de coaliciones del 29-N",
      subtitle = "La cifra etiquetada es la media (suma 350 en cada escenario).",
      x = "Escaños",
      y = NULL
    ) +
    theme_minimal(base_size = 12)

  # provincias con regla: mediana de escaños por partido
  con_regla <- provincias |>
    filter(escenario == "mapa", !is.na(regla), regla != "separadas") |>
    group_by(partido) |>
    filter(max(escanos_mediana) > 0) |>
    ungroup() |>
    mutate(
      partido = etq_mapa[partido],
      partido = factor(partido, levels = rev(levels(resumen$partido))),
      provincia = reorder(
        provincia,
        as.integer(factor(regla, levels = reglas_mapa))
      )
    )
  p_det <- con_regla |>
    ggplot(aes(x = partido, y = provincia, fill = escanos_mediana)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = escanos_mediana), size = 3.2, fontface = "bold") +
    facet_wrap(~regla, scales = "free_y", ncol = 1) +
    scale_fill_gradient(low = "grey95", high = "#A00B85", guide = "none") +
    labs(
      title = "Mapa 29-N: escaños mediana en las provincias con regla de coalición",
      subtitle = "unida: lista conjunta; sin_podemos: a ERC; con_aa: entra Adelante Andalucía; sin_fa: Teruel sin FA",
      x = NULL,
      y = NULL
    ) +
    theme_minimal(base_size = 11)

  p_may <- mayorias |>
    mutate(
      mayoria = factor(
        mayoria,
        levels = c("PP solo >= 176", "PP + VOX >= 176", "izquierda >= 176"),
        labels = c("PP solo ≥ 176", "PP + VOX ≥ 176", "Investidura ≥ 176")
      ),
      escenario = factor(
        escenario,
        c("sin_mapa", "mapa"),
        c("sin mapa (baseline)", "mapa 29-N")
      )
    ) |>
    ggplot(aes(x = mayoria, y = prob, fill = escenario)) +
    geom_col(position = "dodge", width = 0.8) +
    geom_text(
      aes(label = paste0(round(100 * prob, 1), "%")),
      position = position_dodge(width = 0.8),
      vjust = -0.3,
      size = 3
    ) +
    scale_y_continuous(limits = c(0, 1.05), labels = \(x) {
      paste0(100 * x, "%")
    }) +
    labs(title = "Probabilidad de cada mayoría", x = NULL, y = NULL) +
    theme_minimal(base_size = 12)

  ggsave(
    file.path(dir_salida, "escanos_mapa.png"),
    p_esc,
    width = 13,
    height = 6,
    dpi = 200
  )
  ggsave(
    file.path(dir_salida, "escanos_mapa_detalle.png"),
    p_det,
    width = 8,
    height = 8,
    dpi = 200
  )
  ggsave(
    file.path(dir_salida, "escanos_mapa_mayorias.png"),
    p_may,
    width = 7,
    height = 4,
    dpi = 200
  )

  cat("\nguardados escanos_mapa_*.csv y escanos_mapa*.png en", dir_salida, "\n")

  list(
    tot_sin = tot_sin,
    tot_mapa = tot_mapa,
    esc_mapa = mapa$esc,
    regla_prov = setNames(regla_prov, prov_ref$provincia),
    resumen = resumen,
    transicion = transicion,
    mayorias = mayorias,
    provincias = provincias
  )
}

# ---- main ----------------------------------------------------------------------

if (!interactive()) {
  # solo con Rscript; en la consola se llama a simular_mapa a mano
  x <- readRDS(file.path(dir_datos, "estim_escanos_mrp.rds"))
  invisible(simular_mapa(x, dir_proy))
}
