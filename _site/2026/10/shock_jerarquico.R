# ============================================================================
# SHOCK JERÁRQUICO DE GEOGRAFÍA, CALIBRADO CON LA HISTORIA
#
# Sustituye (o acompaña, en diagnóstico) al ruido Dirichlet del baseline.
# En cada muestra, el log-ratio de la familia f en la provincia p se desvía
# de su valor de referencia así:
#
#   log r^(s)_fp = log r_fp + eta_{f,c(p)} + eps_{f,p}
#   eta ~ N(0, tau_c,f)   (compartido por las provincias de la comunidad)
#   eps ~ N(0, tau_p,f)   (propio de la provincia)
#
# tau_c y tau_p se estiman con la deriva histórica: Δ log r entre
# generales consecutivas (20D→26J, 26J→28A, 28A→10N, 10N→23J), por familia,
# con un modelo mixto por familia:
#
#   dlr ~ repeticion + (1 | ccaa)
#
# La dummy de repetición separa las dos transiciones que fueron repetición
# (20D→26J y 28A→10N), cuya deriva es atípicamente baja: si se agruparan
# con las frescas, subestimarían tau. Como el shock se usa para predecir
# una elección FRESCA, tau_p es la sd de los residuos de las transiciones
# frescas y tau_c se reescala por el mismo factor (asume la CCAA-ICC
# constante entre tipos de transición).
#
# Ajuste por familia porque la geografía de cada una se mueve distinto
# (VOX 4 veces más que PP en componente de comunidad).
#
# Salida:
#   2026/10/tau_shock.csv          — taus puntuales (lme4), por familia
#   2026/10/mod_tau_shock.rds      — fit brms con la incertidumbre de los tau
#   2026/10/tau_shock_draws.csv    — draws del posterior (interfaz de simular)
#   2026/10/tau_regionales.csv     — taus de los partidos del BLOQUE (dos etapas)
#   extraer_tau_draws() devuelve UNA FILA POR DRAW DE SIMULACIÓN Y FAMILIA
#   (columnas draw, familia, tau_c, tau_p): la interfaz que espera
#   backtest_prov(modo = "shock", ...). tau_p sale de la celda FRESCA
#   (familia×repeticion) del parámetro sigma; tau_c de las sd de comunidad.
# ============================================================================

library(lme4)

construir_dlr <- function(preps, escala = c("ratio", "logit")) {
  # escala = "ratio": dlr = Δ log(p_prov/p_nac)  (el shock multiplicativo)
  # escala = "logit": dlr = Δ [logit(p_prov) - logit(p_nac)]  (el cambio del
  # log-odds ratio provincia/nación, lo que consume simular(swing = "logit"))
  escala <- match.arg(escala)
  fams <- c("PP", "PSOE", "VOX", "SUMAR")
  trans <- tribble(
    ~de   , ~a    , ~etiqueta              ,
    "20d" , "26j" , "20D→26J (repetición)" ,
    "26j" , "28a" , "26J→28A"              ,
    "28a" , "10n" , "28A→10N (repetición)" ,
    "10n" , "23j" , "10N→23J"
  )
  mapa_ccaa <- provincias(
    tipo_eleccion = "congreso",
    anno = 2023,
    mes = "07"
  ) |>
    filter(codigo_provincia != "99") |>
    distinct(codigo_provincia, codigo_ccaa)

  # ojo: preps tiene nombres duplicados (las elecciones que son objetivo y
  # referencia a la vez); [[id]] devuelve la primera, que es la misma
  d_prov <- bind_rows(map(unique(names(preps)), \(id) {
    p <- preps[[id]]$prov |>
      filter(familia %in% fams) |>
      left_join(mapa_ccaa, by = "codigo_provincia")
    nac <- preps[[id]]$real_nac |> select(familia, pct_nac = pct)
    p |>
      left_join(nac, by = "familia") |>
      select(codigo_provincia, codigo_ccaa, familia, pct_prov, pct_nac) |>
      mutate(eleccion = id, ratio = pct_prov / pct_nac)
  }))

  pmap_dfr(trans, \(de, a, etiqueta) {
    map_dfr(fams, \(f) {
      inner_join(
        d_prov |>
          filter(eleccion == de, familia == f) |>
          select(
            codigo_provincia,
            codigo_ccaa,
            r0 = ratio,
            p0 = pct_prov,
            pn0 = pct_nac
          ),
        d_prov |>
          filter(eleccion == a, familia == f) |>
          select(codigo_provincia, r1 = ratio, p1 = pct_prov, pn1 = pct_nac),
        by = "codigo_provincia"
      ) |>
        filter(r0 > 0, r1 > 0) |>
        mutate(
          dlr = if (escala == "ratio") {
            log(r1 / r0)
          } else {
            (qlogis(p1 / 100) - qlogis(pn1 / 100)) -
              (qlogis(p0 / 100) - qlogis(pn0 / 100))
          },
          familia = f,
          etiqueta = etiqueta,
          transition = paste(de, a, sep = "->")
        )
    })
  }) |>
    filter(!(familia == "VOX" & transition %in% c("20d->26j", "26j->28a"))) |>
    mutate(repeticion = str_detect(etiqueta, "repetición"))
}

ajustar_tau <- function(dlr) {
  map_dfr(sort(unique(dlr$familia)), \(f) {
    d_f <- dlr |> filter(familia == f)
    fit <- lmer(dlr ~ repeticion + (1 | codigo_ccaa), data = d_f)
    vc <- as.data.frame(VarCorr(fit))
    tau_c_pool <- vc$sdcor[vc$grp == "codigo_ccaa"]
    tau_p_pool <- sigma(fit)
    res <- residuals(fit)
    # el shock predice una elección fresca: la varianza provincial es la de
    # los residuos de las transiciones frescas; tau_c se reescala igual
    tau_p <- sd(res[!d_f$repeticion])
    tibble(
      familia = f,
      n_fresh = sum(!d_f$repeticion),
      tau_c_pooled = tau_c_pool,
      tau_p_pooled = tau_p_pool,
      tau_p_fresh = tau_p,
      tau_c = tau_c_pool * tau_p / tau_p_pool,
      tau_p = tau_p,
      tau_total = sqrt(tau_c^2 + tau_p^2)
    )
  })
}

# draws del posterior de brms -> interfaz de backtest_prov(modo = "shock"):
# b filas por familia (draw 1..b), muestreadas con reemplazo de los 4.000
# draws del chain. tau_p = celda FRESCA (FALSE) del sigma; tau_c = sd de
# comunidad (estimada agrupando repeticiones y frescas: las sd de comunidad
# por tipo de transición no se pueden separar en brms de forma sencilla)
# ============================================================================
# PARTIDOS REGIONALES (ERC, Junts, PNV, EH Bildu, BNG, CC, UPN)
#
# Solo compiten en su comunidad, así que la descomposición comunidad/
# provincia del modelo de los grandes no es identificable: la estimación
# va por ANOVA de dos etapas sobre las transiciones frescas (26J→28A y
# 10N→23J, las únicas con candidaturas limpias):
#
#   tau_p = sd de dlr DENTRO de cada salto fresco (dispersión provincial
#           relativa, robusta a cambios de composición uniformes, que
#           desplazan la media pero no la sd)
#   tau_c = sd ENTRE medias de salto, corregida por tau_p^2/P
#           (var(media) = tau_c^2 + tau_p^2/P). Con 2 saltos es frágil,
#           pero es la única señal del movimiento en bloque de la comunidad
#           (ERC 0,16 y Junts 0,11: el bloque catalán; el resto ≈ 0)
#
# Mapa de siglas (cambian de nombre entre elecciones, misma familia):
#   Junts = DL (20D) = CDC (26J) = JxCAT-JUNTS (28A, 10N, 23J); el salto
#           26J→28A es la refundación convergente, se acepta
#   ERC   = ERC-RI.cat (11D), ERC-CATSÍ/CATSI (20D, 26J), ERC-SOBIRAN(ISTES)
#           (28A, 10N), ERC (23J)
#   BNG   = NÓS (20D) = BNG-NÓS (26J) = BNG (28A, 10N) = B.N.G. (11D, 23J)
#   CC    = CC-NC-PNC (11D) = CCa-PNC (20D-28A) = CCa-PNC-NC / NC-CCa-PNC
#           (10N) = CCa (23J). OJO: el salto 10N→23J EXCLUYE por cambio de
#           composición (NCa abandona la coalición y se presenta aparte,
#           NC-bc): su sd provincial (0,33 frente a 0,09) mide esa ruptura,
#           no geografía
#   UPN   = UPN-PP (11D-26J) y NA+ (28A, 10N) son coaliciones con el PP y
#           solo compite en 1 provincia: sin estimación limpia. Hereda el
#           RMS del tau_total de los demás, todo en tau_p (con una provincia
#           el efecto comunidad y el provincial se funden)
#
# RESTO y SALF no se estiman: RESTO es residual por construcción (tau = 0)
# y SALF hereda el RMS de los tau de los 4 grandes en simular() (sin
# historia propia; la afinidad con SUMAR es dudosa — SALF se sitúa más cerca
# del bloque de VOX, que es el de mayor tau_c—, así que herencia neutral).
#
# Salida:
#   2026/10/tau_regionales.csv — una fila por familia; tau_c = 0 para las
#   de una sola comunidad sin señal de bloque. tau puntuales (plug-in): con
#   1-2 saltos frescos la incertidumbre de estimación no se propaga, al
#   contrario que los tau de los grandes (brms); es el precio de tener 4
#   provincias por familia.
# ============================================================================

mapa_ccaa_congreso <- function() {
  provincias(tipo_eleccion = "congreso", anno = 2023, mes = "07") |>
    filter(codigo_provincia != "99") |>
    distinct(codigo_provincia, codigo_ccaa)
}

mapa_regionales <- function(siglas) {
  sn <- stringi::stri_trans_general(siglas, "Latin-ASCII")
  case_when(
    str_starts(sn, "ERC") ~ "ERC",
    sn %in% c("DL", "CDC", "JxCAT-JUNTS", "JxCAT - JUNTS") ~ "Junts",
    sn == "EAJ-PNV" ~ "PNV",
    sn == "EH Bildu" ~ "EH Bildu",
    sn %in% c("NOS", "BNG-NOS", "BNG", "B.N.G.") ~ "BNG",
    sn %in% c("CC-NC-PNC", "CCa-PNC", "CCa-PNC-NC", "NC-CCa-PNC", "CCa") ~ "CC",
    sn %in% c("UPN-PP", "NA+", "U.P.N.") ~ "UPN",
    TRUE ~ NA_character_
  )
}

construir_dlr_regionales <- function(preps) {
  fams_reg <- c("ERC", "Junts", "PNV", "EH Bildu", "BNG", "CC", "UPN")
  trans <- tribble(
    ~de   , ~a    , ~etiqueta              ,
    "20d" , "26j" , "20D→26J (repetición)" ,
    "26j" , "28a" , "26J→28A"              ,
    "28a" , "10n" , "28A→10N (repetición)" ,
    "10n" , "23j" , "10N→23J"
  )
  mapa_ccaa <- mapa_ccaa_congreso()
  d_reg <- bind_rows(map(unique(names(preps)), \(id) {
    p0 <- preps[[id]]$prov
    val_prov <- p0 |> distinct(codigo_provincia, validos)
    p <- p0 |>
      mutate(fam_reg = mapa_regionales(str_remove(familia, "^x_"))) |>
      filter(!is.na(fam_reg)) |>
      group_by(codigo_provincia, familia = fam_reg) |>
      summarise(votos = sum(votos), .groups = "drop") |>
      left_join(val_prov, by = "codigo_provincia") |>
      mutate(pct_prov = 100 * votos / validos) |>
      left_join(mapa_ccaa, by = "codigo_provincia")
    validos_tot <- sum(val_prov$validos)
    p |>
      group_by(familia) |>
      mutate(pct_nac = 100 * sum(votos) / validos_tot) |>
      ungroup() |>
      mutate(eleccion = id, ratio = pct_prov / pct_nac) |>
      select(codigo_provincia, codigo_ccaa, familia, ratio, eleccion)
  }))
  pmap_dfr(trans, \(de, a, etiqueta) {
    map_dfr(fams_reg, \(f) {
      inner_join(
        d_reg |>
          filter(eleccion == de, familia == f) |>
          select(codigo_provincia, codigo_ccaa, r0 = ratio),
        d_reg |>
          filter(eleccion == a, familia == f) |>
          select(codigo_provincia, r1 = ratio),
        by = "codigo_provincia"
      ) |>
        filter(r0 > 0, r1 > 0) |>
        mutate(
          dlr = log(r1 / r0),
          familia = f,
          etiqueta = etiqueta,
          transition = paste(de, a, sep = "->"),
          repeticion = str_detect(etiqueta, "repetición")
        )
    })
  })
}

ajustar_tau_regionales <- function(dlr_reg) {
  # salto excluido por cambio de composición de la coalición
  excl <- dlr_reg |> filter(familia == "CC", transition == "10n->23j")
  d <- dlr_reg |> anti_join(excl, by = c("familia", "transition"))
  est <- d |>
    filter(!repeticion) |>
    group_by(familia, transition) |>
    summarise(
      media = mean(dlr),
      sd_w = sd(dlr),
      P = n(),
      .groups = "drop_last"
    ) |>
    summarise(
      n_fresh_trans = n(),
      P = first(P),
      tau_p = sqrt(mean(sd_w^2, na.rm = TRUE)),
      tau_c = if (n() > 1) sqrt(max(sd(media)^2 - tau_p^2 / P, 0)) else 0,
      .groups = "drop"
    )
  # UPN: una provincia y coaliciones con el PP (UPN-PP, NA+): sin estimación
  # limpia. Hereda el RMS del tau_total de los demás, todo en tau_p
  est <- est |>
    mutate(
      tau_total_otras = tau_c^2 + tau_p^2,
      tau_p = ifelse(
        familia == "UPN",
        sqrt(mean(tau_total_otras[familia != "UPN"])),
        tau_p
      ),
      tau_c = ifelse(familia == "UPN", 0, tau_c),
      tau_total = sqrt(tau_c^2 + tau_p^2),
      fuente = ifelse(familia == "UPN", "heredado (RMS del resto)", "estimado")
    ) |>
    select(-tau_total_otras)
  est
}

extraer_tau_draws <- function(fit, b, seed = 9157, familias = NULL) {
  dr <- as_draws_df(fit)
  sd_cols <- grep("^sd_codigo_ccaa__familia", names(dr), value = TRUE)
  fams <- str_remove(sd_cols, "sd_codigo_ccaa__familia")
  # familias = NULL devuelve todas; con un subconjunto (p. ej. en los LOTO
  # donde a VOX no le queda celda sigma fresca) se filtra antes de indexar
  if (!is.null(familias)) {
    fams <- fams[fams %in% familias]
  }
  set.seed(seed)
  idx <- sample.int(nrow(dr), b, replace = TRUE)
  map_dfr(fams, \(f) {
    tibble(
      draw = seq_len(b),
      familia = f,
      tau_c = dr[[str_c("sd_codigo_ccaa__familia", f)]][idx],
      tau_p = exp(dr[[str_c("b_sigma_celda_sigma", f, "_FALSE")]][idx])
    )
  })
}
