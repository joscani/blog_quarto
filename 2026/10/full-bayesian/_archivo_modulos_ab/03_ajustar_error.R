# ============================================================================
# FULL BAYESIAN 29-N · 03: ajuste del módulo B (error histórico)
#
# 1. Ajuste con las 5 generales y las 39 autonómicas (2018-2026), y
#    diagnósticos. Correlación común a los dos tipos de elección; las
#    generales comparten el patrón de las sigmas con nivel propio (gamma).
# 2. Validación dejando fuera una general cada vez (LOEO): se ajusta con 4
#    generales + todas las autonómicas y se predice el error del consenso de
#    la quinta. Error del consenso =
#    media de z en las encuestas de esa elección. Su predictiva es
#    mu + delta_new + ruido medio de m encuestas (sigma_enc / sqrt(m)).
#    Se mide si cae dentro del IC90.
#
# Salidas (en 2026/10/full-bayesian/):
#   fit_error.rds          objeto CmdStanMCMC (no va a git: release)
#   error_parametros.csv   mu, sigmas, gamma y Omega (mediana, IC90)
#   error_loeo.csv         elección x partido: observado, IC90, dentro (con y
#                          sin el sesgo de las casas)
#   error_casas.csv        sesgo histórico b de las casas de 2026
#   error_diagnostico.csv  Rhat, ESS y divergencias
#
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/03_ajustar_error.R
# ============================================================================

source(here::here("2026/10/full-bayesian/01_datos.R"))
suppressMessages({
  library(cmdstanr)
  library(posterior)
  library(stringr)
  library(purrr)
})

seed_error <- 1129
ajustar <- function(d, prior_only = 0L) {
  mod_b$sample(
    data = stan_error(d, prior_only),
    seed = seed_error,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1500,
    adapt_delta = 0.99,
    refresh = 0,
    show_messages = FALSE
  )
}

mod_b <- cmdstan_model(file.path(dir_fb, "modelo_error.stan"))
db <- datos_error()

# ---- prior predictive ----------------------------------------------------------

fit_prior <- ajustar(db, prior_only = 1L)
cat("\nPrior predictive del error de una elección nueva (logit, q05-q50-q95):\n")
print(
  fit_prior$summary("err_new", ~ quantile(.x, c(0.05, 0.5, 0.95))) |>
    mutate(partido = db$cats, across(where(is.numeric), \(x) round(x, 3)))
)

# ---- 1. ajuste con las 5 generales y las autonómicas -----------------------------------------

fit <- ajustar(db)
fit$save_object(file.path(dir_fb, "fit_error.rds"))

diag <- fit$diagnostic_summary(quiet = TRUE)
resumen <- fit$summary(
  c("mu", "log_s0", "tau_s", "gamma", "sigma_gen", "sigma_aut",
    "sigma_enc", "sigma_b", "b", "delta", "Omega"),
  "rhat",
  "ess_bulk",
  "ess_tail"
)
diag_tab <- tibble(
  divergencias = sum(diag$num_divergent),
  max_treedepth = sum(diag$num_max_treedepth),
  rhat_max = max(resumen$rhat, na.rm = TRUE),
  ess_bulk_min = min(resumen$ess_bulk, na.rm = TRUE),
  ess_tail_min = min(resumen$ess_tail, na.rm = TRUE)
)
cat("\nDiagnósticos:\n")
print(diag_tab)
write_csv(diag_tab, file.path(dir_fb, "error_diagnostico.csv"))

# nombres legibles: mu[k,g] -> mu_gen PSOE; Omega[i,j] -> PSOE-PP
tipo_lbl <- c("gen", "aut")
param <- fit$summary(
  c("mu", "gamma", "sigma_gen", "sigma_aut", "sigma_enc", "sigma_b",
    "b26", "err_new", "err_new_casa", "Omega"),
  "median",
  ~ quantile(.x, c(0.05, 0.95))
) |>
  mutate(
    base = sub("\\[.*", "", variable),
    i = suppressWarnings(as.integer(str_match(variable, "\\[(\\d+)")[, 2])),
    j = suppressWarnings(as.integer(str_match(variable, ",(\\d+)\\]")[, 2]))
  ) |>
  # correlaciones: solo el triángulo inferior
  filter(!grepl("^Omega", base) | i > j) |>
  mutate(
    parametro = case_when(
      base %in% c("mu", "sigma_enc") ~ paste0(base, "_", tipo_lbl[j]),
      TRUE ~ base
    ),
    partido = case_when(
      grepl("^Omega", base) ~ paste(db$cats[i], db$cats[j], sep = "-"),
      !is.na(i) ~ db$cats[i],
      TRUE ~ ""
    )
  ) |>
  select(parametro, partido, mediana = median, q05 = `5%`, q95 = `95%`)
cat("\nParámetros (logit):\n")
print(param |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)
write_csv(param, file.path(dir_fb, "error_parametros.csv"))

# sesgo histórico de las casas que publican en 2026 (logit; positivo = la
# casa se queda corta con ese partido)
n_hist <- table(factor(db$casas[db$casa], levels = db$casas))
casas_tab <- map_dfr(seq_len(db$H26), \(h) {
  i <- db$idx26[h]
  map_dfr(seq_len(db$D), \(k) {
    x <- if (i > 0) {
      as.numeric(fit$draws(sprintf("b[%d,%d]", k, i), format = "draws_matrix"))
    } else {
      NA_real_
    }
    tibble(
      casa = db$casas26[h],
      encuestas_hist = if (i > 0) as.integer(n_hist[i]) else 0L,
      partido = db$cats[k],
      mediana = median(x),
      q05 = quantile(x, 0.05, na.rm = TRUE),
      q95 = quantile(x, 0.95, na.rm = TRUE)
    )
  })
})
write_csv(casas_tab, file.path(dir_fb, "error_casas.csv"))
cat("\nSesgo histórico de las casas de 2026 (logit, mediana [IC90]):\n")
casas_tab |>
  mutate(txt = sprintf("%+.3f [%+.2f, %+.2f]", mediana, q05, q95)) |>
  select(casa, encuestas_hist, partido, txt) |>
  pivot_wider(names_from = partido, values_from = txt) |>
  arrange(desc(encuestas_hist)) |>
  print(n = Inf, width = 200)

# ---- 2. validación dejando fuera una elección ----------------------------------

h <- errores_hist()
loeo <- lapply(elecciones_hist, \(e) {
  d_e <- datos_error(fuera = e)
  fit_e <- ajustar(d_e)
  dr <- fit_e$draws(c("err_new", "sigma_enc", "sigma_b", "b"), format = "draws_matrix")
  en_e <- h$encuestas$eleccion == e
  z_e <- h$z[en_e, , drop = FALSE]
  casas_e <- match(h$casa[en_e], d_e$casas)
  m <- colSums(!is.na(z_e))
  obs_e <- colMeans(z_e, na.rm = TRUE)
  lapply(seq_along(db$cats), \(k) {
    if (m[k] == 0) {
      return(NULL)
    }
    # sesgo medio de las casas de esa elección (encuesta a encuesta, como el
    # consenso); casas sin historia fuera de e: sorteo del prior aprendido
    con <- !is.na(z_e[, k])
    b_e <- sapply(casas_e[con], \(i) {
      if (is.na(i)) {
        rnorm(nrow(dr), 0, dr[, sprintf("sigma_b[%d]", k)])
      } else {
        dr[, sprintf("b[%d,%d]", k, i)]
      }
    })
    ruido <- rnorm(nrow(dr), 0, dr[, sprintf("sigma_enc[%d,1]", k)] / sqrt(m[k]))
    pred_sin <- dr[, sprintf("err_new[%d]", k)] + ruido
    pred_con <- pred_sin + rowMeans(matrix(b_e, nrow(dr)))
    q_sin <- quantile(pred_sin, c(0.05, 0.5, 0.95))
    q_con <- quantile(pred_con, c(0.05, 0.5, 0.95))
    tibble(
      eleccion = e,
      partido = db$cats[k],
      n_encuestas = m[k],
      observado = obs_e[k],
      mediana_sin = q_sin[2],
      dentro_sin = obs_e[k] >= q_sin[1] & obs_e[k] <= q_sin[3],
      q05 = q_con[1],
      mediana = q_con[2],
      q95 = q_con[3],
      dentro = obs_e[k] >= q_con[1] & obs_e[k] <= q_con[3]
    )
  }) |>
    bind_rows()
}) |>
  bind_rows()
write_csv(loeo, file.path(dir_fb, "error_loeo.csv"))
cat("\nLOEO (error del consenso en logit; IC90 de la predictiva):\n")
print(loeo |> mutate(across(where(is.numeric), \(x) round(x, 3))), n = Inf)
cat(
  "\nError absoluto medio de la mediana (logit): sin casas",
  round(mean(abs(loeo$observado - loeo$mediana_sin)), 3),
  "| con casas",
  round(mean(abs(loeo$observado - loeo$mediana)), 3),
  "\nCobertura sin casas:",
  sum(loeo$dentro_sin),
  "de",
  nrow(loeo),
  "\n"
)
cat(
  "\nCobertura LOEO del IC90 (con casas):",
  sum(loeo$dentro),
  "de",
  nrow(loeo),
  "celdas (",
  round(100 * mean(loeo$dentro), 1),
  "%)\n"
)
