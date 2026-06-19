## escanos_v3_stan_completo.R
##
## Modelo unificado en Stan puro:
##   - Meta-análisis: multinomial logístico con efectos aleatorios por empresa
##   - Error histórico: MVN en ILR (especificado vía simulación + método delta)
##   - GQ: predicción día de elección → ILR + error → ratio provincial → Dirichlet → D'Hondt
##
## No se usa brms en ningún punto. Las encuestas entran directamente como datos.

library(tidyverse)
library(cmdstanr)
library(lubridate)
library(ggridges)

# ─── 0. Constantes ────────────────────────────────────────────────────────────

partidos <- c("pp", "psoe", "vox", "por_andalucia", "adelante", "resto")
K        <- length(partidos)   # 6 (resto = categoría de referencia para softmax)
K1       <- K - 1              # 5

fecha_elecciones <- ymd("2026-05-17")

escanos_prov <- c(
  almeria = 12, cadiz = 15, cordoba = 12, granada = 13,
  huelva  = 11, jaen  = 11, malaga  = 17, sevilla = 18
)
P <- length(escanos_prov)

# ─── 1. ILR helpers ───────────────────────────────────────────────────────────

make_ilr_matrix <- function(K) {
  H <- matrix(0, K - 1, K)
  for (k in seq_len(K - 1)) {
    H[k, seq_len(k)] <- sqrt(1 / (k * (k + 1)))
    H[k, k + 1]      <- -sqrt(k / (k + 1))
  }
  H
}

H_ilr  <- make_ilr_matrix(K)
Ht_ilr <- t(H_ilr)     # K × K1, para ilr_inv en Stan: softmax(Ht %*% z)

norm_p  <- function(v) v / sum(v)
ilr     <- function(p) as.numeric(H_ilr %*% log(norm_p(p)))
ilr_inv <- function(z) { lp <- as.numeric(Ht_ilr %*% z); exp(lp) / sum(exp(lp)) }

# ─── 2. Datos de encuestas ────────────────────────────────────────────────────

df_raw <- read_csv(here::here("2026/05/encuestas_andalucia_2026.csv"),
                   show_col_types = FALSE) |>
  mutate(
    fecha = ymd(fecha),
    time  = as.numeric(fecha - fecha_elecciones)
  )

empresas    <- sort(unique(df_raw$empresa))
J           <- length(empresas)
empresa_idx <- match(df_raw$empresa, empresas)

# Conteos por partido (redondeados de proporción × n)
votos_mat <- df_raw |>
  select(all_of(partidos)) |>
  mutate(across(everything(), \(x) round(x * df_raw$n / 100))) |>
  as.matrix()

# n_total = suma de los conteos imputados (puede diferir ±1-2 del n original)
n_total <- rowSums(votos_mat)
N       <- nrow(df_raw)

cat("Encuestas:", N, "| Empresas:", J, "\n")
cat("Rango de time:", range(df_raw$time), "\n")

# ─── 3. Error histórico simulado en ILR ───────────────────────────────────────

p_expand      <- c(pp=0.43, psoe=0.24, vox=0.15, por_andalucia=0.08, adelante=0.05, resto=0.07)
sigma_prop_v1 <- c(pp=0.030, psoe=0.028, vox=0.030, por_andalucia=0.018, adelante=0.015, resto=0.020)

R_v1 <- matrix(c(
  1.0, -0.3, -0.5, -0.1, -0.1,  0.0,
 -0.3,  1.0, -0.2,  0.5,  0.4,  0.0,
 -0.5, -0.2,  1.0, -0.1, -0.1,  0.0,
 -0.1,  0.5, -0.1,  1.0,  0.6,  0.0,
 -0.1,  0.4, -0.1,  0.6,  1.0,  0.0,
  0.0,  0.0,  0.0,  0.0,  0.0,  1.0
), nrow = 6, byrow = TRUE)

Sigma_prop <- diag(sigma_prop_v1) %*% R_v1 %*% diag(sigma_prop_v1)
J_ilr      <- H_ilr / rep(p_expand, each = K1)
Sigma_ilr  <- J_ilr %*% Sigma_prop %*% t(J_ilr)

set.seed(2847)
N_sim           <- 20
hist_errors_ilr <- MASS::mvrnorm(N_sim, mu = rep(0, K1), Sigma = Sigma_ilr)

# ─── 4. Datos provinciales ────────────────────────────────────────────────────

ref_2022 <- read_csv(here::here("2026/05/ref_provincial_2022.csv"),
                     show_col_types = FALSE) |>
  arrange(match(provincia, names(escanos_prov)))

regional_2022 <- c(pp=43.10, psoe=24.10, vox=13.46,
                   por_andalucia=7.68, adelante=4.58, resto=7.08) / 100

votos_prov_2022 <- c(almeria=258942, cadiz=511390, cordoba=382994,
                     granada=360000, huelva=207000,  jaen=247000,
                     malaga=660000,  sevilla=760000)

ref_prov_mat <- ref_2022 |>
  select(all_of(partidos)) |>
  as.matrix() |>
  apply(1, norm_p) |>
  t()

# ─── 5. Stan model ────────────────────────────────────────────────────────────
#
# Modelo:
#
#   [Meta-análisis]
#   phi[i, k] = alpha[k] + beta[k]*time[i]
#               + u_int[empresa[i], k] * sigma_int[k]
#               + u_slope[empresa[i], k] * sigma_slope[k] * time[i]
#   pi[i] = softmax(cat(phi[i,1:K1], 0))   // "resto" es la referencia (log-odds = 0)
#   votos[i] ~ multinomial(n[i], pi[i])
#
#   [Error histórico en ILR]
#   hist_errors[n] ~ MVN(mu_err, Sigma_err)
#
#   [GQ: día de elecciones]
#   pi_enc = softmax(cat(alpha, 0))        // predicción electoral: time=0, empresa desconocida
#   ilr_enc = H_ilr %*% log(pi_enc)       // pasar a ILR
#   error   ~ MVN(mu_err, Sigma_err)       // muestrear error histórico
#   pi_real = softmax(Ht_ilr %*% (ilr_enc + error))  // volver a proporciones
#   → ratio provincial → Dirichlet → D'Hondt

stan_code <- "
data {
  // Encuestas
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> K1;
  int<lower=1> J;
  array[N] int<lower=1,upper=J> empresa;
  vector[N] time;
  array[N] int n_total;
  array[N, K] int votos;

  // ILR
  matrix[K1, K] H_ilr;    // K1 × K
  matrix[K, K1] Ht_ilr;   // K  × K1

  // Error histórico
  int<lower=1> N_hist;
  matrix[N_hist, K1] hist_errors;

  // Provincias
  int<lower=1> P;
  array[P] int seats;
  matrix[P, K] ref_prov;
  vector[K] ref_reg;
  array[P] int votos_prov;
  int<lower=1> scale_factor;
  real<lower=0,upper=1> umbral;
}

parameters {
  // Meta-análisis
  vector[K1] alpha;           // interceptos fijos (escala log-ratio vs. 'resto')
  vector[K1] beta_time;       // tendencias temporales fijas

  matrix[J, K1] z_int;        // efectos aleatorios no centrados (intercepto)
  matrix[J, K1] z_slope;      // efectos aleatorios no centrados (pendiente)
  vector<lower=0>[K1] sigma_int;
  vector<lower=0>[K1] sigma_slope;

  // Error histórico (ILR)
  vector[K1] mu_err;
  vector<lower=0>[K1] sigma_err;
  cholesky_factor_corr[K1] L_Omega_err;
}

transformed parameters {
  // Escalar los efectos aleatorios
  matrix[J, K1] u_int   = z_int   .* rep_matrix(sigma_int',   J);
  matrix[J, K1] u_slope = z_slope .* rep_matrix(sigma_slope', J);
}

model {
  // Priors meta-análisis
  alpha       ~ normal(0, 2);
  beta_time   ~ normal(0, 1);
  sigma_int   ~ exponential(1);
  sigma_slope ~ exponential(1);
  to_vector(z_int)   ~ std_normal();
  to_vector(z_slope) ~ std_normal();

  // Priors error histórico
  mu_err    ~ normal(0, 0.3);
  sigma_err ~ exponential(2);
  L_Omega_err ~ lkj_corr_cholesky(2);

  // Verosimilitud encuestas
  for (i in 1:N) {
    vector[K] log_odds;
    for (k in 1:K1) {
      log_odds[k] = alpha[k] + beta_time[k] * time[i]
                  + u_int[empresa[i], k]
                  + u_slope[empresa[i], k] * time[i];
    }
    log_odds[K] = 0;
    votos[i] ~ multinomial(softmax(log_odds));
  }

  // Verosimilitud errores históricos
  for (n in 1:N_hist)
    hist_errors[n] ~ multi_normal_cholesky(
      mu_err, diag_pre_multiply(sigma_err, L_Omega_err));
}

generated quantities {
  vector[K] escanos;
  vector[K] pi_enc;    // predicción encuestas en día de elección
  vector[K] pi_real;   // predicción real tras corregir por error histórico

  {
    // 1. Predicción encuestas: time=0, empresa desconocida (solo efectos fijos)
    vector[K] log_odds_elec;
    for (k in 1:K1) log_odds_elec[k] = alpha[k];
    log_odds_elec[K] = 0;
    pi_enc = softmax(log_odds_elec);

    // 2. Transformar a ILR: H_ilr %*% log(pi_enc)
    vector[K1] ilr_enc = H_ilr * log(pi_enc);

    // 3. Muestrear error histórico
    vector[K1] err = multi_normal_cholesky_rng(
      mu_err, diag_pre_multiply(sigma_err, L_Omega_err));

    // 4. Predicción real: softmax(Ht_ilr %*% (ilr_enc + error))
    pi_real = softmax(Ht_ilr * (ilr_enc + err));

    // 5. D'Hondt por provincia
    vector[K] total = rep_vector(0.0, K);
    for (p in 1:P) {
      vector[K] raw;
      for (k in 1:K) raw[k] = pi_real[k] * (ref_prov[p, k] / ref_reg[k]);
      vector[K] pi_prov = raw / sum(raw);

      vector[K] alpha_dir;
      for (k in 1:K) alpha_dir[k] = pi_prov[k] * (votos_prov[p] * 1.0 / scale_factor);
      vector[K] pi_dir = dirichlet_rng(alpha_dir);

      vector[K] pi_valid;
      for (k in 1:K)
        pi_valid[k] = (pi_dir[k] >= umbral) ? pi_dir[k] : 0.0;
      if (sum(pi_valid) == 0) pi_valid = pi_dir;
      else pi_valid = pi_valid / sum(pi_valid);

      vector[K] result = rep_vector(0.0, K);
      for (s in 1:seats[p]) {
        int winner = 1;
        real best  = pi_valid[1] / (result[1] + 1);
        for (k in 2:K) {
          real val = pi_valid[k] / (result[k] + 1);
          if (val > best) { best = val; winner = k; }
        }
        result[winner] += 1;
      }
      total += result;
    }
    escanos = total;
  }
}
"

stan_file <- here::here("2026/05/escanos_v3_model.stan")
writeLines(stan_code, stan_file)
mod <- cmdstan_model(stan_file)

# ─── 6. Datos para Stan ───────────────────────────────────────────────────────

stan_data <- list(
  N           = N,
  K           = K,
  K1          = K1,
  J           = J,
  empresa     = empresa_idx,
  time        = df_raw$time,
  n_total     = as.integer(n_total),
  votos       = votos_mat,
  H_ilr       = H_ilr,
  Ht_ilr      = Ht_ilr,
  N_hist      = nrow(hist_errors_ilr),
  hist_errors = hist_errors_ilr,
  P           = P,
  seats       = as.integer(escanos_prov),
  ref_prov    = ref_prov_mat,
  ref_reg     = regional_2022,
  votos_prov  = as.integer(votos_prov_2022[names(escanos_prov)]),
  scale_factor = 1000L,
  umbral       = 0.03
)

# ─── 7. Ajuste ────────────────────────────────────────────────────────────────

fit <- mod$sample(
  data            = stan_data,
  seed            = 2847,
  chains          = 4,
  parallel_chains = 4,
  iter_warmup     = 1000,
  iter_sampling   = 2000,
  refresh         = 200,
  adapt_delta     = 0.95
)

# ─── 8. Diagnósticos ──────────────────────────────────────────────────────────

fit$summary(variables = c("alpha", "beta_time", "sigma_int", "sigma_slope",
                          "mu_err", "sigma_err")) |>
  print(n = 30)

fit$cmdstan_diagnose()

# ─── 9. Resultados ────────────────────────────────────────────────────────────

escanos_draws <- fit$draws(variables = "escanos", format = "df") |>
  as_tibble() |>
  select(starts_with("escanos")) |>
  setNames(partidos)

totales <- rowSums(round(escanos_draws))
if (any(totales != 109)) {
  warning(sprintf("%d draws no suman 109", sum(totales != 109)))
} else {
  message(sprintf("✓ Los %d draws suman exactamente 109 escaños.", nrow(escanos_draws)))
}

cat("\nMedias de escaños:\n")
print(round(sapply(escanos_draws, mean), 1))

pp  <- round(escanos_draws$pp)
vox <- round(escanos_draws$vox)
pvox <- pp + vox

cat(sprintf("\nP(PP mayoría absoluta solo)         : %.1f%%\n", 100 * mean(pp >= 55)))
cat(sprintf("P(PP+Vox mayoría, PP no llega solo) : %.1f%%\n", 100 * mean(pvox >= 55 & pp < 55)))
cat(sprintf("P(PP+Vox < 55, sin mayoría derecha) : %.1f%%\n", 100 * mean(pvox < 55)))

# ─── 10. Ridge plot ───────────────────────────────────────────────────────────

nombres_partido <- c(pp="PP", psoe="PSOE-A", vox="Vox",
                     por_andalucia="Por Andalucía", adelante="Adelante A.", resto="Otros")
orden <- rev(c("PP", "PSOE-A", "Vox", "Por Andalucía", "Adelante A.", "Otros"))

escanos_draws |>
  mutate(sim = row_number()) |>
  pivot_longer(-sim, names_to = "partido", values_to = "escanos") |>
  mutate(partido = factor(nombres_partido[partido], levels = orden)) |>
  ggplot(aes(x = escanos, y = partido, fill = partido)) +
  geom_density_ridges(alpha = 0.8, bandwidth = 0.7, color = NA) +
  scale_x_continuous(breaks = seq(0, 75, 5)) +
  labs(
    title    = "Distribución posterior de escaños (v3: Stan puro)",
    subtitle = "Meta-análisis multinomial + error ILR + Dirichlet provincial + umbral 3%",
    x = "Escaños", y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
