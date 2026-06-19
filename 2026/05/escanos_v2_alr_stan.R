## escanos_v2_alr_stan.R
##
## Error histórico en espacio ILR + D'Hondt (con umbral 3%) como GQ en Stan.
##
## Diferencias clave vs v1:
##   - Espacio de error: ILR (isometric log-ratio) en vez de proporción
##     → sin problema de negativos, correlaciones naturales, invariante a ref
##   - Error simulado desde conocimiento experto (delta method) en vez de hardcoded
##   - Umbral electoral 3% provincial aplicado antes de D'Hondt
##   - P(PP+Vox) condicionada a que PP no llega solo

library(tidyverse)
library(cmdstanr)
library(tidybayes)
library(brms)
library(gtools)

options(brms.backend = "cmdstanr")

# ─── 0. Constantes ────────────────────────────────────────────────────────────

partidos <- c("pp", "psoe", "vox", "por_andalucia", "adelante", "resto")
K  <- length(partidos)   # 6
K1 <- K - 1              # 5

escanos_prov <- c(
  almeria = 12, cadiz = 15, cordoba = 12, granada = 13,
  huelva  = 11, jaen  = 11, malaga  = 17, sevilla = 18
)
P <- length(escanos_prov)

# ─── 1. ILR helpers ───────────────────────────────────────────────────────────
#
# ILR usa una base ortonormal del simplex (partición binaria de Helmert).
# A diferencia de ALR, no hay categoría de referencia: las correlaciones
# reflejan la estructura real del error, no el artefacto del denominador común.
#
# Matriz H (K1 × K):
#   H[k, j≤k]  =  sqrt(1 / (k(k+1)))
#   H[k, k+1]  = -sqrt(k / (k+1))
#   H[k, j>k+1] = 0
#
# ilr(p)     = H %*% log(p)           →  vector en ℝ^(K1)
# ilr_inv(z) = softmax(t(H) %*% z)   →  proporcciones (suman 1)

make_ilr_matrix <- function(K) {
  H <- matrix(0, K - 1, K)
  for (k in seq_len(K - 1)) {
    H[k, seq_len(k)] <- sqrt(1 / (k * (k + 1)))
    H[k, k + 1]      <- -sqrt(k / (k + 1))
  }
  H
}

H_ilr  <- make_ilr_matrix(K)   # (K1 × K), se pasa como dato a Stan
Ht_ilr <- t(H_ilr)             # (K × K1), para la inversa: softmax(Ht %*% z)

norm_p  <- function(v) v / sum(v)
ilr     <- function(p) as.numeric(H_ilr %*% log(norm_p(p)))
ilr_inv <- function(z) { lp <- as.numeric(Ht_ilr %*% z); exp(lp) / sum(exp(lp)) }

# Verificación rápida de ida y vuelta
p_test <- c(0.40, 0.25, 0.15, 0.08, 0.05, 0.07)
stopifnot(max(abs(ilr_inv(ilr(p_test)) - p_test)) < 1e-10)

# ─── 2. Errores simulados en espacio ILR ─────────────────────────────────────
#
# Sigma_ilr = H %*% Sigma_prop %*% H'   (método delta)
# Con esto la covarianza en ILR ya no infla correlaciones por la referencia.

p_expand <- c(pp = 0.43, psoe = 0.24, vox = 0.15,
              por_andalucia = 0.08, adelante = 0.05, resto = 0.07)

sigma_prop_v1 <- c(pp = 0.030, psoe = 0.028, vox = 0.030,
                   por_andalucia = 0.018, adelante = 0.015, resto = 0.020)

R_v1 <- matrix(c(
  1.0, -0.3, -0.5, -0.1, -0.1,  0.0,
 -0.3,  1.0, -0.2,  0.5,  0.4,  0.0,
 -0.5, -0.2,  1.0, -0.1, -0.1,  0.0,
 -0.1,  0.5, -0.1,  1.0,  0.6,  0.0,
 -0.1,  0.4, -0.1,  0.6,  1.0,  0.0,
  0.0,  0.0,  0.0,  0.0,  0.0,  1.0
), nrow = 6, byrow = TRUE)

D_prop     <- diag(sigma_prop_v1)
Sigma_prop <- D_prop %*% R_v1 %*% D_prop

# Delta method: Jacobiano de ILR respecto a proporciones evaluado en p_expand
# d(ILR_k) / d(p_j) = H[k,j] / p_expand[j]
J_ilr      <- H_ilr / rep(p_expand, each = K1)  # (K1 × K) element-wise by column
Sigma_ilr  <- J_ilr %*% Sigma_prop %*% t(J_ilr)

cat("σ en espacio ILR (sqrt diagonal):\n")
print(round(sqrt(diag(Sigma_ilr)), 3))
cat("\nCorrelación en espacio ILR:\n")
print(round(cov2cor(Sigma_ilr), 2))

set.seed(2847)
N_sim           <- 20
hist_errors_ilr <- MASS::mvrnorm(N_sim, mu = rep(0, K1), Sigma = Sigma_ilr)
colnames(hist_errors_ilr) <- partidos[-K]

# ─── 3. Posterior del meta-análisis en espacio ILR ───────────────────────────

model_andalucia <- readRDS(here::here("2026/05/mod_meta_andalucia.rds"))

draws_wide <- tibble(empresa = "votaciones_17mayo", time = 0, n = 1) |>
  add_epred_draws(model_andalucia, allow_new_levels = TRUE) |>
  ungroup() |>
  select(.draw, .category, .epred) |>
  pivot_wider(names_from = .category, values_from = .epred)

meta_ilr_mat <- draws_wide |>
  select(all_of(partidos)) |>
  as.matrix() |>
  apply(1, ilr) |>
  t()

colnames(meta_ilr_mat) <- paste0(partidos[-K], "_ilr")
N_meta <- nrow(meta_ilr_mat)
cat("\nDraws meta-análisis:", N_meta, "\n")

set.seed(2847)
idx_sub  <- sample(N_meta, 2000)
meta_sub <- meta_ilr_mat[idx_sub, ]
N_sub    <- nrow(meta_sub)

# ─── 4. Datos provinciales ────────────────────────────────────────────────────

ref_2022 <- read_csv(here::here("2026/05/ref_provincial_2022.csv"),
                     show_col_types = FALSE) |>
  arrange(match(provincia, names(escanos_prov)))

regional_2022 <- c(
  pp = 43.10, psoe = 24.10, vox = 13.46,
  por_andalucia = 7.68, adelante = 4.58, resto = 7.08
) / 100

votos_prov_2022 <- c(
  almeria = 258942, cadiz = 511390, cordoba = 382994,
  granada = 360000, huelva  = 207000, jaen   = 247000,
  malaga  = 660000, sevilla = 760000
)

ref_2022 <- ref_2022 |>
  mutate(votos_cand = votos_prov_2022[provincia])

ref_prov_mat <- ref_2022 |>
  select(all_of(partidos)) |>
  as.matrix() |>
  apply(1, norm_p) |>
  t()

scale_factor <- 1000L

# ─── 5. Stan model ────────────────────────────────────────────────────────────

stan_code <- "
data {
  int<lower=1> K1;
  int<lower=1> K;
  int<lower=1> N_hist;
  matrix[N_hist, K1] hist_errors;

  int<lower=1> N_sub;
  matrix[N_sub, K1] meta_draws;   // draws del meta-análisis en ILR

  // Matriz ILR transpuesta para la inversa (K × K1)
  matrix[K, K1] Ht_ilr;

  // Provincias
  int<lower=1> P;
  array[P] int seats;
  matrix[P, K] ref_prov;
  vector[K] ref_reg;
  array[P] int votos_prov;
  int<lower=1> scale_factor;

  real<lower=0,upper=1> umbral;   // barrera electoral provincial (0.03)
}

parameters {
  vector[K1] mu_err;
  vector<lower=0>[K1] sigma_err;
  cholesky_factor_corr[K1] L_Omega;
}

model {
  mu_err    ~ normal(0, 0.3);
  sigma_err ~ exponential(2.0);
  L_Omega   ~ lkj_corr_cholesky(2.0);

  for (n in 1:N_hist)
    hist_errors[n] ~ multi_normal_cholesky(
      mu_err, diag_pre_multiply(sigma_err, L_Omega));
}

generated quantities {
  vector[K] escanos;

  {
    // 1. Draw aleatorio del meta-análisis
    int j = categorical_rng(rep_vector(1.0 / N_sub, N_sub));

    // 2. Error en ILR
    vector[K1] err = multi_normal_cholesky_rng(
      mu_err, diag_pre_multiply(sigma_err, L_Omega));

    // 3. ILR electoral = meta + error
    vector[K1] ilr_elec = to_vector(meta_draws[j]) + err;

    // 4. ILR inversa: softmax(Ht_ilr %*% ilr_elec)
    vector[K] pi_reg = softmax(Ht_ilr * ilr_elec);

    // 5. D'Hondt por provincia con umbral 3%
    vector[K] total = rep_vector(0.0, K);

    for (p in 1:P) {
      // Ajuste provincial por ratio 2022
      vector[K] raw;
      for (k in 1:K) raw[k] = pi_reg[k] * (ref_prov[p, k] / ref_reg[k]);
      vector[K] pi_prov = raw / sum(raw);

      // Incertidumbre provincial: Dirichlet
      vector[K] alpha;
      for (k in 1:K) alpha[k] = pi_prov[k] * (votos_prov[p] * 1.0 / scale_factor);
      vector[K] pi_dir = dirichlet_rng(alpha);

      // Umbral electoral: excluir partidos < 3%
      vector[K] pi_valid;
      for (k in 1:K)
        pi_valid[k] = (pi_dir[k] >= umbral) ? pi_dir[k] : 0.0;

      // Si todos quedan excluidos (edge case), usar pi_dir sin umbral
      if (sum(pi_valid) == 0) pi_valid = pi_dir;
      else pi_valid = pi_valid / sum(pi_valid);

      // D'Hondt
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

stan_file <- here::here("2026/05/escanos_v2_ilr_model.stan")
writeLines(stan_code, stan_file)
mod <- cmdstan_model(stan_file)

stan_data <- list(
  K1          = K1,
  K           = K,
  N_hist      = nrow(hist_errors_ilr),
  hist_errors = hist_errors_ilr,
  N_sub       = N_sub,
  meta_draws  = meta_sub,
  Ht_ilr      = Ht_ilr,
  P           = P,
  seats       = as.integer(escanos_prov),
  ref_prov    = ref_prov_mat,
  ref_reg     = regional_2022,
  votos_prov  = as.integer(votos_prov_2022[names(escanos_prov)]),
  scale_factor = scale_factor,
  umbral       = 0.03
)

# ─── 6. Ajuste ────────────────────────────────────────────────────────────────

fit <- mod$sample(
  data            = stan_data,
  seed            = 2847,
  chains          = 4,
  parallel_chains = 4,
  iter_warmup     = 1000,
  iter_sampling   = 2000,
  refresh         = 500,
  adapt_delta     = 0.99
)

# ─── 7. Diagnósticos ──────────────────────────────────────────────────────────

fit$summary(variables = c("mu_err", "sigma_err")) |> print()
fit$cmdstan_diagnose()

# ─── 8. Extraer escaños y validar ────────────────────────────────────────────

escanos_draws <- fit$draws(variables = "escanos", format = "df") |>
  as_tibble() |>
  select(starts_with("escanos")) |>
  setNames(partidos)

totales <- rowSums(round(escanos_draws))
n_mal   <- sum(totales != 109)
if (n_mal > 0) {
  warning(sprintf("%d draws NO suman 109", n_mal))
} else {
  message(sprintf("✓ Los %d draws suman exactamente 109 escaños.", nrow(escanos_draws)))
}

cat("\nMedias de escaños (deben sumar 109):\n")
medias <- round(sapply(escanos_draws, mean), 1)
print(medias)
cat("Suma:", sum(medias), "\n\n")

# Probabilidades de mayoría — correctamente condicionadas
pp  <- round(escanos_draws$pp)
vox <- round(escanos_draws$vox)
pvox_suma <- pp + vox

p_pp_solo     <- mean(pp >= 55)
p_ppvox_sinpp <- mean(pvox_suma >= 55 & pp < 55)   # necesita Vox
p_bloqueo     <- mean(pvox_suma < 55)

cat(sprintf("P(PP mayoría absoluta solo)         : %.1f%%\n", 100 * p_pp_solo))
cat(sprintf("P(PP+Vox mayoría, PP no llega solo) : %.1f%%\n", 100 * p_ppvox_sinpp))
cat(sprintf("P(PP+Vox < 55, sin mayoría derecha) : %.1f%%\n", 100 * p_bloqueo))
cat(sprintf("Suma de probabilidades              : %.1f%%\n",
            100 * (p_pp_solo + p_ppvox_sinpp + p_bloqueo)))

# ─── 9. Ridge plot ────────────────────────────────────────────────────────────

library(ggplot2)
library(ggridges)

nombres_partido <- c(
  pp = "PP", psoe = "PSOE-A", vox = "Vox",
  por_andalucia = "Por Andalucía", adelante = "Adelante A.", resto = "Otros"
)

orden <- rev(c("PP", "PSOE-A", "Vox", "Por Andalucía", "Adelante A.", "Otros"))

p_ridge <- escanos_draws |>
  mutate(sim = row_number()) |>
  pivot_longer(-sim, names_to = "partido", values_to = "escanos") |>
  mutate(partido = factor(nombres_partido[partido], levels = orden)) |>
  ggplot(aes(x = escanos, y = partido, fill = partido)) +
  geom_density_ridges(alpha = 0.8, bandwidth = 0.7, color = NA) +
  scale_x_continuous(breaks = seq(0, 75, 5)) +
  labs(
    title    = "Distribución posterior de escaños (v2: ILR + GQ Stan)",
    subtitle = "Meta-análisis + error histórico (ILR) + Dirichlet provincial + umbral 3%",
    x = "Escaños", y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

print(p_ridge)
