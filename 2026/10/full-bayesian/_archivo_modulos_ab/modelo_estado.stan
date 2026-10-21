// Módulo A: estado latente del voto con paseo aleatorio semanal.
//
// Encuesta i (casa h, semana t), K categorías; la última (BLOQUE) es la
// referencia del log-ratio:
//   y_i ~ multinomial(softmax([eta_t + beta_h + eps_i ; 0]))
//   eta_t = eta_{t-1} + omega_t,   omega_tk ~ N(0, sigma_rw_k)
//   beta_hk ~ N(0, sigma_casa_k),  sum_h beta_hk = 0
//   eps_ik ~ N(0, sigma_enc_k)     (ruido de cada encuesta más allá del
//                                   muestreo: diseño, ponderación, cocina)
// Las semanas sin encuestas, hasta T (día E), las rellena el propio paseo.
// Prior de la semana 1: eta_1 ~ N(log-ratios del 23J, 0,5).
data {
  int<lower=1> N;
  int<lower=2> K;
  int<lower=1> H;
  int<lower=2> T;
  array[N, K] int<lower=0> y;
  array[N] int<lower=1, upper=T> semana;
  array[N] int<lower=1, upper=H> casa;
  vector[K - 1] eta1_mu;      // log-ratios del 23J
  real<lower=0> eta1_sd;
  // escalas de los priors (datos: la sensibilidad no obliga a recompilar)
  real<lower=0> sd_rw;        // sigma_rw ~ N+(0, sd_rw)
  real<lower=0> sd_casa;      // sigma_casa ~ N+(0, sd_casa)
  real<lower=0> sd_enc;       // sigma_enc ~ N+(0, sd_enc)
  int<lower=0, upper=1> prior_only;
}
transformed data {
  int Km1 = K - 1;
}
parameters {
  vector[Km1] eta1;
  matrix[Km1, T - 1] z_rw;
  vector<lower=0>[Km1] sigma_rw;
  array[Km1] sum_to_zero_vector[H] z_casa;
  vector<lower=0>[Km1] sigma_casa;
  matrix[Km1, N] z_enc;
  vector<lower=0>[Km1] sigma_enc;
}
transformed parameters {
  matrix[Km1, T] eta;
  matrix[Km1, H] beta;
  eta[, 1] = eta1;
  for (t in 2:T) {
    eta[, t] = eta[, t - 1] + sigma_rw .* z_rw[, t - 1];
  }
  for (k in 1:Km1) {
    beta[k] = sigma_casa[k] * z_casa[k]';
  }
}
model {
  eta1 ~ normal(eta1_mu, eta1_sd);
  to_vector(z_rw) ~ std_normal();
  sigma_rw ~ normal(0, sd_rw);
  for (k in 1:Km1) {
    z_casa[k] ~ normal(0, 1);
  }
  sigma_casa ~ normal(0, sd_casa);
  to_vector(z_enc) ~ std_normal();
  sigma_enc ~ normal(0, sd_enc);
  if (!prior_only) {
    for (i in 1:N) {
      y[i] ~ multinomial_logit(append_row(
        eta[, semana[i]] + beta[, casa[i]] + sigma_enc .* z_enc[, i], 0));
    }
  }
}
generated quantities {
  // voto de la "media de la industria" por semana (sin casa ni ruido de
  // encuesta), en proporciones; la columna T es el día E
  matrix[K, T] v;
  // réplicas de las encuestas para el posterior predictive
  array[N, K] int y_rep;
  for (t in 1:T) {
    v[, t] = softmax(append_row(eta[, t], 0));
  }
  for (i in 1:N) {
    y_rep[i] = multinomial_logit_rng(
      append_row(eta[, semana[i]] + beta[, casa[i]]
                 + sigma_enc .* to_vector(normal_rng(rep_vector(0, Km1), 1)),
                 0),
      sum(y[i]));
  }
}
