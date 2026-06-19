
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> K1;
  int<lower=1> J;
  array[N] int<lower=1,upper=J> empresa;
  vector[N] time;
  array[N] int n_total;
  array[N, K] int votos;

  matrix[K1, K] H_ilr;
  matrix[K, K1] Ht_ilr;

  int<lower=1> N_hist;
  matrix[N_hist, K1] hist_errors;

  int<lower=1> P;
  array[P] int seats;
  matrix[P, K] ref_prov;
  vector[K] ref_reg;
  array[P] int votos_prov;
  int<lower=1> scale_factor;
  real<lower=0,upper=1> umbral;
}

parameters {
  vector[K1] alpha;
  vector[K1] beta_time;
  matrix[J, K1] z_int;
  matrix[J, K1] z_slope;
  vector<lower=0>[K1] sigma_int;
  vector<lower=0>[K1] sigma_slope;

  vector[K1] mu_err;
  vector<lower=0>[K1] sigma_err;
  cholesky_factor_corr[K1] L_Omega_err;
}

transformed parameters {
  matrix[J, K1] u_int   = z_int   .* rep_matrix(sigma_int',   J);
  matrix[J, K1] u_slope = z_slope .* rep_matrix(sigma_slope', J);
}

model {
  alpha       ~ normal(0, 2);
  beta_time   ~ normal(0, 1);
  sigma_int   ~ exponential(1);
  sigma_slope ~ exponential(1);
  to_vector(z_int)   ~ std_normal();
  to_vector(z_slope) ~ std_normal();

  mu_err    ~ normal(0, 0.3);
  sigma_err ~ exponential(2);
  L_Omega_err ~ lkj_corr_cholesky(2);

  for (i in 1:N) {
    vector[K] log_odds;
    for (k in 1:K1)
      log_odds[k] = alpha[k] + beta_time[k] * time[i]
                  + u_int[empresa[i], k]
                  + u_slope[empresa[i], k] * time[i];
    log_odds[K] = 0;
    votos[i] ~ multinomial(softmax(log_odds));
  }

  for (n in 1:N_hist)
    hist_errors[n] ~ multi_normal_cholesky(
      mu_err, diag_pre_multiply(sigma_err, L_Omega_err));
}

generated quantities {
  vector[K] escanos;
  vector[K] pi_enc;
  vector[K] pi_real;

  {
    vector[K] log_odds_elec;
    for (k in 1:K1) log_odds_elec[k] = alpha[k];
    log_odds_elec[K] = 0;
    pi_enc = softmax(log_odds_elec);

    vector[K1] ilr_enc = H_ilr * log(pi_enc);
    vector[K1] err     = multi_normal_cholesky_rng(
      mu_err, diag_pre_multiply(sigma_err, L_Omega_err));
    pi_real = softmax(Ht_ilr * (ilr_enc + err));

    vector[K] total = rep_vector(0.0, K);
    for (p in 1:P) {
      vector[K] raw;
      for (k in 1:K) raw[k] = pi_real[k] * (ref_prov[p,k] / ref_reg[k]);
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

