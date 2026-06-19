
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

