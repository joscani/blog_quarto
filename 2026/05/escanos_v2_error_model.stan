
data {
  int<lower=1> K1;              // K-1 = 5
  int<lower=1> K;               // 6
  int<lower=1> N_hist;          // elecciones con datos completos
  matrix[N_hist, K1] hist_errors;

  int<lower=1> N_sub;           // draws del meta-análisis submuestra
  matrix[N_sub, K1] meta_draws; // en espacio ALR

  int<lower=1> P;               // provincias
  array[P] int seats;
  matrix[P, K] ref_prov;        // proporciones 2022 por provincia
  vector[K] ref_reg;            // proporciones regionales 2022
  array[P] int votos_prov;
  int<lower=1> scale_factor;
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
  vector[K] escanos;   // escaños totales para este draw MCMC

  {
    // 1. Elegir un draw del meta-análisis al azar
    int j = categorical_rng(rep_vector(1.0 / N_sub, N_sub));

    // 2. Muestrear error en ALR
    vector[K1] err = multi_normal_cholesky_rng(
      mu_err, diag_pre_multiply(sigma_err, L_Omega));

    // 3. Draw electoral: meta + error (en ALR)
    vector[K1] alr_elec = to_vector(meta_draws[j]) + err;

    // 4. Softmax inversa: ALR → proporciones (K dimensiones)
    vector[K] pi_reg;
    {
      vector[K1] e = exp(alr_elec);
      real denom = 1.0 + sum(e);
      for (k in 1:K1) pi_reg[k] = e[k] / denom;
      pi_reg[K] = 1.0 / denom;   // categoría de referencia ('resto')
    }

    // 5. D'Hondt por provincia
    vector[K] total = rep_vector(0.0, K);

    for (p in 1:P) {
      // Ajuste provincial por ratio 2022
      vector[K] raw;
      for (k in 1:K) raw[k] = pi_reg[k] * (ref_prov[p, k] / ref_reg[k]);
      vector[K] pi_prov = raw / sum(raw);

      // Incertidumbre provincial: Dirichlet
      vector[K] alpha;
      for (k in 1:K) alpha[k] = pi_prov[k] * (votos_prov[p] / scale_factor);
      vector[K] pi_dir = dirichlet_rng(alpha);

      // D'Hondt (greedy)
      vector[K] result = rep_vector(0.0, K);
      for (s in 1:seats[p]) {
        int winner = 1;
        real best  = pi_dir[1] / (result[1] + 1);
        for (k in 2:K) {
          real val = pi_dir[k] / (result[k] + 1);
          if (val > best) {
            best   = val;
            winner = k;
          }
        }
        result[winner] += 1;
      }
      total += result;
    }

    for (k in 1:K) escanos[k] = total[k];
  }
}

