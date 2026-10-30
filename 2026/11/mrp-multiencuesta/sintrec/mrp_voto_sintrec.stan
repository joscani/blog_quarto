// MRP conjunto: modelo de voto sin términos de recuerdo de voto.
// Variante de sensibilidad (sintrec): igual que mrp_voto.stan pero sin
// a_rec, a_ccaa_rec ni a_casa_rec. La verosimilitud es idéntica a la de
// mrp_voto.stan con n_rec = 1 (esos términos se cancelan en el softmax), así
// que el posterior de los parámetros comunes es el mismo; aquí no quedan
// parámetros previo-solo (z_rec, sd_rec, ...) que solo gasten MCMC.
//
// Los datos vienen de 06_stan_datos.R con sin_rec = TRUE
// (datos/datos_stan_sintrec.rds): declara rec = "sin_recuerdo", n_rec = 1 y
// P_rec = 1, que este modelo no usa (CmdStan ignora los datos que no declara).

functions {
#include funciones.stan

  // Verosimilitud del voto para reduce_sum, sin recuerdo de voto: igual que
  // voto_parcial() en funciones.stan sin los términos rec, ccaa_rec, casa_rec.
  real voto_parcial_sintrec(array[] int filas, int start, int end,
               array[,] int y, vector mujer,
               array[] int casa, array[] int mes, array[] int edad, array[] int edu,
               array[] int ccaa,
               matrix off, array[] int sin_aa, int j_aa,
               array[] int prov_fila, array[] int ini_fila, array[] int len_fila,
               array[] int ent_prov, vector log_ent_w,
               vector alpha, vector b_mujer, matrix a_casa, matrix a_mes, matrix a_edad,
               matrix a_edu, matrix a_ccaa, matrix a_prov) {
    int J = cols(off);
    real lp = 0;
    for (n in 1:size(filas)) {
      int i = filas[n];
      row_vector[J] eta = alpha' + mujer[i] * b_mujer' + a_casa[casa[i]] + a_mes[mes[i]]
                          + a_edad[edad[i]] + a_edu[edu[i]] + a_ccaa[ccaa[i]] + off[i];
      if (prov_fila[i] > 0) {
        row_vector[J] e = eta + a_prov[prov_fila[i]];
        if (sin_aa[i] == 0) {
          lp += multinomial_logit_lpmf(y[i] | e');
        } else {
          // la encuesta no ofrecía AA: normalizador sin AA
          lp += multinomial_cond(e', y[i], j_aa);
        }
      } else {
        int L = len_fila[i];
        matrix[L, J] lpe;
        for (l in 1:L) {
          int e = ini_fila[i] + l - 1;
          lpe[l] = log_ent_w[e] + log_softmax((eta + a_prov[ent_prov[e]])')';
        }
        vector[J] logp;
        for (j in 1:J) logp[j] = log_sum_exp(lpe[, j]);
        // logp ya está normalizado: softmax(logp) = p
        if (sin_aa[i] == 0) {
          lp += multinomial_logit_lpmf(y[i] | logp);
        } else {
          // la mezcla de provincias, condicionada a no-AA
          lp += multinomial_cond(logp, y[i], j_aa);
        }
      }
    }
    return lp;
  }
}

data {
  int<lower=0, upper=1> prior_only;
  int<lower=0, upper=1> sd_exponencial;
  real<lower=0> s_alpha;
  real<lower=0> s_b_mujer;
  real<lower=0> s_sd_main;
  real<lower=0> s_sd_int;
  int<lower=1> grainsize;               // filas por trozo de reduce_sum
  int<lower=1> N;
  int<lower=2> J;
  array[N, J] int<lower=0> y;
  vector[N] mujer;

  int<lower=2> n_casa;
  int<lower=1> n_mes;
  int<lower=1> n_edad;
  int<lower=1> n_edu;
  int<lower=1> n_ccaa;
  int<lower=1> n_prov;
  array[N] int<lower=1, upper=n_casa> casa;
  array[N] int<lower=1, upper=n_mes> mes;
  array[N] int<lower=1, upper=n_edad> edad;
  array[N] int<lower=1, upper=n_edu> edu;
  array[N] int<lower=1, upper=n_ccaa> ccaa;
  matrix[N, J] off;
  array[N] int<lower=0, upper=1> sin_aa; // 1: la encuesta no ofrecía AA
  int<lower=1, upper=J> j_aa;            // índice de AA en voto_lv

  int<lower=0> N1;
  array[N1] int<lower=1, upper=N> fila1;
  array[N1] int<lower=1, upper=n_prov> prov1;
  int<lower=0> N2;
  array[N2] int<lower=1, upper=N> fila2;
  array[N2] int<lower=1> ini;
  array[N2] int<lower=1> len;
  int<lower=0> E;
  array[E] int<lower=1, upper=n_prov> ent_prov;
  vector<lower=0, upper=1>[E] ent_w;

  int<lower=1> P;
  vector<lower=0>[P] P_N;
  vector[P] P_mujer;
  array[P] int<lower=1, upper=n_edad> P_edad;
  array[P] int<lower=1, upper=n_edu> P_edu;
  array[P] int<lower=1, upper=n_ccaa> P_ccaa;
  array[P] int<lower=1, upper=n_prov> P_prov;
  matrix<lower=0, upper=1>[P, J] P_mascara;
  int<lower=1, upper=n_mes> mes_post;
}

transformed data {
  real esc_casa = sqrt(n_casa / (n_casa - 1.0));

  // Por fila: provincia (0 = mezcla) y dónde están sus entradas de la mezcla
  array[N] int prov_fila = rep_array(0, N);
  array[N] int ini_fila = rep_array(1, N);
  array[N] int len_fila = rep_array(0, N);
  for (n in 1:N1) prov_fila[fila1[n]] = prov1[n];
  for (m in 1:N2) {
    ini_fila[fila2[m]] = ini[m];
    len_fila[fila2[m]] = len[m];
  }
  vector[E] log_ent_w = log(ent_w);
  array[N] int filas = linspaced_int_array(N, 1, N);
}

parameters {
  sum_to_zero_vector[J] alpha;
  sum_to_zero_vector[J] b_mujer;
  array[J] sum_to_zero_vector[n_casa] z_casa;
  matrix[n_mes, J] z_mes;
  matrix[n_edad, J] z_edad;
  matrix[n_edu, J] z_edu;
  matrix[n_ccaa, J] z_ccaa;
  matrix[n_prov, J] z_prov;

  vector<lower=0>[J] sd_casa;
  vector<lower=0>[J] sd_mes;
  vector<lower=0>[J] sd_edad;
  vector<lower=0>[J] sd_edu;
  vector<lower=0>[J] sd_ccaa;
  vector<lower=0>[J] sd_prov;
}

transformed parameters {
  matrix[n_casa, J] a_casa;
  for (k in 1:J) {
    a_casa[, k] = esc_casa * sd_casa[k] * z_casa[k];
  }
  a_casa = centra(a_casa);
  matrix[n_mes, J] a_mes = centra(diag_post_multiply(z_mes, sd_mes));
  matrix[n_edad, J] a_edad = centra(diag_post_multiply(z_edad, sd_edad));
  matrix[n_edu, J] a_edu = centra(diag_post_multiply(z_edu, sd_edu));
  matrix[n_ccaa, J] a_ccaa = centra(diag_post_multiply(z_ccaa, sd_ccaa));
  matrix[n_prov, J] a_prov = centra(diag_post_multiply(z_prov, sd_prov));
}

model {
  alpha ~ normal(0, s_alpha);
  b_mujer ~ normal(0, s_b_mujer);
  for (k in 1:J) z_casa[k] ~ std_normal();
  to_vector(z_mes) ~ std_normal();
  to_vector(z_edad) ~ std_normal();
  to_vector(z_edu) ~ std_normal();
  to_vector(z_ccaa) ~ std_normal();
  to_vector(z_prov) ~ std_normal();
  if (sd_exponencial) {
    sd_casa ~ exponential(1);
    sd_mes ~ exponential(1);
    sd_edad ~ exponential(1);
    sd_edu ~ exponential(1);
    sd_ccaa ~ exponential(1);
    sd_prov ~ exponential(1);
  } else {
    sd_casa ~ normal(0, s_sd_main);
    sd_mes ~ normal(0, s_sd_main);
    sd_edad ~ normal(0, s_sd_main);
    sd_edu ~ normal(0, s_sd_main);
    sd_ccaa ~ normal(0, s_sd_main);
    sd_prov ~ normal(0, s_sd_main);
  }

  if (!prior_only) {
    target += reduce_sum(voto_parcial_sintrec, filas, grainsize,
                         y, mujer, casa, mes, edad, edu, ccaa,
                         off, sin_aa, j_aa, prov_fila, ini_fila, len_fila, ent_prov, log_ent_w,
                         alpha, b_mujer, a_casa, a_mes, a_edad, a_edu, a_ccaa,
                         a_prov);
  }
}

generated quantities {
  // Voto válido (%) en España según cada casa (filas 1..n_casa) y según la
  // casa media (fila n_casa + 1), si votara todo el mundo
  matrix[n_casa + 1, J] v_casa = rep_matrix(0, n_casa + 1, J);
  for (q in 1:P) {
    row_vector[J] base = alpha' + P_mujer[q] * b_mujer' + a_mes[mes_post]
                         + a_edad[P_edad[q]] + a_edu[P_edu[q]]
                         + a_ccaa[P_ccaa[q]] + a_prov[P_prov[q]];
    row_vector[J] casa_media = rep_row_vector(0, J);
    for (h in 1:n_casa) {
      row_vector[J] ef_casa = a_casa[h];
      casa_media += ef_casa / n_casa;
      v_casa[h] += P_N[q] * prob_valida(base + ef_casa, P_mascara[q]);
    }
    v_casa[n_casa + 1] += P_N[q] * prob_valida(base + casa_media, P_mascara[q]);
  }
  v_casa *= 100 / sum(P_N);
}
