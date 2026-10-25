// VERSIÓN ANTERIOR (2026-10-21), con PP como categoría de referencia. La actual es mrp.stan.
// Misma estructura que 04_brms.R, con dos cambios:
//  1. Provincia de More in Common: se mezclan PROBABILIDADES,
//       p_i = sum_m w_im * softmax(eta_i + a_prov[p_im]),
//     como con las encuestas agregadas del post full-bayesian. mm() de brms
//     promedia el predictor lineal, que no es lo mismo.
//  2. Efecto casa con suma cero (sum_to_zero_vector): el intercepto es la
//     "casa media" y cada casa se desvía de ella.
// Categoría de referencia: PP (predictor = 0), como en brms.
// Celdas con todos sus efectos: casa, mes, sexo, edad, estudios, recuerdo,
// CCAA, provincia, CCAA x recuerdo y casa x recuerdo.

functions {
  // probabilidades de voto válido en una celda de población: softmax con
  // la referencia y ceros exactos fuera del territorio de cada partido
  row_vector prob_valida(row_vector eta, row_vector mascara) {
    vector[cols(eta) + 1] p = softmax(append_row(0, eta')) .* mascara';
    return (p / sum(p))';
  }
}

data {
  int<lower=0, upper=1> prior_only;    // 1 = muestrear solo del prior
  // Priors. sd_exponencial = 1: exponential(1) en todas las sd, como en
  // 04_brms.R. Con 0: medias normales con tres escalas (recuerdo, resto de
  // efectos principales e interacciones), elegidas con el prior predictive check.
  int<lower=0, upper=1> sd_exponencial;
  real<lower=0> s_alpha;
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  real<lower=0> s_sd_int;
  int<lower=1> N;                       // celdas de la muestra
  int<lower=2> J;                       // partidos; el 1 (PP) es la referencia
  array[N, J] int<lower=0> y;           // votos por celda y partido
  vector[N] mujer;                      // 1 = mujer

  int<lower=2> n_casa;
  int<lower=1> n_mes;
  int<lower=1> n_edad;
  int<lower=1> n_edu;
  int<lower=1> n_rec;
  int<lower=1> n_ccaa;
  int<lower=1> n_prov;
  array[N] int<lower=1, upper=n_casa> casa;
  array[N] int<lower=1, upper=n_mes> mes;
  array[N] int<lower=1, upper=n_edad> edad;
  array[N] int<lower=1, upper=n_edu> edu;
  array[N] int<lower=1, upper=n_rec> rec;
  array[N] int<lower=1, upper=n_ccaa> ccaa;
  matrix[N, J - 1] off;                 // 0 en su territorio, -20 fuera

  // Provincia. Filas con provincia conocida: una sola provincia.
  int<lower=0> N1;
  array[N1] int<lower=1, upper=N> fila1;
  array[N1] int<lower=1, upper=n_prov> prov1;
  // Filas de More in Common sin provincia: len[m] provincias posibles, en las
  // posiciones ini[m] .. ini[m] + len[m] - 1 de ent_prov y ent_w
  int<lower=0> N2;
  array[N2] int<lower=1, upper=N> fila2;
  array[N2] int<lower=1> ini;
  array[N2] int<lower=1> len;
  int<lower=0> E;
  array[E] int<lower=1, upper=n_prov> ent_prov;
  vector<lower=0, upper=1>[E] ent_w;    // suman 1 en cada fila

  // Postestratificación (último mes, si votara todo el mundo)
  int<lower=1> P;
  vector<lower=0>[P] P_N;
  vector[P] P_mujer;
  array[P] int<lower=1, upper=n_edad> P_edad;
  array[P] int<lower=1, upper=n_edu> P_edu;
  array[P] int<lower=1, upper=n_rec> P_rec;
  array[P] int<lower=1, upper=n_ccaa> P_ccaa;
  array[P] int<lower=1, upper=n_prov> P_prov;
  matrix<lower=0, upper=1>[P, J] P_mascara;
  int<lower=1, upper=n_mes> mes_post;
}

transformed data {
  int K1 = J - 1;                       // predictores lineales (sin PP)
  // celda de las interacciones, índice plano
  array[N] int ccaa_rec;
  array[N] int casa_rec;
  for (i in 1:N) {
    ccaa_rec[i] = (ccaa[i] - 1) * n_rec + rec[i];
    casa_rec[i] = (casa[i] - 1) * n_rec + rec[i];
  }
  // sum_to_zero_vector ~ normal(0, s) tiene desviación típica marginal
  // s * sqrt((n - 1) / n); este factor la devuelve a s
  real esc_casa = sqrt(n_casa / (n_casa - 1.0));
}

parameters {
  vector[K1] alpha;                     // intercepto: la casa media
  vector[K1] b_mujer;

  // Efectos aleatorios no centrados: a = z * sd, con z ~ N(0, 1).
  // Un vector de desviaciones típicas por partido, como brms.
  array[K1] sum_to_zero_vector[n_casa] z_casa;
  matrix[n_mes, K1] z_mes;
  matrix[n_edad, K1] z_edad;
  matrix[n_edu, K1] z_edu;
  matrix[n_rec, K1] z_rec;
  matrix[n_ccaa, K1] z_ccaa;
  matrix[n_prov, K1] z_prov;
  matrix[n_ccaa * n_rec, K1] z_ccaa_rec;
  matrix[n_casa * n_rec, K1] z_casa_rec;

  vector<lower=0>[K1] sd_casa;
  vector<lower=0>[K1] sd_mes;
  vector<lower=0>[K1] sd_edad;
  vector<lower=0>[K1] sd_edu;
  vector<lower=0>[K1] sd_rec;
  vector<lower=0>[K1] sd_ccaa;
  vector<lower=0>[K1] sd_prov;
  vector<lower=0>[K1] sd_ccaa_rec;
  vector<lower=0>[K1] sd_casa_rec;
}

transformed parameters {
  matrix[n_casa, K1] a_casa;
  for (k in 1:K1) {
    a_casa[, k] = esc_casa * sd_casa[k] * z_casa[k];
  }
  matrix[n_mes, K1] a_mes = diag_post_multiply(z_mes, sd_mes);
  matrix[n_edad, K1] a_edad = diag_post_multiply(z_edad, sd_edad);
  matrix[n_edu, K1] a_edu = diag_post_multiply(z_edu, sd_edu);
  matrix[n_rec, K1] a_rec = diag_post_multiply(z_rec, sd_rec);
  matrix[n_ccaa, K1] a_ccaa = diag_post_multiply(z_ccaa, sd_ccaa);
  matrix[n_prov, K1] a_prov = diag_post_multiply(z_prov, sd_prov);
  matrix[n_ccaa * n_rec, K1] a_ccaa_rec = diag_post_multiply(z_ccaa_rec, sd_ccaa_rec);
  matrix[n_casa * n_rec, K1] a_casa_rec = diag_post_multiply(z_casa_rec, sd_casa_rec);
}

model {
  // priors
  alpha ~ normal(0, s_alpha);
  b_mujer ~ normal(0, 1);
  for (k in 1:K1) z_casa[k] ~ std_normal();
  to_vector(z_mes) ~ std_normal();
  to_vector(z_edad) ~ std_normal();
  to_vector(z_edu) ~ std_normal();
  to_vector(z_rec) ~ std_normal();
  to_vector(z_ccaa) ~ std_normal();
  to_vector(z_prov) ~ std_normal();
  to_vector(z_ccaa_rec) ~ std_normal();
  to_vector(z_casa_rec) ~ std_normal();
  if (sd_exponencial) {
    sd_casa ~ exponential(1);
    sd_mes ~ exponential(1);
    sd_edad ~ exponential(1);
    sd_edu ~ exponential(1);
    sd_rec ~ exponential(1);
    sd_ccaa ~ exponential(1);
    sd_prov ~ exponential(1);
    sd_ccaa_rec ~ exponential(1);
    sd_casa_rec ~ exponential(1);
  } else {
    // con <lower=0>, normal(0, s) es una media normal
    sd_rec ~ normal(0, s_sd_rec);
    sd_casa ~ normal(0, s_sd_main);
    sd_mes ~ normal(0, s_sd_main);
    sd_edad ~ normal(0, s_sd_main);
    sd_edu ~ normal(0, s_sd_main);
    sd_ccaa ~ normal(0, s_sd_main);
    sd_prov ~ normal(0, s_sd_main);
    sd_ccaa_rec ~ normal(0, s_sd_int);
    sd_casa_rec ~ normal(0, s_sd_int);
  }

  if (!prior_only) {
    // Predictor lineal sin la provincia, para todas las celdas a la vez.
    // a_edad[edad] coge la fila de cada celda: una matriz N x K1.
    matrix[N, K1] eta = rep_matrix(alpha', N) + mujer * b_mujer'
                        + a_casa[casa] + a_mes[mes] + a_edad[edad] + a_edu[edu]
                        + a_rec[rec] + a_ccaa[ccaa]
                        + a_ccaa_rec[ccaa_rec] + a_casa_rec[casa_rec] + off;

    // 1. Provincia conocida: multinomial con su provincia
    matrix[N1, K1] eta1 = eta[fila1] + a_prov[prov1];
    for (n in 1:N1) {
      y[fila1[n]] ~ multinomial_logit(append_row(0, eta1[n]'));
    }

    // 2. Sin provincia: la probabilidad es la media ponderada de las
    //    probabilidades de cada provincia de la CCAA
    for (m in 1:N2) {
      int i = fila2[m];
      vector[J] p = rep_vector(0, J);
      for (e in ini[m]:(ini[m] + len[m] - 1)) {
        p += ent_w[e] * softmax(append_row(0, (eta[i] + a_prov[ent_prov[e]])'));
      }
      y[i] ~ multinomial(p);
    }
  }
}

generated quantities {
  // Voto válido (%) en España según cada casa (filas 1..n_casa) y según la
  // casa media (fila n_casa + 1): sin efecto casa, y con casa x recuerdo
  // promediado entre casas
  matrix[n_casa + 1, J] v_casa = rep_matrix(0, n_casa + 1, J);
  {
    for (q in 1:P) {
      row_vector[K1] base = alpha' + P_mujer[q] * b_mujer' + a_mes[mes_post]
                            + a_edad[P_edad[q]] + a_edu[P_edu[q]] + a_rec[P_rec[q]]
                            + a_ccaa[P_ccaa[q]] + a_prov[P_prov[q]]
                            + a_ccaa_rec[(P_ccaa[q] - 1) * n_rec + P_rec[q]];
      row_vector[K1] casa_media = rep_row_vector(0, K1);
      for (h in 1:n_casa) {
        row_vector[K1] ef_casa = a_casa[h] + a_casa_rec[(h - 1) * n_rec + P_rec[q]];
        casa_media += ef_casa / n_casa;
        v_casa[h] += P_N[q] * prob_valida(base + ef_casa, P_mascara[q]);
      }
      v_casa[n_casa + 1] += P_N[q] * prob_valida(base + casa_media, P_mascara[q]);
    }
    v_casa *= 100 / sum(P_N);
  }

  // Prior predictive check (solo con prior_only = 1): probabilidad del
  // partido más votado en cada celda de la muestra con provincia conocida
  vector[prior_only ? N1 : 0] p_max;
  if (prior_only) {
    for (n in 1:N1) {
      int i = fila1[n];
      row_vector[K1] eta_i = alpha' + mujer[i] * b_mujer' + a_casa[casa[i]]
                             + a_mes[mes[i]] + a_edad[edad[i]] + a_edu[edu[i]]
                             + a_rec[rec[i]] + a_ccaa[ccaa[i]] + a_prov[prov1[n]]
                             + a_ccaa_rec[ccaa_rec[i]] + a_casa_rec[casa_rec[i]] + off[i];
      p_max[n] = max(softmax(append_row(0, eta_i')));
    }
  }
}
