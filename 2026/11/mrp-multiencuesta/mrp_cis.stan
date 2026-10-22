// MRP solo con los barómetros del CIS, en Stan puro.
// Es mrp.stan sin lo que solo hace falta con varias casas:
//  - sin efecto casa ni casa x recuerdo (una sola casa: su sesgo no se puede
//    separar del intercepto);
//  - sin filas mezcla: el CIS siempre trae la provincia.
// Lo demás es igual: sin categoría de referencia, efectos centrados por filas
// y los mismos priors.
// Celdas con sus efectos: mes, sexo, edad, estudios, recuerdo, CCAA,
// provincia y CCAA x recuerdo.
//
// Participación (con_part = 1): segundo modelo, binomial, en las mismas
// celdas: P(decir "10, seguro que voto"). Se ajusta a la vez que el de voto,
// pero no comparten parámetros. En la postestratificación se calibra: un
// desplazamiento común delta en logit lleva la participación total a
// part_obj (70,4 %, la del 23-J entre residentes). El modelo decide quién
// vota; el 23-J, cuántos.

functions {
  // resta a cada fila su media: en cada nivel, el efecto suma 0 entre partidos
  matrix centra(matrix a) {
    int J = cols(a);
    return a - rep_matrix(a * rep_vector(1.0 / J, J), J);
  }

  // delta tal que sum(N * inv_logit(eta + delta)) = obj * sum(N), por Newton.
  // La función es creciente en delta: converge en pocas iteraciones.
  real calibra(vector eta, vector N, real obj) {
    real delta = 0;
    for (it in 1:50) {
      vector[rows(eta)] p = inv_logit(eta + delta);
      real f = dot_product(N, p) / sum(N) - obj;
      real df = dot_product(N, p .* (1 - p)) / sum(N);
      delta -= f / df;
      if (abs(f) < 1e-10) break;
    }
    return delta;
  }

  // probabilidades de voto válido en una celda de población, con ceros
  // exactos fuera del territorio de cada partido
  row_vector prob_valida(row_vector eta, row_vector mascara) {
    vector[cols(eta)] p = softmax(eta') .* mascara';
    return (p / sum(p))';
  }
}

data {
  int<lower=0, upper=1> prior_only;    // 1 = muestrear solo del prior
  int<lower=0, upper=1> sd_exponencial;
  real<lower=0> s_alpha;
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  real<lower=0> s_sd_int;
  int<lower=1> N;                       // celdas de la muestra
  int<lower=2> J;                       // partidos
  array[N, J] int<lower=0> y;           // votos por celda y partido
  vector[N] mujer;                      // 1 = mujer

  int<lower=1> n_mes;
  int<lower=1> n_edad;
  int<lower=1> n_edu;
  int<lower=1> n_rec;
  int<lower=1> n_ccaa;
  int<lower=1> n_prov;
  array[N] int<lower=1, upper=n_mes> mes;
  array[N] int<lower=1, upper=n_edad> edad;
  array[N] int<lower=1, upper=n_edu> edu;
  array[N] int<lower=1, upper=n_rec> rec;
  array[N] int<lower=1, upper=n_ccaa> ccaa;
  array[N] int<lower=1, upper=n_prov> prov;
  matrix[N, J] off;                     // 0 en su territorio, -20 fuera

  // Participación: celdas con n entrevistas y k que dicen "10"
  int<lower=0, upper=1> con_part;
  int<lower=0> NT;
  array[NT] int<lower=1> t_n;
  array[NT] int<lower=0> t_k;
  vector[NT] t_mujer;
  array[NT] int<lower=1, upper=n_mes> t_mes;
  array[NT] int<lower=1, upper=n_edad> t_edad;
  array[NT] int<lower=1, upper=n_edu> t_edu;
  array[NT] int<lower=1, upper=n_rec> t_rec;
  array[NT] int<lower=1, upper=n_ccaa> t_ccaa;
  array[NT] int<lower=1, upper=n_prov> t_prov;
  real<lower=0, upper=1> part_obj;      // participación total a la que se calibra

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
  array[N] int ccaa_rec;
  for (i in 1:N) {
    ccaa_rec[i] = (ccaa[i] - 1) * n_rec + rec[i];
  }
}

parameters {
  sum_to_zero_vector[J] alpha;
  sum_to_zero_vector[J] b_mujer;

  matrix[n_mes, J] z_mes;
  matrix[n_edad, J] z_edad;
  matrix[n_edu, J] z_edu;
  matrix[n_rec, J] z_rec;
  matrix[n_ccaa, J] z_ccaa;
  matrix[n_prov, J] z_prov;
  matrix[n_ccaa * n_rec, J] z_ccaa_rec;

  vector<lower=0>[J] sd_mes;
  vector<lower=0>[J] sd_edad;
  vector<lower=0>[J] sd_edu;
  vector<lower=0>[J] sd_rec;
  vector<lower=0>[J] sd_ccaa;
  vector<lower=0>[J] sd_prov;
  vector<lower=0>[J] sd_ccaa_rec;

  // Participación: un solo predictor (logit de decir "10"), efectos no centrados
  real alpha_t;
  real b_mujer_t;
  vector[n_mes] z_mes_t;
  vector[n_edad] z_edad_t;
  vector[n_edu] z_edu_t;
  vector[n_rec] z_rec_t;
  vector[n_ccaa] z_ccaa_t;
  vector[n_prov] z_prov_t;
  real<lower=0> sd_mes_t;
  real<lower=0> sd_edad_t;
  real<lower=0> sd_edu_t;
  real<lower=0> sd_rec_t;
  real<lower=0> sd_ccaa_t;
  real<lower=0> sd_prov_t;
}

transformed parameters {
  matrix[n_mes, J] a_mes = centra(diag_post_multiply(z_mes, sd_mes));
  matrix[n_edad, J] a_edad = centra(diag_post_multiply(z_edad, sd_edad));
  matrix[n_edu, J] a_edu = centra(diag_post_multiply(z_edu, sd_edu));
  matrix[n_rec, J] a_rec = centra(diag_post_multiply(z_rec, sd_rec));
  matrix[n_ccaa, J] a_ccaa = centra(diag_post_multiply(z_ccaa, sd_ccaa));
  matrix[n_prov, J] a_prov = centra(diag_post_multiply(z_prov, sd_prov));
  matrix[n_ccaa * n_rec, J] a_ccaa_rec = centra(diag_post_multiply(z_ccaa_rec, sd_ccaa_rec));

  vector[n_mes] a_mes_t = sd_mes_t * z_mes_t;
  vector[n_edad] a_edad_t = sd_edad_t * z_edad_t;
  vector[n_edu] a_edu_t = sd_edu_t * z_edu_t;
  vector[n_rec] a_rec_t = sd_rec_t * z_rec_t;
  vector[n_ccaa] a_ccaa_t = sd_ccaa_t * z_ccaa_t;
  vector[n_prov] a_prov_t = sd_prov_t * z_prov_t;
}

model {
  alpha ~ normal(0, s_alpha);
  b_mujer ~ normal(0, 1);
  to_vector(z_mes) ~ std_normal();
  to_vector(z_edad) ~ std_normal();
  to_vector(z_edu) ~ std_normal();
  to_vector(z_rec) ~ std_normal();
  to_vector(z_ccaa) ~ std_normal();
  to_vector(z_prov) ~ std_normal();
  to_vector(z_ccaa_rec) ~ std_normal();
  if (sd_exponencial) {
    sd_mes ~ exponential(1);
    sd_edad ~ exponential(1);
    sd_edu ~ exponential(1);
    sd_rec ~ exponential(1);
    sd_ccaa ~ exponential(1);
    sd_prov ~ exponential(1);
    sd_ccaa_rec ~ exponential(1);
  } else {
    sd_rec ~ normal(0, s_sd_rec);
    sd_mes ~ normal(0, s_sd_main);
    sd_edad ~ normal(0, s_sd_main);
    sd_edu ~ normal(0, s_sd_main);
    sd_ccaa ~ normal(0, s_sd_main);
    sd_prov ~ normal(0, s_sd_main);
    sd_ccaa_rec ~ normal(0, s_sd_int);
  }

  // participación: alpha_t centrado en logit(0,75), cerca del "10" observado
  alpha_t ~ normal(1, 1.5);
  b_mujer_t ~ normal(0, 1);
  z_mes_t ~ std_normal();
  z_edad_t ~ std_normal();
  z_edu_t ~ std_normal();
  z_rec_t ~ std_normal();
  z_ccaa_t ~ std_normal();
  z_prov_t ~ std_normal();
  sd_rec_t ~ normal(0, s_sd_rec);
  sd_mes_t ~ normal(0, s_sd_main);
  sd_edad_t ~ normal(0, s_sd_main);
  sd_edu_t ~ normal(0, s_sd_main);
  sd_ccaa_t ~ normal(0, s_sd_main);
  sd_prov_t ~ normal(0, s_sd_main);

  if (!prior_only && con_part) {
    t_k ~ binomial_logit(t_n, alpha_t + t_mujer * b_mujer_t + a_mes_t[t_mes]
                              + a_edad_t[t_edad] + a_edu_t[t_edu] + a_rec_t[t_rec]
                              + a_ccaa_t[t_ccaa] + a_prov_t[t_prov]);
  }

  if (!prior_only) {
    matrix[N, J] eta = rep_matrix(alpha', N) + mujer * b_mujer'
                       + a_mes[mes] + a_edad[edad] + a_edu[edu]
                       + a_rec[rec] + a_ccaa[ccaa] + a_prov[prov]
                       + a_ccaa_rec[ccaa_rec] + off;
    for (i in 1:N) {
      y[i] ~ multinomial_logit(eta[i]');
    }
  }
}

generated quantities {
  // Voto válido (%) en España según el CIS. Matriz de una fila, para que
  // 05_stan.R lea igual este modelo y mrp.stan.
  matrix[1, J] v_casa = rep_matrix(0, 1, J);
  // Con participación: votos de cada celda = N * P(votar) * P(partido)
  matrix[1, J] v_part = rep_matrix(0, 1, J);
  real part_bruta = 0;                  // % que diría "10", antes de calibrar
  real delta = 0;                       // desplazamiento de la calibración (logit)
  vector[P] p_vota = rep_vector(1, P);
  if (con_part) {
    vector[P] eta_t;
    for (q in 1:P) {
      eta_t[q] = alpha_t + P_mujer[q] * b_mujer_t + a_mes_t[mes_post]
                 + a_edad_t[P_edad[q]] + a_edu_t[P_edu[q]] + a_rec_t[P_rec[q]]
                 + a_ccaa_t[P_ccaa[q]] + a_prov_t[P_prov[q]];
    }
    part_bruta = 100 * dot_product(P_N, inv_logit(eta_t)) / sum(P_N);
    delta = calibra(eta_t, P_N, part_obj);
    p_vota = inv_logit(eta_t + delta);
  }
  for (q in 1:P) {
    row_vector[J] eta_q = alpha' + P_mujer[q] * b_mujer' + a_mes[mes_post]
                          + a_edad[P_edad[q]] + a_edu[P_edu[q]] + a_rec[P_rec[q]]
                          + a_ccaa[P_ccaa[q]] + a_prov[P_prov[q]]
                          + a_ccaa_rec[(P_ccaa[q] - 1) * n_rec + P_rec[q]];
    row_vector[J] pv = prob_valida(eta_q, P_mascara[q]);
    v_casa[1] += P_N[q] * pv;
    v_part[1] += P_N[q] * p_vota[q] * pv;
  }
  v_casa *= 100 / sum(P_N);
  v_part *= 100 / sum(P_N .* p_vota);

  vector[prior_only ? N : 0] p_max;
  if (prior_only) {
    for (i in 1:N) {
      row_vector[J] eta_i = alpha' + mujer[i] * b_mujer' + a_mes[mes[i]]
                            + a_edad[edad[i]] + a_edu[edu[i]] + a_rec[rec[i]]
                            + a_ccaa[ccaa[i]] + a_prov[prov[i]]
                            + a_ccaa_rec[ccaa_rec[i]] + off[i];
      p_max[i] = max(softmax(eta_i'));
    }
  }
}
