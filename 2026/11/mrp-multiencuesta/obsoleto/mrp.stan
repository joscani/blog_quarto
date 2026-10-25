// MRP conjunto (CIS + 40dB + More in Common) en Stan puro.
// Misma estructura que 04_brms.R, con tres cambios:
//  1. Provincia de More in Common: se mezclan PROBABILIDADES,
//       p_i = sum_m w_im * softmax(eta_i + a_prov[p_im]),
//     como con las encuestas agregadas del post full-bayesian. mm() de brms
//     promedia el predictor lineal, que no es lo mismo.
//  2. Efecto casa con suma cero (sum_to_zero_vector): el intercepto es la
//     "casa media" y cada casa se desvía de ella.
//  3. Sin categoría de referencia: un predictor por partido, centrado (suma
//     cero entre partidos), como el voto latente de full-bayesian. softmax no
//     cambia si se suma lo mismo a todos los partidos, así que centrar no
//     pierde nada y todos los partidos tienen el mismo prior.
//     (La versión con PP de referencia está en mrp_referencia_pp.stan.)
// Celdas con todos sus efectos: casa, mes, sexo, edad, estudios, recuerdo,
// CCAA, provincia, CCAA x recuerdo y casa x recuerdo.
//
// Participación: segundo modelo, binomial, de decir "10, seguro que voto",
// como en mrp_cis.stan pero con efecto casa (el CIS tiene un 79 % de "10";
// 40dB, un 61 %). La provincia de More in Common se mezcla igual que en el
// modelo de voto. En la postestratificación se calibra a part_obj con un
// desplazamiento común: el nivel de cada casa desaparece, así que todas las
// casas usan la misma participación (la de la casa media).

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
  // Priors. sd_exponencial = 1: exponential(1) en todas las sd, como en
  // 04_brms.R. Con 0: medias normales con tres escalas (recuerdo, resto de
  // efectos principales e interacciones), elegidas con el prior predictive check.
  int<lower=0, upper=1> sd_exponencial;
  real<lower=0> s_alpha;
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  real<lower=0> s_sd_int;
  int<lower=1> N;                       // celdas de la muestra
  int<lower=2> J;                       // partidos
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
  matrix[N, J] off;                     // 0 en su territorio, -20 fuera

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

  // Participación: celdas con n entrevistas y k que dicen "10". Provincia
  // conocida (NT1 filas) o mezcla (NT2 filas), como en el modelo de voto.
  int<lower=0> NT;
  array[NT] int<lower=1> t_n;
  array[NT] int<lower=0> t_k;
  vector[NT] t_mujer;
  // Casas con pregunta de probabilidad de votar: las n_casa_t primeras
  // (Ateneo del Dato no la tiene y va la última)
  int<lower=2, upper=n_casa> n_casa_t;
  array[NT] int<lower=1, upper=n_casa_t> t_casa;
  array[NT] int<lower=1, upper=n_mes> t_mes;
  array[NT] int<lower=1, upper=n_edad> t_edad;
  array[NT] int<lower=1, upper=n_edu> t_edu;
  array[NT] int<lower=1, upper=n_rec> t_rec;
  array[NT] int<lower=1, upper=n_ccaa> t_ccaa;
  int<lower=0> NT1;
  array[NT1] int<lower=1, upper=NT> t_fila1;
  array[NT1] int<lower=1, upper=n_prov> t_prov1;
  int<lower=0> NT2;
  array[NT2] int<lower=1, upper=NT> t_fila2;
  array[NT2] int<lower=1> t_ini;
  array[NT2] int<lower=1> t_len;
  int<lower=0> TE;
  array[TE] int<lower=1, upper=n_prov> t_ent_prov;
  vector<lower=0, upper=1>[TE] t_ent_w;
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
  real esc_casa_t = sqrt(n_casa_t / (n_casa_t - 1.0));
}

parameters {
  // intercepto y sexo: suman 0 entre partidos
  sum_to_zero_vector[J] alpha;          // intercepto: la casa media
  sum_to_zero_vector[J] b_mujer;

  // Efectos aleatorios no centrados: a = z * sd, con z ~ N(0, 1), y después
  // centrados por filas con centra(). Una desviación típica por partido.
  array[J] sum_to_zero_vector[n_casa] z_casa;   // suma 0 entre casas
  matrix[n_mes, J] z_mes;
  matrix[n_edad, J] z_edad;
  matrix[n_edu, J] z_edu;
  matrix[n_rec, J] z_rec;
  matrix[n_ccaa, J] z_ccaa;
  matrix[n_prov, J] z_prov;
  matrix[n_ccaa * n_rec, J] z_ccaa_rec;
  matrix[n_casa * n_rec, J] z_casa_rec;

  vector<lower=0>[J] sd_casa;
  vector<lower=0>[J] sd_mes;
  vector<lower=0>[J] sd_edad;
  vector<lower=0>[J] sd_edu;
  vector<lower=0>[J] sd_rec;
  vector<lower=0>[J] sd_ccaa;
  vector<lower=0>[J] sd_prov;
  vector<lower=0>[J] sd_ccaa_rec;
  vector<lower=0>[J] sd_casa_rec;

  // Participación: un solo predictor (logit de decir "10"), efectos no centrados
  real alpha_t;
  real b_mujer_t;
  sum_to_zero_vector[n_casa_t] z_casa_t;
  vector[n_mes] z_mes_t;
  vector[n_edad] z_edad_t;
  vector[n_edu] z_edu_t;
  vector[n_rec] z_rec_t;
  vector[n_ccaa] z_ccaa_t;
  vector[n_prov] z_prov_t;
  real<lower=0> sd_casa_t;
  real<lower=0> sd_mes_t;
  real<lower=0> sd_edad_t;
  real<lower=0> sd_edu_t;
  real<lower=0> sd_rec_t;
  real<lower=0> sd_ccaa_t;
  real<lower=0> sd_prov_t;
}

transformed parameters {
  matrix[n_casa, J] a_casa;
  for (k in 1:J) {
    a_casa[, k] = esc_casa * sd_casa[k] * z_casa[k];
  }
  a_casa = centra(a_casa);              // sigue sumando 0 entre casas
  matrix[n_mes, J] a_mes = centra(diag_post_multiply(z_mes, sd_mes));
  matrix[n_edad, J] a_edad = centra(diag_post_multiply(z_edad, sd_edad));
  matrix[n_edu, J] a_edu = centra(diag_post_multiply(z_edu, sd_edu));
  matrix[n_rec, J] a_rec = centra(diag_post_multiply(z_rec, sd_rec));
  matrix[n_ccaa, J] a_ccaa = centra(diag_post_multiply(z_ccaa, sd_ccaa));
  matrix[n_prov, J] a_prov = centra(diag_post_multiply(z_prov, sd_prov));
  matrix[n_ccaa * n_rec, J] a_ccaa_rec = centra(diag_post_multiply(z_ccaa_rec, sd_ccaa_rec));
  matrix[n_casa * n_rec, J] a_casa_rec = centra(diag_post_multiply(z_casa_rec, sd_casa_rec));

  vector[n_casa_t] a_casa_t = esc_casa_t * sd_casa_t * z_casa_t;
  vector[n_mes] a_mes_t = sd_mes_t * z_mes_t;
  vector[n_edad] a_edad_t = sd_edad_t * z_edad_t;
  vector[n_edu] a_edu_t = sd_edu_t * z_edu_t;
  vector[n_rec] a_rec_t = sd_rec_t * z_rec_t;
  vector[n_ccaa] a_ccaa_t = sd_ccaa_t * z_ccaa_t;
  vector[n_prov] a_prov_t = sd_prov_t * z_prov_t;
}

model {
  // priors
  alpha ~ normal(0, s_alpha);
  b_mujer ~ normal(0, 1);
  for (k in 1:J) z_casa[k] ~ std_normal();
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

  // participación: alpha_t centrado en logit(0,73), cerca del "10" observado
  alpha_t ~ normal(1, 1.5);
  b_mujer_t ~ normal(0, 1);
  z_casa_t ~ std_normal();
  z_mes_t ~ std_normal();
  z_edad_t ~ std_normal();
  z_edu_t ~ std_normal();
  z_rec_t ~ std_normal();
  z_ccaa_t ~ std_normal();
  z_prov_t ~ std_normal();
  sd_rec_t ~ normal(0, s_sd_rec);
  sd_casa_t ~ normal(0, s_sd_main);
  sd_mes_t ~ normal(0, s_sd_main);
  sd_edad_t ~ normal(0, s_sd_main);
  sd_edu_t ~ normal(0, s_sd_main);
  sd_ccaa_t ~ normal(0, s_sd_main);
  sd_prov_t ~ normal(0, s_sd_main);

  if (!prior_only) {
    // Participación sin la provincia; después, provincia conocida o mezcla
    vector[NT] eta_t = alpha_t + t_mujer * b_mujer_t + a_casa_t[t_casa] + a_mes_t[t_mes]
                       + a_edad_t[t_edad] + a_edu_t[t_edu] + a_rec_t[t_rec] + a_ccaa_t[t_ccaa];
    t_k[t_fila1] ~ binomial_logit(t_n[t_fila1], eta_t[t_fila1] + a_prov_t[t_prov1]);
    for (m in 1:NT2) {
      int i = t_fila2[m];
      real p = 0;
      for (e in t_ini[m]:(t_ini[m] + t_len[m] - 1)) {
        p += t_ent_w[e] * inv_logit(eta_t[i] + a_prov_t[t_ent_prov[e]]);
      }
      t_k[i] ~ binomial(t_n[i], p);
    }
  }

  if (!prior_only) {
    // Predictor lineal sin la provincia, para todas las celdas a la vez.
    // a_edad[edad] coge la fila de cada celda: una matriz N x J.
    matrix[N, J] eta = rep_matrix(alpha', N) + mujer * b_mujer'
                       + a_casa[casa] + a_mes[mes] + a_edad[edad] + a_edu[edu]
                       + a_rec[rec] + a_ccaa[ccaa]
                       + a_ccaa_rec[ccaa_rec] + a_casa_rec[casa_rec] + off;

    // 1. Provincia conocida: multinomial con su provincia
    matrix[N1, J] eta1 = eta[fila1] + a_prov[prov1];
    for (n in 1:N1) {
      y[fila1[n]] ~ multinomial_logit(eta1[n]');
    }

    // 2. Sin provincia: la probabilidad es la media ponderada de las
    //    probabilidades de cada provincia de la CCAA
    for (m in 1:N2) {
      int i = fila2[m];
      vector[J] p = rep_vector(0, J);
      for (e in ini[m]:(ini[m] + len[m] - 1)) {
        p += ent_w[e] * softmax((eta[i] + a_prov[ent_prov[e]])');
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
  // Con participación calibrada: votos de cada celda = N * P(votar) * P(partido).
  // P(votar) es la de la casa media (sin efecto casa) para todas las filas.
  matrix[n_casa + 1, J] v_part = rep_matrix(0, n_casa + 1, J);
  real part_bruta;                      // % que diría "10" (casa media), antes de calibrar
  real delta;                           // desplazamiento de la calibración (logit)
  vector[P] p_vota;
  {
    vector[P] eta_pt;
    for (q in 1:P) {
      eta_pt[q] = alpha_t + P_mujer[q] * b_mujer_t + a_mes_t[mes_post]
                  + a_edad_t[P_edad[q]] + a_edu_t[P_edu[q]] + a_rec_t[P_rec[q]]
                  + a_ccaa_t[P_ccaa[q]] + a_prov_t[P_prov[q]];
    }
    part_bruta = 100 * dot_product(P_N, inv_logit(eta_pt)) / sum(P_N);
    delta = calibra(eta_pt, P_N, part_obj);
    p_vota = inv_logit(eta_pt + delta);
  }
  {
    for (q in 1:P) {
      row_vector[J] base = alpha' + P_mujer[q] * b_mujer' + a_mes[mes_post]
                           + a_edad[P_edad[q]] + a_edu[P_edu[q]] + a_rec[P_rec[q]]
                           + a_ccaa[P_ccaa[q]] + a_prov[P_prov[q]]
                           + a_ccaa_rec[(P_ccaa[q] - 1) * n_rec + P_rec[q]];
      row_vector[J] casa_media = rep_row_vector(0, J);
      for (h in 1:n_casa) {
        row_vector[J] ef_casa = a_casa[h] + a_casa_rec[(h - 1) * n_rec + P_rec[q]];
        casa_media += ef_casa / n_casa;
        row_vector[J] pv = prob_valida(base + ef_casa, P_mascara[q]);
        v_casa[h] += P_N[q] * pv;
        v_part[h] += P_N[q] * p_vota[q] * pv;
      }
      row_vector[J] pv_media = prob_valida(base + casa_media, P_mascara[q]);
      v_casa[n_casa + 1] += P_N[q] * pv_media;
      v_part[n_casa + 1] += P_N[q] * p_vota[q] * pv_media;
    }
    v_casa *= 100 / sum(P_N);
    v_part *= 100 / sum(P_N .* p_vota);
  }

  // Prior predictive check (solo con prior_only = 1): probabilidad del
  // partido más votado en cada celda de la muestra con provincia conocida
  vector[prior_only ? N1 : 0] p_max;
  if (prior_only) {
    for (n in 1:N1) {
      int i = fila1[n];
      row_vector[J] eta_i = alpha' + mujer[i] * b_mujer' + a_casa[casa[i]]
                            + a_mes[mes[i]] + a_edad[edad[i]] + a_edu[edu[i]]
                            + a_rec[rec[i]] + a_ccaa[ccaa[i]] + a_prov[prov1[n]]
                            + a_ccaa_rec[ccaa_rec[i]] + a_casa_rec[casa_rec[i]] + off[i];
      p_max[n] = max(softmax(eta_i'));
    }
  }
}
