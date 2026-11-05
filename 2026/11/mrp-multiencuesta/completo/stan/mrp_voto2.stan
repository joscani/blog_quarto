// Modelo completo, parte 3 (PRUEBA de parametrización): copia de
// stan/mrp_voto.stan con dos cambios, para ver si baja el número de pasos
// por iteración (el ajuste de mrp_voto.stan llegó a max_treedepth = 8, 255
// pasos, en las 4.000 iteraciones):
//  a. Todos los efectos por niveles suman cero entre niveles en cada partido
//     (como ya hacía z_casa): sin eso, una constante pasa de alpha a la media
//     de cada efecto sin cambiar las probabilidades (en mrp_ideol.stan esa
//     dirección daba ESS 155 y una cadena atascada).
//  b. Recuerdo y edad centrados (muchos datos y sd grandes); el resto sin
//     centrar. Las escalas esc_* devuelven a cada nivel la sd marginal del
//     grupo.
// Los datos son los mismos que los de stan/mrp_voto.stan.
//
// Cabecera original:
// MRP conjunto: modelo de voto (antes, parte del antiguo mrp.stan conjunto).
//
// Mismo modelo de voto que mrp.stan. Cambios:
//  1. Sin el modelo de participación: va en mrp_part.stan. Los dos modelos
//     no comparten parámetros, así que la posterior es la misma.
//  2. Verosimilitud con reduce_sum (hilos dentro de cada cadena).
//  3. Mezcla de provincias en escala log (log_sum_exp), no sumando probabilidades.
//  4. Prior de b_mujer como dato (s_b_mujer).
// La postestratificación con participación se hace en R con los dos ajustes.

functions {
#include funciones.stan
}

data {
  int<lower=0, upper=1> prior_only;
  int<lower=0, upper=1> sd_exponencial;
  real<lower=0> s_alpha;
  real<lower=0> s_b_mujer;
  real<lower=0> s_sd_rec;
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
  int<lower=1> n_rec;
  int<lower=1> n_ccaa;
  int<lower=1> n_prov;
  array[N] int<lower=1, upper=n_casa> casa;
  array[N] int<lower=1, upper=n_mes> mes;
  array[N] int<lower=1, upper=n_edad> edad;
  array[N] int<lower=1, upper=n_edu> edu;
  array[N] int<lower=1, upper=n_rec> rec;
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
  array[P] int<lower=1, upper=n_rec> P_rec;
  array[P] int<lower=1, upper=n_ccaa> P_ccaa;
  array[P] int<lower=1, upper=n_prov> P_prov;
  matrix<lower=0, upper=1>[P, J] P_mascara;
  int<lower=1, upper=n_mes> mes_post;
}

transformed data {
  array[N] int ccaa_rec;
  array[N] int casa_rec;
  for (i in 1:N) {
    ccaa_rec[i] = (ccaa[i] - 1) * n_rec + rec[i];
    casa_rec[i] = (casa[i] - 1) * n_rec + rec[i];
  }
  real esc_casa = sqrt(n_casa / (n_casa - 1.0));
  real esc_mes = n_mes > 1 ? sqrt(n_mes / (n_mes - 1.0)) : 1;
  real esc_edad = sqrt(n_edad / (n_edad - 1.0));
  real esc_edu = sqrt(n_edu / (n_edu - 1.0));
  real esc_rec = sqrt(n_rec / (n_rec - 1.0));
  real esc_ccaa = sqrt(n_ccaa / (n_ccaa - 1.0));
  real esc_prov = sqrt(n_prov / (n_prov - 1.0));
  real esc_cr = sqrt(n_ccaa * n_rec / (n_ccaa * n_rec - 1.0));
  real esc_hr = sqrt(n_casa * n_rec / (n_casa * n_rec - 1.0));

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
  array[J] sum_to_zero_vector[n_mes] z_mes;
  array[J] sum_to_zero_vector[n_edad] c_edad;      // centrado
  array[J] sum_to_zero_vector[n_edu] z_edu;
  array[J] sum_to_zero_vector[n_rec] c_rec;        // centrado
  array[J] sum_to_zero_vector[n_ccaa] z_ccaa;
  array[J] sum_to_zero_vector[n_prov] z_prov;
  array[J] sum_to_zero_vector[n_ccaa * n_rec] z_ccaa_rec;
  array[J] sum_to_zero_vector[n_casa * n_rec] z_casa_rec;

  vector<lower=0>[J] sd_casa;
  vector<lower=0>[J] sd_mes;
  vector<lower=0>[J] sd_edad;
  vector<lower=0>[J] sd_edu;
  vector<lower=0>[J] sd_rec;
  vector<lower=0>[J] sd_ccaa;
  vector<lower=0>[J] sd_prov;
  vector<lower=0>[J] sd_ccaa_rec;
  vector<lower=0>[J] sd_casa_rec;
}

transformed parameters {
  matrix[n_casa, J] a_casa;
  for (k in 1:J) {
    a_casa[, k] = esc_casa * sd_casa[k] * z_casa[k];
  }
  a_casa = centra(a_casa);
  matrix[n_mes, J] a_mes;
  matrix[n_edad, J] a_edad;
  matrix[n_edu, J] a_edu;
  matrix[n_rec, J] a_rec;
  matrix[n_ccaa, J] a_ccaa;
  matrix[n_prov, J] a_prov;
  matrix[n_ccaa * n_rec, J] a_ccaa_rec;
  matrix[n_casa * n_rec, J] a_casa_rec;
  for (k in 1:J) {
    a_mes[, k] = esc_mes * sd_mes[k] * z_mes[k];
    a_edad[, k] = c_edad[k];
    a_edu[, k] = esc_edu * sd_edu[k] * z_edu[k];
    a_rec[, k] = c_rec[k];
    a_ccaa[, k] = esc_ccaa * sd_ccaa[k] * z_ccaa[k];
    a_prov[, k] = esc_prov * sd_prov[k] * z_prov[k];
    a_ccaa_rec[, k] = esc_cr * sd_ccaa_rec[k] * z_ccaa_rec[k];
    a_casa_rec[, k] = esc_hr * sd_casa_rec[k] * z_casa_rec[k];
  }
  // centrar entre partidos conserva la suma cero entre niveles
  a_mes = centra(a_mes);
  a_edad = centra(a_edad);
  a_edu = centra(a_edu);
  a_rec = centra(a_rec);
  a_ccaa = centra(a_ccaa);
  a_prov = centra(a_prov);
  a_ccaa_rec = centra(a_ccaa_rec);
  a_casa_rec = centra(a_casa_rec);
}

model {
  alpha ~ normal(0, s_alpha);
  b_mujer ~ normal(0, s_b_mujer);
  for (k in 1:J) z_casa[k] ~ std_normal();
  for (k in 1:J) {
    z_mes[k] ~ std_normal();
    c_edad[k] ~ normal(0, esc_edad * sd_edad[k]);
    z_edu[k] ~ std_normal();
    c_rec[k] ~ normal(0, esc_rec * sd_rec[k]);
    z_ccaa[k] ~ std_normal();
    z_prov[k] ~ std_normal();
    z_ccaa_rec[k] ~ std_normal();
    z_casa_rec[k] ~ std_normal();
  }
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
    target += reduce_sum(voto_parcial, filas, grainsize,
                         y, mujer, casa, mes, edad, edu, rec, ccaa, ccaa_rec, casa_rec,
                         off, sin_aa, j_aa, prov_fila, ini_fila, len_fila, ent_prov, log_ent_w,
                         alpha, b_mujer, a_casa, a_mes, a_edad, a_edu, a_rec, a_ccaa,
                         a_prov, a_ccaa_rec, a_casa_rec);
  }
}

generated quantities {
  // Voto válido (%) en España según cada casa (filas 1..n_casa) y según la
  // casa media (fila n_casa + 1), si votara todo el mundo
  matrix[n_casa + 1, J] v_casa = rep_matrix(0, n_casa + 1, J);
  for (q in 1:P) {
    row_vector[J] base = alpha' + P_mujer[q] * b_mujer' + a_mes[mes_post]
                         + a_edad[P_edad[q]] + a_edu[P_edu[q]] + a_rec[P_rec[q]]
                         + a_ccaa[P_ccaa[q]] + a_prov[P_prov[q]]
                         + a_ccaa_rec[(P_ccaa[q] - 1) * n_rec + P_rec[q]];
    row_vector[J] casa_media = rep_row_vector(0, J);
    for (h in 1:n_casa) {
      row_vector[J] ef_casa = a_casa[h] + a_casa_rec[(h - 1) * n_rec + P_rec[q]];
      casa_media += ef_casa / n_casa;
      v_casa[h] += P_N[q] * prob_valida(base + ef_casa, P_mascara[q]);
    }
    v_casa[n_casa + 1] += P_N[q] * prob_valida(base + casa_media, P_mascara[q]);
  }
  v_casa *= 100 / sum(P_N);
}
