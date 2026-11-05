// Modelo completo, parte 3, PROTOTIPO solo con el CIS: voto según el perfil,
// la ideología y el nivel de participación declarados.
//
// Herramienta de trabajo (decisión de José, 2026-11-02): sirve para escribir
// y probar la cadena y la postestratificación rápido. El resultado del post
// sale del modelo con las 4 casas.
//
// Parte de stan/mrp_cis.stan (sin casa ni filas mezcla: el CIS siempre trae
// la provincia). Cambios:
//  - respuesta: voto + simpatía, con BLANCO como categoría propia;
//  - predictores nuevos: ideología (5 tramos) y nivel de participación
//    (improbable, probable, seguro), con un efecto por partido;
//  - sin la interacción CCAA x recuerdo (mediana de |efecto| 0,003 en
//    mrp_voto.stan);
//  - sin el modelo de participación (va en mrp_part3.stan);
//  - todos los efectos por niveles suman cero entre niveles en cada partido
//    (si no, una constante pasa de alpha a la media del efecto); recuerdo y
//    edad centrados, el resto sin centrar.
// Como en mrp_cis.stan, sin partido de referencia: cada efecto se centra
// entre partidos (centra), y los partidos regionales llevan un offset de -20
// fuera de su territorio.

functions {
  // resta a cada fila su media: en cada nivel, el efecto suma 0 entre partidos
  matrix centra(matrix a) {
    int J = cols(a);
    return a - rep_matrix(a * rep_vector(1.0 / J, J), J);
  }
}

data {
  real<lower=0> s_alpha;
  real<lower=0> s_b_mujer;
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  int<lower=1> N;
  int<lower=2> J;
  array[N, J] int<lower=0> y;
  vector[N] mujer;
  int<lower=2> n_mes;
  int<lower=2> n_edad;
  int<lower=2> n_edu;
  int<lower=2> n_rec;
  int<lower=2> n_ccaa;
  int<lower=2> n_prov;
  int<lower=2> n_ideol;
  int<lower=2> n_nivel;
  array[N] int<lower=1, upper=n_mes> mes;
  array[N] int<lower=1, upper=n_edad> edad;
  array[N] int<lower=1, upper=n_edu> edu;
  array[N] int<lower=1, upper=n_rec> rec;
  array[N] int<lower=1, upper=n_ccaa> ccaa;
  array[N] int<lower=1, upper=n_prov> prov;
  array[N] int<lower=1, upper=n_ideol> ideol;
  array[N] int<lower=1, upper=n_nivel> nivel;
  matrix[N, J] off;                     // 0 en su territorio, -20 fuera
}

transformed data {
  real esc_mes = sqrt(n_mes / (n_mes - 1.0));
  real esc_edad = sqrt(n_edad / (n_edad - 1.0));
  real esc_edu = sqrt(n_edu / (n_edu - 1.0));
  real esc_rec = sqrt(n_rec / (n_rec - 1.0));
  real esc_ccaa = sqrt(n_ccaa / (n_ccaa - 1.0));
  real esc_prov = sqrt(n_prov / (n_prov - 1.0));
  real esc_ideol = sqrt(n_ideol / (n_ideol - 1.0));
  real esc_nivel = sqrt(n_nivel / (n_nivel - 1.0));
}

parameters {
  sum_to_zero_vector[J] alpha;
  sum_to_zero_vector[J] b_mujer;
  array[J] sum_to_zero_vector[n_mes] z_mes;
  array[J] sum_to_zero_vector[n_edad] c_edad;     // centrado
  array[J] sum_to_zero_vector[n_edu] z_edu;
  array[J] sum_to_zero_vector[n_rec] c_rec;       // centrado
  array[J] sum_to_zero_vector[n_ccaa] z_ccaa;
  array[J] sum_to_zero_vector[n_prov] z_prov;
  array[J] sum_to_zero_vector[n_ideol] z_ideol;
  array[J] sum_to_zero_vector[n_nivel] z_nivel;

  vector<lower=0>[J] sd_mes;
  vector<lower=0>[J] sd_edad;
  vector<lower=0>[J] sd_edu;
  vector<lower=0>[J] sd_rec;
  vector<lower=0>[J] sd_ccaa;
  vector<lower=0>[J] sd_prov;
  vector<lower=0>[J] sd_ideol;
  vector<lower=0>[J] sd_nivel;
}

transformed parameters {
  matrix[n_mes, J] a_mes;
  matrix[n_edad, J] a_edad;
  matrix[n_edu, J] a_edu;
  matrix[n_rec, J] a_rec;
  matrix[n_ccaa, J] a_ccaa;
  matrix[n_prov, J] a_prov;
  matrix[n_ideol, J] a_ideol;
  matrix[n_nivel, J] a_nivel;
  for (k in 1:J) {
    a_mes[, k] = esc_mes * sd_mes[k] * z_mes[k];
    a_edad[, k] = c_edad[k];
    a_edu[, k] = esc_edu * sd_edu[k] * z_edu[k];
    a_rec[, k] = c_rec[k];
    a_ccaa[, k] = esc_ccaa * sd_ccaa[k] * z_ccaa[k];
    a_prov[, k] = esc_prov * sd_prov[k] * z_prov[k];
    a_ideol[, k] = esc_ideol * sd_ideol[k] * z_ideol[k];
    a_nivel[, k] = esc_nivel * sd_nivel[k] * z_nivel[k];
  }
  // centrar entre partidos conserva la suma cero entre niveles
  a_mes = centra(a_mes);
  a_edad = centra(a_edad);
  a_edu = centra(a_edu);
  a_rec = centra(a_rec);
  a_ccaa = centra(a_ccaa);
  a_prov = centra(a_prov);
  a_ideol = centra(a_ideol);
  a_nivel = centra(a_nivel);
}

model {
  alpha ~ normal(0, s_alpha);
  b_mujer ~ normal(0, s_b_mujer);
  for (k in 1:J) {
    z_mes[k] ~ std_normal();
    c_edad[k] ~ normal(0, esc_edad * sd_edad[k]);
    z_edu[k] ~ std_normal();
    c_rec[k] ~ normal(0, esc_rec * sd_rec[k]);
    z_ccaa[k] ~ std_normal();
    z_prov[k] ~ std_normal();
    z_ideol[k] ~ std_normal();
    z_nivel[k] ~ std_normal();
  }
  sd_rec ~ normal(0, s_sd_rec);
  sd_ideol ~ normal(0, s_sd_rec);       // la ideología puede pesar casi como el recuerdo
  sd_mes ~ normal(0, s_sd_main);
  sd_edad ~ normal(0, s_sd_main);
  sd_edu ~ normal(0, s_sd_main);
  sd_ccaa ~ normal(0, s_sd_main);
  sd_prov ~ normal(0, s_sd_main);
  sd_nivel ~ normal(0, s_sd_main);

  matrix[N, J] eta = rep_matrix(alpha', N) + mujer * b_mujer'
                     + a_mes[mes] + a_edad[edad] + a_edu[edu] + a_rec[rec]
                     + a_ccaa[ccaa] + a_prov[prov] + a_ideol[ideol]
                     + a_nivel[nivel] + off;
  for (i in 1:N) {
    y[i] ~ multinomial_logit(eta[i]');
  }
}
