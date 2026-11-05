// Modelo completo, parte 3, PROTOTIPO solo con el CIS, versión glm: el mismo
// modelo que mrp_voto_cis.stan, pero con una fila por entrevistado y la
// verosimilitud en una sola llamada a categorical_logit_glm (gradientes
// analíticos), en vez del bucle multinomial_logit fila a fila. Las celdas
// casi no agregaban: 1,2 entrevistas por fila en el modelo de las 4 casas.
//
// x es la matriz de diseño (una columna por nivel de cada efecto: sexo, mes,
// edad, estudios, recuerdo, CCAA, provincia, ideología, nivel) y beta se arma
// con los efectos en el mismo orden. El offset de los partidos regionales
// (-20 fuera de su territorio) solo depende de la CCAA: se suma a las filas
// de CCAA de beta (off_ccaa).

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
  array[N] int<lower=1, upper=J> y;
  int<lower=2> n_mes;
  int<lower=2> n_edad;
  int<lower=2> n_edu;
  int<lower=2> n_rec;
  int<lower=2> n_ccaa;
  int<lower=2> n_prov;
  int<lower=2> n_ideol;
  int<lower=2> n_nivel;
  int<lower=1> K;                       // columnas de x
  matrix[N, K] x;                       // sexo, mes, edad, edu, rec, ccaa, prov, ideol, nivel
  matrix[n_ccaa, J] off_ccaa;           // 0 en su territorio, -20 fuera
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
  if (K != 1 + n_mes + n_edad + n_edu + n_rec + n_ccaa + n_prov + n_ideol + n_nivel) {
    reject("K no cuadra con los niveles");
  }
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

  // beta en el mismo orden que las columnas de x
  matrix[K, J] beta = append_row(
    b_mujer',
    append_row(a_mes,
    append_row(a_edad,
    append_row(a_edu,
    append_row(a_rec,
    append_row(a_ccaa + off_ccaa,
    append_row(a_prov,
    append_row(a_ideol, a_nivel))))))));
  y ~ categorical_logit_glm(x, alpha, beta);
}
