// Modelo completo, parte 2: nivel de participación en 3 niveles
// (1 = improbable, 0-4; 2 = probable, 5-9; 3 = seguro, 10) según el perfil y
// la ideología declarada. Logit ordinal, una fila por entrevistado. Solo las
// casas que preguntan la probabilidad de votar (CIS, 40dB, More in Common).
//
// Respecto a mrp_part.stan (binomial de "10, seguro que voto"):
//   - tres niveles en vez de dos;
//   - la ideología (5 tramos) como predictor;
//   - sin provincia (sd 0,11 en mrp_part.stan): así no hay mezcla de
//     provincias y el modelo se vectoriza, como mrp_ideol.stan;
//   - umbrales propios de cada casa en vez de un efecto casa: el CIS tiene
//     la mitad de "probables" que 40dB y More in Common (13,5 % frente a
//     26-29 %) con los improbables parecidos (8-11 %), y un desplazamiento
//     común no puede reproducir eso. La casa media usa la media de los
//     umbrales.
//
// Mes, edad, estudios, recuerdo e ideología suman cero (si no, una constante
// pasa de los efectos a los umbrales). Mes, estudios y CCAA sin centrar; edad
// y recuerdo centrados (muchos datos, sd lejos de 0).

data {
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  real<lower=0> s_c;
  real<lower=0> s_b;
  int<lower=1> N;
  array[N] int<lower=1, upper=3> y;
  vector[N] mujer;
  int<lower=2> n_casa;
  int<lower=2> n_mes;
  int<lower=2> n_edad;
  int<lower=2> n_edu;
  int<lower=2> n_rec;
  int<lower=2> n_ccaa;
  int<lower=2> n_ideol;
  array[N] int<lower=1, upper=n_casa> casa;
  array[N] int<lower=1, upper=n_mes> mes;
  array[N] int<lower=1, upper=n_edad> edad;
  array[N] int<lower=1, upper=n_edu> edu;
  array[N] int<lower=1, upper=n_rec> rec;
  array[N] int<lower=1, upper=n_ccaa> ccaa;
  array[N] int<lower=1, upper=n_ideol> ideol;
}

transformed data {
  // con suma 0, la sd marginal de cada nivel vuelve a ser la sd del grupo
  real esc_mes = sqrt(n_mes / (n_mes - 1.0));
  real esc_edad = sqrt(n_edad / (n_edad - 1.0));
  real esc_edu = sqrt(n_edu / (n_edu - 1.0));
  real esc_rec = sqrt(n_rec / (n_rec - 1.0));
}

parameters {
  array[n_casa] ordered[2] c;            // umbrales de cada casa
  real b_mujer;
  sum_to_zero_vector[n_ideol] a_ideol;   // efecto fijo de la ideología
  sum_to_zero_vector[n_mes] z_mes;
  sum_to_zero_vector[n_edad] a_edad;
  sum_to_zero_vector[n_edu] z_edu;
  sum_to_zero_vector[n_rec] a_rec;
  vector[n_ccaa] z_ccaa;
  real<lower=0> sd_mes;
  real<lower=0> sd_edad;
  real<lower=0> sd_edu;
  real<lower=0> sd_rec;
  real<lower=0> sd_ccaa;
}

transformed parameters {
  vector[n_mes] a_mes = esc_mes * sd_mes * z_mes;
  vector[n_edu] a_edu = esc_edu * sd_edu * z_edu;
  vector[n_ccaa] a_ccaa = sd_ccaa * z_ccaa;
}

model {
  for (h in 1:n_casa) {
    c[h] ~ normal(0, s_c);
  }
  b_mujer ~ normal(0, s_b);
  a_ideol ~ normal(0, s_b);
  z_mes ~ std_normal();
  a_edad ~ normal(0, esc_edad * sd_edad);
  z_edu ~ std_normal();
  a_rec ~ normal(0, esc_rec * sd_rec);
  z_ccaa ~ std_normal();
  sd_rec ~ normal(0, s_sd_rec);
  sd_mes ~ normal(0, s_sd_main);
  sd_edad ~ normal(0, s_sd_main);
  sd_edu ~ normal(0, s_sd_main);
  sd_ccaa ~ normal(0, s_sd_main);

  vector[N] eta = mujer * b_mujer + a_ideol[ideol] + a_mes[mes] + a_edad[edad]
                  + a_edu[edu] + a_rec[rec] + a_ccaa[ccaa];
  // c[casa] da a cada fila los umbrales de su casa (vectorizado)
  y ~ ordered_logistic(eta, c[casa]);
}

generated quantities {
  // umbrales de la casa media
  vector[2] c_media = rep_vector(0, 2);
  for (h in 1:n_casa) {
    c_media += c[h] / n_casa;
  }
}
