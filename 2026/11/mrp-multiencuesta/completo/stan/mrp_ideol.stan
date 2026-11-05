// Modelo completo, parte 1: ideología en 5 tramos (izq, cizq, centro, cder,
// der) según el perfil. Versión sencilla: logit ordinal con sexo, casa, edad,
// recuerdo y CCAA, una fila por entrevistado.
//
// Por qué sencilla (pruebas del 2026-11-02 con mrp_ideol_prov.stan, que
// llevaba además mes, estudios y provincia con mezcla de provincias):
//   - el mes y los estudios apenas mueven la ideología (sd 0,001-0,01) y
//     hacían un embudo: 32 % de transiciones divergentes;
//   - la provincia aporta poco (sd 0,11) y obligaba a mezclar provincias en
//     More in Common y Ateneo: 22-40 min por cadena.
// En el modelo de voto, la provincia y los estudios siguen entrando: aquí
// solo se simplifica cuánta gente de cada ideología hay en cada celda.
//
// Casa, edad y recuerdo suman cero (sin eso, una constante pasa de los efectos
// a los umbrales c). Casa y CCAA sin centrar (pocas casas; sd de la CCAA
// pequeña); edad y recuerdo centrados (muchos datos, sd lejos de 0).

data {
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  real<lower=0> s_c;
  real<lower=0> s_b_mujer;
  int<lower=1> N;                        // entrevistados
  int<lower=3> T;                        // tramos
  array[N] int<lower=1, upper=T> y;
  vector[N] mujer;
  int<lower=2> n_casa;
  int<lower=2> n_edad;
  int<lower=2> n_rec;
  int<lower=2> n_ccaa;
  array[N] int<lower=1, upper=n_casa> casa;
  array[N] int<lower=1, upper=n_edad> edad;
  array[N] int<lower=1, upper=n_rec> rec;
  array[N] int<lower=1, upper=n_ccaa> ccaa;

  // población, para el reparto nacional en generated quantities
  int<lower=1> P;
  vector<lower=0>[P] P_N;
  vector[P] P_mujer;
  array[P] int<lower=1, upper=n_edad> P_edad;
  array[P] int<lower=1, upper=n_rec> P_rec;
  array[P] int<lower=1, upper=n_ccaa> P_ccaa;
}

transformed data {
  // con suma 0, la sd marginal de cada nivel vuelve a ser la sd del grupo
  real esc_casa = sqrt(n_casa / (n_casa - 1.0));
  real esc_edad = sqrt(n_edad / (n_edad - 1.0));
  real esc_rec = sqrt(n_rec / (n_rec - 1.0));
}

parameters {
  ordered[T - 1] c;
  real b_mujer;
  sum_to_zero_vector[n_casa] z_casa;
  sum_to_zero_vector[n_edad] a_edad;
  sum_to_zero_vector[n_rec] a_rec;
  vector[n_ccaa] z_ccaa;
  real<lower=0> sd_casa;
  real<lower=0> sd_edad;
  real<lower=0> sd_rec;
  real<lower=0> sd_ccaa;
}

transformed parameters {
  vector[n_casa] a_casa = esc_casa * sd_casa * z_casa;
  vector[n_ccaa] a_ccaa = sd_ccaa * z_ccaa;
}

model {
  c ~ normal(0, s_c);
  b_mujer ~ normal(0, s_b_mujer);
  z_casa ~ std_normal();
  a_edad ~ normal(0, esc_edad * sd_edad);
  a_rec ~ normal(0, esc_rec * sd_rec);
  z_ccaa ~ std_normal();
  sd_rec ~ normal(0, s_sd_rec);
  sd_casa ~ normal(0, s_sd_main);
  sd_edad ~ normal(0, s_sd_main);
  sd_ccaa ~ normal(0, s_sd_main);

  y ~ ordered_logistic(
    mujer * b_mujer + a_casa[casa] + a_edad[edad] + a_rec[rec] + a_ccaa[ccaa],
    c
  );
}

generated quantities {
  // reparto nacional de la ideología, casa media (efecto casa 0), en %
  vector[T] ideol_nac = rep_vector(0, T);
  for (q in 1:P) {
    real eta_q = P_mujer[q] * b_mujer + a_edad[P_edad[q]] + a_rec[P_rec[q]]
                 + a_ccaa[P_ccaa[q]];
    ideol_nac[1] += P_N[q] * (1 - inv_logit(eta_q - c[1]));
    for (t in 2:(T - 1)) {
      ideol_nac[t] += P_N[q] * (inv_logit(eta_q - c[t - 1]) - inv_logit(eta_q - c[t]));
    }
    ideol_nac[T] += P_N[q] * inv_logit(eta_q - c[T - 1]);
  }
  ideol_nac = 100 * ideol_nac / sum(P_N);
}
