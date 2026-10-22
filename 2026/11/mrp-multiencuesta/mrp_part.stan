// MRP conjunto: modelo de participación (antes, parte de mrp.stan; ahora en obsoleto/).
//
// Binomial de decir "10, seguro que voto", con efecto casa (solo las casas que
// lo preguntan), mes, sexo, edad, estudios, recuerdo, CCAA y provincia. Mismo
// modelo que la parte de participación de mrp.stan, con dos cambios:
//  1. Mezcla de provincias en escala log.
//  2. Priors de alpha_t y b_mujer_t como datos.
//  3. Forma centrada en los efectos con pocos niveles y muchos datos: casa,
//     mes, edad, estudios y recuerdo. CCAA y provincia, sin centrar. La
//     versión sin centrar (obsoleto/mrp_part_no_centrada.stan) llegaba a
//     max_treedepth en el 61 % de las iteraciones; la posterior es la misma.

functions {
#include funciones.stan
}

data {
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  real m_alpha_t;                       // logit del "10" esperado
  real<lower=0> s_alpha_t;
  real<lower=0> s_b_mujer_t;
  int<lower=1> n_mes;
  int<lower=1> n_edad;
  int<lower=1> n_edu;
  int<lower=1> n_rec;
  int<lower=1> n_ccaa;
  int<lower=1> n_prov;

  int<lower=0> NT;
  array[NT] int<lower=1> t_n;
  array[NT] int<lower=0> t_k;
  vector[NT] t_mujer;
  int<lower=2> n_casa_t;
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
  real<lower=0, upper=1> part_obj;

  int<lower=1> P;
  vector<lower=0>[P] P_N;
  vector[P] P_mujer;
  array[P] int<lower=1, upper=n_edad> P_edad;
  array[P] int<lower=1, upper=n_edu> P_edu;
  array[P] int<lower=1, upper=n_rec> P_rec;
  array[P] int<lower=1, upper=n_ccaa> P_ccaa;
  array[P] int<lower=1, upper=n_prov> P_prov;
  int<lower=1, upper=n_mes> mes_post;
}

transformed data {
  real esc_casa_t = sqrt(n_casa_t / (n_casa_t - 1.0));
  vector[TE] log_t_ent_w = log(t_ent_w);
}

parameters {
  real alpha_t;
  real b_mujer_t;
  sum_to_zero_vector[n_casa_t] a_casa_t;
  vector[n_mes] a_mes_t;
  vector[n_edad] a_edad_t;
  vector[n_edu] a_edu_t;
  vector[n_rec] a_rec_t;
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
  vector[n_ccaa] a_ccaa_t = sd_ccaa_t * z_ccaa_t;
  vector[n_prov] a_prov_t = sd_prov_t * z_prov_t;
}

model {
  alpha_t ~ normal(m_alpha_t, s_alpha_t);
  b_mujer_t ~ normal(0, s_b_mujer_t);
  // esc_casa_t: con suma 0, la sd marginal de cada casa vuelve a ser sd_casa_t
  a_casa_t ~ normal(0, esc_casa_t * sd_casa_t);
  a_mes_t ~ normal(0, sd_mes_t);
  a_edad_t ~ normal(0, sd_edad_t);
  a_edu_t ~ normal(0, sd_edu_t);
  a_rec_t ~ normal(0, sd_rec_t);
  z_ccaa_t ~ std_normal();
  z_prov_t ~ std_normal();
  sd_rec_t ~ normal(0, s_sd_rec);
  sd_casa_t ~ normal(0, s_sd_main);
  sd_mes_t ~ normal(0, s_sd_main);
  sd_edad_t ~ normal(0, s_sd_main);
  sd_edu_t ~ normal(0, s_sd_main);
  sd_ccaa_t ~ normal(0, s_sd_main);
  sd_prov_t ~ normal(0, s_sd_main);

  vector[NT] eta_t = alpha_t + t_mujer * b_mujer_t + a_casa_t[t_casa] + a_mes_t[t_mes]
                     + a_edad_t[t_edad] + a_edu_t[t_edu] + a_rec_t[t_rec] + a_ccaa_t[t_ccaa];
  t_k[t_fila1] ~ binomial_logit(t_n[t_fila1], eta_t[t_fila1] + a_prov_t[t_prov1]);
  // Mezcla en escala log: logit(p) = log p - log(1 - p), con
  // log p = log_sum_exp(log w + log_inv_logit(.)), igual para 1 - p
  for (m in 1:NT2) {
    int i = t_fila2[m];
    vector[t_len[m]] l1;
    vector[t_len[m]] l0;
    for (l in 1:t_len[m]) {
      int e = t_ini[m] + l - 1;
      real x = eta_t[i] + a_prov_t[t_ent_prov[e]];
      l1[l] = log_t_ent_w[e] + log_inv_logit(x);
      l0[l] = log_t_ent_w[e] + log1m_inv_logit(x);
    }
    t_k[i] ~ binomial_logit(t_n[i], log_sum_exp(l1) - log_sum_exp(l0));
  }
}

generated quantities {
  // Participación de la casa media en la población, antes y después de calibrar
  real part_bruta;
  real delta;
  {
    vector[P] eta_pt;
    for (q in 1:P) {
      eta_pt[q] = alpha_t + P_mujer[q] * b_mujer_t + a_mes_t[mes_post]
                  + a_edad_t[P_edad[q]] + a_edu_t[P_edu[q]] + a_rec_t[P_rec[q]]
                  + a_ccaa_t[P_ccaa[q]] + a_prov_t[P_prov[q]];
    }
    part_bruta = 100 * dot_product(P_N, inv_logit(eta_pt)) / sum(P_N);
    delta = calibra(eta_pt, P_N, part_obj);
  }
}
