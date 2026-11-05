// Modelo completo, parte 1: ideología en 5 tramos (izq, cizq, centro, cder,
// der) según la celda. Logit ordinal acumulado con los mismos efectos que
// mrp_part.stan: casa (suma cero), mes, sexo, edad, estudios, recuerdo, CCAA y
// provincia.
//
// Datos agregados: cada fila es una celda con el recuento de cada tramo
// (multinomial con las probabilidades del ordinal). Las filas sin provincia
// (More in Common, Ateneo) mezclan las probabilidades de las provincias de su
// CCAA con los pesos de la tabla de postestratificación.
//
// Forma centrada en los efectos con pocos niveles y muchos datos (casa, mes,
// edad, estudios, recuerdo) y sin centrar en CCAA y provincia, como
// mrp_part.stan. El predictor no lleva intercepto: lo hacen los umbrales c.
// Casa, mes, edad, estudios y recuerdo suman cero: sin esa restricción, una
// constante pasa de los efectos a los umbrales sin cambiar las probabilidades
// (primera prueba: correlación 0,96 entre c[1] y la media de a_rec, ESS de
// los umbrales 155 y una cadena atascada con paso 0,004).
// Casa, mes y estudios van sin centrar: sus sd se van a casi 0 (el mes y los
// estudios apenas mueven la ideología) y en forma centrada hacían un embudo
// (segunda prueba: 32 % de transiciones divergentes, con sd_edu mediana
// 0,0007 en las divergentes frente a 0,010 en las demás).
// Verosimilitud con reduce_sum (hilos dentro de cada cadena), como
// mrp_voto.stan. Para eso, cada fila lleva su lista de provincias: una con
// peso 1 si la provincia se conoce, las de su CCAA si no (transformed data).

functions {
  // P(tramo t), t = 1..T, de un logit ordinal acumulado
  vector prob_ord(real eta, vector c) {
    int T = rows(c) + 1;
    vector[T] p;
    p[1] = 1 - inv_logit(eta - c[1]);
    for (t in 2:(T - 1)) {
      p[t] = inv_logit(eta - c[t - 1]) - inv_logit(eta - c[t]);
    }
    p[T] = inv_logit(eta - c[T - 1]);
    return p;
  }

  // log-verosimilitud de las filas start..end (y_trozo son sus recuentos):
  // multinomial con la mezcla de las probabilidades de sus provincias
  real ideol_parcial(array[,] int y_trozo, int start, int end, vector eta,
                     vector a_prov, vector c, array[] int ini_f,
                     array[] int len_f, array[] int prov_f, vector w_f) {
    int T = rows(c) + 1;
    real lp = 0;
    for (i in start:end) {
      vector[T] p = rep_vector(0, T);
      for (l in 1:len_f[i]) {
        int e = ini_f[i] + l - 1;
        p += w_f[e] * prob_ord(eta[i] + a_prov[prov_f[e]], c);
      }
      lp += multinomial_lpmf(y_trozo[i - start + 1] | p);
    }
    return lp;
  }
}

data {
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  real<lower=0> s_c;
  real<lower=0> s_b_mujer;
  int<lower=1> grainsize;               // filas por trozo de reduce_sum
  int<lower=1> N;
  int<lower=3> T;
  array[N, T] int<lower=0> y;
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
  int<lower=1, upper=n_mes> mes_post;
}

transformed data {
  // provincias de cada fila, en un solo juego de índices: fila1 con su
  // provincia (peso 1) y fila2 con las de su CCAA
  int EF = N1 + E;
  array[N] int ini_f;
  array[N] int len_f;
  array[EF] int prov_f;
  vector[EF] w_f;
  {
    int pos = 1;
    for (m in 1:N1) {
      ini_f[fila1[m]] = pos;
      len_f[fila1[m]] = 1;
      prov_f[pos] = prov1[m];
      w_f[pos] = 1;
      pos += 1;
    }
    for (m in 1:N2) {
      ini_f[fila2[m]] = pos;
      len_f[fila2[m]] = len[m];
      for (l in 1:len[m]) {
        prov_f[pos] = ent_prov[ini[m] + l - 1];
        w_f[pos] = ent_w[ini[m] + l - 1];
        pos += 1;
      }
    }
  }
  // con suma 0, la sd marginal de cada nivel vuelve a ser la sd del grupo
  real esc_casa = sqrt(n_casa / (n_casa - 1.0));
  real esc_mes = n_mes > 1 ? sqrt(n_mes / (n_mes - 1.0)) : 1;
  real esc_edad = sqrt(n_edad / (n_edad - 1.0));
  real esc_edu = sqrt(n_edu / (n_edu - 1.0));
  real esc_rec = sqrt(n_rec / (n_rec - 1.0));
}

parameters {
  ordered[T - 1] c;
  real b_mujer;
  sum_to_zero_vector[n_casa] z_casa;
  sum_to_zero_vector[n_mes] z_mes;
  sum_to_zero_vector[n_edad] a_edad;
  sum_to_zero_vector[n_edu] z_edu;
  sum_to_zero_vector[n_rec] a_rec;
  vector[n_ccaa] z_ccaa;
  vector[n_prov] z_prov;
  real<lower=0> sd_casa;
  real<lower=0> sd_mes;
  real<lower=0> sd_edad;
  real<lower=0> sd_edu;
  real<lower=0> sd_rec;
  real<lower=0> sd_ccaa;
  real<lower=0> sd_prov;
}

transformed parameters {
  vector[n_casa] a_casa = esc_casa * sd_casa * z_casa;
  vector[n_mes] a_mes = esc_mes * sd_mes * z_mes;
  vector[n_edu] a_edu = esc_edu * sd_edu * z_edu;
  vector[n_ccaa] a_ccaa = sd_ccaa * z_ccaa;
  vector[n_prov] a_prov = sd_prov * z_prov;
}

model {
  c ~ normal(0, s_c);
  b_mujer ~ normal(0, s_b_mujer);
  z_casa ~ std_normal();
  z_mes ~ std_normal();
  z_edu ~ std_normal();
  a_edad ~ normal(0, esc_edad * sd_edad);
  a_rec ~ normal(0, esc_rec * sd_rec);
  z_ccaa ~ std_normal();
  z_prov ~ std_normal();
  sd_rec ~ normal(0, s_sd_rec);
  sd_casa ~ normal(0, s_sd_main);
  sd_mes ~ normal(0, s_sd_main);
  sd_edad ~ normal(0, s_sd_main);
  sd_edu ~ normal(0, s_sd_main);
  sd_ccaa ~ normal(0, s_sd_main);
  sd_prov ~ normal(0, s_sd_main);

  vector[N] eta = mujer * b_mujer + a_casa[casa] + a_mes[mes] + a_edad[edad]
                  + a_edu[edu] + a_rec[rec] + a_ccaa[ccaa];
  target += reduce_sum(ideol_parcial, y, grainsize, eta, a_prov, c, ini_f,
                       len_f, prov_f, w_f);
}

generated quantities {
  // reparto nacional de la ideología, casa media (efecto casa 0) y último mes
  vector[T] ideol_nac = rep_vector(0, T);
  for (q in 1:P) {
    real eta_q = P_mujer[q] * b_mujer + a_mes[mes_post] + a_edad[P_edad[q]]
                 + a_edu[P_edu[q]] + a_rec[P_rec[q]] + a_ccaa[P_ccaa[q]]
                 + a_prov[P_prov[q]];
    ideol_nac += P_N[q] * prob_ord(eta_q, c);
  }
  ideol_nac = 100 * ideol_nac / sum(P_N);
}
