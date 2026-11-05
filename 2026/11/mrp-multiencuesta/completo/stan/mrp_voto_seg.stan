// Modelo de voto con seguro / dudoso (decisión de José, 2026-11-03).
//
// Copia de completo/stan/mrp_voto3.stan (sin CCAA x recuerdo; suma cero entre
// niveles) con estos cambios:
//  - todos los efectos en forma no centrada, como en el (II). 2026-11-03: con
//    edad y recuerdo centrados (como mrp_voto3), el port a NumPyro no
//    convergió (una cadena atascada con sd_edad de BNG cerca de 0, Rhat 1,59);
//  - respuesta: voto + simpatía, con BLANCO como categoría propia
//    (datos de completo/10_datos_voto_seg.R);
//  - efecto de ser dudoso (no dice "10, seguro que voto") por partido, que
//    cambia con el recuerdo y con la edad, con pooling parcial. El seguro es
//    la referencia: eta_dudoso = eta_seguro + d, con
//    d = b_dud + a_dud_rec[rec] + a_dud_edad[edad]. d es el Δ de
//    15_delta_seguro.R, ahora dentro del modelo grande;
//  - filas sin seg (Ateneo y quien no contesta): mezcla seguro / dudoso con
//    pesos fijos, junto con la de provincias (entradas provincia x seg);
//  - sin generated quantities: la postestratificación, con P(seguro) de
//    stan/mrp_part.stan, se hace en R.
// Sin ideología ni participación en 3 niveles.

functions {
#include funciones.stan

  // Verosimilitud para reduce_sum. Fila con provincia y seg conocidos
  // (prov_fila > 0): multinomial con ellos. Si no, mezcla en escala log de
  // sus entradas provincia x seg.
  real voto_seg_parcial(array[] int filas, int start, int end,
                        array[,] int y, vector mujer,
                        array[] int casa, array[] int mes, array[] int edad,
                        array[] int edu, array[] int rec, array[] int ccaa,
                        array[] int casa_rec, matrix off,
                        array[] int sin_aa, int j_aa,
                        array[] int prov_fila, array[] int seg_fila,
                        array[] int ini_fila, array[] int len_fila,
                        array[] int ent_prov, array[] int ent_seg, vector log_ent_w,
                        vector alpha, vector b_mujer, matrix a_casa, matrix a_mes,
                        matrix a_edad, matrix a_edu, matrix a_rec, matrix a_ccaa,
                        matrix a_prov, matrix a_casa_rec,
                        vector b_dud, matrix a_dud_edad, matrix a_dud_rec) {
    int J = cols(off);
    real lp = 0;
    for (n in 1:size(filas)) {
      int i = filas[n];
      row_vector[J] eta = alpha' + mujer[i] * b_mujer' + a_casa[casa[i]] + a_mes[mes[i]]
                          + a_edad[edad[i]] + a_edu[edu[i]] + a_rec[rec[i]] + a_ccaa[ccaa[i]]
                          + a_casa_rec[casa_rec[i]] + off[i];
      row_vector[J] d = b_dud' + a_dud_edad[edad[i]] + a_dud_rec[rec[i]];
      vector[J] logp;
      if (prov_fila[i] > 0) {
        logp = (eta + a_prov[prov_fila[i]] + (seg_fila[i] == 1 ? d : rep_row_vector(0, J)))';
      } else {
        int L = len_fila[i];
        matrix[L, J] lpe;
        for (l in 1:L) {
          int e = ini_fila[i] + l - 1;
          row_vector[J] x = eta + a_prov[ent_prov[e]];
          if (ent_seg[e] == 1) x += d;
          lpe[l] = log_ent_w[e] + log_softmax(x')';
        }
        for (j in 1:J) logp[j] = log_sum_exp(lpe[, j]);
        // logp ya está normalizado: softmax(logp) = p
      }
      if (sin_aa[i] == 0) {
        lp += multinomial_logit_lpmf(y[i] | logp);
      } else {
        // la encuesta no ofrecía AA: normalizador sin AA
        lp += multinomial_cond(logp, y[i], j_aa);
      }
    }
    return lp;
  }
}

data {
  int<lower=0, upper=1> prior_only;
  real<lower=0> s_alpha;
  real<lower=0> s_b_mujer;
  real<lower=0> s_b_dud;
  real<lower=0> s_sd_rec;
  real<lower=0> s_sd_main;
  real<lower=0> s_sd_int;
  real<lower=0> s_sd_dud;
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

  // filas con provincia y seg conocidos (seg: 1 = dudoso, 2 = seguro)
  int<lower=0> N1;
  array[N1] int<lower=1, upper=N> fila1;
  array[N1] int<lower=1, upper=n_prov> prov1;
  array[N1] int<lower=1, upper=2> seg1;
  // filas con mezcla: entradas ini..ini+len-1 de (ent_prov, ent_seg, ent_w)
  int<lower=0> N2;
  array[N2] int<lower=1, upper=N> fila2;
  array[N2] int<lower=1> ini;
  array[N2] int<lower=1> len;
  int<lower=0> E;
  array[E] int<lower=1, upper=n_prov> ent_prov;
  array[E] int<lower=1, upper=2> ent_seg;
  vector<lower=0, upper=1>[E] ent_w;
}

transformed data {
  array[N] int casa_rec;
  for (i in 1:N) casa_rec[i] = (casa[i] - 1) * n_rec + rec[i];
  real esc_casa = sqrt(n_casa / (n_casa - 1.0));
  real esc_mes = n_mes > 1 ? sqrt(n_mes / (n_mes - 1.0)) : 1;
  real esc_edad = sqrt(n_edad / (n_edad - 1.0));
  real esc_edu = sqrt(n_edu / (n_edu - 1.0));
  real esc_rec = sqrt(n_rec / (n_rec - 1.0));
  real esc_ccaa = sqrt(n_ccaa / (n_ccaa - 1.0));
  real esc_prov = sqrt(n_prov / (n_prov - 1.0));
  real esc_hr = sqrt(n_casa * n_rec / (n_casa * n_rec - 1.0));

  array[N] int prov_fila = rep_array(0, N);
  array[N] int seg_fila = rep_array(0, N);
  array[N] int ini_fila = rep_array(1, N);
  array[N] int len_fila = rep_array(0, N);
  for (n in 1:N1) {
    prov_fila[fila1[n]] = prov1[n];
    seg_fila[fila1[n]] = seg1[n];
  }
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
  array[J] sum_to_zero_vector[n_edad] z_edad;
  array[J] sum_to_zero_vector[n_edu] z_edu;
  array[J] sum_to_zero_vector[n_rec] z_rec;
  array[J] sum_to_zero_vector[n_ccaa] z_ccaa;
  array[J] sum_to_zero_vector[n_prov] z_prov;
  array[J] sum_to_zero_vector[n_casa * n_rec] z_casa_rec;
  // dudoso frente a seguro
  sum_to_zero_vector[J] b_dud;
  array[J] sum_to_zero_vector[n_edad] z_dud_edad;
  array[J] sum_to_zero_vector[n_rec] z_dud_rec;

  vector<lower=0>[J] sd_casa;
  vector<lower=0>[J] sd_mes;
  vector<lower=0>[J] sd_edad;
  vector<lower=0>[J] sd_edu;
  vector<lower=0>[J] sd_rec;
  vector<lower=0>[J] sd_ccaa;
  vector<lower=0>[J] sd_prov;
  vector<lower=0>[J] sd_casa_rec;
  vector<lower=0>[J] sd_dud_edad;
  vector<lower=0>[J] sd_dud_rec;
}

transformed parameters {
  matrix[n_casa, J] a_casa;
  matrix[n_mes, J] a_mes;
  matrix[n_edad, J] a_edad;
  matrix[n_edu, J] a_edu;
  matrix[n_rec, J] a_rec;
  matrix[n_ccaa, J] a_ccaa;
  matrix[n_prov, J] a_prov;
  matrix[n_casa * n_rec, J] a_casa_rec;
  matrix[n_edad, J] a_dud_edad;
  matrix[n_rec, J] a_dud_rec;
  for (k in 1:J) {
    a_casa[, k] = esc_casa * sd_casa[k] * z_casa[k];
    a_mes[, k] = esc_mes * sd_mes[k] * z_mes[k];
    a_edad[, k] = esc_edad * sd_edad[k] * z_edad[k];
    a_edu[, k] = esc_edu * sd_edu[k] * z_edu[k];
    a_rec[, k] = esc_rec * sd_rec[k] * z_rec[k];
    a_ccaa[, k] = esc_ccaa * sd_ccaa[k] * z_ccaa[k];
    a_prov[, k] = esc_prov * sd_prov[k] * z_prov[k];
    a_casa_rec[, k] = esc_hr * sd_casa_rec[k] * z_casa_rec[k];
    a_dud_edad[, k] = esc_edad * sd_dud_edad[k] * z_dud_edad[k];
    a_dud_rec[, k] = esc_rec * sd_dud_rec[k] * z_dud_rec[k];
  }
  // centrar entre partidos conserva la suma cero entre niveles
  a_casa = centra(a_casa);
  a_mes = centra(a_mes);
  a_edad = centra(a_edad);
  a_edu = centra(a_edu);
  a_rec = centra(a_rec);
  a_ccaa = centra(a_ccaa);
  a_prov = centra(a_prov);
  a_casa_rec = centra(a_casa_rec);
  a_dud_edad = centra(a_dud_edad);
  a_dud_rec = centra(a_dud_rec);
}

model {
  alpha ~ normal(0, s_alpha);
  b_mujer ~ normal(0, s_b_mujer);
  b_dud ~ normal(0, s_b_dud);
  for (k in 1:J) {
    z_casa[k] ~ std_normal();
    z_mes[k] ~ std_normal();
    z_edad[k] ~ std_normal();
    z_edu[k] ~ std_normal();
    z_rec[k] ~ std_normal();
    z_ccaa[k] ~ std_normal();
    z_prov[k] ~ std_normal();
    z_casa_rec[k] ~ std_normal();
    z_dud_edad[k] ~ std_normal();
    z_dud_rec[k] ~ std_normal();
  }
  sd_rec ~ normal(0, s_sd_rec);
  sd_casa ~ normal(0, s_sd_main);
  sd_mes ~ normal(0, s_sd_main);
  sd_edad ~ normal(0, s_sd_main);
  sd_edu ~ normal(0, s_sd_main);
  sd_ccaa ~ normal(0, s_sd_main);
  sd_prov ~ normal(0, s_sd_main);
  sd_casa_rec ~ normal(0, s_sd_int);
  sd_dud_edad ~ normal(0, s_sd_dud);
  sd_dud_rec ~ normal(0, s_sd_dud);

  if (!prior_only) {
    target += reduce_sum(voto_seg_parcial, filas, grainsize,
                         y, mujer, casa, mes, edad, edu, rec, ccaa, casa_rec,
                         off, sin_aa, j_aa, prov_fila, seg_fila, ini_fila, len_fila,
                         ent_prov, ent_seg, log_ent_w,
                         alpha, b_mujer, a_casa, a_mes, a_edad, a_edu, a_rec, a_ccaa,
                         a_prov, a_casa_rec, b_dud, a_dud_edad, a_dud_rec);
  }
}
