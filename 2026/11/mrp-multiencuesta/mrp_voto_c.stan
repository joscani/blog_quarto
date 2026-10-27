// MRP conjunto: modelo de voto (antes, parte de mrp.stan; ahora en obsoleto/).
//
// Mismo modelo de voto que mrp.stan. Cambios:
//  1. Sin el modelo de participación: va en mrp_part.stan. Los dos modelos
//     no comparten parámetros, así que la posterior es la misma.
//  2. Verosimilitud con reduce_sum (hilos dentro de cada cadena).
//  3. Mezcla de provincias en escala log (log_sum_exp), no sumando probabilidades.
//  4. Prior de b_mujer como dato (s_b_mujer).
// La postestratificación con participación se hace en R con los dos ajustes.
//
// mrp_voto_c.stan: mrp_voto_vec.stan con casa, mes, edad, estudios y recuerdo
// centrados (como mrp_part.stan). CCAA, provincia y las interacciones siguen
// sin centrar: tienen grupos pequeños. Mismo modelo, otra parametrización.
//
// mrp_voto_vec.stan: el mismo modelo que mrp_voto.stan, con la verosimilitud
// vectorizada por bloques de filas en vez de fila a fila (menos nodos de
// derivada). Quita las constantes de la multinomial: cambia lp__, no la posterior.
//  - Filas con provincia: sum(y .* eta) - n * log_sum_exp(eta) por fila.
//  - Filas mezcla: p = sum_e w_e softmax(eta + a_prov_e), en escala natural.
//    exp() no desborda: |eta| es pequeño y off vale 0 o -20.

functions {
#include funciones.stan

  // predictor de las filas r, sin el efecto provincia
  matrix eta_bloque(array[] int r, vector mujer,
                    array[] int casa, array[] int mes, array[] int edad, array[] int edu,
                    array[] int rec, array[] int ccaa, array[] int ccaa_rec, array[] int casa_rec,
                    matrix off, vector alpha, vector b_mujer, matrix a_casa, matrix a_mes,
                    matrix a_edad, matrix a_edu, matrix a_rec, matrix a_ccaa,
                    matrix a_ccaa_rec, matrix a_casa_rec) {
    return rep_matrix(alpha', size(r)) + mujer[r] * b_mujer'
           + a_casa[casa[r]] + a_mes[mes[r]] + a_edad[edad[r]] + a_edu[edu[r]]
           + a_rec[rec[r]] + a_ccaa[ccaa[r]] + a_ccaa_rec[ccaa_rec[r]]
           + a_casa_rec[casa_rec[r]] + off[r];
  }

  // filas con provincia conocida: r es un trozo de fila1
  real voto1_parcial(array[] int r, int start, int end,
                     matrix Y, vector n_y, array[] int prov_fila, vector mujer,
                     array[] int casa, array[] int mes, array[] int edad, array[] int edu,
                     array[] int rec, array[] int ccaa, array[] int ccaa_rec, array[] int casa_rec,
                     matrix off, vector alpha, vector b_mujer, matrix a_casa, matrix a_mes,
                     matrix a_edad, matrix a_edu, matrix a_rec, matrix a_ccaa,
                     matrix a_ccaa_rec, matrix a_casa_rec, matrix a_prov) {
    matrix[size(r), cols(off)] eta
      = eta_bloque(r, mujer, casa, mes, edad, edu, rec, ccaa, ccaa_rec, casa_rec, off,
                   alpha, b_mujer, a_casa, a_mes, a_edad, a_edu, a_rec, a_ccaa,
                   a_ccaa_rec, a_casa_rec)
        + a_prov[prov_fila[r]];
    vector[size(r)] lse = log(exp(eta) * rep_vector(1.0, cols(off)));
    return sum(Y[r] .* eta) - dot_product(n_y[r], lse);
  }

  // filas mezcla: m es un trozo de 1..N2; sus entradas son contiguas
  real voto2_parcial(array[] int m, int start, int end,
                     array[] int fila2, array[] int ini, array[] int len,
                     array[] int ent_prov, vector ent_w,
                     matrix Y, vector mujer,
                     array[] int casa, array[] int mes, array[] int edad, array[] int edu,
                     array[] int rec, array[] int ccaa, array[] int ccaa_rec, array[] int casa_rec,
                     matrix off, vector alpha, vector b_mujer, matrix a_casa, matrix a_mes,
                     matrix a_edad, matrix a_edu, matrix a_rec, matrix a_ccaa,
                     matrix a_ccaa_rec, matrix a_casa_rec, matrix a_prov) {
    int J = cols(off);
    int n2 = size(m);
    array[n2] int r = fila2[m];
    int e0 = ini[m[1]];
    int e1 = ini[m[n2]] + len[m[n2]] - 1;
    // fila del trozo (1..n2) de cada entrada e0..e1
    array[e1 - e0 + 1] int fila_e;
    for (k in 1:n2) {
      for (l in 1:len[m[k]]) fila_e[ini[m[k]] - e0 + l] = k;
    }
    matrix[n2, J] eta
      = eta_bloque(r, mujer, casa, mes, edad, edu, rec, ccaa, ccaa_rec, casa_rec, off,
                   alpha, b_mujer, a_casa, a_mes, a_edad, a_edu, a_rec, a_ccaa,
                   a_ccaa_rec, a_casa_rec);
    matrix[e1 - e0 + 1, J] pe = exp(eta[fila_e] + a_prov[ent_prov[e0:e1]]);
    // w_e * softmax por fila
    pe = diag_pre_multiply(ent_w[e0:e1] ./ (pe * rep_vector(1.0, J)), pe);
    matrix[n2, J] p;
    for (k in 1:n2) {
      int a = ini[m[k]] - e0 + 1;
      p[k] = rep_row_vector(1.0, len[m[k]]) * pe[a:(a + len[m[k]] - 1)];
    }
    return sum(Y[r] .* log(p));
  }
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

  // para la verosimilitud vectorizada
  matrix[N, J] Y = to_matrix(y);
  vector[N] n_y = Y * rep_vector(1.0, J);
  array[N2] int filas2 = linspaced_int_array(N2, 1, N2);
  // el mismo número de trozos que con grainsize sobre las N filas
  int grain1 = max(1, (grainsize * N1) %/% N);
  int grain2 = max(1, (grainsize * N2) %/% N);
}

parameters {
  sum_to_zero_vector[J] alpha;
  sum_to_zero_vector[J] b_mujer;
  // centrados: c_* ~ normal(0, sd)
  array[J] sum_to_zero_vector[n_casa] c_casa;
  matrix[n_mes, J] c_mes;
  matrix[n_edad, J] c_edad;
  matrix[n_edu, J] c_edu;
  matrix[n_rec, J] c_rec;
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
}

transformed parameters {
  matrix[n_casa, J] a_casa;
  for (k in 1:J) {
    a_casa[, k] = c_casa[k];
  }
  a_casa = centra(a_casa);
  matrix[n_mes, J] a_mes = centra(c_mes);
  matrix[n_edad, J] a_edad = centra(c_edad);
  matrix[n_edu, J] a_edu = centra(c_edu);
  matrix[n_rec, J] a_rec = centra(c_rec);
  matrix[n_ccaa, J] a_ccaa = centra(diag_post_multiply(z_ccaa, sd_ccaa));
  matrix[n_prov, J] a_prov = centra(diag_post_multiply(z_prov, sd_prov));
  matrix[n_ccaa * n_rec, J] a_ccaa_rec = centra(diag_post_multiply(z_ccaa_rec, sd_ccaa_rec));
  matrix[n_casa * n_rec, J] a_casa_rec = centra(diag_post_multiply(z_casa_rec, sd_casa_rec));
}

model {
  alpha ~ normal(0, s_alpha);
  b_mujer ~ normal(0, s_b_mujer);
  // casa: c_casa[k] = esc_casa * sd_casa[k] * z con z ~ std_normal en el
  // subespacio de suma 0 (dimensión n_casa - 1). normal_lpdf sobre n_casa
  // componentes resta n_casa * log(escala); la densidad correcta resta
  // (n_casa - 1) * log(escala): se suma log(escala) por partido.
  for (k in 1:J) c_casa[k] ~ normal(0, esc_casa * sd_casa[k]);
  target += sum(log(esc_casa * sd_casa));
  to_vector(c_mes) ~ normal(0, to_vector(rep_matrix(sd_mes', n_mes)));
  to_vector(c_edad) ~ normal(0, to_vector(rep_matrix(sd_edad', n_edad)));
  to_vector(c_edu) ~ normal(0, to_vector(rep_matrix(sd_edu', n_edu)));
  to_vector(c_rec) ~ normal(0, to_vector(rep_matrix(sd_rec', n_rec)));
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
    target += reduce_sum(voto1_parcial, fila1, grain1,
                         Y, n_y, prov_fila, mujer,
                         casa, mes, edad, edu, rec, ccaa, ccaa_rec, casa_rec,
                         off, alpha, b_mujer, a_casa, a_mes, a_edad, a_edu, a_rec, a_ccaa,
                         a_ccaa_rec, a_casa_rec, a_prov);
    target += reduce_sum(voto2_parcial, filas2, grain2,
                         fila2, ini, len, ent_prov, ent_w,
                         Y, mujer,
                         casa, mes, edad, edu, rec, ccaa, ccaa_rec, casa_rec,
                         off, alpha, b_mujer, a_casa, a_mes, a_edad, a_edu, a_rec, a_ccaa,
                         a_ccaa_rec, a_casa_rec, a_prov);
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
