// Modelo conjunto: todas las campañas generales (2015-2023 y 2026), sus
// resultados y los errores de las autonómicas, en un único posterior.
//
// 1. Campañas. Para la campaña c, voto latente de la media de las casas en
//    la semana t, en la escala de softmax SIN categoría de referencia (un
//    componente por familia, centrado):
//      theta_c,t = theta_c,t-1 + centrado(omega),
//      omega_k ~ N(0, sigma_rw[g(k)] * exp(kappa * campana_t))
//    sigma_rw por grupo de partido; kappa: cuánto más se mueve el voto en
//    las 8 últimas semanas. Con sigma_rw_campana = 1, cada campaña tiene su
//    volatilidad, con pooling parcial:
//      log sigma_rw[c, g] = log sigma_rw[g] + tau_rw * z_rwc[c, g]
//    (las campañas de 2015-2019, con la irrupción de Podemos, Cs y Vox, no
//    imponen su volatilidad a 2026). Con 0, sigma_rw[g] común a todas. Sin referencia, un
//    movimiento de RESTO (Cs en 2019) se queda en RESTO: en la versión con
//    log-ratios frente a RESTO contagiaba a todos (fit_conjunto_v1.rds).
//    Encuesta i (casa h): p_i = softmax(theta_c,t + beta_c,h); cada celda
//    suma sus familias y se normaliza dentro de las celdas de la encuesta
//    (el desglose regional es condicional al BLOQUE).
//      y_i ~ Dirichlet-multinomial(phi_c * q_i)
//    phi_c recoge la dispersión que no es de muestreo (si es grande, las
//    encuestas son multinomiales puras); así esa dispersión no se cuela en
//    sigma_rw. beta centrado por familia dentro de cada encuesta y de cada
//    campaña.
// 2. Resultados. En las campañas pasadas, para los grupos PP, PSOE, VOX y
//    SUMAR (v = suma de sus familias en softmax(theta_c,T)):
//      logit(real_c) - logit(v_c,T) ~ MVN(mu_gen, diag(sigma_gen) Omega diag(sigma_gen))
//    En la campaña objetivo (la última) el error se predice de esa MVN.
// 3. Autonómicas (errores en logit de la última semana):
//      z ~ N(mu_aut + delta_e + b_h, sigma_enc_a),
//      delta_e ~ MVN(0, diag(sigma_aut) Omega diag(sigma_aut)).
//    Omega común; sigma_gen = sigma_aut * exp(gamma) (patrón de las
//    autonómicas con nivel propio). Se probó también una escala propia para
//    las generales (modelo_conjunto_con_modo2.stan.bak): no cambió el
//    backtest y se descartó.
data {
  int<lower=1> C;                       // campañas; la C es el objetivo
  array[C] int<lower=2> K;              // familias por campaña
  array[C] int<lower=2> T;              // semanas por campaña (T = día E)
  int<lower=1> L1;                      // sum(K)
  vector[L1] theta1_mu;                 // log del voto de la elección anterior
  real<lower=0> eta1_sd;
  array[L1] int<lower=1> rwgrp;         // grupo de sigma_rw de cada familia
  int<lower=1> R;                       // número de grupos
  array[sum(T)] int<lower=0, upper=1> campana_sem;

  int<lower=1> N;                       // encuestas
  int<lower=1> S;                       // casas (campaña x casa)
  array[N] int<lower=1, upper=C> enc_campana;
  array[N] int<lower=1> enc_semana;
  array[N] int<lower=1, upper=S> enc_slot;
  array[N] int<lower=1> enc_c_ini;      // primera celda de la encuesta
  array[N] int<lower=1> enc_c_n;        // número de celdas
  array[S] int<lower=1, upper=C> slot_campana;
  int<lower=1> J;                       // celdas
  array[J] int<lower=0> y;
  array[J] int<lower=1> cel_m_ini;      // primera pertenencia de la celda
  array[J] int<lower=1> cel_m_n;
  int<lower=1> M;
  array[M] int<lower=1> memb_fam;       // familia (dentro de su campaña)

  int<lower=0> P;                       // campañas pasadas con resultado (1..P)
  int<lower=1> G;                       // grupos del error (PP, PSOE, VOX, SUMAR)
  matrix[P, G] logit_real;
  array[P, G] int<lower=0, upper=1> obs_real;
  int<lower=1> GM;
  array[GM] int<lower=1, upper=C> gm_campana;
  array[GM] int<lower=1, upper=G> gm_grupo;
  array[GM] int<lower=1> gm_fam;

  int<lower=0> Ja;                      // autonómicas (0 en el backtest del 26J)
  matrix[Ja, G] za;
  array[Ja, G] int<lower=0, upper=1> obsa;
  int<lower=0> Ea;
  array[Ja] int<lower=1, upper=max(Ea, 1)> elec_a;
  int<lower=0> Ha;
  array[Ja] int<lower=1, upper=max(Ha, 1)> casa_a;

  // escalas de los priors
  real<lower=0> sd_rw;
  real<lower=0> sd_kappa;
  real<lower=0> sd_casa;
  real log_phi_mu;                      // log phi ~ N(log_phi_mu, sd_log_phi)
  real<lower=0> sd_log_phi;
  real<lower=0> sd_mu;
  real<lower=0> sd_log_s0;
  real<lower=0> sd_tau_s;
  real<lower=0> sd_gamma;
  real<lower=0> sd_enc_a;
  real<lower=0> sd_b;
  real<lower=0> eta_lkj;
  int<lower=0, upper=1> sigma_rw_campana;
  real<lower=0> sd_tau_rw;
  int<lower=0, upper=1> prior_only;
}
transformed data {
  array[C] int off_th;                  // inicio de la campaña en theta1 / rwgrp
  array[C] int off_z;                   // inicio en z_rw
  array[C] int off_t;                   // inicio en campana_sem
  int Lz = 0;
  {
    int a = 0;
    int t = 0;
    for (c in 1:C) {
      off_th[c] = a;
      off_z[c] = Lz;
      off_t[c] = t;
      a += K[c];
      Lz += K[c] * (T[c] - 1);
      t += T[c];
    }
  }
  array[S] int off_b;
  int Lb = 0;
  for (s in 1:S) {
    off_b[s] = Lb;
    Lb += K[slot_campana[s]];
  }
  int Kmax = max(K);
  int Tmax = max(T);
  // autonómicas en formato largo
  int Na = 0;
  for (j in 1:Ja) {
    Na += sum(obsa[j]);
  }
  vector[Na] za_obs;
  array[Na] int ia_k;
  array[Na] int ia_e;
  array[Na] int ia_h;
  {
    int n = 0;
    for (j in 1:Ja) {
      for (k in 1:G) {
        if (obsa[j, k]) {
          n += 1;
          za_obs[n] = za[j, k];
          ia_k[n] = k;
          ia_e[n] = k + G * (elec_a[j] - 1);
          ia_h[n] = k + G * (casa_a[j] - 1);
        }
      }
    }
  }
}
parameters {
  vector[L1] theta1;
  vector[Lz] z_rw;
  vector<lower=0>[R] sigma_rw;
  matrix[C, R] z_rwc;
  real<lower=0> tau_rw;
  real kappa;
  vector[Lb] z_casa;
  vector<lower=0>[R] sigma_casa;
  vector[C] log_phi;

  vector[G] mu_gen;
  vector[G] mu_aut;
  cholesky_factor_corr[G] L_Omega;
  real log_s0;
  real<lower=0> tau_s;
  vector[G] z_s;
  real gamma;
  matrix[G, Ea] z_delta_a;
  vector<lower=0>[G] sigma_enc_a;
  matrix[G, Ha] z_b;
  vector<lower=0>[G] sigma_b;
}
transformed parameters {
  vector[G] sigma_aut = exp(log_s0 + tau_s * z_s);
  vector[G] sigma_gen = sigma_aut * exp(gamma);
  matrix[G, G] Omega = multiply_lower_tri_self_transpose(L_Omega);
  matrix[G, G] L_gen = diag_pre_multiply(sigma_gen, L_Omega);
  matrix[G, Ea] delta_a = diag_pre_multiply(sigma_aut, L_Omega) * z_delta_a;
  matrix[G, Ha] b = diag_pre_multiply(sigma_b, z_b);

  // volatilidad de cada campaña y grupo
  matrix[C, R] sigma_rw_c;
  for (c in 1:C) {
    for (g in 1:R) {
      sigma_rw_c[c, g] = sigma_rw[g]
                         * (sigma_rw_campana == 1 ? exp(tau_rw * z_rwc[c, g]) : 1);
    }
  }

  // voto latente: theta[c] es K x T (relleno de ceros hasta Kmax x Tmax:
  // Stan no admite matrices de tamaño distinto en un array). Cada columna
  // está centrada (suma cero): softmax no cambia y no hay referencia.
  array[C] matrix[Kmax, Tmax] theta;
  for (c in 1:C) {
    int kc = K[c];
    theta[c] = rep_matrix(0, Kmax, Tmax);
    vector[kc] th = theta1[(off_th[c] + 1):(off_th[c] + kc)];
    theta[c][1:kc, 1] = th - mean(th);
    for (t in 2:T[c]) {
      vector[kc] om;
      real f = exp(kappa * campana_sem[off_t[c] + t]);
      for (k in 1:kc) {
        om[k] = sigma_rw_c[c, rwgrp[off_th[c] + k]] * f
                * z_rw[off_z[c] + (t - 2) * kc + k];
      }
      theta[c][1:kc, t] = theta[c][1:kc, t - 1] + om - mean(om);
    }
  }

  // efectos casa: centrados por familia dentro de cada casa y, después,
  // por familia dentro de cada campaña
  vector[Lb] beta;
  for (s in 1:S) {
    int c = slot_campana[s];
    int kc = K[c];
    vector[kc] bs;
    for (k in 1:kc) {
      bs[k] = sigma_casa[rwgrp[off_th[c] + k]] * z_casa[off_b[s] + k];
    }
    beta[(off_b[s] + 1):(off_b[s] + kc)] = bs - mean(bs);
  }
  for (c in 1:C) {
    int kc = K[c];
    vector[kc] media = rep_vector(0, kc);
    int ns = 0;
    for (s in 1:S) {
      if (slot_campana[s] == c) {
        media += beta[(off_b[s] + 1):(off_b[s] + kc)];
        ns += 1;
      }
    }
    media /= ns;
    for (s in 1:S) {
      if (slot_campana[s] == c) {
        beta[(off_b[s] + 1):(off_b[s] + kc)] -= media;
      }
    }
  }
}
model {
  theta1 ~ normal(theta1_mu, eta1_sd);
  z_rw ~ std_normal();
  sigma_rw ~ normal(0, sd_rw);
  to_vector(z_rwc) ~ std_normal();
  tau_rw ~ normal(0, sd_tau_rw);
  kappa ~ normal(0, sd_kappa);
  z_casa ~ std_normal();
  sigma_casa ~ normal(0, sd_casa);
  log_phi ~ normal(log_phi_mu, sd_log_phi);

  mu_gen ~ normal(0, sd_mu);
  mu_aut ~ normal(0, sd_mu);
  L_Omega ~ lkj_corr_cholesky(eta_lkj);
  log_s0 ~ normal(log(0.1), sd_log_s0);
  tau_s ~ normal(0, sd_tau_s);
  z_s ~ std_normal();
  gamma ~ normal(0, sd_gamma);
  to_vector(z_delta_a) ~ std_normal();
  sigma_enc_a ~ normal(0, sd_enc_a);
  to_vector(z_b) ~ std_normal();
  sigma_b ~ normal(0, sd_b);

  if (!prior_only) {
    // 1. encuestas de todas las campañas
    for (i in 1:N) {
      int c = enc_campana[i];
      int kc = K[c];
      vector[kc] p = softmax(
        theta[c][1:kc, enc_semana[i]]
        + beta[(off_b[enc_slot[i]] + 1):(off_b[enc_slot[i]] + kc)]);
      int nc = enc_c_n[i];
      vector[nc] q;
      array[nc] int yi;
      for (j in 1:nc) {
        int cj = enc_c_ini[i] + j - 1;
        q[j] = 0;
        for (m in cel_m_ini[cj]:(cel_m_ini[cj] + cel_m_n[cj] - 1)) {
          q[j] += p[memb_fam[m]];
        }
        yi[j] = y[cj];
      }
      yi ~ dirichlet_multinomial(exp(log_phi[c]) * q / sum(q));
    }

    // 2. resultados de las campañas pasadas, frente a theta_c,T
    for (c in 1:P) {
      int kc = K[c];
      vector[kc] v = softmax(theta[c][1:kc, T[c]]);
      vector[G] vg = rep_vector(0, G);
      for (m in 1:GM) {
        if (gm_campana[m] == c) {
          vg[gm_grupo[m]] += v[gm_fam[m]];
        }
      }
      int no = sum(obs_real[c]);
      array[no] int idx;
      {
        int n = 0;
        for (g in 1:G) {
          if (obs_real[c, g]) {
            n += 1;
            idx[n] = g;
          }
        }
      }
      vector[no] err = logit_real[c, idx]' - logit(vg[idx]);
      err ~ multi_normal(mu_gen[idx], quad_form_diag(Omega[idx, idx], sigma_gen[idx]));
    }

    // 3. autonómicas
    za_obs ~ normal(mu_aut[ia_k] + to_vector(delta_a)[ia_e] + to_vector(b)[ia_h],
                    sigma_enc_a[ia_k]);
  }
}
generated quantities {
  matrix[K[C], T[C]] v_obj;
  for (t in 1:T[C]) {
    v_obj[, t] = softmax(theta[C][1:K[C], t]);
  }
  vector[G] err_new = mu_gen + multi_normal_cholesky_rng(rep_vector(0, G), L_gen);
}
