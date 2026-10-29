  // Funciones comunes de los modelos MRP. Se incluyen con #include dentro
  // del bloque functions (include_paths = "2026/11/mrp-multiencuesta/stan").

  // resta a cada fila su media: en cada nivel, el efecto suma 0 entre partidos
  matrix centra(matrix a) {
    int J = cols(a);
    return a - rep_matrix(a * rep_vector(1.0 / J, J), J);
  }

  // delta tal que sum(N * inv_logit(eta + delta)) = obj * sum(N), por Newton.
  // La función es creciente en delta. El paso se limita a +-2: sin límite,
  // con eta muy grandes la derivada es casi 0 y delta se dispara (da NaN).
  real calibra(vector eta, vector N, real obj) {
    real delta = 0;
    for (it in 1:50) {
      vector[rows(eta)] p = inv_logit(eta + delta);
      real f = dot_product(N, p) / sum(N) - obj;
      real df = dot_product(N, p .* (1 - p)) / sum(N);
      delta -= fmin(fmax(f / df, -2), 2);
      if (abs(f) < 1e-10) break;
    }
    return delta;
  }

  // probabilidades de voto válido en una celda de población, con ceros
  // exactos fuera del territorio de cada partido
  row_vector prob_valida(row_vector eta, row_vector mascara) {
    vector[cols(eta)] p = softmax(eta') .* mascara';
    return (p / sum(p))';
  }

  // Multinomial condicionada a excluir la categoría j_aa del normalizador:
  // para las encuestas que no ofrecían esa opción (AA), cuyos ceros son de
  // diseño del cuestionario y no informan de su prevalencia. Exige y[j_aa]=0.
  real multinomial_cond(vector eta, array[] int y, int j_aa) {
    int J = num_elements(eta);
    vector[J - 1] sin_jaa;
    int m = 0;
    real n = 0;
    real lp = 0;
    for (j in 1:J) {
      lp += y[j] * eta[j];
      n += y[j];
      if (j != j_aa) {
        m += 1;
        sin_jaa[m] = eta[j];
      }
    }
    return lp - n * log_sum_exp(sin_jaa);
  }

  // Verosimilitud del voto para reduce_sum: suma de log p de las filas
  // filas[1..] (un trozo de 1..N). Cada fila calcula su predictor.
  //   prov_fila[i] > 0: provincia conocida, multinomial con esa provincia.
  //   prov_fila[i] = 0: mezcla de provincias, en escala log:
  //     log p_j = log_sum_exp_e(log w_e + log_softmax(eta + a_prov_e)_j)
  real voto_parcial(array[] int filas, int start, int end,
               array[,] int y, vector mujer,
               array[] int casa, array[] int mes, array[] int edad, array[] int edu,
               array[] int rec, array[] int ccaa, array[] int ccaa_rec, array[] int casa_rec,
               matrix off, array[] int sin_aa, int j_aa,
               array[] int prov_fila, array[] int ini_fila, array[] int len_fila,
               array[] int ent_prov, vector log_ent_w,
               vector alpha, vector b_mujer, matrix a_casa, matrix a_mes, matrix a_edad,
               matrix a_edu, matrix a_rec, matrix a_ccaa, matrix a_prov,
               matrix a_ccaa_rec, matrix a_casa_rec) {
    int J = cols(off);
    real lp = 0;
    for (n in 1:size(filas)) {
      int i = filas[n];
      row_vector[J] eta = alpha' + mujer[i] * b_mujer' + a_casa[casa[i]] + a_mes[mes[i]]
                          + a_edad[edad[i]] + a_edu[edu[i]] + a_rec[rec[i]] + a_ccaa[ccaa[i]]
                          + a_ccaa_rec[ccaa_rec[i]] + a_casa_rec[casa_rec[i]] + off[i];
      if (prov_fila[i] > 0) {
        row_vector[J] e = eta + a_prov[prov_fila[i]];
        if (sin_aa[i] == 0) {
          lp += multinomial_logit_lpmf(y[i] | e');
        } else {
          // la encuesta no ofrecía AA: normalizador sin AA
          lp += multinomial_cond(e', y[i], j_aa);
        }
      } else {
        int L = len_fila[i];
        matrix[L, J] lpe;
        for (l in 1:L) {
          int e = ini_fila[i] + l - 1;
          lpe[l] = log_ent_w[e] + log_softmax((eta + a_prov[ent_prov[e]])')';
        }
        vector[J] logp;
        for (j in 1:J) logp[j] = log_sum_exp(lpe[, j]);
        // logp ya está normalizado: softmax(logp) = p
        if (sin_aa[i] == 0) {
          lp += multinomial_logit_lpmf(y[i] | logp);
        } else {
          // la mezcla de provincias, condicionada a no-AA
          lp += multinomial_cond(logp, y[i], j_aa);
        }
      }
    }
    return lp;
  }
