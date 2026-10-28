# Exporta los datos del modelo de voto (mrp_voto.stan) a .npz para el ajuste
# de numpyro. Lee los mismos datos que stan/ajusta.R (datos/datos_stan.rds, de
# 06_stan_datos.R), así la submuestra del modo "prueba" (set.seed(2026), 10 %)
# es idéntica a la de Stan.
#
# Uso (desde la raíz del repo):
#   Rscript 2026/11/mrp-multiencuesta/numpyro/exporta_npz.R
#
# Salidas: 2026/11/mrp-multiencuesta/datos/voto_numpyro_<modo>.npz y
# part_numpyro_<modo>.npz (participación, para mrp_part_np.py), con el modo
# con el que se generó datos_stan.rds en 06_stan_datos.R.

suppressMessages(library(reticulate))

dir_proy <- here::here("2026/11/mrp-multiencuesta")

x <- readRDS(file.path(dir_proy, "datos", "datos_stan.rds"))
modo <- x$modo
d <- x$datos
# s_b_mujer lo añade stan/ajusta.R a sus datos (el valor es fijo ahí)
d$s_b_mujer <- 1

np <- import("numpy")
f_npz <- here::here(
  "2026/11/mrp-multiencuesta/datos",
  paste0("voto_numpyro_", modo, ".npz")
)
np$savez_compressed(
  f_npz,
  # verosimilitud
  y = d$y,
  mujer = d$mujer,
  off = d$off,
  casa = d$casa,
  mes = d$mes,
  edad = d$edad,
  edu = d$edu,
  rec = d$rec,
  ccaa = d$ccaa,
  # mezcla de provincias (índices 1-based; se convierten en Python)
  fila1 = d$fila1,
  prov1 = d$prov1,
  fila2 = d$fila2,
  ini = d$ini,
  len = d$len,
  ent_prov = d$ent_prov,
  ent_w = d$ent_w,
  # postestratificación
  P_N = d$P_N,
  P_mujer = d$P_mujer,
  P_edad = d$P_edad,
  P_edu = d$P_edu,
  P_rec = d$P_rec,
  P_ccaa = d$P_ccaa,
  P_prov = d$P_prov,
  P_mascara = d$P_mascara,
  mes_post = d$mes_post,
  # escalares
  N = d$N,
  J = d$J,
  P = d$P,
  n_casa = d$n_casa,
  n_mes = d$n_mes,
  n_edad = d$n_edad,
  n_edu = d$n_edu,
  n_rec = d$n_rec,
  n_ccaa = d$n_ccaa,
  n_prov = d$n_prov,
  s_alpha = d$s_alpha,
  s_b_mujer = d$s_b_mujer,
  s_sd_rec = d$s_sd_rec,
  s_sd_main = d$s_sd_main,
  s_sd_int = d$s_sd_int,
  # etiquetas (voto_lv y lv$casa + "casa media", como stan/ajusta.R)
  voto_lv = paste(x$voto_lv),
  casa_lv = c(paste(x$lv$casa), "casa media")
)
cat(
  "escrito",
  f_npz,
  "\n",
  "N:",
  d$N,
  " J:",
  d$J,
  " P:",
  d$P,
  " filas mezcla:",
  d$N2,
  " entradas:",
  d$E,
  "\n"
)

# Participación (mrp_part.stan, para mrp_part_np.py). Los priors de alpha_t y
# b_mujer_t los fija stan/ajusta.R
f_npz_t <- here::here(
  "2026/11/mrp-multiencuesta/datos",
  paste0("part_numpyro_", modo, ".npz")
)
np$savez_compressed(
  f_npz_t,
  # verosimilitud
  t_n = d$t_n,
  t_k = d$t_k,
  t_mujer = d$t_mujer,
  t_casa = d$t_casa,
  t_mes = d$t_mes,
  t_edad = d$t_edad,
  t_edu = d$t_edu,
  t_rec = d$t_rec,
  t_ccaa = d$t_ccaa,
  # mezcla de provincias (índices 1-based)
  t_fila1 = d$t_fila1,
  t_prov1 = d$t_prov1,
  t_fila2 = d$t_fila2,
  t_ini = d$t_ini,
  t_len = d$t_len,
  t_ent_prov = d$t_ent_prov,
  t_ent_w = d$t_ent_w,
  # postestratificación y calibración
  P_N = d$P_N,
  P_mujer = d$P_mujer,
  P_edad = d$P_edad,
  P_edu = d$P_edu,
  P_rec = d$P_rec,
  P_ccaa = d$P_ccaa,
  P_prov = d$P_prov,
  mes_post = d$mes_post,
  part_obj = d$part_obj,
  # escalares
  NT = d$NT,
  n_casa_t = d$n_casa_t,
  n_mes = d$n_mes,
  n_edad = d$n_edad,
  n_edu = d$n_edu,
  n_rec = d$n_rec,
  n_ccaa = d$n_ccaa,
  n_prov = d$n_prov,
  s_sd_rec = d$s_sd_rec,
  s_sd_main = d$s_sd_main,
  m_alpha_t = 1,
  s_alpha_t = 1.5,
  s_b_mujer_t = 1
)
cat(
  "escrito",
  f_npz_t,
  "\n",
  "filas:",
  d$NT,
  " filas mezcla:",
  d$NT2,
  " entradas:",
  d$TE,
  "\n"
)
