# mclogit sobre las celdas del CIS: el candidato frecuentista más cercano al
# multinomial multinivel (mblogit, logit de categoría base por PQL/IRLS).
# Es la sección del post "lo intenté y así fue".
#
# Qué se espera (y es parte del relato):
#   - con las celdas completas el diseño es singular: edad "18-20" y rec
#     "NO_PODIA" son la misma gente (no pudo votar en 23-J), dos ficticias
#     idénticas. El bayesiano lo lleva con priors; PQL, no.
#   - quitando esas celdas, la multinomial sin efectos aleatorios converge
#     (desaparecen los niveles edad 18-20 y rec NO_PODIA).
#   - con random = ~1|prov, convergencia dudosa.
#   - no puede expresar: la verosimilitud condicional de AA, el efecto casa
#     (aquí solo hay CIS) ni las mezclas de provincia (el CIS no las necesita).
#
# Postestratificación: celdas de población de tabla_post_prov.rds, casa CIS
# (el modelo no tiene casa), mes 2026-10 y electorado del 23-J (sin las celdas
# 18-20 x NO_PODIA, que el ajuste sin esos niveles no puede predecir).
#
# Comparación con Stan: datos/estim_stan_nuts_nuevo_cis.rds (ajuste solo-CIS,
# de stan/ajusta_cis.R) si existe; si no, la columna CIS del modelo de 4 casas
# (datos/estim_separado_nuts.rds), que comparte verosimilitud con las otras
# casas vía los efectos comunes.
#
#   Rscript 2026/11/mrp-multiencuesta/13_mclogit.R

library(dplyr)
library(tidyr)
library(mclogit)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")

celdas <- readRDS(file.path(dir_datos, "celdas_brms.rds"))
voto_lv <- colnames(celdas$y) # el orden del modelo: PP es la categoría base
cis <- celdas |>
  filter(casa == "CIS") |>
  mutate(fila = row_number())
cat("celdas CIS:", nrow(cis), " partidos:", length(voto_lv), "\n")

# ---- intento 1: celdas completas (diseño singular) ----------------------------
cat("\n== intento 1: celdas completas ==\n")
m_full <- try(
  mblogit(cis$y ~ sexo + edad + edu + rec + mes + ccaa, data = cis),
  silent = TRUE
)
if (inherits(m_full, "try-error")) {
  cat("falló:", conditionMessage(attr(m_full, "condition")), "\n")
} else {
  cat("convergió inesperadamente\n")
}

# ---- intento 2: sin las celdas colineales, sin efectos aleatorios -------------
# edad 18-20 y rec NO_PODIA son la misma gente: sobra una de las dos ficticias
n_colin <- sum(cis$edad == "18-20" & cis$rec == "NO_PODIA")
cis_fit <- cis |>
  filter(!(edad == "18-20" & rec == "NO_PODIA")) |>
  mutate(across(c(edad, rec), droplevels))
cat(
  "\n== intento 2: sin las",
  n_colin,
  "celdas 18-20 x NO_PODIA, sin efectos aleatorios ==\n"
)
t0 <- Sys.time()
m0 <- mblogit(cis_fit$y ~ sexo + edad + edu + rec + mes + ccaa, data = cis_fit)
t_m0 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("mblogit sin aleatorios:", round(t_m0, 1), "s\n")

# coeficientes: vector "param~partido" -> matriz params x (J-1), por nombres
# (el orden de coef() no está garantizado)
coef_mblogit <- function(m, xnames) {
  b <- coef(m)
  dos <- strsplit(names(b), "~", fixed = TRUE)
  param <- vapply(dos, `[[`, "", 1)
  part <- vapply(dos, `[[`, "", 2)
  B <- matrix(
    NA_real_,
    length(xnames),
    length(voto_lv) - 1,
    dimnames = list(xnames, voto_lv[-1])
  )
  B[cbind(match(param, xnames), match(part, voto_lv[-1]))] <- b
  stopifnot(!anyNA(B))
  B
}
x_fit <- colnames(model.matrix(~ sexo + edad + edu + rec + mes + ccaa, cis_fit))
B <- coef_mblogit(m0, x_fit)
cat("coeficientes:", nrow(B), "params x", ncol(B), "ecuaciones\n")
saveRDS(m0, file.path(dir_datos, "fit_mblogit_cis.rds"))

# ---- intento 3: con random = ~1|prov ------------------------------------------
cat("\n== intento 3: random = ~1|prov ==\n")
t0 <- Sys.time()
m1 <- try(
  mblogit(
    cis_fit$y ~ sexo + edad + edu + rec + mes + ccaa,
    random = ~ 1 | prov,
    data = cis_fit
  ),
  silent = TRUE
)
t_m1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
if (inherits(m1, "try-error")) {
  cat(
    "falló tras",
    round(t_m1, 1),
    "s:",
    conditionMessage(attr(m1, "condition")),
    "\n"
  )
} else {
  cat("acabó en", round(t_m1, 1), "s; convergencia:", m1$convergence, "\n")
}

# ---- postestratificación del ajuste sin aleatorios ----------------------------
# población: la misma de 07_post_celdas.R, sin las celdas 18-20 x NO_PODIA
tabla <- readRDS(file.path(dir_datos, "tabla_post_prov.rds"))
prov_ccaa <- readRDS(file.path(dir_datos, "prov_ccaa.rds"))
post <- as.data.frame.table(tabla, responseName = "N") |>
  filter(N > 1e-6) |>
  left_join(prov_ccaa, by = "prov") |>
  # los factores con los niveles de las celdas, para contrastes idénticos
  mutate(
    sexo = factor(sexo, levels = levels(celdas$sexo)),
    edad = factor(edad, levels = levels(celdas$edad)),
    edu = factor(edu, levels = levels(celdas$edu)),
    rec = factor(rec, levels = levels(celdas$rec)),
    mes = factor(max(levels(celdas$mes)), levels = levels(celdas$mes)),
    ccaa = factor(ccaa, levels = levels(celdas$ccaa))
  )
n_jovenes <- post |> filter(edad == "18-20") |> pull(N) |> sum()
post <- post |>
  filter(!(edad == "18-20" & rec == "NO_PODIA")) |>
  # tras excluir esas celdas, mismos niveles que en el ajuste
  mutate(across(c(edad, rec), droplevels))
cat(
  "\npoblación: ",
  format(sum(post$N), big.mark = ","),
  " (excluidos 18-20: ",
  format(n_jovenes, big.mark = ","),
  ")\n",
  sep = ""
)

X <- model.matrix(~ sexo + edad + edu + rec + mes + ccaa, post)
stopifnot(identical(colnames(X), rownames(B)))
eta <- cbind(0, X %*% B) # PP, la base, a 0
p <- exp(eta - apply(eta, 1, max))
p <- p / rowSums(p)
nacional <- colSums(p * post$N) / sum(post$N) * 100

# ---- comparación con Stan -----------------------------------------------------
f_stan_cis <- file.path(dir_datos, "estim_stan_nuts_nuevo_cis.rds")
if (file.exists(f_stan_cis)) {
  est_stan <- readRDS(f_stan_cis)
  cat("comparación contra", basename(f_stan_cis), "\n")
  stan_col <- est_stan$mrp
  if (is.null(stan_col)) {
    stan_col <- est_stan$media
  }
  nombres_stan <- est_stan$partido
} else {
  est_stan <- readRDS(file.path(dir_datos, "estim_separado_nuts.rds")) |>
    filter(casa == "CIS")
  cat(
    "(sin ajuste solo-CIS de Stan: comparo con la columna CIS del modelo",
    "de 4 casas)\n"
  )
  stan_col <- est_stan$mrp
  nombres_stan <- est_stan$partido
}
i_stan <- match(voto_lv, as.character(nombres_stan))
res <- data.frame(
  partido = voto_lv,
  mclogit = round(nacional, 1),
  stan = round(stan_col[i_stan], 1),
  dif = round(nacional - stan_col[i_stan], 1)
)
cat("\n== nacional, electorado del 23-J (% voto válido) ==\n")
conv <- function(x) {
  c <- x$convergence
  if (is.null(c)) NA else c
}
saveRDS(
  list(
    res = res,
    convergencia = list(
      t_m0 = t_m0,
      t_m1 = t_m1,
      m0_conv = conv(m0),
      m1_conv = if (inherits(m1, "try-error")) NA else conv(m1)
    ),
    n_colin = n_colin
  ),
  file.path(dir_datos, "mclogit_resultado.rds")
)
cat("guardado datos/mclogit_resultado.rds\n")
print(res, row.names = FALSE)
