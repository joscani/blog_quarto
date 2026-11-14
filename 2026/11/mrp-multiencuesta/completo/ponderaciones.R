# Ponderación clásica por raking. La usan 20_preguntas.R (3.12),
# 23_bruto_vs_mrp_edad.R y 24_mapa_ganador_bruto_mrp.R.
#
# Ajusta los pesos de las entrevistas de una comunidad para que sus marginales
# coincidan con la población de 18 y más (la N de tabla_post.rds):
#   pondera_sin_rec(): sexo, edad, estudios y provincia. Lo que hace una casa
#                      que pondera por sociodemografía pero no por voto pasado.
#   pondera_con_rec(): lo mismo y además el recuerdo de voto del 23-J.
#
# Se pondera toda la muestra (también quien no da partido) y después el % se
# calcula sobre quien da partido, como en el bruto.
# Quedan fuera las entrevistas a las que les falta alguna variable del raking:
# no se pueden ponderar por ella.
# Si un nivel no tiene ninguna entrevista (p. ej. 18-20 con estudios altos en
# el País Vasco), no se puede ponderar: ese marginal se ajusta sin él.

library(survey)

# encuestas: entrevistas de la comunidad (con o sin voto_sim)
# celdas:    tabla de postestratificación de la comunidad
# vars:      variables del raking (cod_f es la provincia como factor)
# Devuelve las entrevistas ponderables con una columna `peso_rk`.
pondera_raking <- function(encuestas, celdas, vars) {
  m <- encuestas |>
    mutate(sexo = factor(if_else(sexo == "H", "Hombres", "Mujeres"), c("Hombres", "Mujeres")),
           cod_f = factor(prov)) |>
    filter(if_all(all_of(vars), \(x) !is.na(x))) |>
    mutate(across(all_of(intersect(vars, c("edad", "edu", "rec"))), droplevels))

  # marginales de población, en el formato de survey::rake (variable + Freq)
  margen <- function(v) {
    celdas |>
      mutate(cod_f = factor(cod)) |>
      summarise(Freq = sum(N), .by = all_of(v)) |>
      filter(.data[[v]] %in% unique(m[[v]]))
  }

  diseno <- svydesign(ids = ~1, data = m, weights = rep(1, nrow(m)))
  rk <- rake(diseno,
             sample.margins = lapply(vars, \(v) as.formula(paste("~", v))),
             population.margins = lapply(vars, margen),
             control = list(maxit = 100, epsilon = 1e-6))

  m |> mutate(peso_rk = weights(rk)) |> select(-cod_f)
}

pondera_sin_rec <- function(encuestas, celdas) {
  pondera_raking(encuestas, celdas, c("sexo", "edad", "edu", "cod_f"))
}

pondera_con_rec <- function(encuestas, celdas) {
  pondera_raking(encuestas, celdas, c("sexo", "edad", "edu", "cod_f", "rec"))
}
