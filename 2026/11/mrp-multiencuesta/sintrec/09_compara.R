# Comparación de la sensibilidad sin recuerdo de voto (sintrec/) con la
# producción: % de voto (casa media, con y sin participación) y escaños.

library(dplyr)
library(tidyr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_datos <- file.path(dir_proy, "datos")
dir_sal <- file.path(dir_proy, "sintrec")

prod <- readRDS(file.path(dir_datos, "estim_separado_nuts.rds"))
sint <- readRDS(file.path(dir_sal, "estim_separado_nuts.rds"))

# ---- voto: casa media, con y sin participación --------------------------------
voto <- full_join(
  prod |>
    filter(grepl("casa media", casa)) |>
    select(casa, partido, mrp_prod = mrp, q05_prod = q05, q95_prod = q95),
  sint |>
    filter(grepl("casa media", casa)) |>
    select(casa, partido, mrp_sint = mrp, q05_sint = q05, q95_sint = q95),
  by = c("casa", "partido")
) |>
  mutate(
    dif = mrp_sint - mrp_prod,
    dentro_ic90 = q05_prod >= q05_sint & q95_prod <= q95_sint
  )

cat("\n== voto, casa media: producción frente a sin recuerdo ==\n")
print(
  as.data.frame(
    voto |>
      select(casa, partido, mrp_prod, mrp_sint, dif, dentro_ic90) |>
      mutate(across(where(is.numeric), ~ round(.x, 2)))
  ),
  row.names = FALSE
)
cat(
  "\ndif. máxima en el punto:",
  round(max(abs(voto$dif)), 2),
  " pp\n"
)

# ---- escaños ------------------------------------------------------------------
prod_e <- readRDS(file.path(dir_datos, "estim_escanos_mrp.rds"))
sint_e <- readRDS(file.path(dir_sal, "estim_escanos_mrp.rds"))

esc <- full_join(
  prod_e$resumen |>
    select(
      partido,
      mediana_prod = mediana,
      media_prod = media,
      q05_prod = q05,
      q95_prod = q95
    ),
  sint_e$resumen |>
    select(
      partido,
      mediana_sint = mediana,
      media_sint = media,
      q05_sint = q05,
      q95_sint = q95
    ),
  by = "partido"
) |>
  mutate(dif_media = media_sint - media_prod)

cat("\n== escaños: producción frente a sin recuerdo ==\n")
print(
  as.data.frame(
    esc |>
      mutate(across(where(is.numeric), ~ round(.x, 1)))
  ),
  row.names = FALSE
)

cat("\n== mayorías ==\n")
print(
  as.data.frame(
    full_join(
      prod_e$mayorias |> select(mayoria, prob_prod = prob),
      sint_e$mayorias |> select(mayoria, prob_sint = prob),
      by = "mayoria"
    ) |>
      mutate(across(where(is.numeric), ~ round(.x, 3)))
  ),
  row.names = FALSE
)
