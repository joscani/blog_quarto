## ----simulacion------------------------------------------------------------
library(tidyverse)
library(lme4)

set.seed(34)

grand_mean <- -3
n_prov <- 20
n_mun_het <- 3 # prov_het: pocos municipios
n_mun_resto <- 8 # resto: más municipios
n_obs <- 20

ef_prov <- c(
  0, # prov_het: efecto real = 0
  -5, # prov_hom: efecto real = -5
  rnorm(n_prov - 2, 0, 0.6) # resto: moderadas
)

# prov_het: 3 municipios muy dispares, media = -10  →  media provincia ≈ -13
# pero sabemos que el efecto real de la provincia es 0, , y junto con la
# grand mean debería ser -3,
# pocos municipios, con media -10, pues se va a -13, (-10 - 3)

ef_mun_het <- c(-35, -10, 15) # media = -10, sd ≈ 20.2

# prov_hom: 8 municipios compactos  →  efecto  provincia ≈ -5 y junto con
# efecto global (-3) se va a -8
ef_mun_hom <- rnorm(n_mun_resto, 0, 0.3)

datos <- bind_rows(
  # prov_het
  map_dfr(seq_len(n_mun_het), function(m) {
    tibble(
      provincia = "prov_01",
      municipio = paste0("prov_01_mun_", m),
      tipo = "het (efecto real = 0, n_mun = 3, media ≈ -13)",
      y = grand_mean + ef_prov[1] + ef_mun_het[m] + rnorm(n_obs, 0, 0.5)
    )
  }),
  # prov_hom y resto
  map_dfr(2:n_prov, function(p) {
    n_m <- n_mun_resto
    tipo <- if (p == 2) {
      "hom (efecto real = -5, n_mun = 8, media ≈ -8)"
    } else {
      "resto"
    }
    ef_m <- if (p == 2) ef_mun_hom else rnorm(n_m, 0, 0.3)
    map_dfr(seq_len(n_m), function(m) {
      tibble(
        provincia = paste0("prov_", sprintf("%02d", p)),
        municipio = paste0("prov_", sprintf("%02d", p), "_mun_", m),
        tipo = tipo,
        y = grand_mean + ef_prov[p] + ef_m[m] + rnorm(n_obs, 0, 0.5)
      )
    })
  })
)


## ----plot-municipios-------------------------------------------------------

datos |>
  filter(provincia %in% c("prov_01", "prov_02")) |>
  group_by(provincia) |>
  summarise(mean(y))


datos |>
  filter(provincia %in% c("prov_01", "prov_02")) |>
  ggplot(aes(x = municipio, y = y)) +
  geom_jitter(width = 0.1, alpha = 0.3, color = "steelblue") +
  stat_summary(fun = mean, geom = "point", size = 3, color = "firebrick") +
  stat_summary(
    fun = mean,
    geom = "hline",
    aes(yintercept = after_stat(y), group = provincia),
    linetype = "dashed",
    color = "firebrick"
  ) +
  facet_wrap(~tipo, scales = "free_x") +
  labs(
    x = NULL,
    y = "y",
    title = "Municipios dentro de cada provincia de interés",
    subtitle = "Puntos rojos = media del municipio  |  línea = media de la provincia"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))


## ----modelos---------------------------------------------------------------
# En un modelo no tenemos en cuenta la estructura municipal. pero en el otro si
# No tener en cuenta la estructura palmamos pasta encima de la mesa

mod_sin_mun <- lmer(y ~ 1 + (1 | provincia), data = datos)
mod_con_mun <- lmer(y ~ 1 + (1 | provincia) + (1 | municipio), data = datos)


## ----varianzas-------------------------------------------------------------
get_var <- function(mod, nivel) {
  as.data.frame(VarCorr(mod)) |> filter(grp == nivel) |> pull(vcov)
}

data.frame(
  componente = c(
    "sigma2_provincia  (sin municipios)",
    "sigma2_provincia  (con municipios)",
    "sigma2_municipio  (con municipios)",
    "sigma2_residual   (sin municipios)",
    "sigma2_residual   (con municipios)"
  ),
  varianza = c(
    get_var(mod_sin_mun, "provincia"),
    get_var(mod_con_mun, "provincia"),
    get_var(mod_con_mun, "municipio"),
    attr(VarCorr(mod_sin_mun), "sc")^2,
    attr(VarCorr(mod_con_mun), "sc")^2
  )
)


# cuando no tenemos en cuenta los municipios el sigma de provinaica es 6,
# si lo tenemos en cuenta se va a 2 y el sigma de municipios a casi 10, pero
# si nos fijamos vermos que la varianza residual es mucho menor en el modelo qeu tieene
# en cuenta los municipios. Tener en cuenta los municipios nos explica mayor variabilidad
# de los datos.

## ----predicciones----------------------------------------------------------

# Si queremos hacer predición a nivel de provincia, una forma de hacerlo es usar
# re.form y especificar solo los efectos hasta el nivel máximo que tenemos.
# Esto no reajusta el modelo, sino que simplemeente se queda con el nivel correcto
# (investigar más sobre los re.form y contar mejor. )

datos_prov <- datos |> distinct(provincia, tipo) |> arrange(provincia)

# Construimos dataset con lo que hace falta, que es la provincia como variables
datos_prov

# unimos las dos predicciones

preds <- bind_rows(
  datos_prov |>
    mutate(
      pred = predict(
        mod_sin_mun,
        newdata = datos_prov,
        re.form = ~ (1 | provincia)
      ),
      modelo = "Sin municipios"
    ),
  datos_prov |>
    mutate(
      pred = predict(
        mod_con_mun,
        newdata = datos_prov,
        re.form = ~ (1 | provincia)
      ),
      modelo = "Con municipios"
    )
)

print(preds, n = Inf)


# El modelo que ha tenido en cuenta que hay varianza a nivel de municipio
# ha sido capaz de hacer un shrinkage más agresivo y de -13 pasa a -7.5.
# Con los datos que tiene es lo mejor que puede hacer, la media de provincia
# bruta es de -13, como sabe que hay mucha variabilidad pues contrae.

# De la provincia 2 como los municipios son muy parecidos y heterógeneos contrae
# menos y lo lleva hacia -6. . el efecto real es -5, pero aunque sean pocos
# municipios como son homogeneos, no lo mueve mucho.

preds |>
  mutate(
    provincia = fct_reorder(provincia, pred),
    forma = tipo
  ) |>
  ggplot(aes(x = pred, y = provincia, color = modelo, shape = forma)) +
  geom_vline(xintercept = grand_mean, linetype = "dashed", color = "grey50") +
  geom_point(size = 3, alpha = 0.85, position = position_dodge(width = 0.55)) +
  scale_color_manual(
    values = c(
      "Sin municipios" = "firebrick",
      "Con municipios" = "steelblue"
    )
  ) +
  scale_shape_manual(
    values = c(
      "het (efecto real = 0, n_mun = 3, media ≈ -13)" = 17,
      "hom (efecto real = -5, n_mun = 8, media ≈ -8)" = 15,
      "resto" = 16
    )
  ) +
  labs(
    x = "Predicción a nivel provincia  (re.form = ~(1|provincia))",
    y = NULL,
    color = NULL,
    shape = NULL,
    title = "Predicciones de provincia: sin vs con municipios",
    subtitle = "Línea = media nacional (−3)  |  triángulo = het  |  cuadrado = hom"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom", legend.direction = "vertical")


## ----tabla-estimaciones----------------------------------------------------
preds |>
  pivot_wider(names_from = modelo, values_from = pred) |>
  mutate(
    media_bruta = map_dbl(provincia, ~ mean(datos$y[datos$provincia == .x])),
    cambio = `Con municipios` - `Sin municipios`
  ) |>
  arrange(`Sin municipios`) |>
  mutate(across(where(is.numeric), \(x) round(x, 2))) |>
  select(
    provincia,
    tipo,
    media_bruta,
    `Sin municipios`,
    `Con municipios`,
    cambio
  ) |>
  knitr::kable(
    col.names = c(
      "Provincia",
      "Tipo",
      "Media bruta",
      "Pred sin mun",
      "Pred con mun",
      "Cambio"
    ),
    caption = "Predicciones a nivel provincia (re.form = ~(1|provincia))"
  )

# La lección es que aunque nuestras estimaciones que queresmo sean a un nivle
# superior, si sabemos que hay estructura a niveles inferiores, tenerla en cuenta
# es importante. Si no, considera que todas las observaciones de una provincia
# son observaciones iid de la provincia y como no hay nada que lo estructure, c
# son pseudoréplicas. (catas en mismo sitio)
