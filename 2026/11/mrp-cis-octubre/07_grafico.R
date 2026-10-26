# El gráfico de comparación del post de octubre (plot-comparacion), con el
# estudio 3582. Lee lo que guarda 05_mrp.R.

library(dplyr)
library(tidyr)
library(ggplot2)

dir_datos <- "2026/11/mrp-cis-octubre/datos"
source("2026/11/mrp-cis-octubre/00_partidos.R")   # partidos_lv
voto_lv <- partidos_lv

est <- readRDS(file.path(dir_datos, "estimaciones.rds"))

# Mismos nombres de columna que en el post
comparacion <- est$comparacion %>%
  select(partido, bruto, pesos_cis = peso_cis, mrp, cis_publicado = cis) %>%
  left_join(est$est_mrp %>% select(partido, q05, q95), by = "partido")

print(as.data.frame(comparacion %>% mutate(across(-partido, ~round(.x, 1)))),
      row.names = FALSE)

g <- comparacion %>%
  pivot_longer(
    c(bruto, pesos_cis, mrp, cis_publicado),
    names_to = "metodo",
    values_to = "pct"
  ) %>%
  mutate(
    metodo = factor(
      metodo,
      levels = c("bruto", "pesos_cis", "mrp", "cis_publicado"),
      labels = c("bruto", "pesos CIS", "MRP", "CIS publicado")
    ),
    partido = factor(
      partido,
      levels = rev(voto_lv),
      labels = rev(c(
        "PP", "PSOE", "VOX", "SUMAR y aliados", "ERC", "Junts",
        "Aliança Catalana", "EH Bildu", "PNV", "BNG", "CC", "OTROS"
      ))
    ),
    q05 = if_else(metodo == "MRP", q05, NA_real_),
    q95 = if_else(metodo == "MRP", q95, NA_real_)
  ) %>%
  ggplot(aes(x = pct, y = partido, color = metodo)) +
  geom_pointrange(
    aes(xmin = q05, xmax = q95),
    position = position_dodge(width = 0.6),
    na.rm = TRUE
  ) +
  scale_color_manual(values = c("grey60", "steelblue", "firebrick", "black")) +
  labs(
    x = "% sobre voto válido",
    y = NULL,
    color = NULL,
    title = "Intención de voto según el método",
    subtitle = "CIS 3582, barómetro de octubre 2026. MRP con intervalo al 90 %"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

ggsave(file.path(dir_datos, "comparacion_3582.png"), g,
       width = 8, height = 5, dpi = 300, bg = "white")
cat("guardado comparacion_3582.png\n")
