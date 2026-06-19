## Sin sentidos
library(dplyr)
set.seed(123)

n <- 1000 # número de observaciones

# Simular Evento1 y Evento2 como variables 0/1
Evento1 <- rbinom(n, size = 1, prob = 0.7) # 40% de 1s
Evento2 <- rbinom(n, size = 1, prob = 0.6) # 50% de 1s

# Definir la probabilidad del target según combinaciones
p_target <- ifelse(Evento1 == 1, 0.7, 0.2)

# También puedes añadir efecto de Evento2
p_target <- p_target + ifelse(Evento2 == 1, 0.3, -0.05)
p_target <- pmin(pmax(p_target, 0.01), 0.99) # limitar entre 0 y 1

# Simular target con esa probabilidad
target <- rbinom(n, size = 1, prob = p_target)

# Ponerlo en un data.frame
df <- data.frame(Evento1, Evento2, target)

# Comprobar proporciones
prop.table(table(df$target, df$Evento1), margin = 2)
prop.table(table(df$target, df$Evento2), margin = 2)


df |>
  group_by(Evento1, Evento2) |>
  summarise(
    mean(target)
  )


m_eff_princ <- glm(target ~ Evento1 + Evento2, data = df, family = binomial)
m_eff_interacc <- glm(target ~ Evento1 * Evento2, data = df, family = binomial)

BIC(m_eff_princ, m_eff_interacc)
