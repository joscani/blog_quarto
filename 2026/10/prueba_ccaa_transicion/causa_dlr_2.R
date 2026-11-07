suppressMessages({library(tidyverse)})
suppressMessages({source(here::here("2026/10/backtest_provincial.R")); source(here::here("2026/10/shock_jerarquico.R"))})
preps <- suppressMessages(preparar_preps())
nac <- map_dfr(c("20d","26j","28a","10n","23j"), \(id) preps[[id]]$real_nac |> mutate(eleccion = id))
print(nac |> select(eleccion, familia, votos, pct) |> filter(familia %in% c("PP","VOX")), n = 20)
# Navarra: qué familia tiene UPN / NA+ en cada elección
nav <- map_dfr(c("20d","26j","28a","10n","23j"), \(id) preps[[id]]$prov |> filter(codigo_provincia == "31") |> mutate(eleccion = id))
print(nav |> select(eleccion, familia, votos, pct_prov), n = 60)
cat("infoelectoral:", as.character(packageVersion("infoelectoral")), "\n")
