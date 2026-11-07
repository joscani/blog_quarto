suppressMessages({library(tidyverse)})
suppressMessages({source(here::here("2026/10/backtest_provincial.R")); source(here::here("2026/10/shock_jerarquico.R"))})
preps <- suppressMessages(preparar_preps())
d <- construir_dlr(preps)
f <- readRDS(here::here("2026/10/mod_tau_shock.rds"))$data
x <- d |> mutate(dif = dlr - f$dlr)
x |> group_by(familia, transition) |>
  summarise(min = min(dif), max = max(dif), n_dist = n_distinct(round(dif, 9)), .groups = "drop") |>
  filter(abs(min) > 1e-9 | abs(max) > 1e-9) |> print(width = Inf)
x |> filter(familia == "VOX", transition == "10n->23j") |> arrange(desc(abs(dif))) |>
  select(codigo_provincia, r0, r1, dlr, dif) |> head(6) |> print()
