# Reproducible, simple y distinto del pipeline del curro
library(DBI)
library(duckdb)
library(dplyr)
library(purrr)
set.seed(2025)

con <- dbConnect(duckdb::duckdb(), dbdir=":memory:")

# --- Jerarquía simple: Estado -> Provincia -> Comarca -> Municipio ---
# Codificación mínima por subcadenas: S, Sxx, SxxCyy, SxxCyyMzz
estado_id <- "S"

prov_ids <- sprintf("S%02d", 1:3)                       # 3 provincias
com_por_prov <- c(3, 3, 4)                              # #comarcas por provincia
com_ids <- map2(prov_ids, com_por_prov,
                ~ sprintf("%sC%02d", .x, seq_len(.y))) |> list_c()

# municipios por comarca (reparto sencillo 4-6)
muni_por_com <- sample(4:6, length(com_ids), replace = TRUE)
muni_ids <- map2(com_ids, muni_por_com,
                 ~ sprintf("%sM%02d", .x, seq_len(.y))) |> list_c()

# --- Simulación de anuncios a nivel municipio (sin pesos) ---
# price = nivel_estado + efecto_prov + efecto_com + efecto_muni + ruido anuncio
mu_estado <- 100000
sd_prov   <- 6000
sd_com    <- 8000
sd_muni   <- 9000
sd_ad     <- 12000

sim_ads <- function(mid) {
  prov <- substr(mid, 1, 3)        # Sxx
  com  <- substr(mid, 1, 6)        # SxxCyy

  mu_p <- mu_estado + rnorm(1, 0, sd_prov)
  mu_c <- mu_p      + rnorm(1, 0, sd_com)
  mu_m <- mu_c      + rnorm(1, 0, sd_muni)

  n    <- sample(5:200, 1)
  tibble(
    ad_id   = paste0(mid, "_A", seq_len(n)),
    loc_id  = mid,
    price   = rnorm(n, mu_m, sd_ad)
  )
}

ads <- map_dfr(muni_ids, sim_ads)
dbWriteTable(con, "ads", ads, overwrite = TRUE)

# Tabla de “cortes” para substrings por nivel (longitudes)
cuts <- tibble::tribble(
  ~level, ~cut_len,
  "state",      1,     # "S"
  "province",   3,     # "Sxx"
  "comarca",    6,     # "SxxCyy"
  "municipio",  9      # "SxxCyyMzz"
)
dbWriteTable(con, "cuts", cuts, overwrite = TRUE)

