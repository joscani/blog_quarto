# Tabla de postestratificación vieja (semilla de provincia) frente a la nueva
# (semilla de CCAA, 14 nov 2026), con el mismo posterior.
#
# Solo cambia la N de cada celda; part y el reparto del voto son los de
# posterior_celdas_80.rds (participación del 80 %, casa media). Así se aísla
# el efecto de la tabla.
#
# Entradas: datos/tabla_post_prov_semilla_prov.rds (release mrp-multiencuesta-datos),
#           completo/datos/tabla_post.rds y posterior_celdas_80.rds (21_posterior_celdas.R)
# Salida:   completo/datos/compara_semillas.rds (% medio por grupo, vieja y nueva)
#
#   Rscript 2026/11/mrp-multiencuesta/completo/31_compara_semillas.R   (~40 s)

library(dplyr)

dir_proy <- here::here("2026/11/mrp-multiencuesta")
dir_completo <- file.path(dir_proy, "completo", "datos")

celdas <- readRDS(file.path(dir_completo, "tabla_post.rds"))
vieja <- readRDS(file.path(dir_proy, "datos", "tabla_post_prov_semilla_prov.rds"))

# N de la tabla vieja en las mismas celdas
n_vieja <- as.data.frame.table(vieja, responseName = "N_vieja", stringsAsFactors = FALSE) |>
  transmute(cod = as.integer(prov),
            sexo = factor(sexo, c("H", "M"), c("Hombres", "Mujeres")),
            edad, edu, rec, N_vieja)
celdas <- celdas |>
  mutate(edad = as.character(edad), edu = as.character(edu), rec = as.character(rec)) |>
  left_join(n_vieja, by = c("cod", "sexo", "edad", "edu", "rec")) |>
  mutate(tramo = case_when(edad %in% c("18-20", "21-24", "25-34") ~ "18-34",
                           edad == "65+" ~ "65+",
                           .default = "35-64"))
# las dos tablas suman lo mismo y no queda N vieja fuera de estas celdas
stopifnot(!anyNA(celdas$N_vieja),
          abs(sum(celdas$N_vieja) - sum(vieja)) / sum(vieja) < 1e-6,
          abs(sum(celdas$N) - sum(vieja)) / sum(vieja) < 1e-6)

posterior <- readRDS(file.path(dir_completo, "posterior_celdas_80.rds"))
partidos <- setdiff(names(posterior), c("draw", "celda", "part"))
stopifnot(all(posterior$celda %in% celdas$celda))

# % de voto válido por grupo: votos = N * part * p, sumados en cada draw;
# después, media de los draws
pct_por <- function(grupo, columna_N) {
  posterior |>
    left_join(celdas |> select(celda, grupo = all_of(grupo), N = all_of(columna_N)),
              by = "celda") |>
    group_by(grupo, draw) |>
    summarise(across(all_of(partidos), \(p) sum(N * part * p)), .groups = "drop") |>
    mutate(total = rowSums(across(all_of(partidos))),
           across(all_of(partidos), \(v) 100 * v / total)) |>
    select(-total) |>
    group_by(grupo) |>
    summarise(across(all_of(partidos), mean)) |>
    tidyr::pivot_longer(-grupo, names_to = "partido", values_to = "pct")
}

compara <- function(nombre, grupo) {
  inner_join(pct_por(grupo, "N_vieja"), pct_por(grupo, "N"),
             by = c("grupo", "partido"), suffix = c("_vieja", "_nueva")) |>
    mutate(agregado = nombre, dif = pct_nueva - pct_vieja)
}

celdas <- celdas |> mutate(prov_tramo = paste(provincia, tramo), espana = "España")
res <- bind_rows(
  compara("España", "espana"),
  compara("Provincia", "provincia"),
  compara("Provincia x edad (3 tramos)", "prov_tramo")
)
# en cada grupo los % suman 100 con las dos tablas
stopifnot(all(abs(summarise(res, s = sum(pct_vieja), .by = c(agregado, grupo))$s - 100) < 1e-6),
          all(abs(summarise(res, s = sum(pct_nueva), .by = c(agregado, grupo))$s - 100) < 1e-6))

# Resumen: cuánto se mueven los % y en cuántos grupos cambia el primero
ganador <- res |>
  summarise(primero_vieja = partido[which.max(pct_vieja)],
            primero_nueva = partido[which.max(pct_nueva)],
            .by = c(agregado, grupo))
resumen <- res |>
  summarise(grupos = n_distinct(grupo),
            dif_mediana = median(abs(dif)),
            dif_p95 = quantile(abs(dif), 0.95),
            dif_max = max(abs(dif)),
            .by = agregado) |>
  left_join(ganador |> summarise(cambia_primero = sum(primero_vieja != primero_nueva), .by = agregado),
            by = "agregado")
print(resumen)

cambios <- ganador |> filter(primero_vieja != primero_nueva)
print(cambios, n = Inf)

saveRDS(list(res = res, resumen = resumen, cambios = cambios),
        file.path(dir_completo, "compara_semillas.rds"))
cat("guardado compara_semillas.rds\n")
