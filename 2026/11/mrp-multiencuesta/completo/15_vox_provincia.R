# VOX por provincia: bruto frente a MRP (escenario base 70,4 %, método seg). Ejemplo de pooling: Segovia y Soria.
# Exploración para el post (III), 2026-11-03. Datos sin ponderar de completo/datos/datos_voto_seg.rds.
#   Rscript 2026/11/mrp-multiencuesta/completo/15_vox_provincia.R   (desde la raíz del repo)

suppressMessages({library(dplyr); library(readr)})
x <- readRDS("2026/11/mrp-multiencuesta/completo/datos/datos_voto_seg.rds"); d <- x$datos; lv <- x$lv
r <- d$fila1; Y <- d$y[r,,drop=FALSE]; colnames(Y) <- lv$voto
bruto <- tibble(cod = as.integer(lv$prov[d$prov1]), ccaa = lv$ccaa[d$ccaa[r]],
                n = rowSums(Y), vox = Y[,"VOX"]) |>
  group_by(cod, ccaa) |> summarise(n=sum(n), vox=sum(vox), .groups="drop") |>
  mutate(bruto = 100*vox/n)
ccaa_b <- bruto |> group_by(ccaa) |> summarise(bruto_ccaa = 100*sum(vox)/sum(n), n_ccaa=sum(n))
e <- readRDS("2026/11/mrp-multiencuesta/completo/datos/escenarios_seg/escenarios.rds")
v <- e$votos["seg", 2, , , ]           # base 70,4 %: draw x provincia x partido
pv <- 100 * v[,,"VOX"] / apply(v, c(1,2), sum)
mrp <- tibble(cod = as.integer(e$prov_ref$codigo_provincia), provincia = e$prov_ref$provincia,
              mrp = apply(pv,2,median), q05 = apply(pv,2,quantile,.05), q95 = apply(pv,2,quantile,.95))
ref <- read_csv("2026/10/ref_provincial_23J_13.csv", show_col_types=FALSE) |> filter(partido=="VOX") |>
  transmute(cod = as.integer(codigo_provincia), vox23 = 100*pct_prov)
res <- mrp |> left_join(bruto, by="cod") |> left_join(ccaa_b, by="ccaa") |> left_join(ref, by="cod") |>
  mutate(salto = mrp - bruto, se_bruto = 100*sqrt(bruto/100*(1-bruto/100)/n))
cat("rango de vox23 (para saber la escala de pct_prov):", range(res$vox23), "\n")
print(res |> arrange(n) |> select(provincia, ccaa, n, vox, bruto, mrp, q05, q95, bruto_ccaa, vox23) |>
  mutate(across(where(is.double), \(z) round(z,1))) |> as.data.frame(), row.names=FALSE)
