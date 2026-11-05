# Seguros frente a dudosos: composición (edad, recuerdo) y voto estandarizado edad x recuerdo x casa, comparado con b_dud del modelo.
# Exploración para el post (III), 2026-11-03. Datos sin ponderar de completo/datos/datos_voto_seg.rds.
#   Rscript 2026/11/mrp-multiencuesta/completo/14_vox_seguro_dudoso.R   (desde la raíz del repo)

suppressMessages(library(dplyr)); library(tidyr)
x <- readRDS("2026/11/mrp-multiencuesta/completo/datos/datos_voto_seg.rds"); d <- x$datos; lv <- x$lv
r <- d$fila1; Y <- d$y[r,,drop=FALSE]; colnames(Y) <- lv$voto
cel <- tibble(casa = lv$casa[d$casa[r]], edad = lv$edad[d$edad[r]], rec = lv$rec[d$rec[r]],
              seg = ifelse(d$seg1==1,"dudoso","seguro"))
largo <- bind_cols(cel, as_tibble(Y)) |> pivot_longer(all_of(lv$voto), names_to="voto", values_to="y")
comp <- function(v) largo |> group_by(seg, g = .data[[v]]) |> summarise(n=sum(y), .groups="drop") |>
  group_by(seg) |> mutate(p = round(100*n/sum(n),1)) |> select(-n) |> pivot_wider(names_from=seg, values_from=p)
cat("== composición por edad (%) ==\n"); print(as.data.frame(comp("edad")))
cat("\n== composición por recuerdo (%) ==\n"); print(as.data.frame(comp("rec")))
# % VOX y PP dentro de cada edad, por seg
dentro <- function(v) largo |> group_by(g=.data[[v]], seg) |> summarise(VOX=100*sum(y[voto=="VOX"])/sum(y),
  PP=100*sum(y[voto=="PP"])/sum(y), n=sum(y), .groups="drop") |>
  pivot_wider(names_from=seg, values_from=c(VOX,PP,n)) |> mutate(across(where(is.numeric), \(z) round(z,1)))
cat("\n== % VOX y PP dentro de cada edad ==\n"); print(as.data.frame(dentro("edad")))
cat("\n== % VOX y PP dentro de cada recuerdo ==\n"); print(as.data.frame(dentro("rec")))
# estandarización directa: dudosos con la composición edad x rec x casa de los seguros
t <- largo |> group_by(casa, edad, rec, seg, voto) |> summarise(y=sum(y), .groups="drop") |>
  group_by(casa, edad, rec, seg) |> mutate(n=sum(y)) |> ungroup()
w <- t |> filter(seg=="seguro") |> distinct(casa,edad,rec,n) |> mutate(w=n/sum(n)) |> select(-n)
p_cel <- t |> mutate(p = y/n) |> select(casa,edad,rec,seg,voto,p) |> pivot_wider(names_from=seg, values_from=p)
est <- p_cel |> inner_join(w, by=c("casa","edad","rec")) |> filter(!is.na(dudoso), !is.na(seguro)) |>
  group_by(voto) |> summarise(seguro=sum(w*seguro)/sum(w), dudoso=sum(w*dudoso)/sum(w))
cubre <- w |> semi_join(p_cel |> filter(!is.na(dudoso)), by=c("casa","edad","rec")) |> summarise(sum(w))
bruto <- largo |> group_by(seg, voto) |> summarise(y=sum(y), .groups="drop") |> group_by(seg) |>
  mutate(p=y/sum(y)) |> select(-y) |> pivot_wider(names_from=seg, values_from=p)
lr <- function(a,b){l <- log(b/a); l - mean(l)}
res <- bruto |> rename(seg_b=seguro, dud_b=dudoso) |> inner_join(est, by="voto") |>
  mutate(lr_bruto = lr(seg_b,dud_b), lr_estand = lr(seguro,dudoso)) |>
  mutate(voto=factor(voto, lv$voto)) |> arrange(voto)
fit <- readRDS("2026/11/mrp-multiencuesta/completo/datos/fit_voto_seg.rds")$fit
b_dud <- posterior::as_draws_matrix(fit$draws("b_dud")) # columnas b_dud[j], en el orden de lv$voto
res$b_dud_modelo <- colMeans(b_dud)[paste0("b_dud[", as.integer(res$voto), "]")]
cat("\n== log(dudoso/seguro) centrado entre partidos: bruto, estandarizado (edad x rec x casa), modelo ==\n")
cat("peso de los seguros cubierto por celdas con dudosos:", round(cubre[[1]],3), "\n")
print(as.data.frame(res |> transmute(voto, seg_b=round(100*seg_b,1), dud_b=round(100*dud_b,1),
  seg_est=round(100*seguro,1), dud_est=round(100*dudoso,1), lr_bruto=round(lr_bruto,2), lr_estand=round(lr_estand,2), b_dud_modelo=round(b_dud_modelo,2))))
