# De dónde sale la diferencia de VOX entre el MRP y el bruto: postestratificación por recuerdo x edad, casa media y resto.
# Exploración para el post (III), 2026-11-03. Datos sin ponderar de completo/datos/datos_voto_seg.rds.
#   Rscript 2026/11/mrp-multiencuesta/completo/16_vox_salto_nacional.R   (desde la raíz del repo)

suppressMessages({library(dplyr); library(tidyr); library(posterior)})
dir <- "2026/11/mrp-multiencuesta"
x <- readRDS(file.path(dir, "completo/datos/datos_voto_seg.rds")); d <- x$datos; lv <- x$lv
r <- d$fila1; Y <- d$y[r,,drop=FALSE]; colnames(Y) <- lv$voto
m <- tibble(casa = lv$casa[d$casa[r]], edad = lv$edad[d$edad[r]], rec = lv$rec[d$rec[r]],
            n = rowSums(Y), vox = Y[,"VOX"])
cat("== VOX bruto por casa ==\n")
print(m |> group_by(casa) |> summarise(n=sum(n), vox=round(100*sum(vox)/sum(n),1)) |> as.data.frame())

# población: N y votantes N*pi_cal (base 70,4 %), pi_cal medio sobre 200 draws de la participación
ent <- readRDS(file.path(dir, "completo/datos/entradas_escenarios.rds")); post <- ent$post; lvs <- ent$lv
ap <- readRDS(file.path(dir, "datos/fit_separado_nuts_part.rds"))$fit
dp <- as_draws_matrix(ap$draws(c("alpha_t","b_mujer_t","a_mes_t","a_edad_t","a_edu_t","a_rec_t","a_ccaa_t","a_prov_t")))
dp <- dp[round(seq(1, nrow(dp), length.out=200)), ]; dp <- matrix(as.numeric(dp), nrow(dp), dimnames=list(NULL, colnames(dp)))
ix <- function(v, l) match(as.character(v), l)
ie <- ix(post$edad, lvs$edad); iu <- ix(post$edu, lvs$edu); ir <- ix(post$rec, lvs$rec)
ic <- ix(post$ccaa, lvs$ccaa); ip <- ix(post$prov, lvs$prov); mu <- as.numeric(post$sexo=="M")
mes <- ent$mes_post; obj <- ent$part_obj; N <- post$N
cl <- function(v,i) dp[, paste0(v,"[",i,"]")]
w <- numeric(nrow(post))
for (k in seq_len(nrow(dp))) {
  et <- dp[k,"alpha_t"] + mu*dp[k,"b_mujer_t"] + dp[k,paste0("a_mes_t[",mes,"]")] +
    cl("a_edad_t",ie)[k,] + cl("a_edu_t",iu)[k,] + cl("a_rec_t",ir)[k,] + cl("a_ccaa_t",ic)[k,] + cl("a_prov_t",ip)[k,]
  dl <- uniroot(function(z) sum(N*plogis(et+z))/sum(N) - obj, c(-10,10))$root
  w <- w + N*plogis(et+dl)/nrow(dp)
}
post$V <- w
# distribución por recuerdo y por edad: muestra frente a población (residentes y votantes)
dist <- function(v) {
  a <- m |> group_by(g=.data[[v]]) |> summarise(muestra=sum(n)) |> mutate(muestra=100*muestra/sum(muestra))
  b <- post |> group_by(g=as.character(.data[[v]])) |> summarise(residentes=sum(N), votantes=sum(V)) |>
    mutate(residentes=100*residentes/sum(residentes), votantes=100*votantes/sum(votantes))
  vb <- m |> group_by(g=.data[[v]]) |> summarise(vox_bruto=100*sum(vox)/sum(n))
  full_join(a,b,by="g") |> left_join(vb, by="g") |> mutate(across(where(is.numeric), \(z) round(z,1)))
}
cat("\n== recuerdo: % muestra, % residentes, % votantes, VOX bruto dentro ==\n"); print(as.data.frame(dist("rec")))
cat("\n== edad ==\n"); print(as.data.frame(dist("edad")))
# estandarización directa paso a paso (medias brutas por grupo, pesos de población)
std <- function(vars, peso) {
  cel <- m |> group_by(across(all_of(vars))) |> summarise(p = sum(vox)/sum(n), .groups="drop")
  pob <- post |> mutate(across(all_of(vars), as.character)) |> group_by(across(all_of(vars))) |>
    summarise(W = sum(.data[[peso]]), .groups="drop")
  j <- inner_join(pob, cel, by=vars); round(100*sum(j$W*j$p)/sum(j$W),1)
}
cat("\n== VOX: bruto y reponderado (medias brutas de grupo, pesos de población) ==\n")
cat("bruto:", round(100*sum(m$vox)/sum(m$n),1), "\n")
for (vv in list("rec","edad",c("rec","edad"))) for (pp in c("N","V"))
  cat(paste(vv, collapse=" x "), "| pesos", ifelse(pp=="N","residentes","votantes"), ":", std(vv, pp), "\n")
res_esc <- read.csv(file.path(dir, "completo/datos/escenarios_seg/escenarios_voto.csv"))
base_esc <- paste0(format(round(100*obj, 1), decimal.mark=","), " %")
fila_mrp <- res_esc[res_esc$metodo=="seg" & res_esc$partido=="VOX" & res_esc$participacion==base_esc, ]
stopifnot(nrow(fila_mrp)==1)
cat("MRP (escenario base, seg):", round(fila_mrp$mediana,1), "\n")
cat("\n== por casa: bruto y reponderado rec x edad con pesos de votantes ==\n")
m_all <- m
res <- sapply(unique(m_all$casa), function(h) { m <<- m_all |> filter(casa==h)
  c(bruto = round(100*sum(m$vox)/sum(m$n),1), rec = std("rec","V"), rec_edad = std(c("rec","edad"),"V")) })
m <- m_all; print(res)
cat("media de casas (igual peso), bruto:", round(mean(res["bruto",]),1), " rec x edad:", round(mean(res["rec_edad",]),1), "\n")
cat("casas en el modelo (lv$casa):", paste(lv$casa, collapse=", "), "\n")
