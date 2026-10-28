# Exporta la encuesta unida (datos/encuestas.rds, salida de 01_armoniza.R) a
# SPSS (.sav, con etiquetas de variable y de valor), a CSV (texto, UTF-8) y a
# RDS (R, con factores).
# El codebook está en datos/entrega/codebook.md.
#
#   Rscript 2026/11/mrp-multiencuesta/04_exporta.R

library(haven)
library(dplyr)
library(readr)

dir_datos   <- "2026/11/mrp-multiencuesta/datos"
dir_entrega <- file.path(dir_datos, "entrega")
dir.create(dir_entrega, showWarnings = FALSE)

d <- readRDS(file.path(dir_datos, "encuestas.rds"))

# Nombres de provincia: las etiquetas de PROV del CIS 3582 (códigos del INE)
prov_etq <- attr(read_sav("descargas/MD3582/3582.sav", col_select = "PROV")$PROV, "labels")
prov_etq <- prov_etq[prov_etq %in% 1:52]

# ---------------------------------------------------------------------------
# 1. Versión de texto (CSV): una fila por entrevista, etiquetas en claro
# ---------------------------------------------------------------------------

d <- d |>
  mutate(id = row_number(), .before = 1) |>
  mutate(prov_nombre = unname(setNames(names(prov_etq), prov_etq)[as.character(prov)]),
         .after = prov)

write_csv(d, file.path(dir_entrega, "encuesta_unida.csv"), na = "")
saveRDS(d, file.path(dir_entrega, "encuesta_unida.rds"))   # R: con factores y sus niveles

# ---------------------------------------------------------------------------
# 2. Versión SPSS: factores -> números 1..k con etiquetas de valor
# ---------------------------------------------------------------------------

a_spss <- function(x, etiqueta) {
  if (is.factor(x)) {
    lv <- levels(x)
    x <- labelled(as.integer(x), setNames(seq_along(lv), lv), label = etiqueta)
  } else if (is.logical(x)) {
    x <- labelled(as.integer(x), c(No = 0L, `Sí` = 1L), label = etiqueta)
  } else {
    attr(x, "label") <- etiqueta
  }
  x
}

etiquetas <- c(
  id          = "Identificador de fila (no es el de la encuesta original)",
  casa        = "Casa encuestadora",
  estudio     = "Encuesta",
  mes         = "Mes de campo (AAAA-MM)",
  sexo        = "Sexo",
  edad_n      = "Edad en años (vacía en Ateneo del Dato)",
  edad        = "Tramo de edad",
  edu         = "Nivel de estudios terminados",
  edu_ambigua = "Estudios: la categoría original no separa media y alta",
  ccaa        = "Comunidad autónoma",
  prov        = "Provincia (código INE)",
  prov_nombre = "Provincia (nombre)",
  voto        = "Intención directa de voto en generales (solo voto válido)",
  voto_sim    = "Intención de voto + simpatía (solo voto válido)",
  rec         = "Recuerdo de voto en las generales del 23-J de 2023",
  prob_voto   = "Probabilidad de votar declarada (0-10)",
  vota_seguro = "Dice 10 en la probabilidad de votar",
  ideol01     = "Autoubicación ideológica reescalada a 0-1 (0 = izquierda)",
  peso        = "Peso de la propia encuesta (media 1 en cada encuesta)"
)
stopifnot(setequal(names(etiquetas), names(d)))

sav <- d
sav$casa    <- factor(sav$casa, levels = c("CIS", "40dB", "More in Common", "Ateneo del Dato"))
sav$estudio <- factor(sav$estudio, levels = unique(d$estudio))
sav$sexo    <- factor(sav$sexo, levels = c("H", "M"), labels = c("Hombre", "Mujer"))
sav$prov    <- labelled(sav$prov, setNames(as.integer(prov_etq), names(prov_etq)))
sav$prov_nombre <- NULL   # en el .sav va como etiqueta de valor de prov
sav$vota_seguro <- labelled(sav$vota_seguro, c(No = 0L, `Sí` = 1L))

for (v in names(sav)) {
  et <- etiquetas[[v]]
  if (inherits(sav[[v]], "haven_labelled")) attr(sav[[v]], "label") <- et
  else sav[[v]] <- a_spss(sav[[v]], et)
}

write_sav(sav, file.path(dir_entrega, "encuesta_unida.sav"))

# ---------------------------------------------------------------------------
# 3. Comprobaciones: el .sav se relee igual que el original
# ---------------------------------------------------------------------------

re <- read_sav(file.path(dir_entrega, "encuesta_unida.sav"))
stopifnot(nrow(re) == nrow(d))
for (v in c("casa", "estudio", "edad", "edu", "ccaa", "voto", "voto_sim", "rec")) {
  stopifnot(identical(as.character(as_factor(re[[v]])), as.character(d[[v]])))
}
stopifnot(all.equal(as.numeric(re$peso), d$peso))
re_csv <- read_csv(file.path(dir_entrega, "encuesta_unida.csv"), show_col_types = FALSE)
stopifnot(nrow(re_csv) == nrow(d), identical(names(re_csv), names(d)))
stopifnot(identical(readRDS(file.path(dir_entrega, "encuesta_unida.rds")), d))

cat("== entrevistas por encuesta ==\n")
print(d |> count(casa, estudio, mes), n = Inf)
cat("\nguardado en", dir_entrega, ":\n")
print(file.info(list.files(dir_entrega, full.names = TRUE))["size"])
