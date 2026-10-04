# Definición común de partidos y territorios. La usan 02, 03, 04 y 05.

# SUMAR agrupa a todos los partidos de la coalición del 23-J, Podemos
# incluido. Los nacionalistas y regionalistas van cada uno por su lado.
partidos_lv <- c("PP", "PSOE", "VOX", "SUMAR",
                 "ERC", "JUNTS", "BILDU", "PNV", "BNG", "CCA", "OTROS")
rec_lv <- c(partidos_lv, "ABST", "NO_PODIA")

# Comunidades en las que se presenta cada partido regional (23-J). Fuera de
# ellas su probabilidad es un cero estructural.
territorio <- list(
  ERC   = "Cataluña",
  JUNTS = "Cataluña",
  BILDU = c("País Vasco", "Navarra"),
  PNV   = "País Vasco",
  BNG   = "Galicia",
  CCA   = "Canarias"
)

# TRUE si el partido p se puede votar en la comunidad ccaa
se_presenta <- function(p, ccaa) {
  p <- as.character(p)
  ccaa <- as.character(ccaa)
  out <- rep(TRUE, length(p))
  for (r in names(territorio)) {
    i <- !is.na(p) & p == r
    out[i] <- ccaa[i] %in% territorio[[r]]
  }
  out
}

# Matriz celdas x partidos con 1 donde el partido se presenta y 0 donde no
mascara_territorio <- function(ccaa, partidos = partidos_lv) {
  sapply(partidos, function(p) as.numeric(se_presenta(rep(p, length(ccaa)), ccaa)))
}

# Etiquetas del CIS -> partidos_lv. Lo que no se reconoce va a OTROS.
sumar_cis <- c("Sumar", "Podemos", "Unidas Podemos", "IU", "Compromís",
               "Más Madrid", "CHA")
regionales_cis <- c(ERC = "ERC", JUNTS = "Junts", BILDU = "EH Bildu",
                    PNV = "EAJ-PNV", BNG = "BNG", CCA = "CCa")

recod_cis_partido <- function(x, no_validos) {
  x <- as.character(x)
  out <- dplyr::case_when(
    x == "PP" ~ "PP",
    x == "PSOE" ~ "PSOE",
    x == "VOX" ~ "VOX",
    x %in% sumar_cis ~ "SUMAR",
    x %in% no_validos ~ NA_character_,
    TRUE ~ "OTROS"
  )
  reg <- match(x, regionales_cis)
  out[!is.na(reg)] <- names(regionales_cis)[reg[!is.na(reg)]]
  out
}
