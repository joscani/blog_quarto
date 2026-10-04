# Descarga y guarda en crudo las tres fuentes del post.
# Se ejecuta una vez; luego 02_harmoniza.R trabaja sobre los .rds.

library(ineapir)
library(infoelectoral)

dir_datos <- "2026/10/mrp-raking/datos"

# Los microdatos del CIS (estudio 3577) no se descargan aquí: hay que bajarlos
# a mano desde la web del CIS y dejar 3577.sav en datos/. Las rutas son
# relativas a la raíz del repositorio, así que se ejecuta desde ahí.

# --- 1. Población residente por sexo, edad (años simples) y CCAA -------------
# Estadística Continua de Población, tabla 59238
pob <- get_data_table(idTable = 59238, nlast = 1, unnest = TRUE,
                      metanames = TRUE, metacodes = TRUE, tip = "AM")
saveRDS(pob, file.path(dir_datos, "ine_pob_sexo_edad_ccaa.rds"))
cat("ECP 59238:", nrow(pob), "filas\n")

# --- 2. Nivel de formación por sexo y CCAA ----------------------------------
# EPA, tabla 65288. Hay varias tablas con el mismo nombre: las 66xxx son la
# serie antigua (CNED-2000), que termina en 2013. Conviene comprobar siempre
# la fecha de lo que devuelve la API.
edu_ccaa <- get_data_table(idTable = 65288, nlast = 1, unnest = TRUE,
                           metanames = TRUE, metacodes = TRUE, tip = "AM")
saveRDS(edu_ccaa, file.path(dir_datos, "epa_edu_sexo_ccaa.rds"))
cat("EPA 65288:", nrow(edu_ccaa), "filas, periodo",
    unique(substr(edu_ccaa$Fecha, 1, 10)), "\n")

# --- 4. Resultados de las generales del 23J ---------------------------------
# Ya descargado en datos/prov_23j.rds por infoelectoral::provincias()
if (!file.exists(file.path(dir_datos, "prov_23j.rds"))) {
  r <- provincias(tipo_eleccion = "congreso", anno = "2023", mes = "07")
  saveRDS(r, file.path(dir_datos, "prov_23j.rds"))
}

# Por municipio: solo residentes en España (CER), sin el voto CERA
if (!file.exists(file.path(dir_datos, "mun_23j.rds"))) {
  m <- municipios(tipo_eleccion = "congreso", anno = "2023", mes = "07")
  saveRDS(m, file.path(dir_datos, "mun_23j.rds"))
}

# --- Inventario de categorías de cada fuente --------------------------------
cat("\n===== categorías =====\n")
cat("\n-- ECP: sexo --\n");  print(unique(pob$Sexo))
cat("\n-- ECP: CCAA --\n");  print(unique(pob[["Comunidades.y.Ciudades.Autónomas"]]))
cat("\n-- ECP: edades (primeras) --\n"); print(head(unique(pob$Edad), 4))
cat("\n-- EPA 65288: nivel de formación --\n")
print(unique(edu_ccaa[[grep("ormaci", names(edu_ccaa), value = TRUE)[1]]]))
