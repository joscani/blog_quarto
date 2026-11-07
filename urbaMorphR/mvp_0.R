# 1. INSTALACIÓN DE DEPENDENCIAS ADICIONALES
# install.packages(c("osmdata", "ggplot2", "sf", "geos", "dplyr"))

library(sf)
library(geos)
library(dplyr)
library(osmdata)
library(ggplot2)

# =========================================================================
# 2. NUESTRA FUNCIÓN ULTRA-RÁPIDA (EL MOTOR DE TU LIBRERÍA)
# =========================================================================

library(sf)
library(geos)
library(dplyr)

urbamorph_tessellation_fast <- function(buildings, limit_buffer = 50) {
  old_warn <- options(warn = -1)
  on.exit(options(old_warn))

  # ── 1. Limpieza de input ───────────────────────────────────────────────────
  buildings <- buildings[st_is(buildings, c("POLYGON", "MULTIPOLYGON")), ]
  buildings <- st_make_valid(buildings)
  if (!"id" %in% names(buildings)) {
    buildings$id <- seq_len(nrow(buildings))
  }

  original_crs <- st_crs(buildings)
  geom <- st_geometry(buildings)

  # ── 2. Semillas: vértices del perímetro + centroides ──────────────────────
  # Normalizar a MULTIPOLYGON para que st_coordinates siempre devuelva
  # columnas (X, Y, L1, L2) con L2 como índice de geometría padre.
  geom_multi <- st_cast(geom, "MULTIPOLYGON")
  boundaries <- st_boundary(geom_multi)

  # Contar vértices por edificio de forma vectorizada (sin bucles R)
  coords <- st_coordinates(boundaries)
  lengths_nodes <- tabulate(coords[, ncol(coords)], nbins = length(geom))

  all_ids <- c(rep(buildings$id, times = lengths_nodes), buildings$id)
  all_pts <- c(st_cast(boundaries, "POINT"), st_centroid(geom))

  # ── 3. Voronoi via geos (núcleo C++, sin bucles R) ────────────────────────
  b_geos <- as_geos_geometry(geom)

  # IMPORTANTE: geos_unary_union es element-wise en vectores; hay que
  # colectar primero para obtener la unión global → casco convexo →
  # polígono ÚNICO de recorte (igual que momepy con convex_hull + buffer).
  study_area <- geos_make_collection(b_geos) |>
    geos_unary_union() |>
    geos_convex_hull() |>
    geos_buffer(limit_buffer)

  ext <- geos_extent(study_area)
  envelope <- geos_make_polygon(
    c(ext$xmin, ext$xmax, ext$xmax, ext$xmin, ext$xmin),
    c(ext$ymin, ext$ymin, ext$ymax, ext$ymax, ext$ymin)
  )

  voronoi_sfc <- geos_voronoi_polygons(
    geos_make_collection(as_geos_geometry(all_pts)),
    env = envelope
  ) |>
    st_as_sfc(crs = original_crs) |>
    st_collection_extract("POLYGON")

  # ── 4. Asignar celdas a edificios ─────────────────────────────────────────
  # Celdas de Voronoi son convexas → centroide interior → semilla más cercana
  # = semilla definitoria. Mismo mecanismo que momepy con sjoin_nearest().
  seeds_sf <- st_sf(id = all_ids, geometry = all_pts)
  voronoi_sf <- st_sf(geometry = voronoi_sfc)

  voronoi_centroids <- st_centroid(voronoi_sfc) |> st_sf()
  voronoi_sf$id <- all_ids[st_nearest_feature(voronoi_centroids, seeds_sf)]

  # ── 5. Disolver por edificio y recortar a la zona de estudio ──────────────
  voronoi_sf |>
    group_by(id) |>
    summarise(geometry = st_union(geometry), .groups = "drop") |>
    st_intersection(st_as_sfc(study_area, crs = original_crs))
}

# =========================================================================
# 3. DESCARGA DE DATOS REALES DE OPENSTREETMAP (Misma lógica que OSMnx)
# =========================================================================
message("Descargando manzanas de OpenStreetMap...")

# Elegimos una zona pequeña y densa (ej. Alrededores de la Plaza Mayor de Madrid)
bbox <- opq(bbox = c(-3.709, 40.413, -3.701, 40.418)) |>
  add_osm_feature(key = "building") |>
  osmdata_sf()

# Extraer solo los polígonos de los edificios
edificios_raw <- bbox$osm_polygons

# Momepy requiere una proyección métrica (UTM) para que las distancias tengan sentido físico
# Transformamos a EPSG:25830 (REGINA / UTM Zona 30N para España)
edificios <- st_transform(edificios_raw, 25830) |>
  select(geometry) |>
  mutate(id = row_number())

message(paste("Procesando", nrow(edificios), "edificios reales..."))

# =========================================================================
# 4. EJECUCIÓN Y BENCHMARK VISUAL
# =========================================================================
# Ejecutamos la teselación morfológica con un buffer límite de 35 metros
tic <- Sys.time()
capa_teselacion <- urbamorph_tessellation_fast(edificios, limit_buffer = 35)
toc <- Sys.time()

message(paste(
  "¡Teselación completada en un entorno real en:",
  round(toc - tic, 3),
  "segundos!"
))

# =========================================================================
# 5. MAPA DE RESULTADOS (Idéntico al estilo Matplotlib de Momepy)
# =========================================================================
ggplot() +
  # 1. Dibujar las "Celdas Morfológicas" (Teselación) de fondo en colores aleatorios
  geom_sf(
    data = capa_teselacion,
    aes(fill = as.factor(id %% 10)),
    color = "white",
    size = 0.2,
    alpha = 0.6
  ) +
  # 2. Dibujar las huellas de los edificios encima en gris oscuro
  geom_sf(data = edificios, fill = "#2b2b2b", color = NA) +
  # Estética limpia para papers/presentaciones
  scale_fill_viridis_d(option = "plasma", guide = "none") +
  labs(
    title = "Morphological Tessellation (R Prototype)",
    subtitle = paste(
      "Calculado sobre",
      nrow(edificios),
      "edificios en tiempo récord"
    )
  ) +
  theme_void() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40")
  )
