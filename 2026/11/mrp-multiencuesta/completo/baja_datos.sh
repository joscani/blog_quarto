#!/usr/bin/env bash
# Baja de las releases lo necesario para trabajar en los posts (III) y (IV) en otra máquina.
# Requiere gh con sesión iniciada. Ejecutar desde cualquier carpeta del repositorio:
#   bash 2026/11/mrp-multiencuesta/completo/baja_datos.sh
set -euo pipefail

raiz="$(git rev-parse --show-toplevel)"
dir_proy="$raiz/2026/11/mrp-multiencuesta"
dir_completo="$dir_proy/completo/datos"
mkdir -p "$dir_completo" "$dir_proy/datos"

# ajuste de Stan, datos del modelo, escenarios y entradas
gh release download mrp-multiencuesta-completo --repo joscani/blog_quarto \
  --dir "$dir_completo" --clobber
tar -xzf "$dir_completo/escenarios_seg.tar.gz" -C "$dir_completo"

# la participación del (II), que usan 12_escenarios_voto_seg.R y 16_vox_salto_nacional.R
gh release download mrp-multiencuesta-datos --repo joscani/blog_quarto \
  --pattern fit_separado_nuts_part.rds --dir "$dir_proy/datos" --clobber

# tabla de población y entradas pequeñas (hasta el 14 nov 2026 iban en git).
# tabla_post_prov_semilla_prov.rds: la tabla anterior (semilla por provincia),
# con la que se renderizaron los posts (I) y (II).
for f in tabla_post_prov.rds tabla_post_prov_semilla_prov.rds prov_ccaa.rds \
         celdas_brms.rds celdas_part.rds; do
  gh release download mrp-multiencuesta-datos --repo joscani/blog_quarto \
    --pattern "$f" --dir "$dir_proy/datos" --clobber
done

# post (IV): la release mrp-multiencuesta-completo ya trae tabla_post.rds,
# posterior_celdas_80.rds (unos 800 MB) y las salidas de 31-35. Falta el 23-J por
# municipio, que usa 33_navarra_23j_zonas.R
mkdir -p "$raiz/2026/10/mrp-raking/datos"
gh release download mrp-multiencuesta-datos --repo joscani/blog_quarto \
  --pattern mun_23j.rds --dir "$raiz/2026/10/mrp-raking/datos" --clobber

echo "Datos en $dir_completo"
