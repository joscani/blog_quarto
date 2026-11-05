#!/usr/bin/env bash
# Baja de las releases lo necesario para trabajar en el post (III) en otra máquina.
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

echo "Datos en $dir_completo"
