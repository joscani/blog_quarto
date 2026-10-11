# ============================================================================
# ACTUALIZAR EL META-ANÁLISIS CON ENCUESTAS NUEVAS
#
# 1. actualiza encuestas_con_partidos_regionales.csv en modo incremental
#    (solo las encuestas nuevas; las ya congeladas no se re-scrapean)
# 2. recalcula Otros_residuo y reconstruye 2026/10/encuestas_meta.csv
#
# Ojo: la parte de prensa nueva (artículos sin extracción LLM de regionales)
# sigue requiriendo pasada manual; actualizar_encuestas() lo avisa si queda
# alguna pendiente.
#
# Uso:  Rscript 2026/10/actualizar_meta.R
# ============================================================================

source(here::here("2026/10/preparar_meta.R"))
preparar_todo(actualizar = TRUE)
