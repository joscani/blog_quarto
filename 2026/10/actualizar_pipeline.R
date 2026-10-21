# ============================================================================
# QUÉ TOCAR CUANDO LLEGA UNA ENCUESTA NUEVA
#
# Este fichero no se renderiza (es un .R); es la chuleta del pipeline de
# actualización. Los pasos ejecutables van comentados: descoméntalos o
# córrelos a mano en ese orden.
#
# La cadena completa, de menos a más caro:
#
#   1) preparar_todo(actualizar = TRUE)          [preparar_meta.R]
#      - actualizar_encuestas(solo_nuevas = TRUE): baja las encuestas nuevas
#        de las fuentes scrapeadas. Si la encuesta no está en ninguna fuente,
#        hay que meterla a mano en actualizar_encuestas().
#      - actualizar_objetivo176(): refresca la estimación de ciento76 que
#        entra como covariable del modelo.
#      - reconstruye los CSV de entrada: encuestas_meta_grande.csv,
#        encuestas_meta_grande_sep.csv, encuestas_meta_bloque.csv.
#
#   2) Render del post de meta-análisis (meta-analisis-generales-jlcr.qmd)
#      - Sus chunks llaman a ajustar_grande(refit = TRUE) y
#        ajustar_bloque(refit = TRUE): reajustan brms y refrescan
#        mod_meta_grande.rds y mod_meta_bloque.rds.
#      - El ajuste lleva seed fija (seed_fit): con los mismos datos da el
#        mismo posterior, así que re-renderizar sin encuestas nuevas no
#        mueve las cifras.
#
#   3) Render del post de escaños (meta-analisis-generales-escanos-jlcr.qmd)
#      - simular() llama por dentro a ajustar_grande()/ajustar_bloque()
#        (listas = "unidas", el default) y recoge los fits frescos de la
#        cache. Nada que pasarle.
#
# TRAMPAS:
#   - El ORDEN importa: si se renderiza solo el post de escaños sin pasar
#     antes por el de meta-análisis, simular() lee el fit viejo y la encuesta
#     nueva entra en los CSV pero no en el modelo, sin ningún aviso de error.
#     El chunk de setup del post de escaños compara fechas y lanza un warning
#     si detecta esta situación.
#   - Los escenarios listas = "separadas" y "mapa" usan ajustar_grande_sep()
#     (mod_meta_grande_sep.rds), que no reajusta ningún render. Los reajusta
#     el driver del post del mapa
#     (frente_amplio_y_adelante_andalucia/escenarios_mapa_29N.R, que hace
#     también el paso 1); si se usan fuera de él, refit a mano después del
#     paso 1 (ver abajo). "mapa" lee además la
#     columna regla de coaliciones_izquierda_29N_provincias.csv: si cambian
#     las coaliciones (listas en el BOE el 28 oct), se edita ese CSV.
#
# QUÉ NO HAY QUE TOCAR con una encuesta nueva (todo está calibrado con
# historia y solo cambia si cambia la metodología):
#   - los backtests congelados (backtest_*.csv y sus scripts)
#   - mod_tau_shock.rds y tau_regionales.csv (ruido geográfico)
#   - ref_provincial_23J_14.csv (geografía prestada de elecciones pasadas)
#   - factor_antelacion.csv (error por ventana de antelación)
#
# PENDIENTES conocidos (ver coda del post de escaños):
#   - swing en odds como posible mejora a futuro (candidato a post propio):
#     el backtest pareado (backtest_prov_swing.R) le da mejor cobertura
#     (96% vs 92%) con intervalos un 60% más anchos y el error intacto;
#     el default de simular() sigue siendo el ratio x share. Iría con los
#     tau reestimados en la escala del delta logit
#   - f_antelacion es hoy un mando manual (default 1 = última semana de
#     campaña); la simulación del post corre con 1, no con el factor de los
#     días que faltan hasta el 29N
# ============================================================================

# Descomentar y correr en orden:

source(here::here("2026/10/preparar_meta.R"))
preparar_todo(actualizar = TRUE)

# Solo para los escenarios "separadas" y "mapa":
#   source(here::here("2026/10/ajustar_meta.R"))
#   ajustar_grande_sep(refit = TRUE)

# Los dos renders se lanzan desde Positron/terminal:
#   quarto render 2026/10/meta-analisis-generales-jlcr.qmd
#   quarto render 2026/10/meta-analisis-generales-escanos-jlcr.qmd
