# Guía del post full bayesian

Todo lo que hay que saber para reproducir el post de arriba a abajo: de dónde
salen los datos, qué se ejecuta, qué asume el modelo y qué cosas despistan.
La versión corta es: **lanza el driver y él solo decide qué toca re-ajustar**.

```sh
Rscript 2026/10/full-bayesian/actualizar_full_bayesian.R
```

## 1. De dónde sale cada dato

| Dato | De dónde sale | Quién lo genera |
|---|---|---|
| Encuestas de las generales (2015-2026) | Scrape de las tablas de ciento76 y electomanía; relleno de los 7 regionales con el snapshot de objetivo176.es; prensa con extracción LLM ya congelada | `actualizar_encuestas()` en `descargar_encuestas_ciento76.R`, orquestado por `preparar_todo(actualizar = TRUE)` (2026/10/preparar_meta.R) |
| `encuestas_meta_grande_sep.csv` | Encuestas de 2026 con las 7 categorías (PP, PSOE, VOX, SUMAR, PODEMOS, SALF, BLOQUE), filtros: sin CIS, sin ElectoPanel, sin n sin publicar | `construir_meta_grande(separadas = TRUE)` |
| `encuestas_meta_bloque.csv` | Desglose regional de 2026: las 8 subceldas del BLOQUE (Junts…UPN + RESTO), en formato condicional al total del bloque | `construir_meta_bloque()` |
| `encuestas_meta_agregadas.csv` | Encuestas de casas que no separan toda la izquierda fuera del PSOE (Sigma Dos, Winston, Ateneo del Dato y Data10): PP, PSOE, VOX e izq = Sumar + Podemos, tal como publica la fuente | `construir_meta_agregadas()` (casas_agregadas) |
| Resultados de las 5 generales pasadas | `2026/10/errores_encuestas_historicas.csv` (congelado; lo generó el pipeline de octubre) | leído por `errores_hist()` de `01_datos.R` |
| Errores en 39 autonómicas (2018-2026) | Scrape de Wikipedia (página "Opinion polling" de cada elección) | `06_scrapear_autonomicas.R` → `errores_autonomicas.csv` |
| Geografía del 23J (ratio provincia/nación) | `2026/10/ref_provincial_23J_14.csv`, congelado | pipeline de octubre |
| Mapa de coaliciones de la izquierda (29-N) | `2026/10/coaliciones_izquierda_29N_provincias.csv` | a mano, con las listas del BOE; actualizado el 25 oct con la renuncia de CHA-Frente Amplio en Teruel (regla `sin_fa`: el voto de FA se reparte entre PSOE y fuera) |
| Shock provincial | `2026/10/mod_tau_shock.rds` y `tau_regionales.csv`, congelados | pipeline de octubre |
| Composición del BLOQUE en la simulación | Modelo 2 de brms (`mod_meta_bloque.rds`) | `ajustar_bloque()` (ajustar_meta.R) |

Lo que el modelo **no** usa: los fits de brms (los modelos 1 y 2 del
meta-análisis). El conjunto lee los CSV de encuestas directamente; los brms
solo hacen falta para la comparación con el pipeline de octubre (05).

## 2. La cadena, paso a paso

En orden de ejecución (los detalles de cada script están en las cabeceras):

1. **Datos.** `preparar_todo(actualizar = TRUE)`: scrape incremental (las
   encuestas ya congeladas no se re-scrapean), refresco de objetivo176 y
   reconstrucción de los CSV. Es idempotente.
2. **Modelo conjunto.** `12_ajustar_conjunto.R principal`: prior predictive,
   ajuste de Stan (4 cadenas, semilla 1811), diagnósticos, parámetros,
   tendencia semanal y escaños/mayorías con el mapa. Es lo más caro
   (~10-15 min).
3. **Derivados del fit.** `16_ablacion_error.R` (escaños solo con el voto
   latente) y `17_sensibilidad_omega.R` (la Ω con correlaciones alternativas).
4. **Comparación con octubre.** Refit de `ajustar_grande_sep(refit = TRUE)` y
   `05_comparar.R` (la referencia "Pipeline de octubre" de la figura y la
   tabla de ablación). El driver solo refita los brms cuyas entradas hayan
   cambiado.
5. **Backtest** (opcional, ~40 min). `13_backtest_conjunto.R principal
   26j,28a,10n,23j`: viste al modelo de víspera en las cuatro generales con
   resultado. **Una encuesta nueva de 2026 no lo afecta** (el backtest solo
   usa lo anterior a cada elección); solo hay que rehacerlo si cambian los
   datos históricos, las autonómicas o el propio `11_datos_conjunto.R`.
6. **Render.** `quarto render 2026/10/full-bayesian/full-bayesian.qmd`.
7. **Publicación.** `gh release upload full-bayesian-29n <fits> --clobber`
   (los `.rds` no van a git) y commit + push (los enlaces del post apuntan a
   `blob/master`).

## 3. El driver

`actualizar_full_bayesian.R` automatiza 1-7 con una decisión: solo re-ajusta
si los datos de entrada han cambiado de verdad.

```sh
Rscript 2026/10/full-bayesian/actualizar_full_bayesian.R            # lo normal
Rscript 2026/10/full-bayesian/actualizar_full_bayesian.R --forzar   # reajusta pese a no haber cambios
Rscript 2026/10/full-bayesian/actualizar_full_bayesian.R --con-backtest   # añade el 13
Rscript 2026/10/full-bayesian/actualizar_full_bayesian.R --sin-render     # solo los ajustes
Rscript 2026/10/full-bayesian/actualizar_full_bayesian.R --sin-publicar   # no toca la release
```

Cómo decide: toma la huella md5 de los cuatro ficheros de entrada
(`encuestas_meta_grande_sep.csv`, `encuestas_meta_bloque.csv`,
`encuestas_meta_agregadas.csv`, `errores_autonomicas.csv`), baja las
encuestas nuevas y vuelve a tomarla. Si no ha cambiado nada —porque la
encuesta nueva acaba excluida, como las de NC Report que no publican
SALF— no hay nada que re-ajustar y termina sin tocar nada. Si algo ha
cambiado, reajusta lo que depende de lo cambiado y nada más.

Lo que **no** hace (a mano, y no por encuesta nueva):

- `06_scrapear_autonomicas.R`: la historia autonómica está cerrada.
- Los backtests congelados y los modelos calibrados con historia
  (`mod_tau_shock.rds`, `tau_regionales.csv`, `factor_antelacion.csv`).
- Los posts anteriores (meta-análisis y escaños): su cadena está en
  `2026/10/actualizar_pipeline.R` y `2026/10/actualizar_con_encuestas_nuevas.md`.

## 4. Reglas de inclusión de encuestas (asunciones de datos)

- **Sin CIS ni ElectoPanel** en las categorías de los grandes (se separan
  sistemáticamente del resto; ElectoPanel sí entra en el desglose regional).
- **Sin n publicado → fuera** (en las históricas que no lo decían, se
  asumen 1.000).
- **La fuente tiene que separar todas las categorías del modelo.** Si no
  publica SALF pero sí la izquierda fuera del PSOE (Sigma Dos, Winston,
  Ateneo del Dato y Data10: "Frente Amplio + Podemos" o Sumar y Podemos
  sueltos), la encuesta entra **agregada**: con celdas PP, PSOE, VOX,
  Sumar + Podemos y resto, sin imputar nada — el modelo reparte el
  agregado con su posterior. Si ni siquiera publica ese agregado, fuera.
  Caso NC Report: no publica SALF pero sí las 7 regionales; con las celdas
  gruesas tiraríamos información que sí tenemos, así que de momento se
  queda fuera (merecería su propio esquema fino).
- **Ventanas**: los grandes desde julio 2026; el bloque, desde junio 2025.
- El campo se fecha por el **final del trabajo de campo**.
- En las encuestas con desglose regional, la probabilidad del desglose es
  condicional al total del BLOQUE (no se cuentan dos veces).

## 5. Asunciones del modelo

Las de la tabla "Todos los supuestos principales" del post, en corto:

1. El voto latente (lo que diría la media de las casas cada semana) sigue un
   **paseo aleatorio** en la escala softmax, con volatilidad propia de cada
   campaña (pooling parcial) y un multiplicador $e^\kappa$ en las 8 últimas
   semanas.
2. Cada casa tiene un **efecto propio** ($\beta$), centrado dentro de cada
   campaña: el voto latente es la media de las casas.
3. Las encuestas son **Dirichlet-multinomiales** ($\phi_c$ recoge la
   sobre-dispersión que no es muestreo).
4. El error del consenso (resultado real − voto latente del día E, en logit
   de PP, PSOE, VOX y SUMAR) es **normal multivariante** con sesgo, tamaño y
   correlación ($\mu$, $\sigma$, $\Omega$), aprendidos de las 5 generales con
   resultado y de las 39 autonómicas, que comparten $\Omega$ y el patrón de
   $\sigma$ con un nivel propio ($\gamma$).
5. Para 2026 el error se **sortea** de esa normal, un draw por simulación; el
   resto de familias absorbe la renormalización en proporción a su tamaño.
6. De votos a escaños: shock provincial, reglas del mapa, umbral del 3 % y
   D'Hondt (segunda parte del post).

## 6. Cosas que despistan

- **Warnings de `lkj_corr_cholesky` al principio del warmup** ("proposal is
  about to be rejected… Random variable is 0"): inofensivos. Stan rechaza la
  propuesta y sigue. Solo importan si continúan durante el sampling o van con
  divergencias.
- `renv: the project is out-of-sync`: aviso preexistente, no bloquea.
- "156 encuestas de prensa con regionales incompletos": aviso recurrente del
  scrape; solo importa si la encuesta incompleta es una que querrías usar.
- El scrape baja una encuesta nueva pero el driver dice "sin cambios": la
  encuesta entró en el fichero bruto pero quedó excluida por las reglas de
  arriba (mira la sección 4).
- Una encuesta sin n de Sigma Dos, Winston, Ateneo del Dato o Data10 no entra en `encuestas_meta_agregadas.csv` (`construir_meta_agregadas()` la filtra), así que no mueve el driver. Si ves una fila con la muestra vacía en ese CSV, el filtro se ha perdido.
- Semillas fijas (1811 en los Stan, 2029 en las simulaciones): con los mismos
  datos sale el mismo posterior, así que re-renderizar sin datos nuevos no
  mueve las cifras.
