# Actualizar todo con encuestas nuevas

Documento-chuleta con la cadena completa de actualización: desde que llega una encuesta nueva hasta que los cuatro posts de la serie de generales están re-ajustados y renderizados. Extiende la chuleta de `actualizar_pipeline.R` (que cubre solo los dos primeros posts y el del mapa) con el modelo conjunto del post *full bayesian*.

Para el post de full bayesian en solitario, mejor el driver `2026/10/full-bayesian/actualizar_full_bayesian.R` (detecta si hay cambios de verdad y solo reajusta lo que toca); los detalles, en `2026/10/full-bayesian/GUIA.md`.

Se ejecuta todo **desde la raíz del proyecto** (para que se active `renv`). Fuera de esta cadena quedan `mrp-raking` (datos de INE, EPA y CIS, no de ciento76) y la carpeta `2026/11/`.

## Mapa de dependencias

``` mermaid
flowchart TD
  A["Fuentes scrapeadas<br/>(electomanía, etc.)"] --> B["encuestas_con_partidos_regionales.csv<br/>(actualizar_encuestas, incremental)"]
  C["objetivo176.es<br/>(encuestas.json)"] --> B
  B --> D1["encuestas_meta_grande.csv<br/>(6 categorías)"]
  B --> D2["encuestas_meta_grande_sep.csv<br/>(7 categorías)"]
  B --> D3["encuestas_meta_bloque.csv<br/>(8 subceldas del bloque)"]
  B --> D4["encuestas_meta_agregadas.csv<br/>(casas sin desglose de la izquierda)"]
  D1 --> M1["mod_meta_grande.rds<br/>(modelo 1, post meta-análisis)"]
  D3 --> M2["mod_meta_bloque.rds<br/>(modelo 2, post meta-análisis)"]
  D2 --> M3["mod_meta_grande_sep.rds<br/>(escenarios separadas y mapa)"]
  M1 --> P2["Post de escaños<br/>(meta-analisis-generales-escanos)"]
  M3 --> FA["Driver del mapa<br/>(escenarios_mapa_29N.R)"]
  D1 --> FB["Modelo conjunto<br/>(12_ajustar_conjunto.R)"]
  D2 --> FB
  D3 --> FB
  D4 --> FB
```

Puntos clave que se derivan del mapa:

- El modelo conjunto **no depende de los fits de brms**: lee los CSV de encuestas directamente. Por eso puede ir en cualquier orden *respecto a los modelos 1 y 2*. Lo que sí necesita fresco es el paso 1 (los CSV).
- El escenario `listas = "mapa"` de `simular()` usa `mod_meta_grande_sep.rds`, que **no se reajusta en ningún render**: lo reajusta el driver del mapa. Si se usa `simular(listas = "mapa")` fuera de él, hay que refitarlo a mano.

## Paso 0 · Ver qué hay nuevo (solo lectura)

Antes de tocar nada, compara lo que publica ciento76 con el CSV local. Usa el scraper, pero no escribe nada:

``` r
source(here::here("2026/10/preparar_meta.R"))
cargar_pipeline_encuestas()
web <- scrape_ciento76()
local <- readr::read_csv(here::here("encuestas_con_partidos_regionales.csv"), show_col_types = FALSE)
web |>
  dplyr::filter(!enlace %in% local$enlace) |>
  dplyr::select(Encuestadora, Campo, Muestra, PP, PSOE, VOX, Sumar, Podemos, SALF, enlace)
```

Si no sale ninguna fila, no hay nada que actualizar. Para cada fila, mira si entra según las reglas de la sección 4 de [`full-bayesian/GUIA.md`](full-bayesian/GUIA.md): sin n publicado, fuera de los modelos; sin CIS ni ElectoPanel en las categorías grandes; para el modelo 1 hacen falta las seis principales, SALF incluida. Al reajustar, `ajustar_meta.R` imprime `Fuera (sin n publicado)` con las encuestas excluidas por falta de n.

## Paso 1 · Actualizar los datos (incremental)

``` r
Rscript 2026/10/actualizar_meta.R
```

Que no es más que `source("2026/10/preparar_meta.R"); preparar_todo(actualizar = TRUE)`. Hace, en orden:

1.  `actualizar_encuestas(solo_nuevas = TRUE)`: baja las encuestas nuevas de las fuentes scrapeadas. **Incremental**: las encuestas ya congeladas no se re-scrapean. Si la encuesta nueva no está en ninguna fuente, hay que meterla a mano en `actualizar_encuestas()`.
2.  `actualizar_objetivo176()`: refresca el snapshot de `descargas/objetivo176/encuestas.json`, que es la fuente de relleno de los 7 regionales (y la covariable de ciento176).
3.  Reconstruye los CSV de entrada: `ref_provincial_23J.csv`, `encuestas_meta_grande.csv` (6 categorías), `encuestas_meta_grande_sep.csv` (7 categorías, con SUMAR y PODEMOS separadas; entrada de `ajustar_grande_sep()`), `encuestas_meta_bloque.csv` y `encuestas_meta_agregadas.csv` (casas que no separan la izquierda; entrada del modelo conjunto).

**Argumentos:**

- `preparar_todo(actualizar = FALSE)`: reconstruye los CSV sin bajar nada. Útil si solo has tocado a mano un CSV intermedio.
- Ojo: si queda prensa nueva sin extracción de regionales, `actualizar_encuestas()` lo avisa y hay que hacer esa pasada a mano.
- **Agregadas sin n.** `construir_meta_agregadas()` filtra las encuestas sin `Muestra`, igual que `prep_grande()`. Así `encuestas_meta_agregadas.csv` no cambia por una encuesta que el modelo descarta, y el driver de full bayesian no reajusta por eso. Si el CSV recoge una fila con la muestra vacía, el filtro se ha perdido en `preparar_meta.R`.

## Paso 2 · Modelos 1 y 2 (post de meta-análisis)

El render del post es el que reajusta. No hace falta llamar a las funciones por separado si vas a renderizar igualmente:

``` sh
quarto render 2026/10/meta-analisis-generales-jlcr.qmd
```

Los chunks llaman por dentro a `ajustar_grande(refit = TRUE)` (modelo 1, multinomial de 6 categorías) y `ajustar_bloque(refit = TRUE)` (modelo 2, composición del BLOQUE en 8 subceldas), y refrescan `mod_meta_grande.rds` y `mod_meta_bloque.rds`.

Si prefieres reajustar fuera del render:

``` r
source(here::here("2026/10/ajustar_meta.R"))
ajustar_grande(refit = TRUE)
ajustar_bloque(refit = TRUE)
```

**Argumentos:**

- `refit = FALSE` (default): con caché. Si el `.rds` del fit ya existe, brms devuelve el modelo guardado (`file_refit = "never"`) y **no** ve la encuesta nueva. Es la trampa clásica: por eso en el render se pasa `refit = TRUE`.
- `iter = 4000`, `warmup = 1000`: sampling de brms; los defaults van bien.
- La seed está fijada dentro de las funciones: con los mismos datos sale el mismo posterior, así que re-renderizar sin encuestas nuevas no mueve cifras.

## Paso 3 · Post de escaños (pipeline de octubre)

``` sh
quarto render 2026/10/meta-analisis-generales-escanos-jlcr.qmd
```

`simular()` llama por dentro a `ajustar_grande()`/`ajustar_bloque()` (listas = "unidas", el default) y recoge los fits frescos del paso 2.

**La trampa del orden**: si se renderiza solo este post sin pasar antes por el de meta-análisis, `simular()` lee el fit viejo: la encuesta nueva entra en los CSV pero no en el modelo, sin ningún error. El chunk de setup compara fechas y lanza un warning si detecta esta situación, pero mejor no llegar a él.

## Paso 4 · Escenarios y mapa de coaliciones (post de frente amplio)

``` sh
Rscript 2026/10/frente_amplio_y_adelante_andalucia/escenarios_mapa_29N.R
```

Es un driver autocontenido que repite el paso 1 y reajusta los **tres** modelos (unidas, bloque y separadas, con seed fija) y simula los tres escenarios (`unidas`, `separadas`, `mapa`) con la misma semilla y el mismo número de draws (`n_draws = 6000`), de forma que los draws están emparejados y las diferencias entre escenarios son solo de la regla provincial. En el mapa, la transferencia catalana del voto de Podemos y la share andaluza de Adelante Andalucía se sortean draw a draw de priors explícitos. Guarda los CSV (`escenarios_29N_*.csv`) y los PNG del post en su carpeta.

Después:

``` sh
quarto render 2026/10/frente_amplio_y_adelante_andalucia/frente_amplio_y_adelante_andalucia.qmd
```

**Argumentos del driver** (van en el `Rscript`, no en el `quarto render`,
que solo maqueta los resultados que el driver ya ha escrito en CSV y PNG):

- `--sin-refit`: no reajusta los tres modelos y usa los fits en caché. Solo tiene sentido si ya los has reajustado (fase 2a del procedimiento completo).
- `--sin-actualizar`: se salta el paso 1 (útil si acabas de correr `actualizar_meta.R` y no quieres repetir el scrape incremental). Sin él, hace el paso 1 entero, que es idempotente.

- Si solo necesitas `mod_meta_grande_sep.rds` refrescado (por ejemplo, para la comparación del post full bayesian, que usa `simular(listas = "mapa")`) y no quieres todo el driver:

  ``` r
  source(here::here("2026/10/ajustar_meta.R"))
  ajustar_grande_sep(refit = TRUE)
  ```

## Paso 5 · El modelo conjunto (post full bayesian)

``` sh
Rscript 2026/10/full-bayesian/12_ajustar_conjunto.R principal
```

Un solo script que hace las cuatro cosas: prior predictive, ajuste de Stan (`modelo_conjunto.stan`, 4 cadenas × 1000 warmup + 1500 sampling, `adapt_delta = 0.95`), resumen de parámetros y tendencia semanal, y simulación de escaños con el escenario mapa + probabilidades de mayoría. Deja `fit_conjunto.rds` y `conjunto_sim.rds` (que no van a git) y los `conjunto_*.csv` que lee el post. Con `seed_cj = 1811` fijada: mismos datos, mismo posterior.

**Argumentos:**

- `principal` (default, sin sufijo): volatilidad propia de cada campaña (`sigma_rw_campana = 1`); es la variante que usa el post.
- `v2`: volatilidad común a todas las campañas; escribe con sufijo `_v2` y no pisa los ficheros principales.

Después, los dos complementos que leen el fit:

``` sh
Rscript 2026/10/full-bayesian/16_ablacion_error.R   # conjunto_sin_error.csv
```

Y, si quieres tener al día también la comparación con el pipeline de octubre que aparecen en la figura y la tabla de ablación (los `comparar_*.csv`):

``` sh
Rscript 2026/10/full-bayesian/05_comparar.R
```

Que simula `simular(listas = "mapa")` con la semilla 2029: necesita `mod_meta_grande_sep.rds` fresco (lo reajusta el driver del paso 4, o a mano con `ajustar_grande_sep(refit = TRUE)`).

Opcional: `17_sensibilidad_omega.R` (sensibilidad de la Ω). Con encuestas nuevas apenas debería moverse; se puede saltar.

## Paso 6 · Render del post full bayesian

``` sh
quarto render 2026/10/full-bayesian/full-bayesian.qmd
```

## Paso 7 · Releases de GitHub

Los posts enlazan a releases para los datos de salida. Cuando esos datos cambian, súbelos con `--clobber` y comprueba que el enlace del post sigue apuntando a esa release.

| Post | Release | Qué subir | Quién lo sube |
|---|---|---|---|
| Full bayesian | `full-bayesian-29n` | `fit_conjunto.rds`, `conjunto_sim.rds` (y los `fit_conjunto_bt_*` si rehaces el backtest) | a mano, con `gh release upload` (fase 4 del procedimiento) |
| Frente amplio y AA | `frente-amplio-29n` | `escenarios_29N_*.csv` (6), `coaliciones_izquierda_29N_provincias.csv`, `aa_autonomicas2026_provincias.csv`; los `sensibilidad_29N_*.csv` solo si los has regenerado | a mano. Ni el driver del mapa ni el render suben nada |
| Meta-análisis y escaños | ninguna con datos de encuestas | nada | la release `generales-2026-descargas` son microdatos del CIS, que no cambian con encuestas nuevas |

Subida a mano del post del mapa:

``` sh
gh release upload frente-amplio-29n \
  2026/10/frente_amplio_y_adelante_andalucia/escenarios_29N_*.csv \
  2026/10/coaliciones_izquierda_29N_provincias.csv \
  2026/10/aa_autonomicas2026_provincias.csv \
  --clobber
```

## Paso 8 · Textos y cifras escritas a mano

Las cifras calculadas en línea (`r ...` o chunks) se actualizan solas al renderizar. Las escritas a mano no. Para localizarlas:

``` sh
grep -nE "[0-9]+ encuestas|última encuesta|hasta el [0-9]+|Actualización del" \
  2026/10/meta-analisis-generales-jlcr.qmd \
  2026/10/meta-analisis-generales-escanos-jlcr.qmd \
  2026/10/frente_amplio_y_adelante_andalucia/frente_amplio_y_adelante_andalucia.qmd \
  2026/10/full-bayesian/full-bayesian.qmd
```

Sitios conocidos:

- `meta-analisis-generales-jlcr.qmd`: los callouts "Actualización del …" recogen cada encuesta nueva con sus recuentos ("el modelo 1 pasa de 47 a 48 encuestas"). Para el número nuevo, cuenta en los CSV con `dplyr::n_distinct(encuestadora, campo)`, sobre `encuestas_meta_grande.csv` (modelo 1) y `encuestas_meta_bloque.csv` (modelo 2). Añade un callout nuevo; no reescribas los antiguos.
- `full-bayesian.qmd`, pie de `fig-tendencia`: cita la fecha de la última encuesta. La línea y la etiqueta del gráfico sí se calculan (`ultima`); comprueba que la fecha del pie coincide con ella.
- `full-bayesian.qmd`, sección de volatilidad: "entre la última encuesta y el 29-N quedan casi  semanas". Falta el número.
- `meta-analisis-generales-escanos-jlcr.qmd`: "124 encuestas" del error histórico son las de las generales pasadas; no cambian.
- `frente_amplio_y_adelante_andalucia.qmd`: la lista de ficheros de la release (sección de datos) tiene que coincidir con la tabla del paso 7.

## Paso 9 · Publicar y comprobar

1. Los renders no deben mostrar el aviso "es más nuevo que" del post de escaños. Si lo muestran, el post de meta-análisis no se ha renderizado antes.
2. Revisa en el log del driver del mapa la línea `encuestas en el modelo separado: N, última: fecha`: N y la fecha tienen que cuadrar con lo que esperabas.
3. Commit y push: los scripts, CSV y `.qmd` van a `master` en el remoto, porque los enlaces apuntan a `blob/master`. Después, `git status -- 2026/10` debe quedar limpio, salvo lo que no quieras publicar.
4. Comprueba que cada release tiene lo mismo que tu disco: `gh release download <tag> -D <dir>` y compara md5 con los ficheros locales.

## Qué NO hay que tocar con encuestas nuevas

Calibrado con historia; solo cambia si cambia la metodología:

- Los backtests congelados (`backtest_*.csv` y sus scripts, incluidos los `13_backtest_conjunto.R` y `10_backtest.R` de full-bayesian).
- `mod_tau_shock.rds` y `tau_regionales.csv` (ruido geográfico).
- `ref_provincial_23J_14.csv` (geografía prestada de elecciones pasadas).
- `factor_antelacion.csv` (error por ventana de antelación).
- `06_scrapear_autonomicas.R` y los errores autonómicos (historia cerrada).

## Procedimiento completo, en orden

Desde la raíz del proyecto. Primero los datos, después todos los modelos, al final los posts y las releases. Así cada post lee fits ya reajustados.

``` sh
# Fase 0 · ver qué hay nuevo (solo lectura: snippet del paso 0)

# Fase 1 · datos
Rscript 2026/10/actualizar_meta.R

# Fase 2 · reajustar todos los modelos
# 2a. meta-análisis: grande (6 categorías), bloque y grande_sep (7 categorías)
Rscript -e 'source(here::here("2026/10/ajustar_meta.R")); invisible(ajustar_grande(refit = TRUE)); invisible(ajustar_bloque(refit = TRUE)); invisible(ajustar_grande_sep(refit = TRUE))'
# 2b. modelo conjunto y derivados
Rscript 2026/10/full-bayesian/12_ajustar_conjunto.R principal
Rscript 2026/10/full-bayesian/16_ablacion_error.R
Rscript 2026/10/full-bayesian/17_sensibilidad_omega.R
# 2c. comparación con octubre (usa grande_sep y bloque de 2a)
Rscript 2026/10/full-bayesian/05_comparar.R
# 2d. escenarios del mapa: solo simula, con los fits de 2a
Rscript 2026/10/frente_amplio_y_adelante_andalucia/escenarios_mapa_29N.R --sin-actualizar --sin-refit

# Fase 3 · renders (cada uno lee los resultados de la fase 2)
quarto render 2026/10/meta-analisis-generales-jlcr.qmd
quarto render 2026/10/meta-analisis-generales-escanos-jlcr.qmd
quarto render 2026/10/frente_amplio_y_adelante_andalucia/frente_amplio_y_adelante_andalucia.qmd
quarto render 2026/10/full-bayesian/full-bayesian.qmd

# Fase 4 · releases
gh release upload full-bayesian-29n 2026/10/full-bayesian/fit_conjunto.rds 2026/10/full-bayesian/conjunto_sim.rds --clobber
gh release upload frente-amplio-29n 2026/10/frente_amplio_y_adelante_andalucia/escenarios_29N_*.csv 2026/10/coaliciones_izquierda_29N_provincias.csv 2026/10/aa_autonomicas2026_provincias.csv --clobber
```

Después, la fase 5: los textos escritos a mano (paso 8), las comprobaciones, el commit y el push (paso 9).

Notas:

- Los renders de la fase 3 vuelven a reajustar los modelos 1 y 2 dentro de sus chunks (`refit = TRUE`). Con la misma semilla el resultado es idéntico; es coste extra, no un error. Para evitarlo habría que cambiar esos chunks.
- `actualizar_full_bayesian.R` hace de una vez las fases 2b, 2c, el render y la release del full bayesian, pero solo reajusta si la huella md5 de sus CSV cambia. Como la fase 1 ya los ha reconstruido, lanzado después da "sin cambios". Si lo usas tras la fase 1, añade `--forzar`. Esta guía no lo usa.
- El backtest (`13_backtest_conjunto.R`) y `06_scrapear_autonomicas.R` no entran: ver "Qué NO hay que tocar".

Nota: los pasos 4 y 5 son independientes entre sí (el conjunto no usa brms); el orden entre ellos da igual. Los pasos 2 → 3 no: el de escaños debe ir después del de meta-análisis.