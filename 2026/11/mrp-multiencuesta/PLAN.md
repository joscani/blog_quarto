# MRP con varias encuestas (CIS + 40dB + More in Common)

Proyecto de noviembre 2026. Continúa `2026/10/mrp-raking/`.
Idea: juntar microdatos de varias casas y varios meses en un solo modelo MRP.
Cada encuesta aporta información; el modelo estima el efecto de cada casa y la evolución temporal.

## Inventario de microdatos (comprobado el 2026-10-21)

Todos los ficheros están fuera de git (`descargas/` está en `.gitignore`).

| Casa | Estudio | Fichero | n |
|---|---|---|---|
| CIS | 3571, barómetro julio 2026 | `2026/10/mrp-raking/datos/MD3571/3571.sav` | 4.020 |
| CIS | 3577, barómetro septiembre 2026 | `descargas/MD3577/3577.sav` | 4.042 |
| CIS | 3582, barómetro octubre 2026 | `descargas/MD3582/3582.sav` | 5.024 |
| 40dB | Barómetro septiembre 2026 | `2026/10/mrp-raking/datos/2026_09_barometro_descargables/03_Datos_septiembre_2026.sav` | 2.000 |
| 40dB | Barómetro octubre 2026 | `2026/10/mrp-raking/datos/2026_10_barometro_descargables/03_Datos_octubre_2026.sav` | 2.000 |
| 40dB | Flash octubre 2026 (8 oct) | `descargas/2026_10_08_barometro_flash_descargables/03_Datos_octubre_2026_flash.sav` | 800 |
| More in Common | Estimación electoral abril 2026 | `descargas/microdatos-2026-05/microdatos-mayo-2026.sav` | 2.547 |
| More in Common | Estimación electoral junio 2026 | `descargas/microdatos-2026-07/microdatos-julio-2026.sav` | 2.277 |
| More in Common | Estimación electoral septiembre | `descargas/microdatos-2026-09/microdatos-septiembre-2026.sav` | 2.144 |

Total: 24.854 entrevistas en 9 encuestas.

Documentación adicional:
- `descargas/informe-metodologico-electoral.pdf`: metodología de More in Common (abril 2026).
- Fichas técnicas y cuestionarios del CIS en cada `MDxxxx/`. Notas metodológicas de 40dB en cada carpeta.

## Variables que hay que armonizar

| Concepto | CIS | 40dB | More in Common |
|---|---|---|---|
| Sexo | `SEXO` | `sexo` | `genero` |
| Edad | `EDAD` | `edad`, `edad_r` | `edad`, `grupo_edad` |
| CCAA | `CCAA` | `ccaa` | `comunidad_autonoma` |
| Provincia | `PROV` | `prov` | no tiene |
| Estudios | `ESTUDIOS` | `educacion_r` | `nivel_estudios` |
| Recuerdo 23J | `RECUVOTOG`, `RECUERDO` | `p6` (sep), `p9` (flash oct) | `voto_2023_raw`, `participacion_2023_raw` |
| Intención | `INTENCIONG` (+ `VOTOSIMG`) | `p3`+`p4` (sep), `p2`+`p3` (flash) | `intencion_voto_raw`, `intencion_voto_simpatia` |
| Prob. de votar | `PROBVOTO` | `p2` (sep), `p1` (flash) | `prob_voto_declarada` |
| Peso propio | `PESO` | `ponde` | `peso` |

Las preguntas de 40dB cambian de número entre oleadas. Hay que mapear cada fichero por la etiqueta, no por el nombre.

## Incidencias detectadas

- More in Common: el nombre del fichero no es la fecha de campo. Decisión (2026-10-21): usar el mes de la portada del codebook. `mayo` → abril 2026, `julio` → junio 2026, `septiembre` → septiembre 2026. Los codebooks no dan días.
- More in Common: usar las variables `_raw`. Las `_final` ya están imputadas por su modelo.
- No he comprobado aún la variable de recuerdo del barómetro de octubre de 40dB (n = 2.000).

## Paso 1 hecho: armonización (2026-10-21)

Script `01_armoniza.R` → `datos/encuestas.rds` (24.854 entrevistas, una fila por entrevista). Log en `datos/log_01.txt`.

Decisiones (reversibles):
- Intención = voto directo, sin simpatía: `INTENCIONG` (CIS), `p3`/`p2` (40dB, localizada por etiqueta), `intencion_voto_raw` (MIC).
- Voto + simpatía (`voto_sim`), misma regla en las 3 casas: intención directa; si es "no sabe", "no contesta" o "no votaría", el partido de simpatía. Blanco y nulo no cambian.
  - CIS: rehecho con `SIMPATIA`, no con `VOTOSIMG`, porque `VOTOSIMG` mete Aliança Catalana y otros pequeños en "Otro partido".
  - 40dB: pregunta la simpatía a todos (`p4`, `p3` en el flash); solo se usa para quien no da partido.
  - MIC: `intencion_voto_simpatia`, que ya sigue esa regla.
  - Comprobado: la simpatía nunca cambia un partido declarado. Recupera de 38 (40dB flash) a 571 (CIS 3582) entrevistas por encuesta.
- Blanco → OTROS en intención y recuerdo; nulo → OTROS solo en recuerdo (como el pipeline del CIS).
- Podemos y todas las marcas de Sumar → SUMAR. AC solo existe en el CIS; las otras casas la meten en "Otro".
- Estudios: FP de MIC y "segundo grado, 2º ciclo" de 40dB → media, con `edu_ambigua = TRUE`. En 40dB la marca es demasiado amplia (incluye Bachillerato): 39 % de su muestra. Pendiente: tratarlo como agregado, igual que la provincia.
- MIC participación: "No pudo" y "No suele votar" → ABST; "Sin edad" → NO_PODIA si tiene 18-20; "No recuerda" → NA.
- 18-20 años → NO_PODIA en todas las casas, por edad.
- Provincia de MIC rellenada en las uniprovinciales; NA en el resto (5.232 entrevistas).

Hallazgos:
- 40dB no tiene entrevistas en Ceuta ni Melilla.
- 40dB oct casi no tiene 18-20 (9 casos, ninguno de 20 años); deduzco cuotas por tramo 18-24.
- Recuerdo PSOE en la muestra: 29-34 % en las 9 encuestas, frente a 21,1 % real. ABST: 8-16 %, frente a 28,2 %. El sesgo de recuerdo no es solo del CIS.
- VOX bruto: 13-15 % en el CIS, 19-23 % en 40dB y MIC.

Siguiente: paso 2, tabla de postestratificación provincial.

## Propuesta de modelo (borrador)

1. Armonizar las 9 encuestas a un formato común: casa, estudio, fecha de campo, sexo, edad, CCAA, estudios, recuerdo, intención.
2. Reutilizar la tabla de postestratificación de `2026/10/mrp-raking/` (sexo × edad × estudios × CCAA × recuerdo).
3. Ajustar un multinomial en `brms` agregado por celda con `trials()`, con efectos aleatorios para casa (efecto casa), mes y CCAA.
4. Postestratificar sin efecto casa (o con su media) y comparar con la estimación publicada por cada casa.
5. Comprobar cuánto cambia la estimación al quitar cada casa (dejar una fuera).

## Efectos de CCAA y provincia con More in Common sin provincia

Estructura: efecto aleatorio de CCAA y de provincia anidada, `(1 | ccaa) + (1 | prov)`.
More in Common no tiene provincia, pero sí la conocemos en las CCAA uniprovinciales:
Asturias, Cantabria, La Rioja, Madrid, Murcia, Navarra, Baleares, Ceuta y Melilla.
Eso cubre el 24,6 % (abril), 24,6 % (junio) y 25,7 % (septiembre) de su muestra (comprobado).

Para el resto, la provincia es una variable discreta que falta. Dos formas de tratarla:

| | A. Multi-membership (`brms`) | B. Agregar probabilidades (`cmdstanr`) |
|---|---|---|
| Predictor | η = … + u_ccaa + Σ_p w_p · u_p | η_p = … + u_ccaa + u_p, para cada p |
| Verosimilitud | multinomial(n, softmax(η)) | multinomial(n, Σ_p w_p · softmax(η_p)) |
| Pesos w_p | proporción de la celda en cada provincia (tabla de postestratificación provincial) | igual |
| Exactitud | aproximada: promedia en escala logit | exacta si cada entrevistado está en su provincia de forma independiente |
| Coste | bajo, fórmula de `brms` | Stan a mano, hasta 9 softmax por fila (Castilla y León) |

- En A, la fórmula es `(1 | mm(p1, …, p9, weights = cbind(w1, …, w9)))`. Las filas con provincia conocida llevan peso 1 en una casilla y 0 en el resto. He comprobado que `make_standata` acepta pesos cero (prueba con `bernoulli`; no lo he probado aún con `multinomial`).
- B es lo mismo que hace `2026/10/full-bayesian/modelo_conjunto.stan` con las encuestas que solo publican agregados (Sigma Dos, Winston…): el modelo trabaja a nivel fino y la observación usa la suma de probabilidades. Ver `GUIA.md:91-97` de ese post.
- Con B, que More in Common venga por CCAA y sin provincia no es un problema: su fila es una observación del agregado.
- Corrección (2026-10-21): una versión anterior ponía en B Σ_p w_p · multinomial(…), una mezcla. Eso supone que todos los de la celda están en la misma provincia. Lo correcto es la multinomial con las probabilidades promediadas.
- Plan: en `brms`, ajustar A. Al pasar a `cmdstanr`, usar B como en full bayesian. Comparar A, B y el modelo sin More in Common: la diferencia es material del post.

Consecuencia para la postestratificación: la tabla de `mrp-raking` es por CCAA. Hace falta una por provincia.
Fuentes: ECP del INE (sexo × edad × provincia) y resultados del 23J por provincia (`prov_23j.rds`). No he comprobado si hay estudios × provincia en la EPA.

## Rendimiento de brms (estimación del 2026-10-21)

Referencia medida: MRP del 3577, 1.442 celdas, ~22 min por cadena (10 calentamiento + 6–12 muestreo), 4 cadenas en paralelo.
Mac: 8 núcleos de rendimiento, 4 de eficiencia, 48 GB. La memoria no limita.

Estimación para las 9 encuestas (deducida, no medida):
- ~18.500 entrevistas con voto válido (75 %, como en el 3582) → ~10.000–12.000 celdas (0,55–0,65 celdas por entrevista; en el 3582 son 0,41).
- ~3–5 h por cadena. `mm()` multiplica por 9 el término de provincia.
- Lo que más pesa: la multinomial tiene 11 predictores lineales y cada efecto aleatorio se repite en los 11.

Medido en el 3582 (3.554 entrevistas con voto válido y recuerdo sin imputar):
- CCAA × sexo × edad × estudios × recuerdo: 1.489 filas (0,42 por entrevista).
- Provincia × sexo × edad × estudios × recuerdo: 2.136 filas (0,60 por entrevista), de 28.392 celdas posibles.
- La encuesta tiene que ser clave del `group by` (efecto casa y mes), así que las filas no se juntan entre encuestas: total ≈ suma por encuesta.
- Palanca para reducir filas: edad en 4 tramos en vez de 7.

Plan:
1. Prototipar con `algorithm = "pathfinder"` o con un 20 % de las celdas.
2. Ajuste final con 4 cadenas y `threads = threading(2)` (8 núcleos). Ganancia esperada de 1,5 a 1,8 veces.
3. Si sigue lento: agrupar los partidos regionales pequeños en el modelo conjunto.
4. Alternativa: la mezcla exacta (opción B) en `cmdstanr` directo, con mejor vectorización.

Siempre `refresh = 100, silent = 0` en `brm()` para ver el avance de las cadenas en el log.

## Pasos 2 y 3 (2026-10-21)

- Paso 2 hecho: `02_poblacion.R` (Censo 2024, tabla 66600, por provincia; 23-J por municipio) y `03_raking.R` → `datos/tabla_post_prov.rds`. Cuadra con los dos targets (error máximo 0,002 personas).
- Corrección también en `2026/10/mrp-raking/03_targets.R`: los targets compartían márgenes que no coincidían (18-20 vs NO_PODIA; sexo por CCAA de ECP vs EPA). Las estimaciones del post de octubre y del 3582 cambian menos de 0,013 puntos. No se ha republicado.
- Paso 3 en curso: `04_brms.R` (pathfinder / nuts). 19.092 entrevistas, 11.888 filas agregadas.

Pendientes del paso 3:
- `posterior_epred` necesita `allow_new_levels = TRUE`: Ceuta × OTROS está en la tabla sin datos.
- Sensibilidad de la semilla del raking: ahora solo CIS. Probar semilla con las 3 casas (MIC repartido por los pesos de `mm()`) y comparar.
- Decidido (2026-10-21): `casa:rec` y `ccaa:rec` los dos. `casa:rec` mide si el efecto casa depende del recuerdo (p. ej., votantes del PSOE de 2023 más fieles en el CIS); `ccaa:rec` hace falta para las estimaciones por CCAA.
- Decidido (2026-10-21): en el siguiente ajuste, casa como `(1 | casa)` en vez de efecto fijo con `contr.sum(3)`. Motivo: más didáctico y coherente con `(1 | casa:rec)`. El ajuste en curso sigue con el efecto fijo; comparar los dos. En Stan puro: efectos casa aleatorios con `sum_to_zero_vector`, como en full bayesian.
- Siguiente ajuste de brms: `(1 | casa)` + `voto_sim` + recuerdo imputado + modelo de participación conjunto.
- Ficheros de modelo: usar `file_refit = "on_change"` en `brm()` (con el valor por defecto, `"never"`, brms carga el `.rds` guardado aunque cambien fórmula, datos o priors, sin avisar). Y un nombre de fichero por versión (`mod_brms_nuts_voto`, `mod_brms_nuts_votosim`…) para poder comparar. Aplicar en `04_brms.R` cuando termine el ajuste en curso.
- Pathfinder descartado: Pareto k = 12 y error `path__` de brms 2.23.0 (corregido solo en GitHub, 2.23.2).
- Prueba NUTS (10 % de filas, 1 cadena, 300 iteraciones): script completo OK en 3,7 min. Ajuste completo lanzado a las 18:44 con `casa:rec` → `datos/mod_brms_nuts.rds`.
- Modelo de participación conjunto e imputación del recuerdo que falta (ahora quedan fuera 5.762 entrevistas, sin voto válido o sin recuerdo; no he separado cuántas por cada motivo).

## Resultado del ajuste brms completo (2026-10-21, 19:07–21:56)

- `datos/mod_brms_nuts.rds`, 168,6 min. `(1 | casa)`, `init = 0`, `max_treedepth = 8`, priors `exponential(1)` en las sd.
- 0 divergencias, pero 4.000 de 4.000 iteraciones en el límite de profundidad (255 pasos). Rhat máximo 1,037 (15 de 110 sd y b > 1,01); ESS bulk mínimo 143. Sirve para explorar, no para publicar.
- Por casa (2026-10, todos votan, % voto válido): `datos/estim_brms_nuts.rds`. Casa media (efecto casa = 0, `04b_casa_media_brms.R`): `datos/estim_brms_nuts_casa_media.rds`. PP 25,7 (22,2–28,9), PSOE 24,7 (20,6–29,5), VOX 21,2 (14,9–29,5). Intervalos anchos: con 3 casas, el centro de la población de casas está poco identificado.
- Ya aplicado `file_refit = "on_change"` en el modo nuts de `04_brms.R`.

## Paso 4: Stan puro (2026-10-21)

- `mrp.stan` + `05_stan.R` (modos prueba, prior, nuts; segundo argumento priors = nuevo | brms). Usa `datos/celdas_brms.rds`.
- More in Common sin provincia: probabilidad de la celda = suma ponderada de las probabilidades de sus provincias (como las encuestas agregadas de full-bayesian). Peso = proporción de la provincia en la celda, de la tabla de postestratificación. El usuario lo validó.
- Casa con `sum_to_zero_vector`: casa media = media de las 3 casas. Postestratificación en `generated quantities` (4.000 iteraciones).
- Prueba (10 %, 1 cadena, 300 it., priors brms): 112 s frente a 129 s de brms; mediana 127 pasos leapfrog.
- Prior predictive check: con `exponential(1)` el 44 % de las celdas tiene un partido > 90 % (mediana 0,86). Fidelidad observada por recuerdo: 63–88 %. Priors nuevos: sd_rec N+(0, 2), resto N+(0, 0,5), interacciones N+(0, 0,25), alpha N(0, 1) → mediana 0,66, 20 % > 90 %.
- Ajuste completo `05_stan.R nuts nuevo`: lanzado a las 21:57 y **parado a las 22:10 a petición del usuario** (iba por la iteración 100, ~7 min por 100). Relanzar con: `Rscript 2026/11/mrp-multiencuesta/05_stan.R nuts nuevo`. Estimado ~140 min.

- 2026-10-22: sin categoría de referencia (petición del usuario). Un predictor por partido, centrado entre partidos (`centra()`; alpha y b_mujer como `sum_to_zero_vector[J]`), como en full-bayesian. Versión anterior en `mrp_referencia_pp.stan`; sus salidas renombradas con `_refpp`. Prior predictive: ahora simétrico (mediana nacional 14–16 % para PP, PSOE, VOX y SUMAR; antes PP 8 %); 22 % de celdas > 90 %. Prueba (10 %, 1 cadena, priors nuevos): 90,6 s, 0 divergencias, 0 en max_treedepth, mediana 127 pasos.

## Estimación para las generales del 29-N (propuesta, no iniciada)

Voto 29-N = casa media del MRP + participación + cambio octubre → 29-N + error de las encuestas, draw a draw como en full-bayesian.
1. Modelo de participación conjunto (prob_voto, vota_seguro), calibrado.
2. `mes` como paseo aleatorio en `mrp.stan`, un paso más hasta noviembre.
3. Error: `err_new` de full-bayesian como supuesto declarado (mide el error de las estimaciones publicadas, no del MRP), o backtest con preelectorales del CIS (23J, 10N, 28A). Pendiente de decidir por el usuario.
4. Escaños: D'Hondt por provincia con el voto provincial del MRP (secundario).

Otros pendientes: función `log_post_r` en R para seguir el código Stan y compararla con `fit$log_prob()` (propuesta, sin respuesta); modelo por piezas M0–M4 y posterior predictive checks.

## Efecto casa de More in Common a favor del PSOE (2026-10-21)

Con el MRP, More in Common da más PSOE que las otras casas (26,7 frente a 25,5 del CIS y 23,1 de 40dB), aunque en bruto el CIS sobrerrepresenta más el recuerdo PSOE (37,7 % de los votantes frente a 35,8 y 35,0; 23J real 31,7). Causa: fidelidad PSOE 2023 → PSOE del 80,6 % en More in Common frente al 75 % en CIS y 40dB, y abstención 2023 → PSOE del 25 % frente al 12–19 %. PP → PP igual en las tres (81–82 %).
- Descartado (en bruto, `encuestas.rds` sin pesos): mes de campo (se mantiene en septiembre, el único mes común), ideología (misma media del recuerdo PSOE que 40dB; diferencia mayor en centro-izquierda: 86,9 frente a 79,9), estudios (se mantiene en los tres niveles) y selección de indecisos (10,0 % sin intención válida frente a 9,5 % en 40dB).
- Sin comprobar: composición del panel online; cuestionario (orden de recuerdo e intención, recuerdo en dos pasos con más abstención 2023: 13,2 %, lista de partidos); que More in Common acierte (no identificable sin elecciones).
- Posible siguiente paso: revisar orden y redacción de las preguntas en los tres codebooks de `descargas/`.
- Estimación publicada (substack, 3 sep 2026): PP 28,7, PSOE 27,0, Vox 20,3, Sumar 6,6, Podemos 3,6. Reproducida con los microdatos (`intencion_voto_final`, peso × `prob_voto_estimada`, sin blanco/nulo/abstención): PP 28,2, PSOE 26,8, Vox 20,4. Nuestro MRP para More in Common (brms, todos votan): PP 24,9, PSOE 26,7. Su cocina pone al PP 1,4 por delante; nuestro MRP, 1,8 por detrás.
- **Fecha por revisar:** el fichero `microdatos-2026-09` (n = 2.144) tiene trabajo de campo del 24 al 28 de agosto de 2026 según el substack; el codebook dice "Septiembre 2026" (generado el 1 de septiembre). Con la regla del usuario (fecha de campo) sería 2026-08, no 2026-09. Pendiente de confirmar y, si procede, cambiarlo en `01_armoniza.R` y reajustar. Decisión del usuario (2026-10-22): se queda en 2026-09.
- Informe metodológico de More in Common: `descargas/informe-metodologico-electoral.pdf` (abril 2026). Raking sin estudios; XGBoost para recuerdo, indecisos y participación; participación sin calibrar; Rubin M = 50. Sus microdatos traen `peso`, `prob_voto_estimada`, `voto_2023_final` e `intencion_voto_final`: permiten reproducir su estimación.

## Experimento futuro: causact (NumPyro)

Orden: solo **después** de tener el modelo en Stan puro (cmdstanr, paso 4). Es algo a investigar, no parte del plan principal.

Idea del usuario (2026-10-21): repetir el modelo con `causact`, que define el modelo como un DAG en R y lo ajusta con NumPyro (JAX) a través de `reticulate`.
- Motivo: el ajuste en brms tarda horas (~2,7 s por iteración estimado con 11.888 filas). NumPyro con JAX podría ser más rápido; no lo he medido.
- Estado: `causact` 0.6.0 está en CRAN; no está instalado. Necesita un entorno de Python con NumPyro (`causact::install_causact_deps()`), y habría que añadirlo a `renv`.
- Por comprobar antes de empezar: si `causact` admite la multinomial con `trials` y una suma ponderada de efectos de provincia como la de `mm()` / la agregación de probabilidades. Si no, la alternativa es NumPyro directo en Python.
- Comparación natural: mismo modelo en brms, cmdstanr y causact; tiempo y resultados.

## Pendiente de decidir

- Universo de partidos: qué agrupar en "otros".
- Tratamiento de indecisos y de la abstención.
- Si el tiempo entra como efecto aleatorio por mes o como tendencia suave.

## 2026-10-22: More in Common de 2026-04 fuera; MRP solo CIS

- Quitado MIC abr (2026-04, `microdatos-mayo-2026.sav`) de `01_armoniza.R`: muy antiguo. Rehechos `encuestas.rds`, raking y `celdas_brms.rds` (`04_brms.R celdas`). La tabla de postestratificación no cambia (mismo md5). Copia de los ficheros anteriores en `datos/con_mic_abr/`.
- Los ajustes conjuntos anteriores (brms nuts, Stan prueba) llevan MIC abr. Hay que repetirlos.
- Nuevo `mrp_cis.stan` y `05_stan.R ... cis`: MRP solo con barómetros del CIS, sin efecto casa ni filas mezcla. Para comparar con el CIS dentro del modelo conjunto.
- Añadido CIS 3567 (campo 1-4 jun 2026, 4.000 entrevistas, 2.888 con voto válido u OTROS) como 2026-06. `sumar_cis` incluye ahora "En Comú Podem" y "Por Andalucía". La semilla del raking usa todo el CIS, así que la tabla de postestratificación cambia (diferencia relativa máxima de 413 % en celdas de más de 1.000 personas).
- Añadido 40dB "julio" (`descargas/92_Julio_2026`, campo 26-29 jun 2026, 2.000 entrevistas) como 2026-06. La tabla de postestratificación no cambia (la semilla es solo CIS). Junio tiene ya las tres casas.

## 2026-10-22: orden de los scripts y modelo de participación

Cada script hace una sola cosa:

1. `01_armoniza.R`: lee las encuestas → `encuestas.rds`.
2. `02_poblacion.R`: censo y 23-J → targets.
3. `03_raking.R`: tabla de postestratificación (IPF con semilla del CIS).
4. `03b_celdas.R`: agrega las entrevistas en celdas → `celdas_brms.rds` (voto) y `celdas_part.rds` (participación). Antes estaba dentro de `04_brms.R`.
5. `04_brms.R`: ajuste brms (solo modelo de voto).
6. `05_stan.R`: ajuste Stan. `mrp.stan` (las tres casas) o `mrp_cis.stan` (solo CIS, con participación).

Participación en `mrp_cis.stan`: binomial de decir "10, seguro que voto", con mes, sexo, edad, estudios, recuerdo, CCAA y provincia. Se calibra con un desplazamiento común en logit a la participación del 23-J entre residentes (70,4 % = votos emitidos / censo INE). Pendiente pasarlo a `mrp.stan`, con efecto casa.

## 2026-10-22: prueba de sensibilidad a la semilla del raking

- `03_raking.R "<estudio>" <alpha>` construye tablas alternativas en `datos/sensibilidad/`: semilla sin cada barómetro del CIS (alpha 0,5) y semilla completa con alpha 0,25 y 0,75.
- `05_stan.R gq nuevo cis` con `TABLA_POST=<tabla>` postestratifica la posterior ya ajustada con otra tabla (generate_quantities), sin reajustar.
- Diferencia entre tablas, celdas de más de 1.000 personas: mediana 12–15 %, p90 41–53 %. Personas que cambian de celda: 6–9 %.
- Resultado (2026-10-22, `mrp_cis.stan`, 1.000 muestras): la estimación nacional cambia como mucho 0,07 puntos entre tablas (0,02 sin alpha 0,25 y 0,75). El ruido de la semilla no importa a escala nacional. No probado: CCAA y provincias, ni el sesgo de la semilla (todas las semillas son del CIS).
- `gq` postestratifica en R: `generate_quantities` falla porque relee las muestras con 6 cifras y `sum_to_zero_vector` exige suma 0 exacta.

## 2026-10-22: participación en el modelo conjunto (`mrp.stan`)

- Binomial de decir "10" con efecto casa (`sum_to_zero_vector`), mes, sexo, edad, estudios, recuerdo, CCAA y provincia; la provincia de More in Common se mezcla como en el voto.
- Calibración al 70,4 % con la casa media: el nivel de cada casa lo absorbe la calibración, así que todas las casas usan la misma P(votar). Salida `v_part[n_casa + 1, J]`.
- Prueba (10 % de filas, 1 cadena, 4,3 min): 0 divergencias, 150 de 150 en max_treedepth 8. Participación de la casa media antes de calibrar: 60,5 % (CIS solo: 70,9 %).
- Pendiente: `05_stan.R nuts nuevo` cuando haya núcleos libres. El ajuste solo CIS ya guardado sigue valiendo, pero la huella de sus datos ha cambiado (orden de la lista): relanzar `nuts nuevo cis` lo reajustaría.

## Futuro: error histórico de las encuestas para el 29-N

1. Error histórico: sumar `err_new` (de `2026/10/full-bayesian`, ajuste de `modelo_conjunto.stan`) al `eta` de PP, PSOE, VOX y SUMAR en todas las celdas, muestra a muestra, y postestratificar (variante del modo `gq`). `err_new` = logit(real) − logit(voto latente el día E), sorteado de MVN(mu_gen, Σ): el usuario confirma que está bien definido. Supuesto declarado: el MRP falla como el consenso.
2. Tiempo hasta el 29-N: `err_new` se mide el día E, así que el salto octubre → noviembre va aparte. Sencillo: un mes nuevo ~ N(0, sd_mes). Bueno: efecto mes como paseo aleatorio (reajuste).
3. Backtest del MRP con barómetros del CIS anteriores al 23-J (y 28-A/10-N 2019) antes de publicar nada del 29-N.

## 2026-10-22: Ateneo del Dato como cuarta casa

- Fichero `descargas/atdt-dat-expresidentes-corrupcion.csv`: 1.500 entrevistas, campo 9-12 jun 2026 (dato del usuario), panel online, peso `ponde`. Mes `2026-06`.
- Lector `lee_atdt()` en `01_armoniza.R`. Las otras 10 encuestas salen idénticas (comprobado con `all.equal`). Total: 29.831 entrevistas.
- Edad solo en tramos, con 18-24 junto: "No tenía edad para votar" → 18-20 (28 casos); el resto de 18-24 → 21-24 (111).
- Estudios: "Secundaria 2do ciclo" → media, `edu_ambigua = TRUE` (294, 20 %). Sin provincia: rellenada en las uniprovinciales; NA en 1.141.
- `ideol` = 11 → NA (66 casos). Deduzco que es NS/NC; el CSV no trae codebook.
- Sin probabilidad de votar: no entra en el modelo de participación (filtro `!is.na(vota_seguro)` de `03b_celdas.R`).
- Pendiente: en `mrp.stan`, el efecto casa de la participación es `sum_to_zero_vector` sobre `n_casa`. Ateneo no tiene filas ahí: comprobar que no fuerza a las otras casas.
- Hecho (2026-10-22): `n_casa_t` en `mrp.stan`. El efecto casa de participación solo lleva CIS, 40dB y More in Common. Prueba: 3,1 min, 0 divergencias, 0 en max_treedepth.

## 2026-10-22: voto y participación separados (prueba)

- Ajuste completo `05_stan.R nuts nuevo` parado a las 15:56 (iteración 1.100 de 2.000, ~10 min por 100) a petición del usuario, para probar mejoras.
- Nuevos: `stan/funciones.stan` (`calibra()` con paso limitado a ±2: sin él da NaN con eta muy grandes, comprobado), `mrp_voto.stan` (reduce_sum, mezcla en log), `mrp_part.stan` (mezcla en log, priors como datos), `05b_separado.R`. `mrp.stan` y `mrp_cis.stan` sin tocar.
- Prueba (10 % del voto, todas las filas de participación, 1 cadena, 300 it.): conjunto 3,1 min. Separado: voto 2,1 min con 1 hilo y 1,0 min con 3 hilos; participación 2,3 min. Voto igual que el conjunto (dif. media 0,08 puntos, máx. 0,45).
- Problema: la participación sola llega a max_treedepth en 92 de 150 iteraciones (mediana 255 pasos); dentro del conjunto, 0. Sin explicar.
- Participación centrada (`mrp_part_c.stan`: casa, mes, edad, estudios y recuerdo centrados; CCAA y provincia sin centrar): 1,7 min, 0 en max_treedepth, mediana 127 pasos, ESS bulk de part_bruta 251 (antes 155). Resuelve el problema. Voto con 3 hilos: 1,0 min.
- Flujo nuevo (2026-10-22): `05b_separado.R` ajusta `mrp_voto.stan` y `mrp_part.stan` (centrada) y postestratifica con participación en R (`v_part`), emparejando las muestras de los dos ajustes. Comprobado: `v_casa` en R = Stan (dif. máx. 1e-6). Prueba (3 hilos): voto 1,3 min, participación 1,9 min; casa media con participación PP 26,1, PSOE 26,4, VOX 21,2 (conjunto: 26,0, 26,4, 21,4).
- Movidos a `obsoleto/`: `mrp.stan` (conjunto), `mrp_referencia_pp.stan`, `mrp_part_no_centrada.stan` y sus salidas (`obsoleto/datos/`, fuera de git). `05_stan.R` ya no ajusta las 4 casas: solo prepara datos (`SOLO_DATOS`) y sigue con `mrp_cis.stan` y el modo `gq`. `mrp_cis.stan` mantiene la `calibra()` antigua (sin el límite de paso).

## 2026-10-23: PODEMOS fuera de SUMAR; reajuste pendiente

- `00_partidos.R` (staged): PODEMOS separado de SUMAR, porque desde el mapa del 29-N compite por su lado. En el recuerdo del 23-J sigue sin existir como categoría propia (`rec_lv` lo excluye). `01_armoniza.R` adaptado; `encuestas.rds`, `celdas_brms.rds` (14.717 filas) y `celdas_part.rds` regenerados a las 09:58.
- Los ajustes nuts de esa mañana (09:07–09:19) y todo lo posterior (`post_celdas_casa_media_part.rds`, `estim_escanos_mrp.rds`) son de la definición antigua: movidos a `obsoleto/datos/podemos_en_sumar/`.
- Prueba de `05b_separado.R prueba 3` con los celdas nuevos: OK end to end. Voto 1,5 min, participación 1,8 min, 0 divergencias; el voto llega a max_treedepth en 150/150 (la participación, limpia).
- El treedepth no lo introduce PODEMOS: el ajuste nuts completo de la mañana (definición antigua) ya iba 4.000/4.000 en el límite (255 pasos), 0 divergencias, Rhat máx 1,014, ESS bulk mín 431.
- Decisión del usuario: reajustar con `max_treedepth` 8, como esa mañana.
- `05b_separado.R`: la comprobación v_casa R frente a Stan pasa de 9,7e-07 (J=11) a 1,2e-06 puntos (J=12) y rompía el `stopifnot` de 1e-6; relajado a 1e-4 (un error real sería del orden de 0,1).
- `04_brms.R` tiene ahora un modo `bench` (10 % de filas, 1 cadena, 150+150, threads 2): métrica diag_e/dense_e × priors exponential(1)/nuevos (sd_rec N(0,2), principales N(0,0,5), interacciones N(0,0,25), como en Stan). Mide tiempo, divergencias, iteraciones en max_treedepth y leapfrog mediano; guarda `datos/bench_brms.csv` tras cada variante.
- Resultado del bench (2026-10-23, log_04_bench.txt): diag_e × exponential(1) 5,2 min, 150/150 en treedepth, 255 pasos; diag_e × nuevos 4,3 min, 0 en treedepth, 127 pasos; dense_e × exponential(1) 10,8 min, sin mejora (255 pasos; brms ya es no-centered). dense_e descartada. Los priors nuevos son la palanca: en el ajuste completo esperaría ~85–90 min frente a los 168,6 medidos. La fila dense_e × nuevos se perdió (el proceso acabó al cerrarla; ~9 min de muestreo, pierde por tiempo).
- Nota técnica: con backend cmdstanr, `mod$fit` es un stanfit S4 (rstan) y `nuts_params()` da la columna `Value`; `mod$fit$sampler_diagnostics` no vale en brms 2.23.0.
- Limpieza (2026-10-23): `datos/` pasa de ~2 GB a 23 MB. A `obsoleto/datos/podemos_en_sumar/` se han movido los artefactos de la definición antigua que quedaban: `fit_stan_nuts_nuevo_cis.rds` y su estim (mrp_cis), `fit_stan_prueba_nuevo_cis.rds`, los dos prior checks de Stan y sus fits (`fit_stan_prior_{brms,nuevo}.rds`, `prior_check_*.rds`), `mod_brms_prueba.rds`, `estim_brms_prueba.rds` y los `sensibilidad/estim_*.rds` (derivados del fit antiguo). Las tablas de raking de `sensibilidad/` se quedan: el raking no depende de la definición de partidos. Los logs se quedan en `datos/` como registro. Los prior predictive checks habrá que repetirlos con J = 12.
- Decidido con el bench: el modo `nuts` de `04_brms.R` usa ahora los priors nuevos (diag_e). Lanzado a las 13:12 (PID 15351, log `log_04_nuts_nuevo.txt`); medido ~6-8 s por iteración en warmup, ETA ~16:00-16:45 (la estimación previa de ~90 min era optimista: el warmup va a árboles grandes). Encadenador en marcha (espera al PID y lanza `07_post_celdas.R nuts` en modo brms — sin participación — → `08_escanos.R` → `09_escanos_mapa.R`; log `log_cadena_post.txt`). La variante con participación llega con el refit de `05b`. Con `file_refit = "on_change"` reajustó al cambiar datos y priors; el `mod_brms_nuts.rds` antiguo (definición vieja de partidos) está en `obsoleto/datos/podemos_en_sumar/`.
- Lanzado (2026-10-23 19:07, PID 61625, log `log_05b_nuts.txt`) el refit `05b_separado.R nuts 3` con los celdas nuevos. Encadenador detrás (`log_cadena_stan.txt`): al acabar corre `07 nuts stan` (sin participación) → `08` → renombra salidas a `*_sin_part.csv/png` y `estim_escanos_mrp_sin_part.rds` → `07 nuts stan part` → `08` → `09` (mapa sobre con-participación). Así la comparación con/sin participación sale del mismo posterior. ETA ~21:15.
- Refit Stan acabado a las 21:23 (voto 121 min, participación 11,5 min): 0 divergencias; el voto 4.000/4.000 en max_treedepth (también Stan satura con todos los datos), Rhat máx 1,017, ESS mín 466; comprobación R↔Stan 1,2e-06.
- Comparación con/sin participación (mismo posterior, casa media): PP 114→120, PSOE 111→114, **VOX 91→79 (−12)**. Resuelto el misterio VOX 76 (mañana, con participación) frente a 92 (brms, sin participación): era la ponderación de participación, no el backend ni los priors ni PODEMOS; el votante de VOX tiene participación estimada bajo la media. La participación estrecha los intervalos (VOX 84–98 → 73–85). P(PP+VOX ≥ 176) = 100 % en las dos versiones. Salidas: `escanos_mrp_*_sin_part.csv` (sin participación) y `escanos_mrp_*.csv` (con participación, las finales).
- Efecto del mapa 29-N sobre la versión con participación: PSOE −3,3, SUMAR +2,6, AA +3,5, VOX −1,7, PP −1,2, ERC +1,3, PODEMOS −1,1; P(PP+VOX ≥ 176) = 100 % con y sin mapa.
- `simular_mapa()` tiene ahora el argumento `cat_mult` para sensibilizar el Dirichlet de Cataluña. Prueba con α×0,5, α×1 y α×2 (misma semilla, 1.000 draws, salidas en temporal): ningún escaño mediano cambia, ni en el Congreso ni por provincia (ERC 10, SUMAR 13 en las tres), ni los IC90. Motivo: multiplicar α no cambia la media del Dirichlet (~66 % ERC, ~29 % SUMAR) y el pool transferido es el ~2-3 % del voto catalán, demasiado pequeño para mover el último escaño de D'Hondt. El supuesto es robusto: solo su media importa, y esa viene del post del 29-N.
- Misma prueba con el Dirichlet de AA (`aa_mult`, sobre `aa_caniba_alpha` PODEMOS 4/SUMAR 4/PSOE 1): tampoco mueve nada (PSOE 111, AA 3, SUMAR 13 en las tres variantes; dif_mapa de PSOE −3,34 idéntico). Casi toda la masa de AA sale de OTROS y la parte que toca a PSOE es marginal. Los parámetros que gobiernan el efecto AA son `aa_pct_and` (9,62 %) y `aa_sd_logit` (0,2), no el destino del excedente.
- Sensibilizados `aa_pct` y `aa_sd` (nuevos argumentos de `simular_mapa()`, one-at-a-time, misma semilla): share andaluz de AA 7 % → AA 2,4 (IC90 1–3); 9,62 % → 3,5 (3–6); 12,5 % → 5,4 (3–10), ~0,36 escaños por punto. PSOE solo cede 0,7 escaños entre 7 % y 12,5 % (OTROS es el colchón y no compite). `aa_sd` controla la anchura, no el centro: IC90 de AA 3–4 (sd 0,1) frente a 2–8 (sd 0,3). P(PP+VOX ≥ 176) = 100 % en los cinco casos. Citable: «AA: 3 escaños (3–6); 2–10 en el rango plausible de su resultado andaluz».
- Comparación brms/Stan en escaños (2026-10-24, regenerado `07` brms + `08` → `escanos_mrp_*_brms.csv`, `estim_escanos_mrp_brms.rds`, `post_celdas_casa_media_brms.rds`): medias y medianas idénticas (±1-2 escaños; PP 114/114, PSOE 110/111, VOX 92/91) y voto nacional igual de preciso, pero los IC90 de escaños de brms son 2-3× más anchos (PP 98–126 vs 108–120; VOX 69–118 vs 84–98): la sd draw a draw de la share provincial es un 60-70 % mayor en brms para los partidos grandes (PSOE 2,59 vs 1,60 pp; VOX 3,86 vs 2,18), con ruido común dentro de cada draw, y D'Hondt lo amplifica. Descartado: la sd del efecto provincia (0,080 brms vs 0,108 Stan), la mezcla mm y el número de draws. Explica una parte el modo común no identificado del grupo mm sin centrar (sd 0,013 en logit ≈ 0,25 pp; en Stan centra() lo fija a 0); el resto apunta a correlaciones posterior entre efectos anidados distintas por parametrización — por descomponer si hace falta. Consecuencia para el post: los escaños y sus intervalos salen de la vía Stan; la comparación brms/Stan queda como comprobación de implementación con este hallazgo del ancho.
- `09_escanos_mapa.R` arreglado y estrenado (fallaba: `gtools::rdirichlet` no pone nombres de columna y faltaba `library(purrr)`). Salidas: `escanos_mapa_*.csv/png`. Efecto del mapa (sobre el fit brms sin participación): PP −1,6, PSOE −1,8, VOX −1,6, SUMAR +1,3, ERC +1,1, AA +3,7. Baseline brms: PP 114, PSOE 110, VOX 92; P(PP+VOX ≥ 176) = 99,6 %. VOX 92 frente a 76 del fit de la mañana (Stan con participación): probablemente por quitar la ponderación de participación; comprobar con la comparación con/sin participación de esta cadena.

## 2026-10-25: el ancho de brms era un artefacto del epred sin casa

- Motivación: explicar los IC90 de escaños 2-3× más anchos en brms. Scripts nuevos: `10_descomp_bayes.R` (descomposición de la varianza del share por término en una celda, con el ensamblado del eta validado contra `posterior_linpred` a 1e-15) y `10c_agregado_fits.R` (agregado nacional y provincial calculado directamente de los dos fits, sin pasar por post_celdas; guarda `datos/agregado_fits.rds`). Logs: `datos/log_10_descomp.txt`, `datos/log_10c.txt`.
- Resultado: **Stan y brms coinciden en todo** con el ensamblado completo (casa media, como el gq de Stan). Voto nacional, sd: PSOE 0,68 (Stan) vs 0,64 (brms), VOX 0,75 vs 0,83, PP 0,63 vs 0,56. Share provincial, sd media en las 52: PSOE 1,60 vs 1,59, VOX 2,18 vs 2,34, PP 1,68 vs 1,31. Correlación de desviaciones entre provincias: 0,13-0,21 en los dos. A nivel de celda (Madrid modal): 3,22 vs 3,15 (PSOE), 4,6 vs 4,8 (VOX). Las `sd_` por grupo y partido también cuadran (casa, casa:rec, rec, prov, ccaa, ccaa:rec, edad, edu, mes). Por término hay diferencias (el intercepto y `rec` de PSOE más ruidosos en brms, `edad` más en Stan) que se anulan en el total.
- Causa del hallazgo de ayer: `post_celdas_casa_media_brms.rds` (rama brms de `07`, herencia de `04b`) evaluaba `posterior_epred` con `re_formula` **sin los términos de casa**. En la parametrización de brms (interceptos libres por dpar, sin centrar) el modo común intercepto↔efectos solo queda anulado con todos los términos: sin casa, la sd del eta de celda pasa de 0,20 a 0,60 (sd del `b_Intercept` de PSOE: 0,59) y el agregado nacional de PSOE de 0,64 a 2,10. Reproducido exactamente con la llamada de `04b` a escala completa: nacional 2,098, Madrid 1,995, cor Madrid-nacional 0,943. D'Hondt amplifica y de ahí los IC 2-3× más anchos. No era diferencia de modelos ni del sampler: era la especificación del epred.
- Consecuencias: los `escanos_mrp_*_brms.csv` de ayer están contaminados; si el post menciona la comparación brms/Stan, o se regenera la vía brms con el epred incluyendo casa (casa media, como el gq de Stan) o se cuenta solo la coincidencia de centros. Los escaños finales (Stan) no cambian. La decisión de usar Stan se mantiene (más rápido), pero ya no por el ancho.
- Notas técnicas: (1) el orden interno de los dos fits es `voto_lv` = `colnames(celdas$y)`; el orden de las columnas de recuento de `celdas_brms.rds` es OTRO (VOX, PSOE, PODEMOS, PP...) — usar siempre `setdiff(partidos_lv, "AC")`. (2) Al rellenar un array draws×B×J por expansión de filas, los draws deben quedar como dim más rápida: `rep(seq_len(nd), times = B)`, no `each` (con `each` las draws se rebarajan y la media sale bien pero las draws sin sentido). (3) `as_draws_matrix`/subset de `draws_matrix` no es una matriz plana: convertir con `matrix(as.numeric(m), ...)` antes de aritmética mixta. (4) Comparar draws de submuestras distintas (ndraws=1000 de epred frente a thinning propio) da cor≈0 por emparejamiento arbitrario: comparar distribuciones, no draws fila a fila.
- Borrados `10b_agregado_provincial.R` y `10d_debug_stan.R` (diagnósticos intermedios, con el orden de partidos mal puesto).

## Ejecutar el ajuste completo en otro PC

Entradas en git: `datos/celdas_brms.rds`, `celdas_part.rds`, `tabla_post_prov.rds`, `prov_ccaa.rds` y `00_partidos.R`. Con ellas basta para 05b; no hacen falta los microdatos (`descargas/`, fuera de git) salvo para rehacer los pasos 01–03.

Para rehacer 02–03b: release `mrp-multiencuesta-datos` (`encuestas.rds`, `censo_66600.rds`, `targets_prov.rds`, `mun_23j.rds`, `targets.rds`, `tabla_postestratificacion.rds`; la descripción dice dónde copiar cada uno). Bajar con `gh release download mrp-multiencuesta-datos -D <carpeta>`.

1. `renv::restore()` y CmdStan instalado (`cmdstanr::install_cmdstan()`; aquí, 2.40.0).
2. Desde la raíz del repo: `Rscript 2026/11/mrp-multiencuesta/05b_separado.R prueba 2` (unos 3 min).
3. `Rscript 2026/11/mrp-multiencuesta/05b_separado.R nuts H > 2026/11/mrp-multiencuesta/datos/log_05b_nuts.txt 2>&1`, con H = hilos por cadena (4 cadenas × H núcleos). Estimado aquí: ~70 min el voto con 3 hilos, ~11 min la participación (deducido de la prueba, sin medir).
4. Salidas: `datos/fit_separado_nuts_voto.rds`, `_part.rds`, `estim_separado_nuts.rds`.
