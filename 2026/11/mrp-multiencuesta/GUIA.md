# MRP multiencuesta: guía del pipeline

MRP de intención de voto con **9 encuestas** (CIS 3567/3571/3577/3582, 40dB
jul/sep/oct + flash, More in Common jun/sep, Ateneo jun): 29.831 entrevistas,
14.717 filas de voto. Postestratificación por provincia × sexo × edad ×
estudios × recuerdo del 23-J (tabla IPF con el Censo 66600 y los resultados
del 23-J por municipio). Escaños del Congreso con D'Hondt sobre el mapa del
29-N.

Este post es la continuación de `2026/10/mrp-raking` (una encuesta, CCAA) y
`2026/10/full-bayesian` (todo bayesiano con error histórico; aquí no hay error
histórico ni volatilidad: es voto latente medido por las encuestas del mes, con
la postestratificación y el shrinkage bayesiano de las áreas pequeñas).

## Estructura y orden de ejecución

| Script | Qué hace | Salida en `datos/` |
|---|---|---|
| `00_partidos.R` | definiciones comunes (partidos, territorio regional). Lo hacen `source()` todos | — |
| `01_armoniza.R` | fase 1: une y armoniza las 9 encuestas | `encuestas.rds` |
| `02_poblacion_raking.R` | fase 1: targets (censo 66600 + 23-J) y raking IPF | `targets_prov.rds`, `tabla_post_prov.rds`, `prov_ccaa.rds` |
| `03_celdas.R` | fase 1: agrega las encuestas en celdas | `celdas_brms.rds`, `celdas_part.rds` |
| `04_exporta.R` | fase 1: entrega SPSS/CSV/RDS + codebook | `entrega/` |
| `05_brms_mm.R` | fase 2: primer intento, brms con `mm()` para la provincia de More in Common | `mod_brms_nuts.rds`, `estim_brms_nuts.rds` |
| `06_stan_datos.R` | fase 3: listas de datos de modelo para los tres backends | `datos_stan.rds` |
| `stan/ajusta.R` | fase 3: ajusta `mrp_voto.stan` + `mrp_part.stan` (cmdstanr, reduce_sum) | `fit_separado_<modo>_voto/part.rds`, `estim_separado_<modo>.rds` |
| `stan/ajusta_cis.R` | fase 3: solo CIS (`mrp_cis.stan`), la referencia de una sola encuesta | `fit_stan_<modo>_nuevo_cis.rds`, `estim_stan_…_cis.rds` |
| `numpyro/exporta_npz.R` | fase 3: los datos a `.npz` | `voto_numpyro_<modo>.npz`, `part_numpyro_<modo>.npz` |
| `numpyro/mrp_voto_np.py`, `mrp_part_np.py` | fase 3: los mismos modelos en numpyro/JAX | `voto/part_numpyro_*.npz` |
| `numpyro/celdas_a_rds.R` | fase 3: los draws de numpyro a formato `post_celdas` | `post_celdas_casa_media_numpyro_part.rds` |
| `rtmb/ajusta.R` | fase 3: los mismos modelos con RTMB (Laplace; llama a `rtmb/mrp_*_rtmb.R` si faltan los fits) | `fit_rtmb_<modo>_voto/part.rds`, `estim_rtmb_<modo>.rds` |
| `11_mgcv_cis.R` | fase 3: contraste ML (`mgcv::gam(multinom)`), solo CIS | `estim_gam_cis.rds` |
| `07_post_celdas.R` | fase 3: probabilidad de voto por celda y draw (casa media, último mes). Constantes: `fuente`, `con_part` | `post_celdas_casa_media(_rtmb)_part.rds` |
| `08_escanos.R` | fase 3: D'Hondt y mayorías sobre el posterior por celda | `escanos_mrp_*.csv/png`, `estim_escanos_mrp*.rds` |
| `09_mapa_29N.R` | fase 8: mapa de coaliciones del 29-N sobre el posterior de 08 | `escanos_mapa_*` |
| `10_compara_stan_brms.R` | fase 5: por qué los IC de brms son 2-3× más anchos; agregado patrón oro | `agregado_fits.rds` |

Los tres backends comparten `06_stan_datos.R`: mismo modelo (multinomial con la
verosimilitud bien, UPN como categoría), mismos datos, mismo post. Para una
prueba rápida: `modo <- "prueba"` en `06_stan_datos.R` y en el ajuste; en
producción `"nuts"` (constantes al principio de cada script, sin argumentos).

Orden: `01 → 02 → 03 → 06 → (stan | numpyro | rtmb) → 07 → 08 → 09`.

## Decisiones y pendientes

- **UPN** como categoría propia (ceros estructurales fuera de Navarra).
- **AA (Adelante Andalucía)**: DECIDIDO (2026-10-29): categoría propia solo en
  Andalucía (como UPN en Navarra; las 96 menciones de los microdatos están
  todas en Andalucía) y **multinomial condicional** para las casas que no la
  preguntaron (40dB mensual, MIC, Ateneo): su verosimilitud normaliza
  excluyendo AA (`Σ y_j·η_j − n·log Σ_{j≠AA} exp(η_j)`), porque sus ceros son
  de diseño del cuestionario, no evidencia. AA queda identificada por CIS +
  flash (6 menciones). Hecho (2026-10-29): `estudio` en la clave de
  agregación de las celdas de voto (necesario para que la máscara sea
  constante por fila: el flash, que sí ofrece AA, comparte casa con los
  mensuales; +278 filas; no cambia el efecto casa, que sigue siendo
  predictor, ni la verosimilitud salvo donde cambia la máscara),
  `sin_aa` en `03_celdas.R`/`06_stan_datos.R` y la verosimilitud
  condicional en `funciones.stan` (`multinomial_cond`) y en las dos
  variantes de Stan (`mrp_voto.stan` fila a fila; `mrp_voto_vec.stan`,
  álgebra exacta con `log1m_exp`). Probado en modo prueba con
  `mrp_voto.stan`: 0 divergencias y la postestratificación en R cuadra
  con la de Stan (1,3e-06). Pendiente: portar la máscara a numpyro
  (`numpyro/mrp_voto_np.py`) y RTMB (`rtmb/mrp_voto_rtmb.R`), reajustar
  (solo el modelo de voto; el de participación no tiene categorías de
  partido) y subir los fits a la release. Nota: `celdas_brms.rds` es
  ahora a nivel estudio; el fit de brms guardado es de la agregación
  anterior y no se reajusta. Para más tarde: el recuerdo de las
  autonómicas de mayo (9,6 % en la serie de octubre) como predictor
  natural de AA. brms no se reajusta: no puede expresar la
  verosimilitud condicional y en el post es el intento con `mm()`, no
  resultados. Para más tarde: el recuerdo de las autonómicas de mayo (9,6 %
  en la serie de octubre) como predictor natural de AA.
- Los fits de Stan/brms/numpyro que hay en `datos/` son de la era **sin AA**;
  solo la cadena RTMB está al día con AA.

## Datos y releases

Los datos y fits van en la release `mrp-multiencuesta-datos` (no en git;
`datos/*` está en .gitignore salvo `celdas_brms.rds`, `celdas_part.rds`,
`tabla_post_prov.rds` y `prov_ccaa.rds`). Actualizar con
`gh release upload mrp-multiencuesta-datos <ficheros> --clobber`. Los scripts
se enlazan desde `blob/master`.

En la release (2026-10-28): `censo_66600.rds`, `encuestas.rds`,
`mun_23j.rds`, `tabla_postestratificacion.rds`, `targets.rds`,
`targets_prov.rds`, `fit_separado_nuts_voto.rds` (286 MB, ~2 h),
`fit_separado_nuts_part.rds`, `mod_brms_nuts.rds` (99 MB),
`voto_numpyro_nuts.npz`, `part_numpyro_nuts.npz`.
Los `post_celdas_*` (358-672 MB) no se suben: se regeneran con
`07_post_celdas.R` en minutos.

## Plan del post (borrador)

- **(I) Unir, armonizar y postestratificar** (scripts 00-04): Introducción /
  Las encuestas (quién pregunta qué) / Armonización / Población y raking / La
  tabla de postestratificación / Coda.
- **(II) Por qué no lo hago con un glm** (05, 06, backends, 07, 08,
  10, 11). Hilo conductor: el multinivel estima en sitios pequeños, lo
  bayesiano da bien la incertidumbre y la postestratificación corrige el
  sesgo de muestreo (no el de casa). Arco:
  1. Por qué multinivel (estimación en celdas pequeñas).
  2. No hay multinomial en lme4; el truco de Poisson como efectos fijos
     por celda es demasiado (sin pooling, celdas vacías muertas) y como
     GLMM Poisson lme4 no lo aguanta. Aquí entra `11_mgcv_cis.R`
     (`mgcv::gam(multinom)`) como lo más parecido a un GLMM multinomial
     de catálogo, solo CIS.
  3. `mclogit` (PQL). DECIDIDO (2026-10-29): probamos a correrlo (si no
     tira, discusión). Su límite real no es la provincia que falta,
     sino el reparto de MIC entre 9 provincias (mezcla de probabilidades,
     no de logits) que ningún GLMM de catálogo maneja.
  4. `brms` con `mm()` (aproximación en logit).
  5. La versión real en Stan: agregación exacta de probabilidades. El caso
     AA (ceros estructurales + encuestas que no la ofrecían) va dentro de
     esta sección: el cuestionario es parte del modelo.
  6. numpyro como alternativa rápida (JIT/XLA de toda la verosimilitud;
     misma geometría, misma posterior, otra implementación).
  7. Anexo RTMB (Laplace) y comprobaciones de igualdad entre backends.
  8. De celdas a escaños / Coda: el multinivel corrige el muestreo, no la
     casa; remite a (III).
- **(III)**: insights bayesianos (escenarios de participación, dónde votan
  VOX los jóvenes, el escaño más disputado, mapa 29-N) y los sesgos que la
  postestratificación no corrige: fidelidad por casa (MIC y sus votantes
  superfieles al PSOE; el sesgo de recuerdo medido, no solo del CIS),
  autoselección en los paneles online y no-respuesta en el CIS (se
  seleccionan los votantes del PSOE más fieles/militantes entre los que
  contestan). Además:
  - Estimaciones por provincia × nivel de estudios con intervalos, y
    comparación con lo que dice la encuesta cruda (la reproducción de la
    estimación publicada de MIC ya está en PLAN.md).
  - Contraste con un multinomial de efectos fijos (edad × sexo × edu ×
    recuerdo × provincia, sin pooling): en celdas ralas, estimaciones
    salvajes con intervalos enormes; en celdas vacías no existe. Dos
    fallos por separado: el sesgo de casa lo arregla modelar la casa, la
    varianza en celdas pequeñas solo el pooling. Mostrar también una celda
    rica donde todo coincide (el shrinkage no distorsiona donde hay datos).
  - En la (II), una figura de pooling en la sección "por qué multinivel":
    celda rala o vacía, encuesta cruda vs efectos fijos vs multinivel.
  - No duplicar lo del post de full-bayesian (2026/10): allí ya están el
    voto latente nacional, el error del consenso (sesgo, Omega,
    contracción), las manías de las casas (contra resultados de
    autonómicas), el rebaño y backtests; de escaños, solo mediana + IC90
    con shares nacionales y P(PP+Vox >= 176). Todo lo territorial,
    demográfico, la transferencia y los escenarios de participación son
    nuevos aquí; histograma de escaños y probabilidades de bloques se
    presentan como la versión provincial/completa de lo que allí rozó
    ("la segunda parte" que aquel post prometía). La fidelidad por casa
    conecta con las manías: enlazar, no repetir.

El historial de decisiones está en PLAN.md.
