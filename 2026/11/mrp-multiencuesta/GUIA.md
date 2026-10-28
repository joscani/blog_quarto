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
- **AA (Adelante Andalucía)**: PENDIENTE. Ahora mismo va como categoría propia
  en el ajuste RTMB, pero está infraestimado: 40dB mensual y More in Common no
  preguntaban por AA y sus ceros arrastran su efecto de casa a 0 (nacional
  0,1%; en Andalucía 0,3-0,9%; 0 escaños). Opciones sobre la mesa: tratar AA
  como missing en las casas que no lo preguntan (efecto de casa estimado solo
  con CIS + flash) y qué hacer con el recall de las autonómicas de mayo (9,6%
  en la serie de octubre). El refit RTMB con AA está en local, no en la release.
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
- **(II) El multinomial, en Stan, numpyro y RTMB** (05, 06, backends, 07, 08,
  10, 11): Intro / El modelo / El intento con brms y `mm()` / El multinomial
  con la verosimilitud bien / Tres backends (Stan la referencia, numpyro
  rápido, RTMB Laplace y casi igual) / De celdas a escaños / Coda.
- **(III) opcional**: insights bayesianos (escenarios de participación, dónde
  votan VOX los jóvenes, el escaño más disputado, mapa 29-N) y límites de la
  postestratificación (fidelidad por casa, ad hoc al escribir).

El historial de decisiones está en PLAN.md.
