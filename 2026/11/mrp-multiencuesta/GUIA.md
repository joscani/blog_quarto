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
| `11_mgcv_cis.R` | fase 3: contraste ML (`mgcv::gam(multinom)`), solo CIS | `estim_gam_cis.rds` (no sale en el post) |
| `07_post_celdas.R` | fase 3: probabilidad de voto por celda y draw (casa media, último mes). Constantes: `fuente`, `con_part` | `post_celdas_casa_media(_rtmb)_part.rds` |
| `08_escanos.R` | fase 3: D'Hondt y mayorías sobre el posterior por celda | `escanos_mrp_*.csv/png`, `estim_escanos_mrp*.rds` |
| `09_mapa_29N.R` | fase 8: mapa de coaliciones del 29-N sobre el posterior de 08 | `escanos_mapa_*` |
| `10_compara_stan_brms.R` | fase 5: por qué los IC de brms son 2-3× más anchos; agregado patrón oro | `agregado_fits.rds` |

Los tres backends comparten `06_stan_datos.R`: mismo modelo (multinomial con la
verosimilitud bien, UPN como categoría), mismos datos, mismo post. Para una
prueba rápida: `modo <- "prueba"` en `06_stan_datos.R` y en el ajuste; en
producción `"nuts"` (constantes al principio de cada script, sin argumentos).

Orden: `01 → 02 → 03 → 06 → (stan | numpyro | rtmb) → 07 → 08 → 09`.

## Estado al 2026-11-05: base del post (III)

**El modelo de voto del post (III) es el de seguro / dudoso** (decisión de
José, 2026-11-03), en `completo/`. **Las cifras del post salen del ajuste de
Stan** (decisión de José, 2026-11-05). Detalle y resultados en
`PARTICIPACION.md`, sección "El modelo hecho".

**Para seguir en otra máquina** (código en `master`, datos en la release
`mrp-multiencuesta-completo`): `bash
2026/11/mrp-multiencuesta/completo/baja_datos.sh`. No hace falta compilar ni
reajustar nada, ni `post_celdas_casa_media_part.rds`, ni Python.

- Orden: `completo/10_datos_voto_seg.R` → `completo/11_ajusta_voto_seg.R`
  (`completo/stan/mrp_voto_seg.stan`; 256 min con `HILOS=2`) →
  `completo/11b_entradas_escenarios.R` → `completo/12_escenarios_voto_seg.R`
  (~4 min) → `14`, `15`, `16` (cifras de apoyo).
- Ajuste de Stan: 4 × 1.000, `max_treedepth = 8`, `adapt_delta = 0.95`,
  0 divergencias de 4.000, Rhat máx 1,012, ESS bulk mín 535, E-BFMI 0,83–0,95
  (`completo/datos/log_voto_seg_h2.txt`). Con `max_treedepth = 10` iba a
  ~25 s por iteración (más de 10 h).
- NumPyro, para iterar: `completo/numpyro/exporta_voto_seg_npz.R` →
  `completo/numpyro/mrp_voto_seg_np.py --modo nuts` (~50 min) →
  `FUENTE=numpyro completo/12_escenarios_voto_seg.R` (salidas en
  `completo/datos/escenarios_seg_numpyro/`). Medias casi iguales a Stan en los
  2.895 valores (diferencia máx 0,095 sd) y media de escaños con diferencia
  máx 0,15 en las 104 combinaciones de método, escenario y partido.
- Optimizar el `.stan` no compensa (medido el 2026-11-05, una corrida de 40
  iteraciones): saltarse las casillas con offset −20, 23,9 → 25,4 ms por paso;
  `STAN_NO_RANGE_CHECKS` + `STAN_CPP_OPTIMS`, 22,4 ms; `--territorio` en NumPyro
  sigue en 255 pasos, sin mejora clara de ESS.
- Lecciones del día: (1) forma no centrada en todo (la centrada no
  convergió); (2) en NumPyro, guardar los draws antes de arviz (arviz 1.3
  cambió `from_dict` y se perdió un ajuste de 46 min); (3) bajo consumo en
  el Mac: la mitad de núcleos y el doble de tiempo.

**Guion del post (III)** (discutido con José el 2026-11-03; lo escribe él,
previsto para el finde 7–8 nov 2026). El texto es de José; aquí solo la
estructura y las cifras de apoyo. Criterio: pocos insights, y cada uno
enseña una pieza del método (bruto frente a MRP, y por qué difieren).

0. **Pregunta de entrada**: ¿qué pasa si vota más gente? Empezar por la
   pregunta, no por el método.
1. **Dónde estábamos**: (I) y (II) en tres frases, con enlaces.
2. **Voto + simpatía** (`voto_sim`): por qué, y cuántas entrevistas se
   ganan (tabla de n con y sin simpatía; cifra pendiente de sacar).
3. **Seguro / dudoso en los datos** (no b_dud del modelo: aún no se ha
   presentado). Seguro = dice 10. Secuencia:
   - Tabla bruta de % de voto (20.552 entrevistas con seg y provincia;
     fuera 4.201 de Ateneo y MIC sin provincia): seguros / dudosos,
     PP 25,4 / 21,0; PSOE 31,2 / 25,9; VOX 15,6 / 14,7; SUMAR 8,6 / 7,7;
     OTROS 6,4 / 11,6; BLANCO 2,2 / 8,1; partidos pequeños en "resto".
   - Tabla de composición: los dudosos son otra gente. 18-34 años: 30,8 %
     de los dudosos frente a 16,5 % de los seguros; recuerdo ABST: 19,6 %
     frente a 3,1 %.
   - Una frase: con el mismo perfil, VOX baja entre los dudosos (en 6 de 7
     tramos de edad; entre ABST, 17,7 % frente a 27,8 %). En bruto parecía
     que no (0,94 frente a 0,83 de PP). Estandarizado edad x recuerdo x
     casa, VOX -0,28 frente a PP -0,16; el modelo, -0,40 frente a -0,14.
     La tabla estandarizada no va al post (paso de método que sobra).
   - Ojo: % de dudosos por casa: 40dB 35, MIC 29, CIS 17 (¿cómo pregunta
     cada casa la escala?, sin comprobar en los cuestionarios). Una frase.
   - Script: `completo/14_vox_seguro_dudoso.R`.
4. **Una variable que no está en la tabla** (núcleo didáctico): primero
   una celda con números (cuántos seguros y dudosos, cómo vota cada grupo),
   después P(voto | celda, seg) · P(seg | celda). Ejemplo en
   `PARTICIPACION.md` ("la enmienda, en corto"). Aquí entra b_dud (BLANCO
   +1,00, VOX -0,40), mejor como probabilidades de una celda típica que en
   log-odds.
5. **Escenarios de participación**: titular en la contraintuición. De 65 %
   a 80 %, PP+VOX pasa de 199,4 a 197,7 escaños (método `seg`), intervalos
   casi iguales. Después, quién entra (70,4 % → 80 %): VOX 18,5 %, OTROS
   10,9 %, BLANCO 8,9 %. Un solo método en el texto (`seg`); `celda` en
   nota o caja plegada. PENDIENTE: por qué `seg` y `celda` difieren ya en
   la participación base (198,6 frente a 195,1).
6. **Lo que la encuesta no ve** (dos insights):
   - *Multinivel (pooling)*: Segovia y Soria, VOX. Bruto: Segovia 10,0 %
     (5 de 50), Soria 22,2 % (8 de 36), 12 puntos de hueco decididos por
     3 personas. MRP: 21,3 [18,0, 25,3] y 20,2 [16,8, 24,9]. Castilla y
     León bruto 15,2 %. Control: Madrid (n = 3.331) 15,5 → 20,1. Poner
     la fila España (15,4 → 19,7): ese +4,3 es común a todas y no es
     pooling. Script: `completo/15_vox_provincia.R`.
   - *Postestratificación*: la muestra tiene de más a los del PSOE de 2023
     y de menos a los abstencionistas. Recuerdo PSOE: muestra 33,5 %,
     votantes 25,9 % (VOX dentro: 2,7 %); ABST: 6,8 % frente a 13,5 % (VOX
     dentro: 21,3 %); NO_PODIA: 2,4 % frente a 3,6 % (23,5 %). Enlaza con
     la sección 3 (los abstencionistas otra vez).
     Descomposición del +4,3 (pasos sucesivos, depende del orden): bruto
     15,4 → recuerdo x edad 16,8 (+1,4) → casa media 18,2 (+1,4) → MRP
     19,7 (+1,5 sin separar: Ateneo, edu/sexo/provincia, pooling, mezcla).
     Script: `completo/16_vox_salto_nacional.R`.
7. **Supuestos y límites**: los dudosos que entran votan como los dudosos
   de su celda; `c_mezcla`; casa media es una elección (CIS 12,8 % de VOX
   en bruto frente a 40dB 20,9 % y MIC 19,8 %; el CIS es el 68 % de la
   muestra); el modelo no tiene mujer x edad, así que no puede ver una
   brecha de género propia de los jóvenes (la postestratificación no
   inventa).

Fuera del post (o caja plegada al final): NumPyro frente a Stan, forma no
centrada, glmmTMB como comprobación. Fuera: matriz de transferencias
completa, VOX por estudios, pesos por celda y xgboost (como mucho, un
párrafo de "otras formas de hacerlo"; si se hace el xgboost, tidymodels
con `finetune::tune_race_anova()`, `mn_log_loss` y CV agrupada por
encuesta).

Pendientes antes de escribir: n con y sin simpatía (sección 2); por qué
`seg` y `celda` difieren en la base (sección 5); la referencia
`2026/10/ref_provincial_23J_13.csv` tiene VOX = 0 en Santa Cruz de
Tenerife (no usada; revisar si se usa).

**Apuntes para discutir (José, 2026-11-03), con la valoración hablada:**
- Pesos por celda (N_celda / n_celda) a cada entrevistado y un modelo
  ponderado, recortando pesos altos. Valoración: funciona, pero pierde lo
  que da el MRP. Celdas pequeñas con pesos enormes (recortar mete sesgo),
  celdas vacías sin peso (hay que fundirlas, como en el raking), y con
  pesos en la verosimilitud la "posterior" no está bien calibrada. No
  arregla seguro / dudoso: los pesos no saben quién es seguro. Útil como
  contraste en el post ("con pesos sale X, con MRP Y, la diferencia está
  en las celdas pequeñas"), no como modelo principal.
- Un xgboost multinomial como modelo de voto (`multi:softprob`, 15
  clases). Hay literatura de árboles en lugar del multinivel en el MRP (de
  memoria, sin comprobar: BARP, Bisbee 2019; autoMrP, Broniecki, Leemann y
  Wüest 2022). A favor: encuentra solo las interacciones, rápido, seguro /
  dudoso es una variable más. En contra: sin posterior (bootstrap), riesgo
  de sobreajuste en partidos pequeños (CV con log-loss), efecto casa sin
  pooling (predecir con cada casa y promediar), no admite la mezcla de
  provincias, máscara de territorio después de predecir, calibración en
  clases raras. Valoración: como comprobación, no para las cifras; si
  encuentra una interacción fuerte que el multinivel no tiene, es la
  pista. Como glmmTMB: herramientas rápidas para decidir la estructura, el
  bayesiano para las cifras.

## Semilla de la tabla a nivel de CCAA (14 nov 2026): qué falta por cambiar

**Qué cambió.** En `02_poblacion_raking.R`, `nivel_semilla <- "ccaa"`. La
asociación sexo × edad × estudios × recuerdo de la semilla sale ahora del CIS de
la comunidad, reescalado al número de entrevistas de cada provincia. Antes salía
del CIS de la provincia (`"prov"` lo recupera). Los targets (Censo y 23-J por
provincia) no cambian.

**Por qué.** En X señalaron que el mapa de ganador por edad ponía al PSOE primero
entre los jóvenes de Ourense. Venía de la semilla: 16 jóvenes del CIS en Ourense
(8 recuerdan PSOE, 1 PP), que daban a los jóvenes de Ourense más recuerdo PSOE
(22 %) que PP (16 %). Con la semilla de Galicia: PP 23 %, PSOE 17 %. Se probó
antes mezclar provincia y CCAA con 100 entrevistas ficticias (`k_ccaa = 100`):
no bastaba (PSOE seguía primero en el 88 % de los draws).

**Hecho (fase 1, sin reajustar Stan).**
- `02_poblacion_raking.R` → `datos/tabla_post_prov.rds`.
- Release `mrp-multiencuesta-datos`: `tabla_post_prov.rds` y
  `tabla_postestratificacion_provincia.csv` (enlazado en el post I) son ya la
  tabla nueva; las anteriores quedan como `tabla_post_prov_semilla_prov.rds` y
  `tabla_postestratificacion_provincia_semilla_prov.csv`.
- `datos/*.rds` fuera de git (antes iban `celdas_brms`, `celdas_part`,
  `tabla_post_prov` y `prov_ccaa`); se bajan con `completo/baja_datos.sh`.
- `completo/11b_entradas_escenarios.R` toma N de `tabla_post_prov.rds`.
- `completo/12_escenarios_voto_seg.R`: `stopifnot` del delta relajado de 1e-4 a
  5e-3 (diferencia observada con la tabla nueva: 1,4e-3 en logit).
- Rehechos `11b`, `12`, `21` y `28` (`mapa_ganador_seguridad.png`).

**Pendiente.**
1. Fase 2, reajustar con la tabla nueva:
   - `completo/10_datos_voto_seg.R`: reparte las entrevistas sin provincia
     (More in Common, 2.861 de 3.802; Ateneo, 1.024 de 1.338) entre las
     provincias de su CCAA con pesos de la tabla. Rehacer y reajustar
     `11_ajusta_voto_seg.R` (Stan, ~4 h).
   - Participación (`fit_separado_nuts_part.rds`, `mrp_part.stan`): su `delta`
     calibra la participación del 23-J con las N de la tabla. Rehacer sus datos
     y reajustar.
   - Después, volver el `stopifnot` del delta en `12` a 1e-4 y rehacer 11b, 12, 21.
2. Post I (`index.qmd`): el texto de la semilla (l. 256-293) dice "por
   provincia". El CSV de la release (l. 344) ya es el nuevo: decir en el post
   que la tabla cambió el 14 nov y enlazar la anterior (`_semilla_prov.csv`).
3. Post II (`ii-modelo.qmd`): usa la tabla (l. 160) y habla de la semilla (l. 823).
4. Post III (`iii-participacion.qmd` vía `iii-funciones.R`): lee
   `entradas_escenarios.rds` y `escenarios_seg/`; re-renderizar y revisar cifras.
5. Mapas por edad publicados en X: sustituir por `mapa_ganador_seguridad.png` nuevo.
6. Release `mrp-multiencuesta-completo` (ficheros de `completo/datos/`): subir los
   nuevos.
7. Los posts están congelados (`freeze`): borrar su `_freeze/` antes de renderizar.
8. Commit de `02`, `11b`, `12`, `28`, `GUIA.md`, `.gitignore`, `baja_datos.sh` y
   el borrado de git de los 4 `.rds` de `datos/`.

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
  con la de Stan (1,3e-06). Hecho (2026-10-30): máscara portada a numpyro
  (`numpyro/mrp_voto_np.py`) y RTMB (`rtmb/mrp_voto_rtmb.R`); reajustado el
  modelo de voto en producción y fits en la release. Nota: `celdas_brms.rds` es
  ahora a nivel estudio; el fit de brms guardado es de la agregación
  anterior (sin AA y con UPN dentro de OTROS). Para más tarde: el recuerdo
  de las autonómicas de mayo (9,6 % en la serie de octubre) como predictor
  natural de AA. brms no puede expresar la verosimilitud condicional de AA:
  en el post es el intento con `mm()`, no resultados; se reajusta igualmente
  con las celdas actuales (ver pendientes).
- Los fits de Stan/numpyro/RTMB de voto en `datos/` (y en la release) están al
  día con AA (2026-10-30). El de brms no (ver pendientes).
- **Idea para explorar (José, 2026-11-03): el voto como Poisson en glmmTMB**
  (truco multinomial–Poisson). Sin hacer ni probar. Para explorar
  estructuras rápido (¿sobra dudoso × edad?, ¿casa × recuerdo?), no para
  las cifras del post: es máxima verosimilitud con Laplace, sin draws.
  - Datos en largo: una fila por celda × partido, con los ceros. Las
    casillas imposibles (regionales fuera de su territorio, AA en
    encuestas sin AA) se quitan, sin offset de −20 ni condicionado.
  - Hace falta un **efecto fijo por celda** (φ_celda), no basta el offset
    log(n): con solo el offset, Σ_partido exp(η) no vale 1 y el modelo
    intenta explicar también el tamaño de cada celda. Con φ_celda la
    verosimilitud de η es la del multinomial. El offset log(N) de binomial
    a Poisson es solo una aproximación para sucesos raros (p pequeña).
  - Solo cuentan los términos con `partido`: lo que no interactúa con
    partido lo absorbe φ_celda.
  - `glmer` no sirve: ~14.700 columnas de `celda` en una matriz densa de
    ~220.000 filas (~26 GB), y con factores no da una sd por partido.
    glmmTMB sí: `sparseX = c(cond = TRUE)` y `diag(0 + partido | edad)`
    (una sd por partido, sin correlaciones). Citado de memoria, sin
    comprobar la versión instalada.
  - Esbozo: `y ~ 0 + celda + partido + partido:mujer + partido:dudoso +
    diag(0 + partido | casa) + ... + diag(0 + partido | prov) +
    diag(0 + partido | casa:rec) + diag(0 + partido:dudoso | edad) +
    diag(0 + partido:dudoso | rec)`, `family = poisson`, con `dudoso`
    numérico 0/1. Sin la mezcla de provincias (solo filas con provincia).
  - Mejor que `0 + celda` (~14.700 parámetros en el optimizador externo):
    **`(1 | celda)` con la sd fija y grande** (p. ej. 10) con `start` y
    `map` de glmmTMB. Deducido (sin probar): con prior plano, Laplace
    sobre φ_celda es exacto para η, porque lo que depende de η es
    S^(−n), con S = Σ_partido exp(η), igual que la integral exacta
    Γ(n)/S^n; Laplace solo falla en la constante de Stirling, que no
    depende de parámetros. Cada φ toca solo su celda: hessiano casi
    diagonal, barato. Con la sd estimada (no fija) deja de ser
    equivalente. INLA hace lo mismo con un `iid` de precisión fija
    (de memoria).
  - **Decisión (José, 2026-11-03): glmmTMB solo como prueba; RTMB no** (el
    RTMB del modelo completo agotó 48 GB el 2026-11-02). Script:
    `completo/13_glmmtmb_voto_seg.R`. Prueba con el 20 % de las celdas: 7,3 min,
    1,3 GB, un núcleo (sin OpenMP), hessiano no definido positivo, muchas
    sd en 0 y b_dud extremos en partidos pequeños (UPN −4,27, PNV +2,27);
    BLANCO igual que NumPyro (+1,00 frente a +1,01).
    Completo (14.692 celdas, 112.380 filas en largo, sin filas con mezcla):
    17,7 min, 3,0 GB, un núcleo; hessiano no definido positivo. b_dud:
    correlación 0,97 con NumPyro (VOX −0,38 frente a −0,40; BLANCO +1,00
    frente a +1,01; diferencias mayores en BILDU −0,61 / −0,43 y JUNTS).
    sd de dudoso × recuerdo en 0 en 10 de 15 partidos (NumPyro: 0,24–0,55):
    máxima verosimilitud lleva al borde las sd poco identificadas. Sirve
    para los efectos fijos y para explorar; las sd, del bayesiano.
  - **Decisión anterior (José, 2026-11-03): probar glmmTMB.** Alternativa en `glmer`
    (deducido, sin probar): `(1 | celda)` es efecto aleatorio, va en la Z
    dispersa y no tiene el problema de memoria de `0 + celda`. Con
    `nAGQ = 0` β se estima en PIRLS con φ y la equivalencia se mantiene
    (la curvatura de φ_i en su máximo es n_i, sin η; perfilar φ deja el
    multinomial). Fijar la sd de celda exige la interfaz modular
    (`glFormula` → `mkGlmerDevfun(nAGQ = 0)` → `bobyqa` con esa sd fija),
    y una sd por partido exige ~150 términos a mano y `bobyqa` sin
    gradientes: viable solo con una sd por efecto (`(1 | partido:edad)`).

## Pendientes para el Mac (2026-10-30)

Tras subirlos a la release / commitearlos, hacer `git pull` en el otro PC y
seguir con el post (II).

1. **brms**: reajustar `05_brms_mm.R` con las celdas actuales (a nivel
   estudio, UPN como categoría). AA no se puede expresar (verosimilitud
   condicional): decidir cómo la trata el fit (fundida en OTROS, o fuera)
   y documentarlo. Subir `mod_brms_nuts.rds` a la release con `--clobber`
   y, si se regeneran, los `escanos_mrp_brms_*` a git.
2. **numpyro**: el npz ya está en era AA en la release; no hay que reajustar,
   solo regenerar (minutos): `numpyro/celdas_a_rds.R` con
   `datos/voto_numpyro_nuts.npz` → `08_escanos.R` con `fuente <- "numpyro"`
   → commitear los `escanos_mrp_numpyro_*` nuevos (los de git son del 10-28,
   sin AA). Si se prefiere reajustar de cero, `numpyro/mrp_voto_np.py --modo nuts`.
3. Con ambos, cerrar la tabla de igualdad Stan/numpyro/RTMB del post (II)
   y el texto de la sección brms (ii-modelo.qmd).

## Estado del post (II) al 2026-10-31: dónde lo dejo

Hecha una revisión de estilo, claridad y rigor de `ii-modelo.qmd` (aceptada
entera por José). Renderiza sin errores en el PC de casa. José quizá lo
reescriba después con su voz: **no reescribir el texto por iniciativa propia**;
editar solo lo que se pida, con cambios quirúrgicos.

Qué cambió (para no deshacerlo sin querer):

- Supuestos explícitos añadidos: pesos $w_p$ de la mezcla salen de la
  población (asume que More in Common/Ateneo reparten por provincia como la
  población dentro de la CCAA y perfil); multinomial condicionada de AA
  (el votante de AA sin AA en el cuestionario no se va en bloque a otro
  partido); recuerdo de voto (no se separa no respuesta de mala declaración);
  la tabla de población es fabricada por raking (4 de 10 cruces los pone la
  encuesta); máscara a cero exacto además del offset −20.
- Párrafo nuevo del modelo de participación (binomial sobre "10, seguro que
  voto", efecto casa sin Ateneo, casa media 60,5 % → calibrada a 70,4 %
  moviendo el intercepto por draw, draws emparejados por índice).
- Tabla de mayorías: formato "> 99,9 %" / "< 0,1 %", definición de
  Investidura, y párrafo de aviso: es la casa media **sin error del
  consenso**, no comparable con las horquillas del post todo bayesiano.
- Caption de la tabla de escaños explica `escanos` y `p_mas_esc`.
- Cifras corregidas contra PLAN.md: Rhat máx Stan 1,017 y ESS mín 466 (antes
  ponía 1,008, que es de numpyro); 120 comparaciones = 5 casas × 12 partidos
  × con/sin participación; `max_treedepth = 8` es elección propia (255 pasos
  es ese tope, no el de Stan por defecto); la ablación descrita como se hizo
  (sin mezcla; sd de prov, ccaa_rec, casa_rec a 0 una a una; todas a su
  mediana; cribados 200+200); "subir max_treedepth no compra nada" pasa a
  deducción no probada; prior predictivo 22 % (no 20 %); Stan participación
  con 1 hilo.
- RTMB: Laplace (no "analítico") y las muestras dejan las sd fijas en la moda.
- "7.817 celdas vacías" estaba escrito a mano y no cuadraba: ahora se calcula
  en el chunk `soria` (`n_celdas_vacias`, celdas con población sin ninguna
  entrevista con provincia) y da 6.781. "12.300 celdas" también es inline.
  Si José confirma otra definición, ajustar.
- Quitada la frase que justificaba interacciones que el modelo no tiene
  (provincia × recuerdo, edad × estudios).
- Quitado el callout «Una lección de brms por el camino» (el epred y el
  `re_formula`): la lección vive ahora en la sección «Lección de brms» de esta
  guía.
- Estructura: "Tres backends, un modelo" es ahora sección (6), con
  subsección "El árbol de NUTS, siempre a tope", **antes** de la Coda
  (reescrita). Después solo queda "Apéndice: cómo se ejecuta".
- Referencias: "post de octubre" ambiguo sustituido por "post de raking" o
  "post de todo bayesiano" con enlace.
- La tabla de población ya no se atribuye al post de raking de octubre: sale
  del raking de la primera parte de esta serie (targets oficiales, semilla CIS).
- Tras la figura de pooling, párrafo y tabla didácticos sobre la primera fila:
  la única encuestada de la celda (Segovia, M 65+, baja, rec. PSOE) contesta
  PP (cruda 0 %, n = 1) y el modelo da 90 % a PSOE porque una persona no
  representa a su perfil (chunk `primera-celda`).
- La figura de pooling pasa de 10 a 6 celdas: Segovia (n = 1), Las Palmas
  rec. PSOE (n = 1, el 13,1 % del ejemplo del texto), Las Palmas rec. PP
  (n = 2, el 43,9 %), València (n = 25), Madrid VOX (n = 46) y Madrid PP
  (n = 72); se pierden las de n = 4-15. El texto dice "seis celdas".
- Reescrito el "el modelo corrige el muestreo, no la casa" (en «De las celdas
  a los escaños» y en la Coda): el efecto de casa absorbe las diferencias
  entre casas (incluida casa×recuerdo), pero no lo que todas comparten; con
  4 casas aprende poco, harían falta más (microdatos); sin microdatos, solo
  queda usar las pocas publicadas o irse al meta-análisis. La Coda añade que
  la casa media va sin error del consenso.

Lo que sigue pendiente en el post (II):

1. Lo de la sección "Pendientes para el Mac" de abajo (brms y numpyro en era
   AA) y el `<!-- TODO: tabla de igualdad Stan/numpyro/RTMB -->` de la
   sección 6.
2. `<!-- TODO: pista del post -->` en el callout Listening.
3. Enlace de la tercera parte vacío (`[tercera parte]()`) en la Coda.
4. Al renderizar en el Mac, `datos/datos_pooling.rds` y
   `datos/datos_postestrat.rds` están en .gitignore pero **ya están en la
   release** (subidos el 2026-10-31): bajarlos con
   `gh release download mrp-multiencuesta-datos -p 'datos_p*.rds' -D 2026/11/mrp-multiencuesta/datos --clobber`.
   Si faltan, el post los regenera con `07b_pooling_datos.R` y
   `07c_postestrat_datos.R`, pero esos leen
   `datos/post_celdas_casa_media_part.rds` (700 MB, NO está en la release;
   sale de `07_post_celdas.R` con el fit de Stan).

## Lección de brms: el epred y el `re_formula` (quitada del post II, 2026-11-01)

Antes de rendirse con `brms` se llegó a ajustar el modelo entero
([`05_brms_mm.R`](https://github.com/joscani/blog_quarto/blob/master/2026/11/mrp-multiencuesta/05_brms_mm.R)).
Lección que no está en la documentación de brms: **el epred tiene que llevar
todos los términos aleatorios.**

- La primera postestratificación evaluaba `posterior_epred()` con un
  `re_formula` sin los términos de casa.
- En la parametrización de brms (interceptos libres por categoría, sin
  centrar), el modo común entre intercepto y efectos solo se anula con *todos*
  los términos. Sin los de casa: la sd del predictor de celda pasa de 0,20 a
  0,60 y la sd del agregado nacional de PSOE, de 0,64 a 2,10. D'Hondt
  amplifica, y los intervalos de escaños salían 2-3 veces más anchos que los
  de Stan. La culpa no era del sampler: era de cómo se pedía el epred.
- `re_formula` no añade términos a los del modelo: **sustituye** la lista
  completa.
  - `NULL` — el default; basta con no pasar el argumento —: mantiene todos
    los efectos aleatorios. Es lo que hace el `posterior_epred` final de
    `05_brms_mm.R` (líneas ~238-244).
  - `NA`: lo contrario, quita todos (solo efectos fijos).
  - Una fórmula parcial tipo `~ (1 | ccaa) + ...`: deja fuera todo lo que no
    aparezca en ella — ese era el fallo.

## Siguiente tarea (pedida el 2026-10-31, para hacer el 2026-11-01)

**Hacer el post (II) más empírico**: enseñar más los datos con tablas
pequeñas y ejemplos reales, sin alargarlo mucho. **Primero proponer (qué
tabla, dónde va, qué script la genera) y NO modificar `ii-modelo.qmd` hasta
que José elija.** Mejor 2-3 incorporaciones muy buenas que muchas. Solo
datos y resultados reales del proyecto: nada inventado ni aproximado para
rellenar. Nada de tablas enormes ni gráficos decorativos: cada ejemplo
responde a una pregunta concreta. Mantener su voz (cercano, directo, técnico,
contando lo que hace con los datos).

Candidatos que pidió:

1. Tabla de 6-8 celdas reales de postestratificación: provincia, sexo, edad,
   estudios, recuerdo, población estimada (N) y n.º de entrevistas.
   Provincias grandes y pequeñas, celdas con y sin entrevistas.
2. Para algunas, probabilidad posterior de voto con IC90, junto a la cruda:
   que se *vea* el partial pooling.
3. Ejemplo real de los pesos provinciales de la mezcla para una encuesta sin
   provincia, preferiblemente Canarias, con la cuenta de la probabilidad
   mezclada de un partido (CC).
4. Si cabe sin pisar la parte III: una provincia pequeña de celdas → % de
   voto provincial → D'Hondt.

Exploración hecha (el 31, sin escribir nada en el post):

- Fuentes: `datos/tabla_post_prov.rds` (array prov × sexo × edad × edu × rec,
  N); `datos/celdas_brms.rds` (entrevistas por fila, con `casa`, `prov` NA en
  las filas sin provincia, `p1..p9`/`w1..w9` = provincias y pesos de la
  mezcla, de `03_celdas.R::pesos_mm`); `datos/post_celdas_casa_media_part.rds`
  (700 MB, no está en la release; `p` = draws × celda × partido, `post` =
  metadatos con N y `pv_media`, `pv` = participación por draw; sale de
  `07_post_celdas.R`); `datos/estim_escanos_mrp.rds` (`votos` 1000 × 52 × 14,
  `esc` 1000 × 52 × 13, `prov_ref` con diputados, `totales`).
- Filas sin provincia: More in Common 1.556 y Ateneo 639 (con provincia 514
  y 202, uniprovinciales).
- Idea para 1+2 (una sola tabla): **un mismo perfil en varias provincias**
  (p. ej. Madrid, Sevilla, Las Palmas, Segovia, Soria y una con 0
  entrevistas), con N, n, cruda e IC90 del modelo. Así el pooling se ve sin
  explicarlo: donde hay n, el modelo se acerca a la cruda; donde no, tira
  hacia la estructura. Ojo: la cruda de `datos_pooling.rds` es solo del CIS,
  y la figura actual usa las nueve encuestas (`cruda9` en el post); y el
  modelo es la casa media.
- Idea para 3: coger una fila real de More in Common en Canarias de
  `celdas_brms.rds`, sus `w1, w2` (Las Palmas / S. C. de Tenerife) y la
  P(CC | provincia) de esa celda en `post_celdas`, y hacer la suma
  ∑ w_p · P(CC | p) a la vista. Con casa media, no con la casa More in
  Common: decirlo.
- Idea para 4: Soria (2 escaños) o Segovia/Teruel (3) con `est$votos` y
  `est$esc`: % provincial por draw → D'Hondt → distribución de repartos
  (p. ej. cuántos draws dan 2-1 frente a 1-1-1). Comprobar que no se pisa
  "el escaño más disputado" de la parte III.
- Probable implementación: un script nuevo `07d_ejemplos_datos.R` que lea
  `post_celdas` una vez y guarde un `datos/datos_ejemplos.rds` pequeño (como
  07b/07c), y subirlo a la release para que el post renderice sin los 700 MB.

## Datos y releases

Lo del post (III) va en la release `mrp-multiencuesta-completo`
(`completo/datos/*` está en .gitignore); se baja con `completo/baja_datos.sh`.
Lleva `fit_voto_seg.rds` (Stan; sus draws van dentro, no hacen falta los CSV),
`voto_seg_numpyro_draws_nuts.npz`, `datos_voto_seg.rds`, `entradas_escenarios.rds`,
`encuestas_completo.rds`, `escenarios_seg.tar.gz` (Stan y NumPyro) y los logs.
No lleva los ajustes de ideología y 3 niveles (`fit_ideol.rds`, `fit_part.rds`),
fuera del post.

Los datos y fits van en la release `mrp-multiencuesta-datos` (no en git;
`datos/*` está en .gitignore salvo `celdas_brms.rds`, `celdas_part.rds`,
`tabla_post_prov.rds` y `prov_ccaa.rds`). Actualizar con
`gh release upload mrp-multiencuesta-datos <ficheros> --clobber`. Los scripts
se enlazan desde `blob/master`.

En la release (2026-10-30): `censo_66600.rds`, `encuestas.rds`,
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
  10). Hilo conductor: el multinivel estima en sitios pequeños, lo
  bayesiano da bien la incertidumbre y la postestratificación corrige el
  sesgo de muestreo (no el de casa). DECIDIDO (2026-10-30): `mgcv` y
  `mclogit` no se mencionan en el post (el script 11 se arregló — la
  respuesta de la lista de fórmulas solo lleva la primera con `y ~` — pero
  queda como ejercicio interno; mclogit ni se corre). Arco:
  1. El problema: las provincias (escaños, celdas ralas, mediana n = 1 en
     el CIS), por qué multinivel y por qué bayesiano (variabilidad,
     ejemplos reales de celdas), con la figura de pooling (`07b`).
  2. More in Common sin provincia: mezcla exacta de probabilidades.
  3. El cuestionario es parte del modelo: UPN y AA (multinomial
     condicional).
  4. Por qué no lo hago con un glm: efectos fijos por celda, truco de
     Poisson (glmmTMB: Hessian no definida, sd colapsadas), y la razón de
     fondo: verosimilitudes a medida.
  5. `brms` con `mm()` (aproximación en logit) y sus dos lecciones.
  6. El modelo en Stan: especificación, priors, saturación de treedepth.
  7. Los escaños: D'Hondt draw a draw, mayorías.
  8. Los tres backends como servicio (no el core): tiempos e igualdad.
  9. Coda: el multinivel corrige el muestreo, no la casa; remite a (III).
- **(III)**, estructura decidida por José (2026-11-02):
  1. Enmienda corta al (II), sin mucha historia: participación aparte del
     voto, y el modelo de voto con `voto` en vez de `voto_sim`.
  2. Quizá el ejemplo de cómo usar ideología y participación en el MRP
     aunque no estén en la población (ejemplo numérico de una celda).
  3. Directo a insights que solo puede contestar el bayesiano: voto de las
     mujeres por edad, provincias, la transferencia (23-J → hoy).
  **Antes del (III), el modelo completo** («da tiempo de sobra»): un solo
  reajuste del voto con `voto_sim`, ideología y nivel de participación
  como predictores, y el blanco fuera de OTROS; ideología y participación
  como modelos aparte (ver `PARTICIPACION.md`). **Separado del flujo
  principal** (petición de José): todo en `completo/` (scripts, Stan y
  salidas en `completo/datos/`); solo lee `datos/` y no toca
  `00_partidos.R`, `01_armoniza.R` ni los scripts 03–08.
  Estado de `completo/` (2026-11-02):
  - Parte 1, ideología: `stan/mrp_ideol.stan` (sencillo: sexo, casa,
    edad, recuerdo, CCAA; sin mes, estudios ni provincia). 5,9 min, Rhat
    1,005, 9 divergencias de 4.000 (subir adapt_delta en el final).
    Reparto nacional igual que el modelo con provincia (±0,2 puntos).
  - Parte 2, participación: `stan/mrp_part3.stan` (3 niveles, ideología
    como predictor, umbrales por casa, sin provincia). 10 min, 1
    divergencia, Rhat 1,007. U por ideología: centro −0,45, der +0,60.
  - Parte 3, voto: **decisión de José: sin la interacción CCAA ×
    recuerdo** (3.724 de ~6.000 parámetros; mediana de |efecto| 0,003,
    máximo 0,25). Prueba de parametrización en curso con
    `stan/mrp_voto3.stan` (suma cero entre niveles; recuerdo y edad
    centrados) frente a 255 pasos por iteración de `stan/mrp_voto.stan`.
    Pendiente de comprobar: voto por CCAA con y sin la interacción (<1
    punto en los partidos grandes).
  Plan de trabajo (José, 2026-11-02): **prototipo con solo el CIS** (sin
  casa ni mezcla de provincias; rápido) para escribir y probar la cadena
  y la postestratificación; **final con las 4 casas, ajustado de noche**.
  En el post, las cifras de las 4 casas; el del CIS, como mucho un
  párrafo de comparación («con solo el CIS, el PSOE saldría X puntos
  más»), que refuerza la idea multiencuesta.
  Backend (José, 2026-11-02): por ahora Stan. NumPyro queda como opción
  para el voto final de las 4 casas (en el (II): 18,4 min frente a 121 min
  en Stan; port en `numpyro/mrp_voto_np.py`, sin adaptar).
  **Estado al cerrar (2026-11-02, 19:00):** ajustados y guardados en
  `completo/datos/`: `fit_ideol.rds` (parte 1) y `fit_part.rds` (parte 2).
  Parados sin terminar (lentos, ~12 s por iteración en warmup): la prueba
  B de las 4 casas (`stan/mrp_voto3.stan`, 1 cadena, iteración 100 de
  300) y el prototipo del CIS (`stan/mrp_voto_cis.stan`, 4 cadenas, menos
  de 50 de 2.000). Mañana: relanzar el voto mientras José está en la
  piscina; primero el prototipo del CIS (`08_ajusta_voto_cis.R`).
  Prueba glm (19:08–19:18): `stan/mrp_voto_cis_glm.stan` (CIS, una fila
  por entrevistado, `categorical_logit_glm`) no aceleró claramente: ~6 s
  por iteración en warmup, como el bucle. Deducción sin confirmar: el
  freno son los pasos por iteración (geometría), no el coste de cada
  paso. Parado en la iteración 100 de 1.000. Mañana: ver pasos por
  iteración en el sampling antes de decidir (Stan, NumPyro o más
  simplificación).
  **Decisión de José (2026-11-02, cierre):** el modelo de voto de las 4
  casas se itera en NumPyro o RTMB (ports ya existentes en `numpyro/` y
  `rtmb/`; en el (II): 18,4 min y 5,6 min frente a 121 min en Stan).
  Solo cuando la estructura esté decidida se hace el ajuste final en
  Stan.
  Primer intento RTMB de las 4 casas (19:29–19:40,
  `completo/rtmb/mrp_voto_completo_rtmb.R`): se quedó sin memoria (48 GB
  ocupados, 36 GB comprimidos; en 10 min ni una línea de `nlminb`).
  Parado. Deducción sin comprobar: la cinta de AD de la mezcla provincia
  x nivel (hasta 27 combinaciones por fila, antes 9) y los 2.998 efectos.
  Siguiente intento: sin Ateneo (máximo 9 combinaciones, como el (II)) o
  quitando de la mezcla las provincias con peso pequeño.
  Intentos RTMB 2 y 3 (19:41–19:44): con la mezcla reducida (21.194
  entradas, como el (II)) y con 2 hilos en vez de 8, la memoria
  comprimida pasa de 16 GB en 1–2 min (vigilante en
  `vigila_rtmb.sh` del scratchpad; no persiste). Descartados el relleno
  y los hilos. Sin saber: memoria del RTMB del (II). Mañana: medir en
  modo prueba (10 %) el del (II) y el nuevo; si el nuevo gasta mucho más
  por fila, quitar piezas nuevas de una en una (casa x nivel, casa x
  ideología, mezcla de nivel). `HILOS=n` como variable de entorno.
  Detalle anterior:
  abre con una enmienda corta al (II) sobre la participación
  (decisión de José, 2026-11-02: que ocupe poco y pase a los insights);
  borrador con el ejemplo numérico en `PARTICIPACION.md`, sección "Para
  el post (III)". La idea general (ampliar el MRP con variables sin
  censo: cadena ideología → participación → voto, ajuste por partes,
  indecisos dentro) está en la sección "La idea general" del mismo
  fichero; a José le parece la idea más potente de la serie. Después, insights bayesianos (escenarios de participación, dónde votan
  VOX los jóvenes, el escaño más disputado, mapa 29-N) y los sesgos que la
  postestratificación no corrige: fidelidad por casa (MIC y sus votantes
  superfieles al PSOE; el sesgo de recuerdo medido, no solo del CIS),
  autoselección en los paneles online y no-respuesta en el CIS (se
  seleccionan los votantes del PSOE más fieles/militantes entre los que
  contestan). Además:
  - Escenarios de participación (planteado por José, 2026-11-01): la
    calibración al 23-J es un solo intercepto en logit (casa media 60,5 %
    → 70,4 % residentes). Mover ese intercepto, en vez de anclarlo, da una
    familia de escenarios (más/menos participación) que conserva las
    diferencias entre perfiles — sumar lo mismo a todos los logits preserva
    los odds ratios — y se propaga draw a draw a escaños y mayorías.
    «Lo mismo lo hago para el siguiente post».
    **Problema (2026-11-02):** así, más participación favorece a la
    derecha, y José sabe que no es así. Causa: `mrp_voto.stan` no usa
    `prob_voto` y el votante nuevo vota como la media de su celda, que
    dominan los seguros. Datos crudos, abstencionistas del 23-J: derecha
    54,4 % en seguros frente a 38,1 % en no seguros. Meter la abstención
    como categoría del multinomial no lo arregla (el softmax reparte en
    proporción al voto de la celda). Dos alternativas; elegida B, con
    prueba antes del finde del 7–8 nov. Explicación sencilla de las dos en
    `PARTICIPACION.md`; scripts `15_delta_seguro.R` y
    `16_escenarios_part.R`, con salidas fuera de `datos/` (`DIR_SALIDA`):
    - **A. Modelo conjunto (reajusta todo).** Propensión ordinal en 3
      niveles (10 / 5–9 / 0–4) con los efectos de celda de
      `mrp_part.stan`; voto condicionado al nivel (efecto del nivel +
      nivel × recuerdo, con pooling). Casas sin `prob_voto` (Ateneo):
      nivel marginalizado con las probabilidades de la ordinal.
      Calibración al 23-J con un intercepto común. Escenarios = mover los
      umbrales: entra un "probable" con su voto. Sensibilidad γ para quien
      no contesta. Coste: reajustar el voto (hoy 121 min con 3 hilos) y
      algo más por la interacción. Pendiente: confirmar los cortes.
    - **B. Corrección en la postestratificación (sin reajustar).** q_s =
      P("10") por celda, de los draws de `mrp_part.stan`. Modelo pequeño
      sobre microdatos con `prob_voto`: multinomial seguro/no seguro ×
      recuerdo (quizá edad), con pooling → δ = log-odds no seguros −
      seguros. Por celda y draw, separar v en v_s y v_n con δ y la
      condición q_s·v_s + (1 − q_s)·v_n = v (el escenario base no cambia).
      En los escenarios, los votantes nuevos votan con v_n. Limitaciones:
      δ solo varía con recuerdo (y edad); su incertidumbre se combina como
      independiente de los otros dos modelos. Recomendada para el (III).
      Detalles de B (conversación desde el móvil, 2026-11-02):
      - Escenarios con un peso α sobre los no seguros, en vez de δ en logit
        sobre todos: voto_k ∝ Σ_j N_j·[p_j·P(k|j,s) + α·(1 − p_j)·P(k|j,n)],
        con α calibrado a la participación del escenario (60 %, 70 %...).
        Es otro supuesto sobre quién entra a votar, y se explica mejor.
      - Tilt: P(k|j,l) = P(k|j)·exp(Δ_kl) / Σ_k' P(k'|j)·exp(Δ_k'l), con Δ
        centrado para que la mezcla de niveles devuelva P(k|j).
      - Ordinal (probable / dudoso) solo si se quieren escenarios tipo «se
        movilizan los probables, no los dudosos»: reajusta solo
        `mrp_part.stan`, que es pequeño (11,5 min). Enseñar dos variantes.
      - Sensibilidad más barata: Δ fijado a mano por escenario (p. ej. los
        movilizados votan como los de 18–34, o como los abstencionistas
        del 23-J).
      - **Comprobado (2026-11-02, `datos/encuestas.rds` con el filtro de
        `03_celdas.R:31`):** el modelo de voto entra con todos los que
        declaran partido, sea cual sea su `prob_voto`. P(k|j) es una mezcla
        con mayoría de seguros: CIS 85,8 % (n = 12.473), More in Common
        72,2 % (n = 3.559), 40dB 65,7 % (n = 5.506); Ateneo sin
        `prob_voto` (n = 1.260). Esa proporción es mayor que la del modelo
        de participación, que cuenta también a quien no da partido
        (comprobado, % de "10" sin filtrar por voto, todas las entrevistas
        con `prob_voto`): CIS 78,5 %, More in Common 64,7 %, 40dB 60,6 %.
        Entre quien no da partido, solo 32–56 %. El centrado de Δ debe usar
        la mezcla del modelo de voto, no p_j.
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
