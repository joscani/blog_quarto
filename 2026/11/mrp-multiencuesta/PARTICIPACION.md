# Escenarios de participación

Nota de trabajo (2026-11-02) para el post (III). Explica qué hacemos para los
escenarios de participación y qué sería lo correcto si hubiera tiempo.

## El problema

El modelo de participación (`stan/mrp_part.stan`) da, para cada perfil
(celda), la probabilidad de decir "10, seguro que voto". La calibramos al 23-J
con un solo desplazamiento en logit: la casa media pasa del ~60,5 % al 70,4 %
de residentes.

Si movemos ese desplazamiento para hacer escenarios, más participación
favorece a la derecha. No nos lo creemos, y la razón es esta:

- El modelo de voto (`stan/mrp_voto.stan`) no sabe si el entrevistado está
  seguro de votar o no. En cada perfil da un solo reparto de voto.
- Ese reparto lo dominan los seguros. Entre los que dan partido, dicen "10" el
  85,4 % en el CIS, el 72,1 % en More in Common y el 66,1 % en 40dB.
- Al subir la participación, entra gente de los perfiles con más dudosos
  (jóvenes, abstencionistas del 23-J). El modelo los hace votar como los
  seguros de su perfil.
- Y no votan igual. Entre los abstencionistas del 23-J, la derecha saca el
  54,4 % de los seguros y el 38,1 % de los no seguros (datos crudos).

## Lo que vamos a hacer: corregir el voto después, sin reajustar

En cada perfil hay dos tipos de gente: los seguros y los dudosos. El modelo de
participación ya dice cuántos hay de cada tipo. Lo que falta es saber en qué
se diferencia su voto. Pasos:

1. **Medir la diferencia con un modelo pequeño aparte**
   (`15_delta_seguro.R`). Es un multinomial del partido sobre seguro/no
   seguro, con microdatos de CIS, 40dB y More in Common. La diferencia entre
   seguros y dudosos puede cambiar con el recuerdo de voto y con la edad, con
   pooling parcial. La casa entra como control. Partidos agrupados en PP,
   PSOE, VOX, SUMAR, PODEMOS y RESTO. Da Δ: cuánto más (o menos) vota cada
   partido un dudoso que un seguro, en log-odds.
2. **Separar el voto de cada perfil en dos** (`16_escenarios_part.R`). Del
   reparto que ya da el modelo de voto sacamos uno para los seguros y otro
   para los dudosos, con Δ. La mezcla de los dos tiene que devolver el
   reparto original, para que el escenario base no cambie. La mezcla usa la
   proporción de seguros entre los que dan partido, que es más alta que la
   del modelo de participación (ver más abajo).
3. **Mover la participación con un peso α sobre los dudosos.** Los seguros
   votan siempre. De los dudosos vota una fracción α, que subimos hasta la
   participación del escenario (65 %, 70,4 %, 75 %, 80 %).
4. **Comparar con lo de ahora** (desplazar el logit de todos), draw a draw:
   voto nacional, escaños, mayorías y a quién votan los que entran.

Resultado esperado: al subir la participación entra gente con sus propias
preferencias, y más en los perfiles con muchos dudosos. Ahora mismo, en
cambio, los que entran votan igual que los que ya votaban.

### Supuestos de esta versión

- La diferencia entre seguros y dudosos depende solo del recuerdo y la edad.
  Es la misma en todas las provincias y niveles de estudios.
- Los votantes nuevos salen solo de los dudosos.
- La incertidumbre de Δ se combina como si fuera independiente de los dos
  modelos grandes. No lo es del todo: comparten datos.
- Proporción de seguros entre los que dan partido: no la estima ningún modelo.
  Se aproxima desplazando el logit de la probabilidad de seguro de cada celda
  una cantidad fija c. c es la media, entre las tres casas, de la diferencia
  observada en logit entre "seguros entre los que dan partido" y "seguros
  entre todos".
- Ateneo del Dato no pregunta la probabilidad de votar: no entra en el modelo
  pequeño.

### Comprobado al preparar la prueba

| Casa | % "10" con partido | % "10" en todas | % "10" sin partido |
|---|---|---|---|
| CIS | 85,4 | 78,5 | 55,9 |
| More in Common | 72,1 | 64,7 | 32,1 |
| 40dB | 66,1 | 60,6 | 31,8 |

Fuente: `datos/encuestas.rds`, entrevistas con `prob_voto`. El modelo de voto
entra con todos los que dan partido (filtro de `03_celdas.R:31`), sea cual sea
su probabilidad de votar.

## Lo que habría que haber hecho: un solo modelo

La forma correcta es estimar participación y voto juntos, en un solo modelo de
Stan:

1. **Propensión a votar en varios niveles** con un logit ordinal: seguro (10),
   probable (5–9) e improbable (0–4), con los mismos efectos de perfil que el
   modelo de participación.
2. **Voto que depende del nivel**: el multinomial del voto lleva el nivel
   como predictor, con efecto distinto por partido y su interacción con el
   recuerdo, todo con pooling parcial.
3. **Casas sin pregunta de propensión** (Ateneo): el nivel se trata como
   desconocido y se promedia con las probabilidades del ordinal.
4. **Calibración al 23-J** con un desplazamiento común de la ordinal.
5. **Escenarios**: se mueven los umbrales de la ordinal. El que entra a
   votar es un "probable", con el voto que le corresponde a su perfil y a su
   nivel.

Qué gana frente a lo que hacemos:

- La diferencia entre seguros y dudosos puede variar por provincia, estudios o
  lo que sea, y no solo por recuerdo y edad.
- Toda la incertidumbre sale de un solo posterior, sin combinar modelos como
  si fueran independientes.
- La proporción de seguros entre los que dan partido sale del modelo, no de
  una aproximación.

Por qué no lo hacemos: hay que reajustar el modelo de voto entero. Hoy tarda
121 min con 3 hilos y la interacción nivel × recuerdo añade parámetros. Para
el post, la corrección posterior basta si se cuentan sus supuestos.

### Ideología en el modelo completo (idea de José, 2026-11-02)

La ideología no está en el censo, pero se puede usar sin postestratificar por
ella como dato conocido. Se modela su distribución en cada celda y se suma
sobre ella (la idea de MrsP, Leemann y Wasserfallen, 2017):

P(ideología, nivel, voto | celda) = P(ideología | celda) ·
P(nivel | celda, ideología) · P(voto | celda, ideología, nivel)

1. Ideología: logit ordinal (5 tramos) con los efectos de celda y efecto casa.
2. Nivel de participación: logit ordinal de 3 niveles.
3. Voto: multinomial con ideología y nivel como predictores, con `voto_sim`.

Ventaja: los indecisos entran. Tienen ideología y nivel aunque no voto, y su
voto queda implícito (votan como los decididos de su celda, ideología y
nivel). Cuidado: efecto casa en la ideología y casas que no la preguntan
(marginalizar).

## La idea general: ampliar el MRP con variables que el censo no tiene

Conversación del 2026-11-02, para el post (III). A José le parece la idea más
potente de la serie: «en mi mente creo que ya iba barruntando que eso se
podía hacer, pero no acababa de entender el cómo».

**La clave.** La tabla de postestratificación no necesita contener una
variable para poder usarla. Basta con un modelo que diga cómo se reparte esa
variable dentro de cada celda. El censo da cuántos hay en cada celda; los
modelos dan el resto, en cadena, y cada paso condiciona al siguiente.

**La cadena para el voto.**

1. Ideología: ideología_i ~ efectos de celda (sexo, edad, estudios,
   recuerdo, CCAA, provincia) + casa. Logit ordinal.
2. Participación: nivel_i ~ efectos de celda + casa + ideología_i. Logit
   ordinal de 3 niveles.
3. Voto: voto_i ~ efectos de celda + casa + ideología_i + nivel_i, con
   `voto_sim`.

Las partes 2 y 3 usan la ideología y el nivel **que declara cada
entrevistado** (datos), no las predicciones de la parte anterior. Las
predicciones solo entran en la postestratificación y cuando falta un dato.
El orden ideología → participación → voto es una elección; cualquier orden
da la misma distribución conjunta, y este es el natural (lo más estable
primero).

**Por qué es barato: se ajusta por partes.** Si las tres partes no comparten
parámetros, la verosimilitud conjunta es el producto
P(ideol | celda) · P(nivel | celda, ideol) · P(voto | celda, ideol, nivel), y
ajustar cada una por separado da el mismo posterior que ajustarlas juntas.
Coste (deducido, sin medir): las partes 1 y 2 tienen un solo predictor lineal
y unos umbrales, como `mrp_part.stan` (11,5 min); la 3 es el modelo de voto
actual (121 min) con dos predictores más. Lo caro del voto son sus 14
categorías.

**Los indecisos entran.** Tienen ideología y nivel aunque no tengan voto:
aportan a las partes 1 y 2 y no aparecen en la 3. Su voto queda implícito
bajo un supuesto: faltan al azar dada su celda, ideología y nivel (MAR).

**Casas que no preguntan algo.** Ateneo no pregunta la probabilidad de
votar: en la parte 3 su nivel se promedia con P₂. Eso acopla las partes 2 y
3; salida práctica en dos etapas (P₂ ya ajustada como pesos fijos, solo para
sus 1.444 entrevistas). Quien no da ideología: promedio con P₁, o un tramo
"ns".

**Postestratificación.** Cada celda se parte en ideología × nivel (5 × 3 =
15 grupos), draw a draw:

votos_k(celda) = N · Σ_ideol Σ_nivel P₁(ideol) · P₂(nivel | ideol) ·
r_nivel · P₃(k | ideol, nivel)

r_nivel es la tasa de voto de cada nivel: la de los seguros se calibra al
23-J, y los escenarios mueven r_probable y r_improbable. Dos variantes para
enseñar: «se movilizan los probables» y «se movilizan también los
improbables».

**Generaliza.** Cualquier variable que se pregunte en las encuestas puede
entrar como un eslabón más: interés por la política, valoración de líderes,
medio de información. El límite es práctico: cada variable multiplica los
grupos por celda y añade un modelo.

**De dónde viene (citado de memoria, sin comprobar las referencias).**

- MrsP, Leemann y Wasserfallen (2017, *AJPS*): ampliar la tabla de
  postestratificación con variables sin censo, estimando su distribución
  conjunta.
- Kastellec, Lax, Malecki y Phillips (2015): postestratificación ampliada con
  la identificación partidista estimada a partir de otra encuesta.
- Ghitza y Gelman (2013), «Deep interactions with MRP»: participación y voto
  modelados juntos (presidenciales de EE. UU. de 2008).

Lo que añade esta versión: la cadena con ajuste por partes, los indecisos
dentro del modelo y varias casas que preguntan cosas distintas. En el post,
basta una línea al final de la enmienda con el enlace a Leemann y
Wasserfallen.

## Para el post (III): la enmienda, en corto

Decisión de José (2026-11-02): el (III) abre con una enmienda breve al (II)
sobre la participación, y pasa enseguida a los insights. Borrador:

> En el post anterior, la participación iba en un modelo aparte. Cada perfil
> tenía un solo reparto de voto, y si subía la participación, los nuevos
> votantes votaban como la media de su perfil. Pero en un mismo perfil no
> votan igual los que están seguros de ir a votar que los que dudan.
>
> Un ejemplo con números inventados. Una celda de 10.000 personas: 6.000 de
> izquierda y 4.000 de derecha. Entre los de izquierda, el 30 % está seguro de
> votar; entre los de derecha, el 50 %. Los seguros votan todos. De los
> dudosos vota una fracción r.
>
> | Grupo | Personas | Votan a A | Votantes con r = 0,3 | Votantes con r = 0,6 |
> |---|---|---|---|---|
> | Izquierda, seguro | 1.800 | 85 % | 1.800 | 1.800 |
> | Izquierda, dudoso | 4.200 | 75 % | 1.260 | 2.520 |
> | Derecha, seguro | 2.000 | 10 % | 2.000 | 2.000 |
> | Derecha, dudoso | 2.000 | 20 % | 600 | 1.200 |
> | **A sobre el voto** | | | **49,4 %** | **51,3 %** |
>
> Al subir la participación del 56,6 % al 75,2 %, A pasa del 49,4 % al
> 51,3 %: los 1.860 votantes nuevos le votan en un 57,3 %, porque entre los
> dudosos hay más gente de izquierda. Con un solo reparto por perfil, A se
> quedaría en el 49,4 %.
>
> En el modelo, cada celda se parte así en ideología × nivel de
> participación, aunque el censo no tenga ninguna de las dos:
>
> $$\text{votos}_k = N \sum_{\text{ideol}} \sum_{\text{nivel}}
> P(\text{ideol} \mid \text{celda}) \, P(\text{nivel} \mid \text{celda},
> \text{ideol}) \, r_\text{nivel} \, P(k \mid \text{celda}, \text{ideol},
> \text{nivel})$$

Comprobado a mano: 1.800·0,85 + 1.260·0,75 + 2.000·0,10 + 600·0,20 = 2.795
de 5.660 (49,4 %); con r = 0,6, 3.860 de 7.520 (51,3 %); entrantes 1.065 de
1.860 (57,3 %).

## Para el post (III): qué cambia en el modelo

Borrador (2026-11-02). Regla: en el cuerpo, solo lo que cambia el resultado
o la lectura; lo técnico, en un anexo o un desplegable.

**En el cuerpo (tres viñetas, después de la enmienda):**

- El voto usa voto + simpatía (`voto_sim`): la simpatía rellena a quien no
  da partido. El voto en blanco va aparte (antes iba en OTROS).
- Tres modelos encadenados: ideología según el perfil, nivel de
  participación (seguro / probable / improbable) según el perfil y la
  ideología, y voto según el perfil, la ideología y el nivel. Cada uno usa
  lo que declara el entrevistado; se encadenan al postestratificar.
- Se quita la interacción CCAA × recuerdo del modelo de voto: con el pooling,
  sus efectos se quedaban en casi cero (mediana 0,003 en log-odds, máximo
  0,25) y era más de la mitad de los parámetros.

**En el anexo o desplegable:**

- Por qué la ideología y la participación no llevan provincia, mes ni (la
  ideología) estudios: aportan poco (sd 0,001-0,11) y la provincia obligaba a
  mezclar provincias en las casas que no la dan. El reparto nacional de la
  ideología no cambia (±0,2 puntos).
- Umbrales por casa en la participación: el CIS tiene la mitad de
  "probables" (13,5 % frente a 26-29 %) con los improbables parecidos. Tabla
  de logits de 40dB frente al CIS (−0,11 en el primer corte, −0,84 en el
  segundo).
- Parametrización: suma cero entre niveles (si no, una constante pasa del
  intercepto a los efectos) y forma centrada o no centrada según el efecto.
  Con cifras de las pruebas: ESS 155 en los umbrales, 32 % de divergencias por
  el embudo, 255 pasos por iteración en el modelo de voto anterior.
- Cómo se comprobó que una interacción sobra: medias posteriores de los
  efectos (percentiles y rango por partido), con la advertencia de que una
  media cerca de 0 puede ser incertidumbre; la prueba definitiva es comparar
  el voto por CCAA con y sin ella.

**Hallazgos que pueden ir como insight y no como cambio técnico:**

- El que duda es de centro: efecto de la ideología en la seguridad de votar
  izq +0,25, cizq −0,30, centro −0,45, cder −0,10, der +0,60.
- El dudoso de centro vota en blanco más que el seguro (juguete: 18,9 %
  frente a 5,9 %).

## Resultados de la prueba (2026-11-02)

Escaños de PP + VOX (media, 1.000 draws, casa media, octubre de 2026):

| Participación | logit | α con `voto` | α con `voto_sim` | α, indecisos imputados |
|---|---|---|---|---|
| 65,0 % | 198,4 | 202,6 | 202,1 | 201,9 |
| 70,4 % (base) | 199,0 | 203,1 | 202,0 | 201,7 |
| 75,0 % | 199,6 | 203,4 | 201,9 | 201,4 |
| 80,0 % | 200,4 | 203,7 | 201,8 | 201,2 |

Indecisos imputados (`15b_imputa_ideologia.R`): `voto_sim` + el voto de los
indecisos no seguros imputado con su ideología. Casi no cambia: entre los
abstencionistas no seguros, PP + VOX pasa de 35,6 % (decididos) a 34,4 %
(todos). Bajo MAR dada la ideología, la intuición de que la abstención
beneficia a la derecha no aparece en estos datos; si es cierta, vendría de
algo que la ideología no recoge (voto oculto), que estas encuestas no pueden
comprobar.

- Con `voto_sim`, más participación deja de favorecer a PP + VOX, pero no
  favorece a la izquierda: el efecto queda plano.
- El 56 % de los que entran viene de celdas de recuerdo ABST. La mayor parte
  del efecto viene de quién entra (el perfil), no de cómo vota dentro de su
  celda.
- Pendiente: el 48,5 % de los abstencionistas no seguros sigue sin partido
  con `voto_sim`. Son más de izquierda por ideología (media 0,412 frente a
  0,464 de los que dan partido). Siguiente paso posible: imputarlos con la
  ideología en el modelo pequeño.
- El escenario base cambia con α (199 → 202–203 escaños): con α = 0,25 el
  electorado tiene más seguros que la muestra que da partido. Es una
  decisión de supuesto, sin resolver.

## Juguete con datos reales (2026-11-02)

`2026/11/mrp-ampliado-juguete/juguete.R`: CIS 3582, celdas sexo × edad ×
recuerdo (98), cuatro brms (clásico; ideología → seguro → voto). Voto con
`voto_sim` y el blanco separado (en la armonización iba a OTROS; recuperado de
`descargas/MD3582/3582.sav`).

- Con todo el mundo votando, clásico y ampliado difieren menos de 1 punto.
- Insight para el post: **el que duda es de centro.** % seguros: izquierda
  75,3, centro 59,4, derecha 81,1. Entre los dudosos, 46,9 % de centro, 36,2 %
  de izquierda, 16,9 % de derecha (entre los seguros: 27,3 / 44,0 / 28,8).
  Ojo: `m_seguro` no tiene ideología × edad ni × recuerdo, así que la U es la
  misma en todas las celdas por construcción.
- **El dudoso de centro vota en blanco:** 18,9 % frente a 5,9 % del seguro de
  centro (celda mujeres 25–34, ABST). Con el ampliado, el blanco pasa del
  2,5 % (solo seguros) al 5,6 % (todos); con el clásico, del 3,6 % al 4,7 %.
- Lo que gana la izquierda en composición lo pierde en voto: los dudosos de
  izquierda votan menos PSOE (24,9 % frente a 33,5 %) y más OTROS y blanco.

Hallazgo para el proyecto grande, sin tocar: `recod_cis_partido()`
(`00_partidos.R:107`) manda "En blanco" a OTROS (161 casos en el 3582, 3,7 %
del voto con partido). El comentario de `08_escanos.R` dice que el voto válido
no lleva blancos. No he mirado 40dB ni More in Common.

## El modelo hecho: voto con seguro / dudoso (2026-11-03)

**Base del post (III)** (decisión de José, 2026-11-03). Ni ideología ni
participación en 3 niveles: el voto lleva seguro (dice "10") / dudoso como
predictor y la participación es la de `stan/mrp_part.stan` del (II), sin
reajustar. Factorización exacta: P(voto | celda, seguro) · P(seguro | celda),
sin parámetros comunes.

- Datos: `completo/10_datos_voto_seg.R`. `voto_sim` (voto + simpatía),
  BLANCO aparte, 4 casas, 24.753 entrevistas en 17.537 filas. Filas sin
  provincia o sin seguro (Ateneo y 79 que no contestan): mezcla
  provincia × seguro con pesos fijos; P(seguro) de `mrp_part.stan`
  desplazada `c_mezcla` = 0,254 en logit (de "entre todos" a "entre los que
  dan partido").
- Modelo: `completo/stan/mrp_voto_seg.stan` (el ajuste del post) y su port
  `completo/numpyro/mrp_voto_seg_np.py` (para iterar). `mrp_voto3` + dudoso:
  d = b_dud + a_dud_edad[edad] + a_dud_rec[rec], con una sd por partido.
  **Todo en forma no centrada**: con edad y recuerdo centrados una cadena se
  atascó con `sd_edad` de BNG cerca de 0 (Rhat 1,59).
- Ajuste Stan (`completo/datos/fit_voto_seg.rds`, 2026-11-05): 256,1 min,
  4 × 1.000, 0 divergencias, Rhat máx 1,012, ESS bulk mín 535. Medias casi
  iguales a NumPyro (diferencia máx 0,095 sd).
- Ajuste NumPyro (`completo/datos/voto_seg_numpyro_draws_nuts.npz`): 51,6 min,
  4 × 1.000, 0 divergencias, Rhat máx 1,009, ESS bulk mín 630, 255 pasos en
  todas las iteraciones. `target_accept_prob` 0,8 no gana (ESS mín por
  minuto 13,1 frente a 12,2) y da lo mismo: se queda 0,95.
- b_dud (log-odds dudoso frente a seguro; media [90 %], Stan): BLANCO +1,00
  [+0,78, +1,22], OTROS +0,29, SUMAR +0,21, PSOE −0,09, PP −0,14,
  VOX −0,40 [−0,63, −0,15].
- Escenarios (`completo/12_escenarios_voto_seg.R`, salidas en
  `completo/datos/escenarios_seg/`): votan los seguros (N · π) y, si la
  participación del escenario pasa de π, los dudosos que entran
  (N · (π_cal − π)) con el voto de los dudosos de su celda. Sin α ni
  `c_mezcla` en la postestratificación.

Escaños de PP + VOX (Stan; media [90 %], 1.000 draws, casa media, 2026-10):

| Participación | seg (entran como dudosos) | celda (como la media de su celda) |
|---|---|---|
| 65,0 % | 199,4 [194, 205] | 195,1 [190, 200] |
| 70,4 % (base) | 198,6 [193, 204] | 195,1 [190, 201] |
| 75,0 % | 198,0 [192, 204] | 195,3 [190, 201] |
| 80,0 % | 197,7 [192, 203] | 195,6 [190, 201] |

- Con seguro / dudoso, más participación resta ~2 escaños a PP + VOX; con
  el reparto medio de la celda, suma algo. Dirección esperada por José.
- Los que entran (70,4 % → 80 %, % mediana): seg PP 20,7, PSOE 23,9,
  VOX 18,5, OTROS 10,9, BLANCO 8,9; celda: VOX 21,1, BLANCO 6,5.
- La base cambia con el método (198,6 frente a 195,1): seg fija quién vota
  con el modelo de participación (86 % de seguros entre los votantes);
  celda pondera con los seguros entre los que dan partido.
- No comparable uno a uno con la tabla de la prueba de 2026-11-02 (allí,
  voto del (II): `voto`, sin BLANCO, con CCAA × recuerdo).
- Supuesto: MAR dentro de celda y seguro / dudoso (el dudoso sin partido
  vota como el dudoso con partido de su celda).

## Ficheros

- `15_delta_seguro.R`: modelo pequeño (brms, multinomial con conteos
  agregados). Guarda Δ draw a draw.
- `16_escenarios_part.R`: escenarios con los dos métodos.
- Salidas de la prueba: en el directorio de `DIR_SALIDA` (variable de
  entorno), fuera de `datos/`, para no tocar lo publicado en el post (II).
