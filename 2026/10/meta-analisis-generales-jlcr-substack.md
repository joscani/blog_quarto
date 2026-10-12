# Meta-análisis. Agregando encuestas IV. Generales 2026 (con escaños)

*Meta-análisis bayesiano de encuestas para las generales del 29-N y simulación de escaños con D'Hondt. Versión editada del post original del [blog](https://muestrear-no-es-pecado.es/2026/10/meta-analisis-generales-jlcr), con todo el código y los detalles técnicos.*

**Para escuchar mientras se lee**: [Tabletom — "De voto a escaños"](https://open.spotify.com/track/3pXal0lz6nsqLStop3HpjX) (o el tema que sea, la referencia del título es la que es).

## Introducción

Ya hice metaanálisis antes: en las [generales de julio de 2023](https://muestrear-no-es-pecado.es/2023/metanalisis) y para las [europeas de junio de 2024](https://muestrear-no-es-pecado.es/2024/06/meta-analisis-europeas). Ahora toca hacerlo para las elecciones del próximo **29 de noviembre de 2026**.

Las anteriores veces hice un meta-análisis para estimar el porcentaje de voto, con el efecto de la casa encuestadora como efecto aleatorio y el tiempo hasta el día de las elecciones como efecto fijo. En las [andaluzas de mayo de 2026](https://muestrear-no-es-pecado.es/2026/05/meta-analisis-andalucia) añadí una [estimación bayesiana de escaños](https://muestrear-no-es-pecado.es/2026/05/escanos-andalucia-2026). Teniendo solo el agregado de encuestas a nivel de Andalucía, tuve que hacer una asunción sobre cómo se distribuye el voto a nivel provincial. Quiero hacer lo mismo para las generales, pero aparte del problema de no tener el voto por provincias, no se informa siquiera por comunidad autónoma (recordemos que lo que usamos es el resultado que publican las casas encuestadoras, no los microdatos).

Y por si fuera poco, en la mayoría de las encuestas solo se informa de la estimación de los principales partidos, obviando los regionales (PNV, BNG, Junts, etc.), y tener una estimación de estos partidos es necesario para hacer una estimación provincial medianamente coherente.

Así que el plan tiene 4 pasos:

1. **Agregamos encuestas** a nivel nacional
2. **Rescatamos los regionales** en las encuestas en las que venga informado, o leyendo los enlaces a los medios donde se publican los resultados detallados
3. **Modelamos las series** con brms: dos modelos, uno para los principales partidos y otro para modelar los partidos regionales solamente
4. **Traducimos voto a escaños** provincia por provincia, con umbral del 3% y D'Hondt

Todo este ejercicio es incompleto. En el paso 4 añado una capa de error histórico agregado —cómo se equivocan las encuestas en bloque el día de la votación—, pero lo que de verdad haría falta es calibrar por empresa encuestadora: es posible que una casa se equivoque siempre a favor de los mismos, y eso el meta-análisis no lo puede ver. Esa calibración por casa queda pendiente.

## Los datos

La fuente principal es la tabla de encuestas de [ciento76](https://www.ciento76.es/encuestas), que descargo con un script y luego trato mínimamente para construir el dataset. Añadí una columna `regionales_completos`: es `FALSE` en la mayoría, salvo para algunas en las que pude obtener estimaciones regionales:

- **electomanía**: publican la tabla completa de cada encuesta; scrapeo el render de Datawrapper
- **prensa**: algunos artículos publican la tabla completa; la extraigo con LLM y reviso a mano
- **yo**: lo que no sale en ningún lado, se va a buscar leyendo artículos. Así conseguí los regionales de IMOP, SocioMétrica, Data10 y NC Report de primera semana de octubre

## La arquitectura en cascada

Si tuviera datos de estimaciones de partidos regionales en la mayoría de las encuestas podría plantearme un modelo multinomial con 13 categorías (los 5 grandes + SALF + 7 regionales + resto). Pero como no es así, propongo hacer dos modelos en cascada:

- **Modelo 1**: multinomial de 6 categorías — PP, PSOE, VOX, SUMAR (= Sumar + Podemos), SALF y **BLOQUE** (todo lo demás). Entran todas las encuestas que reportan los 6 principales.
- **Modelo 2 (la composición)**: multinomial condicional de 8 subceldas dentro del BLOQUE — Junts, ERC, PNV, EH Bildu, BNG, CC, UPN y RESTO. Solo entran las encuestas completas.

En la simulación, cada muestra del posterior del modelo 1 fija el tamaño del BLOQUE y se reparte con la composición del modelo 2. Resultado: 13 valores que suman 100 con la distribución conservada.

### El encaje de los dos modelos, con un ejemplo de juguete

**Qué entrena el modelo 2.** Es un multinomial *condicional*: el `trials` de su fórmula no es la muestra completa de la encuesta, sino los votos totales del BLOQUE en esa encuesta. Por ejemplo, una encuesta con n = 1.000 y un bloque del 11% entra al modelo 2 como 110 votos repartidos en 8 celdas:

```r
# una encuesta con n = 1.000 y BLOQUE = 11% entra al modelo 2 así:
encuesta <- c(
  ERC = 42, Junts = 18, PNV = 13, EH_Bildu = 15,
  BNG = 8, CC = 5, UPN = 2, RESTO = 7
)
sum(encuesta)   # 110 <- el trials del modelo 2: los votos del bloque, no la muestra
```

El modelo 2 ni sabe ni le importa cuánto pesa el bloque: solo aprende en qué proporción se reparte *dentro* de él (y cómo deriva con el tiempo). El peso del bloque lo estima el modelo 1, que ve muchas más encuestas.

**Cómo se pegan.** En la simulación, cada muestra del modelo 1 aporta el *tamaño* del BLOQUE y una muestra del modelo 2 aporta su *composición* interna. Se multiplican y punto:

```r
# una muestra (inventada) del modelo 1: los 6 grandes
d1 <- c(PP = 32.6, PSOE = 27.1, VOX = 19.2, SUMAR = 8.0, SALF = 1.8, BLOQUE = 11.3)
draw13 <- c(d1[1:5], d1["BLOQUE"] * c2)
#>       PP    PSOE     VOX   SUMAR    SALF     ERC   Junts     PNV EH_Bildu
#>     32.6    27.1    19.2     8.0     1.8     4.3     1.8     1.3     1.5
#>      BNG      CC     UPN   RESTO
#>      0.8     0.5     0.2     0.7        # suma: 100
```

Las muestras de los dos modelos se generan por separado y se emparejan por índice: se asume que el tamaño del bloque y su composición interna no están correlacionadas. Un modelo único de 13 categorías las acoplaría si en la realidad lo estuvieran, pero es justo lo que no podemos ajustar: solo pocas encuestas reportan las 13 categorías y casi todas de la misma casa (ElectoPanel). Hacerlo en dos modelos divide el problema en dos; puede que no sea lo más estadísticamente correcto, pero es práctico.

## La ventana temporal (y el peso de ElectoPanel)

Mi primer intento tenía 185 encuestas para agregar, y pensé que no tenía sentido añadir encuestas de hace 2 años. Pero el principal problema es que ElectoPanel tenía demasiado peso:

| Ventana | M1 (encuestas, casas) | M2 (encuestas, casas) | % ElectoPanel M1 / M2 |
|---|---|---|---|
| Todo (jun 25 →) | 185, 22 | 68, 7 | 38% / **84%** |
| Desde jul 2026 | 40, 16 | 14, 7 | 35% / 43% |

ElectoPanel es un panel continuo de electomanía con releases muy frecuentes: sus 71 encuestas no son 71 informaciones independientes. Y en el modelo 2 (el del bloque de otros) era el **84% del dataset**, porque casi es la única casa que reporta los regionales de forma sistemática.

Así que tomé una decisión que puede ser discutible: **ElectoPanel queda fuera del modelo 1**, donde sobran encuestas de otras casas y un panel con un tercio del peso solo metería inercia. **En el modelo 2 se queda**, porque sin él la composición del bloque se estimaría con un puñado de encuestas.

Del modelo 1 también sale **el CIS**: quizá lo meta más tarde, si hago una estimación propia partiendo de sus microdatos (su cocina electoral es muy distinta y habría que corregir el sesgo por recuerdo de voto, que el CIS decidió no hacer).

Además, en octubre y noviembre habrá una oleada de encuestas y seguramente captarán mejor los cambios que se puedan producir en el electorado. La idea es irlas incorporando y reajustar los modelos. **Iré actualizando el post.**

## El modelo 1

**Los datos.** Cada encuesta se convierte en conteos: el porcentaje de cada una de las 6 categorías por el tamaño muestral publicado, redondeado. Una encuesta de 1.000 entrevistas con el PP al 33% aporta 330 "votos" al PP. Es una decisión útil para modelar el multinomial como conteo y, de paso, hace que las encuestas con mayor *n* pesen más.

**El modelo.** Un multinomial con el PP como categoría de referencia. Para cada una de las otras cinco categorías $k$ (PSOE, VOX, SUMAR, SALF y BLOQUE), el log-odds frente al PP en la encuesta $i$, hecha por la casa $h$, es:

$$
\log \frac{\pi_{k,i}}{\pi_{PP,i}} = \alpha_k + \beta_k \, t_i + u_{k,h},
\qquad u_{k,h} \sim \mathcal{N}(0, \sigma_k)
$$

- $t_i$ son los días transcurridos desde el 1 de julio de 2026 hasta el fin del trabajo de campo de la encuesta $i$
- $\alpha_k$ es el nivel de partida: el log-odds de $k$ frente al PP al 1 de julio
- $\beta_k$ es la tendencia diaria de ese log-odds, común para todas las casas. Toda la dinámica de campaña que el modelo ve entra por aquí
- $u_{k,h}$ es el **efecto casa**: el sesgo sistemático de la empresa $h$ en el log-odds de $k$ frente al PP, constante a lo largo de la ventana. Al ser jerárquico, las casas con una sola encuesta se encogen hacia 0 y las que publican a menudo dejan ver su sesgo

Los priors son Normal(0, 2) —en desviación típica— para $\alpha_k$ y $\beta_k$ y Exponential(1) para $\sigma_k$, todo en escala log-odds. Para $\beta_k$, una pendiente por día, Normal(0, 2) es prácticamente plano: una dinámica de campaña creíble se mueve en centésimas de log-odds al día, así que el prior no frena nada y mandan los datos. Para $\alpha_k$ es débilmente informativo: $\alpha_k = 2$ significaría que la categoría $k$ vale $e^2 \approx 7$ veces el PP, de sobra para cualquier partido español.

**Asunciones.** Los efectos casa están centrados en 0, así que el modelo estima *la media de las encuestadoras*, no el resultado real. Si todas se equivocan en la misma dirección, no tiene forma de verlo. La corrección de verdad sería calibrar por casa con elecciones pasadas; lo que hago en el paso 4 es una versión tosca: un error agregado del conjunto de encuestas, estimado con solo cinco generales.

**El muestreo.** 4 cadenas de 4.000 iteraciones (1.000 de warmup) con cmdstanr, 12.000 muestras del posterior en total. Para que Stan trague: `init = 0`, `adapt_delta = 0.95` y `max_treedepth = 12`. En fórmula: `cell_counts | trials(n) ~ time + (1 | encuestadora)`.

Con 40 encuestas del modelo 1, el ajuste converge sin problemas (Rhat máximo 1.002, ESS mínimo 3.170, 0 transiciones divergentes). Los interceptos estimados (log-odds frente al PP al 1 de julio):

| parámetro | Estimate | Est.Error |
|---|---|---|
| muPSOE_Intercept | -0.199 | 0.039 |
| muVOX_Intercept | -0.615 | 0.046 |
| muSUMAR_Intercept | -1.194 | 0.065 |
| muSALF_Intercept | -2.797 | 0.135 |
| muBLOQUE_Intercept | -0.957 | 0.054 |

Las pendientes de `time` vienen en escala de cambio por día, por eso son valores muy pequeños. Evaluando el modelo hoy y en el día E (efectos casa en su media), el cambio mediano más grande entre hoy y el día E es de **0.6 puntos** y **ningún** intervalo al 95% se separa del 0: **el modelo ve una carrera plana desde julio**. PP, PSOE y demás están (según las encuestas) donde estaban en verano.

| partido | cambio | IC 95% |
|---|---|---|
| PP | +0.12 | [-0.79, 1.03] |
| PSOE | +0.04 | [-0.82, 0.96] |
| VOX | +0.62 | [-0.16, 1.49] |
| SUMAR | -0.34 | [-0.87, 0.31] |
| SALF | -0.12 | [-0.31, 0.16] |
| BLOQUE | -0.38 | [-0.93, 0.29] |

### Estimación para el día de las elecciones

Evaluamos el modelo en el `time` de la fecha electoral con los efectos casa en su media: lo que el modelo sabe decir es dónde estaría la media de las encuestadoras ese día, y la distancia hasta el resultado real la pone la capa de error histórico del paso 4.

| partido | mediana | IC 90% |
|---|---|---|
| PP | 32.6 | [31.1, 34.1] |
| PSOE | 26.6 | [25.1, 28.1] |
| VOX | 19.3 | [18.0, 20.8] |
| SUMAR | 8.7 | [7.7, 9.8] |
| SALF | 1.6 | [1.2, 2.0] |
| BLOQUE | 11.2 | [10.2, 12.3] |

![Meta-análisis generales 2026. Estimación día de las elecciones](https://muestrear-no-es-pecado.netlify.app/2026/10/meta-analisis-generales-jlcr_files/figure-html/unnamed-chunk-13-1.png)

### El modelo 2

El modelo 2 es un multinomial condicional de las 8 subceldas del BLOQUE, con `trials` igual a los votos totales del bloque de cada encuesta: solo aprende cómo se reparte *dentro* del bloque (`cell_counts | trials(bloque) ~ time`). Lleva la misma familia de priors que el modelo 1, pero con dos diferencias importantes:

- **No lleva efecto casa**: en su ventana (desde junio de 2025) solo hay un puñado de casas que reporten los 7 regionales, casi todas ElectoPanel. Un efecto aleatorio con 2-3 niveles y ese desbalance sería ruido, no información
- **La ventana es larga** (jun 2025 →) a diferencia del modelo 1 (jul 2026 →): con 14 encuestas en tres meses no hay de dónde colgar una tendencia, así que de momento le dejamos el histórico

La primera subcelda, **Junts, es la referencia** del multinomial: los coeficientes son log-odds de cada regional frente a Junts y las pendientes de `time` describen cómo cambia ese reparto interno con el tiempo.

## De voto a escaños

Vamos a la parte chunga, como el título de la canción de Tabletom.

En España los escaños se reparten **provincia a provincia** con D'Hondt, con un **umbral legal del 3%** de los votos válidos de cada circunscripción (LOREG art. 163), y Ceuta y Melilla eligen un diputado cada una por mayoría simple.

### El truco de la distribución provincial

Mismo truco que en [Andalucía](https://muestrear-no-es-pecado.es/2026/05/escanos-andalucia-2026), pero con referencia del 23J: si en 2023 ERC sacó en Barcelona 6.5 veces su media nacional, asumo que ese desvío se mantiene. Formalmente:

$$\tilde{\pi}_{k,p} = \pi_k \cdot r_{k,p}, \qquad r_{k,p} = \frac{\pi^{23J}_{k,p}}{\pi^{23J}_{k}}$$

y luego normalizo dentro de cada provincia. El que tenga una idea mejor, que la comente o me escriba.

> **El caso SALF**: "Se Acabó La Fiesta" se registró en 2024, así que en el 23J **no existía**. No hay manera de sacar su geografía de esa referencia. Su patrón provincial lo tomo de las **Europeas de junio de 2024**, su única cita con las urnas: 803.545 votos y 3 escaños, que en el Congreso se reparten muy mal para un partido así.

### El ruido Dirichlet, con Granada de ejemplo

Si aplicara los ratios del 23J tal cual, la geografía de 2026 sería un calco exacto de la de 2023 y toda la incertidumbre de los escaños vendría del voto nacional. No me lo creo: así que añadimos un poquito de ruido. En cada muestra de la simulación cada provincia usa unos ratios ligeramente distintos, muestreados alrededor de los del 23J. Veámoslo con Granada (7 diputados). Esto es lo que la referencia guarda para Granada:

| partido | votos | pct prov | pct nac | ratio |
|---|---|---|---|---|
| PP | 185.996 | 37.3 | 33.3 | 1.12 |
| PSOE | 165.862 | 33.2 | 31.9 | 1.04 |
| VOX | 81.080 | 16.3 | 12.5 | 1.30 |
| SUMAR | 58.567 | 11.7 | 12.4 | 0.94 |
| SALF | 20.674 | 6.02 | 4.62 | 1.30 |
| RESTO | 7.283 | 1.46 | 2.48 | 0.59 |

El PP sacó en Granada un 37.3% frente al 33.3% nacional: ratio 1.12. VOX, un 16.3 frente a un 12.5: ratio 1.30. Si una muestra del posterior da al PP un 32.6% nacional, Granada le asigna de partida 32.6 × 1.12 ≈ 36.5% (antes de normalizar la provincia para que sume 100).

**La idea: volver a votar el 23J en Granada, pero con una muestra de unos 500 "votantes".** El ruido solo toca el numerador del ratio: en cada muestra sustituyo los porcentajes de voto provinciales del 23J por unos "re-sorteados", los que saldrían si se repitiera aquella elección en Granada con ~500 personas en lugar de los ~520.000 votantes reales. Ese re-sorteo es una Dirichlet centrada en los porcentajes del 23J:

$$
\boldsymbol{\pi}^{(s)}_p \sim \text{Dirichlet}\left(\frac{V_p}{1000}\,\boldsymbol{\pi}^{23J}_p\right),
\qquad
r^{(s)}_{k,p} = r_{k,p} \cdot \frac{\pi^{(s)}_{k,p}}{\pi^{23J}_{k,p}}
$$

donde $V_p$ son los votos de la provincia. El "500" sale de ahí: una Dirichlet con concentración total $a$ se comporta, en dispersión, como los porcentajes de una muestra con $a$ encuestados, y Granada tuvo unos 520.000 votos: $520.000 / 1.000 \approx 520$. En esperanza la Dirichlet devuelve los porcentajes del 23J, así que en media el ratio no cambia: lo único que añade es dispersión. Cinco muestras para Granada:

| partido | ratio 23J | m1 | m2 | m3 | m4 | m5 |
|---|---|---|---|---|---|---|
| PP | 1.12 | 1.02 | 1.15 | 1.21 | 1.11 | 1.18 |
| PSOE | 1.04 | 1.12 | 1.06 | 1.04 | 1.01 | 1.11 |
| VOX | 1.30 | 1.00 | 0.98 | 1.22 | 1.39 | 1.30 |
| SUMAR | 0.94 | 1.20 | 1.14 | 0.84 | 1.05 | 0.66 |
| SALF | 1.30 | 1.35 | 1.26 | 1.16 | 1.09 | 1.23 |
| RESTO | 0.59 | 1.00 | 0.70 | 0.58 | 0.50 | 0.52 |

Cada columna es una distribución posible de los ratios de cada partido en Granada respecto de lo nacional. El PP oscila entre 1.02 y 1.21 alrededor de su 1.12; SUMAR, entre 0.66 y 1.20 alrededor de 0.94. Los partidos pequeños se mueven más en términos relativos, igual que en una encuesta pequeña el error relativo es mayor para los partidos con poco voto.

**¿Por qué la concentración depende de los votos de la provincia?** Básicamente, a mayor población que vota en una provincia, más nos fiemos de su deriva respecto de lo nacional. Es una hipótesis: si Granada fue más de derechas en 2023 que el conjunto de España, supongo que sigue siendo así.

El PP tiene el mismo ratio en Granada y en Soria, pero no la misma dispersión:

| provincia | diputados | votantes equivalentes | ratio 23J | media | sd |
|---|---|---|---|---|---|
| Granada | 7 | 519 | 1.12 | 1.12 | 0.07 |
| Soria | 2 | 52 | 1.12 | 1.12 | 0.20 |

Las dos provincias están centradas en 1.12, pero Granada (unos 520 "votantes equivalentes") tiene una desviación típica de 0.07 y Soria (unos 52) de 0.20, unas tres veces más, como corresponde a una muestra diez veces menor. Y es en las provincias pequeñas, donde se reparten 2 o 3 escaños, donde ese ruido decide si un partido entra o se queda fuera.

Ojo: el divisor de 1.000 es un mando de incertidumbre que fijé yo (lo heredo de Andalucía), no sale de ningún dato: con 500 la geografía bailaría más y con 5.000 casi nada.

### El error histórico de las encuestas

En este apartado es donde tengo bastantes dudas de lo que he hecho; creo que necesita ser más trabajado. La idea es que puede que las encuestas hayan subestimado o sobreestimado a algún partido de forma sistemática, y corregir en cierta forma ese sesgo.

Para cada elección pasada, comparo el resultado real con la media de las encuestas cuyo trabajo de campo terminó en los últimos días de campaña — **124 encuestas** de las cinco últimas generales, sacadas de los anexos de sondeos de Wikipedia. Quedan fuera los sondeos internos de partidos: no son estimaciones públicas. Para el PSOE sale esto:

| Elección | Encuestas | Real | Error |
|---|---|---|---|
| 20D-2015 | 21.1 | 22.0 | +0.94 |
| 26J-2016 | 21.2 | 22.6 | +1.37 |
| 28A-2019 | 28.8 | 28.7 | -0.09 |
| 10N-2019 | 27.1 | 28.0 | +0.89 |
| 23J-2023 | 28.2 | 31.7 | +3.49 |

Positivo en cuatro de las cinco. De los cuatro partidos principales, el único cuyo sesgo es consistente es **el PSOE (+1.3 puntos)**. Para PP, VOX y Sumar la media cambia de signo según qué elecciones entren, así que la pongo a 0. A cada muestra del posterior se le suma, pues, un error con media +1.3 para el PSOE y 0 para el resto, una desviación típica por partido (entre 1.3 y 2.6 puntos) y una correlación PP–VOX de +0.5. El BLOQUE de otros absorbe la diferencia para que los porcentajes sigan sumando 100, y SALF se queda fuera: no existía antes de 2024 y no tiene historia.

La capa, en números. Con cada muestra del posterior se sortea un vector de errores $\boldsymbol{\varepsilon}$ de una normal multivariante y se le suma a los porcentajes:

$$
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}),
\qquad
\pi_k \leftarrow \pi_k + \varepsilon_k,
\qquad
\pi_{BLOQUE} \leftarrow \pi_{BLOQUE} - \sum_k \varepsilon_k
$$

| partido | mu | sigma |
|---|---|---|
| PP | 0 | 2.57 |
| PSOE | 1.32 | 1.32 |
| VOX | 0 | 1.29 |
| SUMAR | 0 | 1.89 |
| SALF | 0 | 0 |

Cómo se lee: la **diagonal** es la varianza del error de cada partido. Fuera de la diagonal está la covarianza entre los errores de dos partidos. Para PP y VOX, $0.5 \times 2.6 \times 1.3 \approx 1.7$. Todas las demás covarianzas son cero, que es una forma educada de decir que no sé cómo se correlacionan: con cinco elecciones, solo la de PP–VOX es estable.

Lo de la correlación PP–VOX positiva: en las muestras en las que el error favorece al PP, tiende a favorecer también a VOX (y viceversa). Si la pusiera a cero, habría muestras con la derecha encuestada por arriba *y* por abajo a la vez, que es justo el tipo de ruido que no se observa en la realidad. La correlación empírica es +0.92, que con n = 3 elecciones encogí a +0.5.

Nota: esto es la contraria de lo que puse a mano en Andalucía (allí ρ(PP, VOX) = −0.5, pensando en el trasvase de voto útil). Pero el trasvase es **dinámica real de campaña** — el voto moviéndose de PP a VOX y viceversa — y eso el modelo ya lo captura con las pendientes temporales. Esta capa de incertidumbre va sobre otra cosa: cómo se equivocan las encuestas el día de la votación. Y ahí los datos dicen que cuando las encuestas se pasan con el bloque del PP, se pasan también con VOX.

Sobre el impacto: esta incertidumbre no solo ensancha los intervalos, **desplaza la estimación** — el PSOE gana algunos escaños frente a no aplicar la corrección, y la probabilidad de mayoría de PP+VOX baja. Es una corrección estimada con solo 5 elecciones: mejor que nada, peor que calibrar por casa, que sigue pendiente.

### El umbral del 3% y las trampas del agregado

Un par de detalles:

- El **umbral del 3% se aplica por provincia** antes de D'Hondt. Con provincias de 3-5 escaños, el umbral efectivo es mucho mayor que el legal: un partido con el 10% en Soria puede quedarse sin nada
- El **RESTO** (CUP, PACMA, etc.) como categoría agregada gana **escaños fantasma**: D'Hondt con el agregado le regala diputados que en la realidad ninguna de sus componentes conseguiría. Solución: sus votos van a no representados

### Validación: reproducir el 23J

Para ver que el mecanismo de asignación de escaños funciona bien, lo aplico a los datos del 23J: le doy a la simulación los porcentajes nacionales reales del 23J y sus ratios de referencia, sin ruido, y debe reproducir los escaños oficiales. Los reproduce exactos: PP 137, PSOE 121, VOX 33, SUMAR 31, Junts 7, ERC 7, PNV 5, EH Bildu 6, BNG 1, CC 1, UPN 1 — escaño por escaño.

## La simulación

Con cada muestra del posterior (voto nacional del modelo 1 + composición del bloque del modelo 2) se aplica el error histórico de las encuestas, se distribuye por provincias con su ruido Dirichlet, se aplica el umbral del 3% y D'Hondt provincia a provincia. 6.000 muestras en unos segundos. Mediana e intervalo del 80%:

| partido | escaños | IC 80% |
|---|---|---|
| PP | 131 | [118, 145] |
| PSOE | 111 | [100, 121] |
| VOX | 66 | [57, 75] |
| SUMAR | 17 | [7, 30] |
| ERC | 8 | [2, 12] |
| EH Bildu | 6 | [2, 9] |
| PNV | 4 | [1, 6] |
| Junts | 3 | [1, 7] |
| BNG | 2 | [0, 5] |
| CC | 1 | [0, 3] |
| UPN | 0 | [0, 1] |
| SALF | 0 | [0, 0] |
| RESTO | 0 | [0, 0] |

![Escaños estimados para el 29 de noviembre de 2026](https://muestrear-no-es-pecado.netlify.app/2026/10/meta-analisis-generales-jlcr_files/figure-html/unnamed-chunk-23-1.png)

La lectura: con las encuestas con trabajo de campo desde julio, **PP + VOX tendrían la mayoría con probabilidad 96.3%** — mediana de 198 escaños para la suma — y el PP solo se queda a 45 escaños de la mayoría (probabilidad de que gobierne el PP solo: 0.0%; de que sume la izquierda: 2.9%).

Es la capa de error histórico la que baja esa probabilidad del ~100% que daban los posteriors a secas: si las encuestas se vuelven a equivocar como en las últimas cinco elecciones, el margen ya no sobra. Nada de esto es una predicción: es una fotografía de lo que dicen las encuestas disponibles, con toda la incertidumbre que el modelo es capaz de cuantificar, y que iremos ampliando conforme lleguen encuestas entre ahora y el día E.

### Y en bloques

El bloque de investidura de 2023 (los grupos que apoyaron a Sánchez) contra la derecha, que es quizá el escenario más operativo: si las encuestas no se mueven, la repetición de la aritmética de la anterior legislatura no llega (derecha = PP + VOX + UPN; investidura = PSOE + SUMAR + ERC + Junts + EH Bildu + PNV + BNG + CC).

![Bloque de la derecha y bloque de investidura](https://muestrear-no-es-pecado.netlify.app/2026/10/meta-analisis-generales-jlcr_files/figure-html/unnamed-chunk-25-1.png)

## Coda

- El modelo 2 (composición del bloque) sigue con la ventana larga y el 84% de ElectoPanel del principio. Con la ola de octubre-noviembre a la vista, habrá que decidir qué hacer con él
- El supuesto del ratio provincial merecería una versión con las encuestas que publican estimaciones por CCAA (ciento76 tiene algunas)
- La ola de encuestas que viene: se re-descargan, se reajustan los modelos y se vuelve a simular; la baseline de este post queda guardada para medir cuánto se mueve todo

Todo el código —el scraping, los dos modelos con brms y la simulación de escaños— está en el [post original del blog](https://muestrear-no-es-pecado.es/2026/10/meta-analisis-generales-jlcr) y en el [repo de GitHub](https://github.com/joscani/blog_quarto).

Feliz sábado.
