# Notas para escribir el post "Generales 2026, todo bayesiano (I): el voto"

Una entrada por decisión del modelo: qué es, por qué, qué dijeron los datos y
una propuesta de cómo contarlo. Las propuestas son borradores para reescribir
con tus palabras. Al lector no se le cuenta la versión por módulos: las
comparaciones son con el pipeline de octubre.

Cifras: las del modelo principal a 18 oct 2026 (`conjunto_*.csv`). En el post,
siempre calculadas en línea desde los CSV.

---

## 1. Dirichlet-multinomial (ruido de cada encuesta)

- **Qué es:** cada encuesta saca sus votos de su propia urna, una copia algo
  movida de la real. φ dice cuánto se mueven las copias (φ grande = sorteo
  perfecto).
- **Por qué:** el modelo solo tiene dos formas de explicar que las encuestas
  no coincidan: el azar del muestreo o que el voto se mueva. Sin ruido extra,
  todo lo que pase del punto y medio del muestreo lo convierte en movimiento,
  y la volatilidad se infla.
- **Alternativas:** multinomial pura (infla la volatilidad); un efecto por
  encuesta (~3.500 parámetros que los datos no informan).
- **Cómo se estima φ:** comparando cuánto se separan las encuestas del voto
  latente con lo que daría el puro sorteo (conocido, porque sabemos el n). Un
  cambio real lo ven todas las casas a la vez; el ruido es de una encuesta
  suelta.
- **Datos:** φ entre 61 y 209 en 2015–abril 2019; decenas de miles desde el
  10N (sorteo perfecto).
- **Límite:** solo añade ruido, no lo quita. En 2026 las encuestas del PSOE y
  Vox se parecen más de lo que permite el azar (rebaño: 0,44 y 0,51).
- **Ejemplo de juguete (ya en el post):** diez encuestas de 1.000 la misma
  semana, PP al 32 %: sorteo perfecto 30–34,4 %; con φ = 200, 29,4–39,8 %.
- **Cómo contarlo:** "Las encuestas no son urnas perfectas. Cada una tiene su
  cocina: a quién consigue llamar esa semana, cómo pondera, qué hace con los
  que no recuerdan a quién votaron… Imagina tres semanas seguidas: 30 %, 35 %,
  31 %. Con sorteos perfectos, la única explicación es que el PP subió cinco
  puntos y bajó cuatro. El modelo aprende que el voto es muy nervioso y, al
  proyectar siete semanas sin encuestas, la horquilla se dispara. Lo bonito
  es que φ no lo pongo yo: lo estima el modelo, campaña a campaña."

## 2. Tres errores, tres piezas (efecto casa, ruido, error común)

- **Tabla de juguete (ya en el post):** casa A siempre +1 → efecto casa;
  las tres suben a la vez en la semana 3 → voto; la C da 35 una vez → ruido.
- **Qué va dónde:** universos, métodos y ponderaciones de cada casa → efecto
  casa; lo que cambia de encuesta a encuesta → ruido (φ); lo que les pasa a
  todas a la vez (voto oculto, rebaño) → error del consenso.
- **Lo que la tabla no ve:** que todas fallen a la vez. Por eso el voto
  latente es la media de las casas, no la verdad.
- **Cómo contarlo:** "Cada encuesta se equivoca por tres motivos. Por la casa
  que la hace: su método, su universo, su forma de ponderar; eso se repite en
  todas sus encuestas. Por la propia encuesta: quién contestó esa semana;
  eso va y viene. Y por lo que les pasa a todas a la vez, como el voto que
  nadie confiesa. Ese último no se ve hasta el día de la votación."

## 3. El consenso es la media de las casas, no de las encuestas

- Efecto casa con suma cero dentro de cada campaña: cada casa pesa igual,
  publique una encuesta o cinco.
- **Datos 2026:** media de las encuestas del PP 32,5 %; media de las casas
  32,2 %. Las que más publican le dan algo más (efecto débil, correlación 0,23).
- **Por qué:** una casa muy activa no impone su sesgo.

## 4. Error del consenso en logit, partido a partido

- **Alternativas descartadas:** en puntos (2 puntos no significan lo mismo
  para un partido del 32 % que para uno del 3 %); log-ratio frente al resto
  (el resto incluía a Cs en 2015–2019 y su error contaminaba a todos).
- **Elegida:** logit de cada partido frente a todos los demás: proporcional
  al tamaño, comparable entre elecciones.
- **Ejemplo:** 28A, PP encuestas 19,7 % → resultado 16,7 % (3 puntos, −0,20
  en logit); 23J, 34,2 % → 33,1 % (1 punto, −0,05). En puntos, el triple; en
  logit, cuatro veces: fallar tres puntos con un PP pequeño es más grave.
- **Precio:** al volver a porcentajes, el resto de partidos se reescala en
  proporción (supuesto; se medirá su efecto en la segunda parte).

## 5. Dónde entra la verdad: los resultados de las elecciones pasadas

- De cinco generales sabemos el resultado; entra como dato. Es el único
  sitio donde entra la verdad.
- En cada una: error del consenso = logit(resultado) − logit(voto latente
  del día E). De ahí aprende el sesgo (μ), el tamaño (σ) y la correlación (Ω).
- Para 2026: voto latente del 29-N + un error sorteado de lo aprendido.
- **Por qué contra el voto latente:** en octubre medía el error contra la
  media de las encuestas de la última semana y lo sumaba a la tendencia del
  modelo. En la media de las encuestas, la casa que más publica pesa más; en
  el voto latente, todas igual. Ejemplo inventado: casa A con cinco
  encuestas al 35 %, otras cuatro al 32 %, resultado 33 %: contra la media de
  las encuestas (33,7 %) "lo sobrestimaron"; contra la de las casas (32,6 %)
  "lo infraestimaron". Cambia hasta el signo.
- **Extra del modelo conjunto:** el resultado ayuda a situar el voto latente
  de campañas antiguas con pocas encuestas, y esa duda entra en el error.
- **Cómo contarlo:** "Las encuestas solas no pueden saber si se equivocan
  todas a la vez. Para eso hay que mirar al pasado. En cinco generales sabemos
  qué votó la gente, y el modelo compara ese resultado con lo que decía la
  media de las casas al final de cada campaña. Es como el hombre del tiempo
  que, antes de dar la previsión, mira cuánto se equivocó otros años."

## 6. Las autonómicas

- **Por qué:** con 5 generales (3 con Vox) hay que estimar 14 números del
  error. Las casas son las mismas en autonómicas: 39 elecciones, 217
  encuestas de la última semana (sin encuestas a pie de urna).
- **Qué se comparte:** la correlación Ω entera; el patrón de tamaños por
  partido (con nivel propio de las generales, γ). No se comparten el sesgo
  μ ni el ruido de cada encuesta.
- **Datos:** PP–Vox −0,50 (bien aprendida, contracción 0,87); γ ≈ 0,03: las
  generales fallan más o menos lo mismo que las autonómicas.
- **Probado y descartado:** correlación propia de las generales (no se
  aprendía); tamaño de error propio de las generales (mismo resultado).
- **Cuidado al contarlo:** no es que PP y Vox "compitan"; es que las
  encuestas se equivocan al repartir entre los dos. Hipótesis: recuerdo de
  voto. El modelo solo ve que pasa.
- **Límites:** Sumar no está en autonómicas (sus correlaciones las pone casi
  el prior); Vox < 3 % fuera.
- **Analogía:** el portero del que solo has visto cinco penaltis en la
  selección: miras también los de su club.
- **Cómo contarlo:** "Cinco generales son muy pocas para saber cómo se
  equivocan las encuestas. Pero las casas son las mismas en las autonómicas,
  y ahí hay 39 elecciones. No las mezclo a lo loco: de las autonómicas tomo
  cómo se reparten los errores entre partidos y qué partidos se fallan más;
  el nivel general del error y el sesgo, cada tipo de elección el suyo. Y
  aprendo una cosa curiosa: en autonómicas, cuando las encuestas se pasan con
  el PP, se quedan cortas con Vox."

## 7. Efecto casa frente a sesgo histórico de las casas

- **Dos preguntas distintas:** el efecto casa (β) responde "¿esta casa se
  aparta del grupo?", y no necesita resultados. El sesgo histórico (b)
  responde "¿esta casa acierta con lo que se vota?", y sí los necesita.
- **Datos (modelo final, PP, en puntos aprox.):** las dos cosas se parecen
  (correlación 0,6): GESOP −2,0 frente al grupo en 2026 y −1,9 frente al
  resultado en el pasado; NC Report +0,7 / +1,3; Data10 +0,8 / +1,1;
  Celeste-Tel +0,6 / +1,2. Las manías son reales y estables.
- **Por qué no se usa para predecir:** el efecto casa ya devuelve a cada casa
  al grupo; la historia solo añadiría si el grupo entero se pasa o se queda
  corto. La media histórica de las 14 casas de 2026 es +0,3 puntos con el
  PP. El error común de una elección es de unos 3 puntos (sd). Corregir por
  casas mueve diez veces menos que esa incertidumbre. En una prueba, la
  predicción del consenso no mejoró.
- **Cuándo importaría:** pocas casas publicando, o todas sesgadas hacia el
  mismo lado.
- **Posible mejora no hecha:** enlazar el efecto de cada casa entre
  campañas (ayudaría a las casas con pocas encuestas en 2026).
- **Script de la tabla:** scratchpad `casas_dos_preguntas.R` (pasarlo al post).
- **Cómo contarlo:** "¿No está el efecto casa para esto? Sí y no. El efecto
  casa detecta qué casas van por libre: GESOP da al PP dos puntos menos que
  el resto, DYM casi dos más. Y mirando al pasado, esas manías son de verdad.
  Pero el efecto casa solo devuelve a cada casa al grupo; no sabe si el grupo
  entero acierta. Eso solo lo dice el resultado. Si junto la historia de las
  catorce casas que publican ahora, en conjunto se pasan con el PP unas tres
  décimas. Tres décimas, cuando el error que cometen todas a la vez suele ser
  de tres puntos."

## 8. Sumar + Podemos y el reescalado del resto

- **Sumar + Podemos:** la historia solo conoce la familia entera (Podemos+IU,
  Unidas Podemos, Sumar con Podemos dentro). El error se aplica a la suma y
  se reparte con el peso de cada uno en esa simulación, antes del error.
  Supuesto: el error no distingue entre los dos.
- **El resto (SALF, regionales, otros):** sin error propio (en 2015–2019 el
  resto era sobre todo Cs). Tras aplicar el error a los cuatro grupos, todo
  se divide por el total: el resto absorbe el ajuste en proporción.
- **Ejemplo:** grandes 85 % + resto 15 %; el error sube a los grandes 2
  puntos → 87 %; el resto baja a 13 % y ERC pasa de 2 a 2 × 13/15 = 1,73.
- **Alternativas:** error propio del resto (sin datos); resto fijo (supone
  que las encuestas aciertan con los regionales). La proporcional es neutral.
- **Pendiente (segunda parte):** sensibilidad de los escaños regionales a
  "resto fijo".
- Las reglas del mapa (Cataluña, Canarias, C. Valenciana, AA) van en la
  simulación de escaños, segunda parte.
- **Cómo contarlo:** "Las encuestas del pasado solo me dicen cuánto fallan
  con la familia de Sumar entera, así que el error se aplica a Sumar y
  Podemos juntos y luego se reparte entre los dos. Y de los regionales no
  tengo historia: en 2015–2019 el 'resto' era sobre todo Ciudadanos. Así que
  no les pongo error propio: cuando los grandes suben o bajan, el resto se
  ajusta en proporción para que todo sume cien. Es un supuesto, y en la
  segunda parte miro si cambia algún escaño."

## 9. Los priors

- **Filosofía:** débilmente informativos, para descartar lo absurdo, no para
  dar la respuesta. Se eligen por órdenes de magnitud (0,1 en logit ≈ 2
  puntos con un partido del 30 %), no con los datos.
- **Tabla en lenguaje llano** (ya en el post, desplegable "Los priors").
- **Comprobaciones con el modelo final:** prior predictive (PP entre 9 % y
  56 %, Vox entre 3 % y 28 % solo con los priors) y contracción (lo que más
  pesa en la predicción lo deciden los datos: 0,85–0,94; lo de Sumar en
  autonómicas y algo de Vox lo pone sobre todo el prior).
- **NO hecho con el modelo final:** sensibilidad ×0,5 / ×2 (solo con una
  versión anterior: 94–96 %). El usuario decidió no repetirla: no afirmarla
  en el post para el modelo final.
- **Cómo contarlo:** "Los priors no los saco de los datos, sino de lo que sé
  de encuestas: que el voto no se mueve diez puntos en una semana, que una
  casa no se separa veinte puntos de las demás, que las encuestas suelen
  fallar un par de puntos y no diez. Son priors para descartar locuras, no
  para dar la respuesta. ¿Cómo sé que no deciden ellos? Porque con solo los
  priors el modelo deja al PP entre el 9 % y el 56 %: no sabe nada. Y porque
  lo que más pesa en la predicción lo aprende de los datos."

## Pendientes de explicar

- 8. Sumar + Podemos y el reescalado del resto.
- 9. Los priors.
- 10. La validación (backtest, contracción, sensibilidad).
- El paseo aleatorio y la volatilidad por campaña ya están en el post
  (sección 3); falta repasarlos contigo.

## Para la segunda parte: gráfico de escaños por bloques (19 oct 2026)

Quitado de la primera parte (allí solo estimación de voto). Usa `sim_cj` y `num_es` del setup.

````
```{r}
#| label: fig-bloques
#| fig-height: 3.5
#| fig-cap: "Escaños de cada bloque el 29-N (6.000 simulaciones del modelo conjunto). Derecha = PP + Vox. Investidura = los grupos que apoyaron a Sánchez en 2023: PSOE, Sumar, Podemos, Adelante Andalucía, ERC, Junts, EH Bildu, PNV, BNG y CC. Sin UPN ni SALF. Barra gruesa: intervalo del 50 %; fina: del 90 %. Etiqueta: mediana. Línea roja: mayoría absoluta (176)."
bloques_draws <- sim_cj$draws |>
  transmute(
    Derecha = PP + VOX,
    Investidura = PSOE + SUMAR + PODEMOS + AA + ERC + Junts + `EH Bildu` +
      PNV + BNG + CC
  ) |>
  pivot_longer(everything(), names_to = "bloque", values_to = "escanos") |>
  mutate(bloque = fct_relevel(bloque, "Investidura", "Derecha"))
ggplot(bloques_draws, aes(x = escanos, y = bloque, fill = bloque)) +
  # escaños: enteros, histograma con bins de 2 alineados con los enteros
  ggdist::stat_halfeye(
    density = "histogram",
    breaks = ggdist::breaks_fixed(width = 2),
    align = ggdist::align_center(at = 0),
    normalize = "groups",
    .width = c(0.5, 0.9),
    alpha = 0.8
  ) +
  geom_label(
    data = bloques_draws |>
      summarise(escanos = median(escanos), .by = bloque),
    aes(label = escanos),
    size = 3,
    fill = "white",
    nudge_y = -0.25
  ) +
  geom_vline(xintercept = 176, linetype = "dashed", colour = "red") +
  scale_fill_manual(
    values = c(Derecha = "#005999", Investidura = "#A00B85"),
    guide = "none"
  ) +
  labs(x = "Escaños", y = NULL)
```
````
