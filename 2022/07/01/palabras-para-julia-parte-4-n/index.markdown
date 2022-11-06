---
title: Palabras para Julia (Parte 4 /n). Predicción con Turing
author: jlcr
date: '2022-07-01'
slug: palabras-para-julia-parte-4-n
categories:
  - bayesian
  - Julia
tags:
  - Julia
  - julia
  - análisis bayesiano
description: ''
topics: []
---


En [Palabras para Julia parte 3](https://muestrear-no-es-pecado.netlify.app/2022/03/20/palabras-para-julia-parte-3-n/) hablaba de modelos bayesianos con [Turing.jl](https://turing.ml/stable/), y me quedé con una espinita clavada, que era la de poder predecir de forma relativamente fácil con Turing, o incluso guardar de alguna forma la "posterior samples" y poder usar mi modelo en otra sesión de Julia. 

Empiezo una serie de entradas cuyo objetivo es ver si puedo llegar a la lógica para poner "en producción" un modelo bayesiando con Turing, pero llegando incluso a crear un binario en linux que me permita predecir con un modelo y desplegarlo incluso en entornos dónde no está instalado Julia. La verdad, que no sé si lo conseguiré, pero al menos aprendo algo por el camino. 

Si, ya sé que existen los dockers y todo eso, pero no está de más saber que existen alternativas que quizá sean mejores.  Ya en el pasado he tratado temas de cómo productivizar modelos  de h2o sobre spark [aquí](https://muestrear-no-es-pecado.netlify.app/2019/03/12/productivizando-modelos-binarios-con-h20/) o  con Julia [aquí](https://muestrear-no-es-pecado.netlify.app/2021/08/16/palabras-para-julia-parte-2-n/).  El objetivo final será llegar a tener un binario en linux que tome como argumento la ruta dónde se haya guardado las posterior samples de un modelo bayesiano y la ruta con especificación de dicho modelo en texto (para que Turing sepa como usar esas posterior samples) y que nos genere la posterior predictive para nuevos datos. 


Así que vamos al lío. Empezamos por ver como entrenamos un modelo bayesiano con Turing y como se puede guardar y utilizar posteriormente.

## Entrenamiento con Julia

Vamos a hacer un ejemplo sencillo, entrenando una regresión lineal múltiple de forma bayesiana. 
El dataset forma parte del material del libro Introduction to Statistical Learning. [Advertising](https://www.statlearning.com/s/Advertising.csv)


```julia

using LinearAlgebra, Plots
using Turing
using ReverseDiff, Memoization 
using DataFrames
using CSV
using Random
using StatsPlots
using Distributions
using StatsBase



import Logging
Logging.disable_logging(Logging.Warn)

mm = DataFrame(CSV.File("data/Advertising.csv"))
describe(mm)


```

```bash

200×5 DataFrame
 Row │ Column1  TV       radio    newspaper  sales   
     │ Int64    Float64  Float64  Float64    Float64 
─────┼───────────────────────────────────────────────
   1 │       1    230.1     37.8       69.2     22.1
   2 │       2     44.5     39.3       45.1     10.4
   3 │       3     17.2     45.9       69.3      9.3
  ⋮  │    ⋮        ⋮        ⋮         ⋮         ⋮
 198 │     198    177.0      9.3        6.4     12.8
 199 │     199    283.6     42.0       66.2     25.5
 200 │     200    232.1      8.6        8.7     13.4
                                     194 rows omitted

julia> describe(mm)
5×7 DataFrame
 Row │ variable   mean      min   median   max    nmissing  eltype   
     │ Symbol     Float64   Real  Float64  Real   Int64     DataType 
─────┼───────────────────────────────────────────────────────────────
   1 │ Column1    100.5      1     100.5   200           0  Int64
   2 │ TV         147.043    0.7   149.75  296.4         0  Float64
   3 │ radio       23.264    0.0    22.9    49.6         0  Float64
   4 │ newspaper   30.554    0.3    25.75  114.0         0  Float64
   5 │ sales       14.0225   1.6    12.9    27.0         0  Float64

```

Especificamos el modelo,  y aquí tengo que comentar un par de cosas. Una que julia gracias a que implementa eficazmente el  [Multiple dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), podemos tener una misma función que devuelva cosas diferentes dependiendo de que le pasemos, así una función puede tener diferentes métodos.  El otro aspecto es el uso del condition en Turing (alias `|`) se puede especificar el modelo sin pasar como argumento la variable dependiente y usarla solo para obtener la posterior, lo cual nos va a permitir hacer algo como `predict( modelo(Xs), cadena_mcmc)`, y no tener que pasar la `y` como un valor perdido. 


```julia

@model function mm_model_sin_sales(TV::Real, radio::Real, newspaper::Real)
    # Prior coeficientes
    a ~ Normal(0, 0.5)
    tv_coef  ~ Normal(0, 0.5)
    radio_coef  ~ Normal(0, 0.5)
    newspaper_coef  ~ Normal(0, 0.5)
    σ₁ ~ Gamma(1, 1)
    
    # antes 
    mu = a + tv_coef * TV + radio_coef * radio + newspaper_coef * newspaper 
    sales ~ Normal(mu, σ₁)
end

@model function mm_model_sin_sales(TV::AbstractVector{<:Real},
    
    radio::AbstractVector{<:Real},
    newspaper::AbstractVector{<:Real})
    # Prior coeficientes
    a ~ Normal(0, 0.5)
    tv_coef  ~ Normal(0, 0.5)
    radio_coef  ~ Normal(0, 0.5)
    newspaper_coef  ~ Normal(0, 0.5)
    σ₁ ~ Gamma(1, 1)
           
    mu = a .+ tv_coef .* TV .+ radio_coef .* radio .+ newspaper_coef .* newspaper 
    sales ~ MvNormal(mu, σ₁^2 * I)
end


``` 

Ahora tenemos el mismo modelo, que me a servir tanto para pasarle  como argumentos escalares como vectores, nótese que la función Normal tomo como argumento la desviación típica, mientrar que MvNormal toma una matriz de varianzas/covarianzas. Se aconseja el uso de MvNormal en Turing pues mejora el tiempo de cálculo de la posteriori.


Obtenemos la posteriori de los parámetros, pasándole como datos el dataset de Advertising. Es importante que la columna de la variable dependiente se pase como `NamedTuple`, esto se puede hacer en julia usando `(; vector_y)` . 


```julia

# utilizamos 4 cadenas con n_samples = 2000  para cada una

# usamos | para pasarle los datos de Y que no habiamos pasado en la especificacion del modelo

chain = sample(mm_model_sin_sales(mm.TV, mm.radio, mm.newspaper) | (; mm.sales),
    NUTS(0.65),MCMCThreads(),
    2_000, 4)
    
```

Y en unos 18 segundos tenemos nuestra MCMC Chain. 

```bash
Chains MCMC chain (2000×17×4 Array{Float64, 3}):

Iterations        = 1001:1:3000
Number of chains  = 4
Samples per chain = 2000
Wall duration     = 18.06 seconds
Compute duration  = 71.54 seconds
parameters        = a, tv_coef, radio_coef, newspaper_coef, σ₁
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
      parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
          Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

               a    2.0952    0.2712     0.0030    0.0038   5123.1176    0.9999       71.6139
         tv_coef    0.0481    0.0013     0.0000    0.0000   7529.0954    0.9998      105.2461
      radio_coef    0.1983    0.0087     0.0001    0.0001   5230.9995    1.0000       73.1220
  newspaper_coef    0.0040    0.0059     0.0001    0.0001   6203.9490    1.0002       86.7224
              σ₁    1.7205    0.0874     0.0010    0.0011   5441.9631    1.0000       76.0709

Quantiles
      parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
          Symbol   Float64   Float64   Float64   Float64   Float64 

               a    1.5635    1.9104    2.0982    2.2788    2.6182
         tv_coef    0.0455    0.0472    0.0481    0.0489    0.0508
      radio_coef    0.1814    0.1923    0.1983    0.2042    0.2155
  newspaper_coef   -0.0077    0.0001    0.0040    0.0078    0.0157
              σ₁    1.5585    1.6607    1.7169    1.7781    1.8997
```

Vale, estupendo,en `chain` tenemos las 8000 samples para cada uno  de los 5 parámetros , y también las de temas del ajuste interno por HMC, de ahí lo de (2000×17×4 Array{Float64, 3}).  
Pero ¿cómo podemos predecir para nuevos datos? 

Pues podemos pasarle simplemente  3 escalares correspondientes a las variables TV, radio y newspaper. 

Es necesario pasarle a la función predict  la llamada al modelo con los nuevos datos  `mm_model_sin_sales(tv_valor, radio_valor,newspaper_valor) `  y las posterioris (la cadena MCMC) de los parámetros.


```bash

julia> predict(mm_model_sin_sales(2, 5, 7), chain)
Chains MCMC chain (2000×1×4 Array{Float64, 3}):

Iterations        = 1:1:2000
Number of chains  = 4
Samples per chain = 2000
parameters        = sales
internals         = 

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64 

       sales    3.2203    1.7435     0.0195    0.0176   8053.9924    1.0000

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

       sales   -0.1863    2.0441    3.2553    4.4030    6.6547
```

También podemos pasarle más valores


```bash

julia> mm_last = last(mm, 3)
3×5 DataFrame
 Row │ Column1  TV       radio    newspaper  sales   
     │ Int64    Float64  Float64  Float64    Float64 
─────┼───────────────────────────────────────────────
   1 │     198    177.0      9.3        6.4     12.8
   2 │     199    283.6     42.0       66.2     25.5
   3 │     200    232.1      8.6        8.7     13.4


julia> predicciones = predict(mm_model_sin_sales(mm_last.TV, mm_last.radio, mm_last.newspaper), chain)
Chains MCMC chain (2000×3×4 Array{Float64, 3}):

Iterations        = 1:1:2000
Number of chains  = 4
Samples per chain = 2000
parameters        = sales[1], sales[2], sales[3]
internals         = 

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64 

    sales[1]   12.5192    1.7427     0.0195    0.0170   8270.6268    1.0000
    sales[2]   24.3266    1.7560     0.0196    0.0222   7720.4172    1.0001
    sales[3]   14.9901    1.7327     0.0194    0.0188   8039.4940    0.9999

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

    sales[1]    9.0888   11.3344   12.5241   13.6990   15.9571
    sales[2]   20.8369   23.1519   24.3414   25.4967   27.7429
    sales[3]   11.6549   13.8304   14.9617   16.1471   18.3733

``` 

Podría quedarme con las predicciones para sales[1] y calcular el intervalo de credibilidad el 80%

```bash
julia> quantile(reshape(Array(predicciones["sales[1]"]), 8000), [0.1, 0.5, 0.9])
3-element Vector{Float64}:
 10.28185973755853
 12.524091380928425
 14.74877121738519

```

## Guardar cadena y predecir

Ahora viene la parte que nos interesa a los que nos dedicamos a esto y queremos usar un modelo entrenado hace 6 meses sobre datos de hoy. Guardar lo que hicimos y predecir sin necesidad de reentrenar.



Guardamos la posteriori 

```julia


write( "cadena.jls", chain)
``` 

Y ahora, cerramos julia y abrimos de nuevo. 


```julia

using LinearAlgebra, Plots
using Turing
using ReverseDiff, Memoization 
using DataFrames
using CSV
using Random
using StatsPlots
using Distributions
using StatsBase

import Logging
Logging.disable_logging(Logging.Warn)

# posteriori guardada
chain = read("cadena.jls", Chains)

# Especificación del modelo (esto puede ir en otro fichero .jl)

# Si tengo en un fichero jl el código de @model, lo puedo incluir ahí. 


# ruta = "especificacion_modelo.jl"
# include(ruta)


@model function mm_model_sin_sales(TV::Real, radio::Real, newspaper::Real)
    # Prior coeficientes
    a ~ Normal(0, 0.5)
    tv_coef  ~ Normal(0, 0.5)
    radio_coef  ~ Normal(0, 0.5)
    newspaper_coef  ~ Normal(0, 0.5)
    σ₁ ~ Gamma(1, 1)
    
    # antes 
    mu = a + tv_coef * TV + radio_coef * radio + newspaper_coef * newspaper 
    sales ~ Normal(mu, σ₁)
end

@model function mm_model_sin_sales(TV::AbstractVector{<:Real},
     radio::AbstractVector{<:Real},
      newspaper::AbstractVector{<:Real})
    # Prior coeficientes
    a ~ Normal(0, 0.5)
    tv_coef  ~ Normal(0, 0.5)
    radio_coef  ~ Normal(0, 0.5)
    newspaper_coef  ~ Normal(0, 0.5)
    σ₁ ~ Gamma(1, 1)
           
    mu = a .+ tv_coef .* TV .+ radio_coef .* radio .+ newspaper_coef .* newspaper 
    sales ~ MvNormal(mu, σ₁^2 * I)
end





```

Y aqui viene la parte importante. En la que utilizamos el modelo guardado, que no es más que las  posterioris de los parámetros que hemos salvado en disco previamente.


```julia

## predecimos la misma observación , fila 198 del dataset

predict(mm_model_sin_sales(177, 9.3, 6.4 ), chain)

```


```bash
Chains MCMC chain (2000×1×4 Array{Float64, 3}):

Iterations        = 1:1:2000
Number of chains  = 4
Samples per chain = 2000
parameters        = sales
internals         = 

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64 

       sales   12.4723    1.7285     0.0193    0.0186   8326.7650    0.9998

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

       sales    9.0844   11.3106   12.4727   13.6334   15.7902
       
```

Y voilá. Sabiendo que se puede guardar la posteriori y  usarla luego , veo bastante factible poder llegar al objetivo de crear un "motor de predicción " de modelos bayesianos con Turing, que sea un ejecutable y que tome como argumentos la posteriori guardada de un modelo ajustado y en texto (con extensión jl ) y escriba el resultado en disco.  Y lo dicho, que pueda desplegar este ejecutable en cualquier sistema linux, sin tener que instalar docker ni nada, solo hacer un `unzip `






