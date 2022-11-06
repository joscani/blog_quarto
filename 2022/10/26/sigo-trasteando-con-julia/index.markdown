---
title: Sigo trasteando con julia
author: jlcr
date: '2022-10-26'
categories:
  - julia
  - produccion
tags:
  - julia
  - linux
  - produccion
slug: sigo-trasteando-con-julia
---


Siguiendo con lo que contaba [aquí](https://muestrear-no-es-pecado.netlify.app/2021/08/16/palabras-para-julia-parte-2-n/) me he construido un binario para predecir usando un modelo de xgboost con Julia. La ventaja es que tengo un tar.gz que puedo descomprimir en cualquier linux (por ejemplo un entorno de producción sin acceso a internet y que no tenga ni vaya a tener julia instalado, ni docker ni nada de nada), descomprimir y poder hacer un `miapp_para_predecir mi_modelo_entrenado.jls csv_to_predict.csv resultado.csv` y que funcione y vaya como un tiro. 


Pongo aquí los ficheros relevantes. 

Por ejemplo mi fichero para entrenar un modelo y salvarlo . 

Fichero `train_ boston.jl`


```julia 
# Training model julia
using  CSV,CategoricalArrays, DataFrames, MLJ, MLJXGBoostInterface


df1 = CSV.read("data/boston.csv", DataFrame)

df1[:, :target] .= ifelse.(df1[!, :medv_20].== "NG20", 1, 0)
const target = CategoricalArray(df1[:, :target])

const X = df1[:, Not([:medv_20, :target])]

Tree = @load XGBoostClassifier pkg=XGBoost
tree_model = Tree(objective="binary:logistic", max_depth = 6, num_round = 800)
mach = machine(tree_model, X, target)

Threads.nthreads()
evaluate(tree_model, X, target, resampling=CV(shuffle=true),measure=log_loss, verbosity=0)
evaluate(tree_model, X, target,
                resampling=CV(shuffle=true), measure=bac, operation=predict_mode, verbosity=0)



train, test = partition(eachindex(target), 0.7, shuffle=true)

fit!(mach, rows=train)

yhat = predict(mach, X[test,:])

evaluate(tree_model, X[test,:], target[test], measure=auc, operation=predict_mode, verbosity=0)

niveles = levels.(yhat)[1]
niveles[1]

log_loss(yhat, target[test]) |> mean

res = pdf(yhat, niveles)
res_df = DataFrame(res,:auto)

MLJ.save("models/boston_xg.jls", mach)

```


Y luego los ficheros que uso para construirme la app binaria .. Recordemos del [post que mencionaba](https://muestrear-no-es-pecado.netlify.app/2021/08/16/palabras-para-julia-parte-2-n/) que lo que necesito es el código del programa principal (el main) y un fichero de precompilación que sirve para que al crear la app se compilen las funciones que voy a usar. 


fichero `precomp.jl`, 

```julia
using CSV, DataFrames, MLJ, MLJBase, MLJXGBoostInterface

# uso rutas absolutas
df1 = CSV.read("data/iris.csv", DataFrame)
X = df1[:, Not(:Species)]

predict_only_mach = machine("models/mimodelo_xg_binario.jls")

ŷ = predict(predict_only_mach, X) 


predict_mode(predict_only_mach, X)

niveles = levels.(ŷ)[1]

res = pdf(ŷ, niveles) # con pdf nos da la probabilidad de cada nivel
res_df = DataFrame(res,:auto)
rename!(res_df, ["target_0", "target_1"])

CSV.write("data/predicciones.csv", res_df)

```


fichero `xgboost_predict_binomial.jl` , aquí es dónde está el main 

```julia
module xgboost_predict_binomial

using CSV, DataFrames, MLJ, MLJBase, MLJXGBoostInterface

function julia_main()::Cint
    try
        real_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

# ARGS son los argumentos pasados por consola 

function real_main()
    if length(ARGS) == 0
        error("pass arguments")
    end

# Read model
    modelo = machine(ARGS[1])
# read data. El fichero qeu pasemos tiene que tener solo las X.(con su nombre)
    X = CSV.read(ARGS[2], DataFrame, ntasks= Sys.CPU_THREADS)
# Predict    
    ŷ = predict(modelo, X)            # predict
    niveles = levels.(ŷ)[1]           # get levels of target
    res = pdf(ŷ, niveles)             # predict probabilities for each level
    
    res_df = DataFrame(res,:auto)     # convert to DataFrame
    rename!(res_df, ["target_0", "target_1"])          # Column rename
    CSV.write(ARGS[3], res_df)        # Write in csv
end


end # module

```

y si todo está correcto y siguiendo las instrucciones del post anterior, se compilaría haciendo por ejemplo esto

```julia
using PackageCompiler
create_app("../xgboost_predict_binomial", "../xg_binomial_inference",
 precompile_execution_file="../xgboost_predict_binomial/src/precomp_file.jl", force=true, filter_stdlibs = true, cpu_target = "x86_64")
```


Y esto me crea una estructura de directorios dónde está mi app y todo lo necesario para ejecutar julia en cualqueir linux. 

```bash

╰─ $ ▶ tree -L 2 xg_binomial_inference
xg_binomial_inference
├── bin
│   ├── julia
│   └── xgboost_predict_binomial
├── lib
│   ├── julia
│   ├── libjulia.so -> libjulia.so.1.8
│   ├── libjulia.so.1 -> libjulia.so.1.8
│   └── libjulia.so.1.8
└── share
    └── julia
```

y poner por ejemplo en el `.bashrc` el siguiente alias. 

```bash
alias motor_xgboost=/home/jose/Julia_projects/xgboost_model/xg_binomial_inference/bin/xgboost_predict_binomial
```

y ya está listo. 

Ahora tengo un dataset a predecir de 5 millones de filas 

```bash

╰─ $ ▶ wc -l data/test.csv 
5060001 data/test.csv

```

```bash

 head -n4 data/test.csv 
crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,lstat
0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,4.98
0.02731,0,7.07,0,0.469,6.421,78.9,4.9671,2,242,17.8,9.14
0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,4.03

```


y bueno, tardo unos 11 segundos en obtener las predicciones y escribir el resultado
 
```bash
╰─ $ ▶ time motor_xgboost models/boston_xg.jls data/test.csv pred.csv

real	0m11,091s
user	0m53,293s
sys	0m2,321s


```

y comprobamos que lo ha hecho bien

```bash

╰─ $ ▶ wc -l  pred.csv 
5060001 pred.csv


╰─ $ ▶ head -n 5 pred.csv 
target_0,target_1
0.9999237,7.63197e-5
0.99120975,0.008790266
0.99989164,0.00010834133
0.99970543,0.00029458306

```


Y nada, pues esto puede servir para subir modelos a producción en entornos poco amigables (sin python3, sin R, sin julia, sin spark, sin docker, sin internet). Es un poco `old style` que me diría mi arquenazi favorito Rubén, pero 


Os dejo el tar.gz para que probéis, también os dejo el `Project.toml`y el `Manifest.toml` y el fichero con el que he entrenado los datos.  para que uséis el mismo entorno de julia que he usado yo. 


[enlace_drive](https://drive.google.com/drive/folders/1jQW-QQNoABlMdUHhlwHvY9MQnZh1x_Yi?usp=sharing)



