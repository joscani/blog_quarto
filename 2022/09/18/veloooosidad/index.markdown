---
title: Veeelooosidad
author: jlcr
date: '2022-09-18'
categories:
  - R
  - python
  - C++
  - Rust
tags:
  - R
  - python
slug: veloooosidad
---


No, este post no va sobre la canción de [Medina Azahara](https://www.youtube.com/watch?v=hEj1z_ihVX8) sino  de comparar un par de librerías para lectura y procesamiento de datos. A saber, [polars](https://pola-rs.github.io/polars-book/user-guide/index.html) escrita en Rust y con api en __python__ versus [vroom](https://vroom.r-lib.org/) en combinación con librerías como [data.table](https://github.com/Rdatatable/data.table) o [collapse](https://sebkrantz.github.io/collapse/) en R. Estas últimas usan por debajo C++, así que tanto por el lado de python como por el de R el principal mérito se debe a usar Rust y C++. 



## Datos, hardware y entornos

Para hacer la comparación vamos a usar un dataset de 100 millones de filas y 9 columnas, el mismo que se usa en [h2o.ai db-benchmark](https://github.com/h2oai/db-benchmark). 

Lo voy a probar en mi pc, que es un [slimbook](https://slimbook.es/) de justo antes de la pandemia, con 1gb de ssd, 32Gb de RAM y procesador Intel i7-9750H (12) @ 4.500GHz  con 6 núcleos (12 hilos) y corriendo Linux Mint 20.

### R

Para R voy a chequear vroom y data.table para leer los datos y data.table, tidytable y collapse para el procesamiento

R:  Uso R version 4.2.1 (2022-06-23) -- "Funny-Looking Kid"
vroom: 1.5.7
data.table: 1.14.2
tidytable: 0.8.1.9
collapse: 1.8.8

### Python
Uso un entorno de conda con python 3.6.12
polars: '0.12.5'



## Scripts 

### R 


En R voy a usar microbenchmark para realizar varias ejecuciones

Fichero: tests.R


```r
library(tidyverse)
library(data.table)
library(collapse)
library(vroom)
library(tidytable)

library(microbenchmark)

# Check lectu

setDTthreads(0L)

lectura <- microbenchmark(
    vroom  = vroom::vroom("/home/jose/Rstudio_projects/db-benchmark/data/G1_1e8_1e1_5_1.csv", show_col_types = FALSE), 
    data.table = data.table::fread("/home/jose/Rstudio_projects/db-benchmark/data/G1_1e8_1e1_5_1.csv"),
    times = 3L
)

print(lectura)

# group by sum

x <- vroom::vroom("/home/jose/Rstudio_projects/db-benchmark/data/G1_1e8_1e1_5_1.csv", show_col_types = FALSE)

# x= sample_frac(x, size = 0.1)
x_dt <- qDT(x)

group_by_performance <- microbenchmark(
    data.table = x_dt[, lapply(.SD, mean, na.rm = TRUE), keyby = id1, .SDcols = 7:9],
    # dplyr      = x %>%
    #     group_by(id1, id2) %>%
    #     summarise(v1 = sum(v1, na.rm = TRUE)) %>% 
    #     ungroup(),
    tidytable = x_dt %>%
        summarize.(v1 = sum(v1),
                   v2 = sum(v2),
                   v3 = sum(v3),
                   .by = c(id1, id2)),
    # base_R = tapply(x$v1, list(x$id1, x$id2), sum, na.rm = TRUE),

    collapse= x_dt %>%
        fgroup_by(id1, id2) %>%
        fsummarise(v1 = fsum(v1),
                   v2 = fsum(v2),
                   v3 = fsum(v3)),

    collapse_pure = {
        g <- GRP(x, ~ id1 +id2)
        fsum(x$v1, g)
        fsum(x$v2, g)
    },
    times = 5L
)

print(group_by_performance)
```


### Python

Fichero: tests.py


```python
import polars as pl
import time

start = time.time()
df = pl.read_csv("/home/jose/Rstudio_projects/db-benchmark/data/G1_1e8_1e1_5_1.csv")
end = time.time()

print(end -start)

start = time.time()

(
    df
.lazy()
    .groupby(['id1','id2'])
    .agg(
        [
            pl.col("v1").sum().alias('v1_sum'),
            pl.col("v2").sum().alias('v2_sum'),
            pl.col("v3").sum().alias('v3_sum')
        ]
    )
.collect()
)
end = time.time()
print(end - start)
```

## Resultados 

Para comparar, ejecuto los scripts desde consola y teniendo cerrado navegadores, ides y demás. 

### R 

```bash
Rscript tests.R
```

__Lectura en R__

```
Unit: seconds
       expr       min        lq      mean    median        uq       max neval
      vroom  7.783958  7.953598  8.185716  8.123239  8.386596  8.649953     3
 data.table 41.914928 42.809751 45.213309 43.704575 46.862499 50.020424     3

```

Group by y sum en R. 

```
Unit: seconds
          expr      min       lq     mean   median       uq       max neval cld
    data.table 1.469617 1.476545 1.550360 1.486647 1.633409  1.685581     5   a
     tidytable 1.182273 1.189111 1.291734 1.279313 1.314287  1.493686     5   a
      collapse 1.799175 1.813744 6.255215 1.891603 2.076616 23.694936     5   a
 collapse_pure 1.553002 1.555598 1.570758 1.566454 1.571605  1.607132     5   a

```

Por lo que más o menos, usar vroom para leer y tidytable, data.table o collapse para hacer el cálculo sale por unos 10 segundos o un poco menos. 



### Python

```bash
python tests.py 
```
```
7.755492448806763
1.8228027820587158

```
Y vemos que con polars tenemos más o menos los mismos tiempos. 


## Conclusiones

Tanto en R como en Python tenemos librerías muy rápidas que , si tenemos suficiente RAM podemos trabajar con conjunto de datos bastante tochos  y hacer cosas en tiempos más que razonables.  

Polars es una librería muy nueva y muy bien hecha, ojalá hagan api para R. No obstante, data.table lleva tiempo en R y su desempeño es consistente en múltiples situaciones. Mi consejo es echarle un ojo al [fastverse](https://github.com/fastverse/fastverse).
