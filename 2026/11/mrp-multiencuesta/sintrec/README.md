# Sensibilidad: el modelo sin recuerdo de voto (sintrec)

Copia autocontenida del pipeline de producción para estimar qué pasaría si el
modelo no usara el recuerdo de voto, ni en el ajuste ni en la
postestratificación. **No forma parte del pipeline de producción**: los
scripts de `../06_stan_datos.R`, `../stan/ajusta.R`, `../07_post_celdas.R` y
`../08_escanos.R` están limpios y no dependen de esta carpeta.

Qué cambia frente a la producción:

- `06_datos.R` (copia de `../06_stan_datos.R`): marginaliza la tabla de
  postestratificación en `rec` (los márgenes del censo no cambian) y colapsa
  el recuerdo a un único nivel en las celdas de encuesta. Escribe
  `sintrec/datos_stan.rds`.
- `mrp_voto_sintrec.stan`: igual que `../stan/mrp_voto.stan` pero sin los
  términos `a_rec`, `a_ccaa_rec` ni `a_casa_rec` (con su propia
  `voto_parcial_sintrec`, para no tocar `funciones.stan`). La verosimilitud es
  idéntica a la de `mrp_voto.stan` con `n_rec = 1`, así que el posterior de
  los parámetros comunes es el mismo; aquí no quedan parámetros previo-solo.
- `ajusta.R` (copia de `../stan/ajusta.R`): reajusta voto (con el modelo
  sintrec) y participación (misma `mrp_part.stan`, con el recuerdo colapsado:
  con `n_rec = 1` su efecto es constante y lo absorbe el intercepto). Escribe
  los fits y `estim_separado_nuts.rds` en esta carpeta.
- `07_post_celdas.R` (copia de `../07_post_celdas.R`): postestratifica sobre
  la tabla marginalizada, con los efectos de recuerdo a 0. Escribe
  `post_celdas_casa_media_part.rds` aquí.
- `08_escanos.R` (copia de `../08_escanos.R`): D'Hondt sobre el posterior
  sintrec. Escribe los CSV, el PNG y `estim_escanos_mrp.rds` aquí.
- `09_compara.R`: compara con la producción (`../datos/estim_separado_nuts.rds`
  y `../datos/estim_escanos_mrp.rds`).

Orden de ejecución, desde la raíz del repo:

```
Rscript 2026/11/mrp-multiencuesta/sintrec/06_datos.R
Rscript 2026/11/mrp-multiencuesta/sintrec/ajusta.R
Rscript 2026/11/mrp-multiencuesta/sintrec/07_post_celdas.R
Rscript 2026/11/mrp-multiencuesta/sintrec/08_escanos.R
Rscript 2026/11/mrp-multiencuesta/sintrec/09_compara.R
```

Nota: los resultados no dependen de si el ajuste se hace con este modelo
dedicado o con `mrp_voto.stan` y `n_rec = 1` (marginalización exacta); la
diferencia es solo de eficiencia de muestreo.
