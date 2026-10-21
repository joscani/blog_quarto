# Archivo de la arquitectura de dos módulos

La versión original del análisis full bayesian (octubre 2026) iba en dos
modelos Stan independientes que se combinaban draw a draw:

- **Módulo A** (`modelo_estado.stan`, `02_ajustar_estado.R`): estado latente
  con paseo aleatorio para las encuestas de 2026.
- **Módulo B** (`modelo_error.stan`, `03_ajustar_error.R`): error histórico
  jerárquico de las encuestas (generales 2015-2023).

`04_simular.R` los combinaba y `05/07/08/09/10` comparaban, diagnosticaban y
validaban. Todo eso lo sustituye el modelo conjunto (`modelo_conjunto.stan`,
`11`-`17`), que estima las dos cosas a la vez y mide el error contra el voto
latente, no contra la media de encuestas.

**Nota:** estos scripts sourceaban `01_datos.R` tal como estaba antes de la
limpieza (con `datos_estado()`, `stan_estado()`, `stan_error()`,
`priors_estado()`, `priors_error()`, `lr_23j()` y `semana()`). Esas piezas se
quitaron de `01_datos.R`; para re-ejecutar algo de aquí haría falta
recuperarlas de la versión previa a la limpieza.
