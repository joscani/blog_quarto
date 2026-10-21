# Preferencias del proyecto

- En todas las llamadas a ficheros (read_csv, write_csv, read_rds, saveRDS, etc.) usar rutas construidas con `here::here()`, nunca rutas relativas crudas (`./...`, `../...`) ni absolutas.
- Si cambian datos que un post usa (CSV, RDS u otros ficheros de salida), actualizar la release de GitHub de ese post con `gh release upload <tag> <ficheros> --clobber` y comprobar que los enlaces del post siguen apuntando a ella. Los scripts se enlazan desde `blob/master`, así que deben estar commiteados y en el remoto.
