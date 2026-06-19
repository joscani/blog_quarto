# Convenciones de frontmatter en posts

## Categorías: el año debe ir entre comillas

### ¿Qué?

El tag del año en `categories` debe escribirse siempre entre comillas (`"2026"`, `"2025"`, etc.), nunca como entero sin comillas.

### ¿Por qué?

Sin comillas, YAML interpreta el valor como número entero. Las listing pages del blog (p.ej. `2026.qmd`) filtran por string — si el tag es un entero, el post no aparece en el listado aunque el año coincida.

Esto causó que `meta-analisis-andalucia` se viera en la URL de deploy de Netlify pero no en el blog de producción.

### Correcto

```yaml
categories:
  - "2026"
  - muestreo
  - encuestas electorales
```

### Incorrecto

```yaml
categories:
  - 2026
  - muestreo
  - encuestas electorales
```

### Ejemplos en el codebase

- `2026/03/sim_encuestas_claude_code.qmd:9`
- `2026/05/meta-analisis-andalucia.qmd:5`
- `2026/nochevieja-cachitos-2025/cachitos_2025_tercera_parte.qmd:7`

## Re-renderizar la listing page del año

Tras publicar un post nuevo, hay que re-renderizar `YYYY.qmd` (p.ej. `2026.qmd`) para que el post aparezca en el listado del blog. Sin este paso el post es accesible por URL directa pero no aparece en el índice.

```r
quarto::quarto_render("2026.qmd")
```

### Excepciones

Ninguna — el año siempre va entre comillas, independientemente del año.
