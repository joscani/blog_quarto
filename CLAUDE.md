# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conventions

- `docs/quarto-frontmatter-conventions.md` — frontmatter rules (year categories must be quoted)

## Project Overview

This is a Quarto-based blog website ("Muestrear no es pecado" - "Sampling is not a sin") focused on statistics, data science, and big data topics. The blog features computational content with R (primary), Python, and Julia, including interactive browser-based applications via WebAssembly.

**Deployment**: Netlify at https://muestrear-no-es-pecado.netlify.app

## Common Commands

### Building and Development

```bash
# Render the entire site
quarto render

# Preview site with live reload
quarto preview

# Render specific post
quarto render 2025/01/my-post/index.qmd

# Publish to Netlify (configured in _publish.yml)
quarto publish netlify
```

### R Environment Management

The project uses `renv` for R package dependency management:

```r
# Restore R packages from lockfile
renv::restore()

# Update lockfile after installing new packages
renv::snapshot()

# Check package status
renv::status()
```

### Development Workflow

1. Start RStudio or R session (renv activates automatically via .Rprofile)
2. Create new post in appropriate year/month directory (e.g., `2026/02/my-post.qmd`)
3. Develop content with live preview: `quarto preview`
4. Render locally to test: `quarto render`
5. Deploy: `quarto publish netlify`

## Architecture

### Content Organization

**Chronological structure**: Posts organized by `YYYY/MM/post-name/` or `YYYY/MM/post-name.qmd`

```
blog_quarto/
├── 2019/ through 2026/    # Year-based directories
│   ├── 01/ through 12/    # Month subdirectories
│   │   └── *.qmd          # Blog posts
│   └── _metadata.yml      # Year-level metadata (author, freeze, comments)
├── posts/                 # Legacy posts (pre-chronological structure)
└── _site/                 # Generated output (gitignored)
```

**Content aggregation**: Multiple listing pages pull from different sources:
- `blog.qmd` - Main feed from all years (2019-2026) + legacy posts
- `archive.qmd` - Table view of all posts
- `cachitos.qmd` - Series filter (path: "*cachitos*")
- `julia.qmd` - Julia-specific posts (path: "*julia*")
- `2023.qmd`, `2024.qmd`, etc. - Year-specific listings with custom JSON metadata

### Post Metadata Structure

Posts inherit common metadata from year-level `_metadata.yml`:

```yaml
author: 'José Luis Cañadas Reche'
freeze: true              # Cache computational output
title-block-banner: false
page-layout: article
toc: true
toc-depth: 3
comments:
  giscus:
    repo: joscani/blogComments
```

Individual posts add specific frontmatter:

```yaml
---
title: "Post Title"
date: '2026-02-27'
categories:
  - Category1
  - Category2
description: 'Brief description'
execute:
  message: false
  warning: false
  echo: true
---
```

### Computational Freezing

**Critical**: `freeze: true` caches computational results in `_freeze/` directory to avoid re-running expensive computations on every build. The freeze directory structure mirrors the source structure.

When to clear freeze cache:
- Code chunks changed: Delete specific `_freeze/YYYY/MM/post-name/`
- Dependency updates: Delete entire `_freeze/` directory
- Never commit changes without testing if freeze affects output

### Quarto Extensions

Located in `_extensions/`, enabled via:

1. **shinylive** (`quarto-ext/shinylive`)
   - Serverless Shiny apps running in browser via WebAssembly
   - Supports R and Python
   - Filter enabled in `_quarto.yml`: `filters: [shinylive]`
   - Example: `2023/shinylive-R.qmd`

2. **webR** (`coatless/webr`)
   - Browser-based R execution
   - Interactive, editable code chunks
   - No server required
   - Example: `2023/webr.qmd`

3. **fontawesome** (`quarto-ext/fontawesome`)
   - Icon support: `{{< fa icon-name >}}`

### Site Configuration

**Themes**: Dual light/dark mode
- Light: Flatly (Bootswatch)
- Dark: Darkly (Bootswatch)
- Custom styles: `theme.scss` (fonts: Oleo Script, Prata, Source Sans Pro)

**Navigation**:
- Main navbar: Blog, About, GitHub, Bluesky, Twitter, Mastodon, RSS, Resources dropdown, Archive
- Resources section links to R Weekly, R Bloggers, Datanalytics

**Comments**: Giscus (GitHub Discussions-backed) configured per-year in `_metadata.yml`

## Special Content Series

**"Cachitos Nochevieja"**: Annual analysis series extracting/analyzing subtitles from Spanish TV show using:
- `ffmpeg` - Video/audio extraction
- `mplayer` - Subtitle extraction
- `tesseract` - OCR on subtitle images
- GNU parallel - Parallel processing
- R - Text analysis and visualization

Posts typically split into multiple parts (primera_parte, segunda_parte, tercera_parte).

## Working with Posts

### Creating New Posts

1. **Location**: Create in `YYYY/MM/` matching publication date
2. **File naming**: Either `post-slug.qmd` or `post-slug/index.qmd` (latter for posts with assets)
3. **Frontmatter**: Include title, date, categories, description
4. **Test locally**: Use `quarto preview` to see changes live

### Code Execution Options

Standard knitr options used across posts:

```yaml
knitr:
  opts_chunk:
    out.width: 80%
    fig.showtext: TRUE    # Enable custom fonts in figures
    collapse: true
    comment: "#>"
```

Figure settings:

```yaml
format:
  html:
    fig-height: 5
    fig-dpi: 300
    fig-width: 8.88
    fig-align: center
```

### Multilingual Content

Blog contains both Spanish and English posts. No automatic translation system - language determined per-post.

## R Environment

**renv version**: 1.1.4
**Bioconductor**: 3.21
**Snapshot type**: implicit (captures packages actually used)

The `.Rprofile` automatically activates renv when R starts in this directory.

When adding R dependencies:
1. Install packages normally: `install.packages("pkg")`
2. Update lockfile: `renv::snapshot()`
3. Commit `renv.lock` changes
4. Other developers use `renv::restore()` to sync

## Deployment

**Target**: Netlify (ID: 24378fa6-b968-4147-b957-8caab77fd700)

**Process**:
1. Build site locally: `quarto render`
2. Review `_site/` output
3. Deploy: `quarto publish netlify` (uses `_publish.yml` configuration)

No CI/CD automation - deployment is manual from local machine.

## Content Guidelines

**Mathematics**: Use LaTeX syntax in `$` or `$$` blocks (Quarto handles MathJax)

**Code display**: Common patterns:
```yaml
code-fold: true       # Collapsible code blocks
code-tools: true      # Show/hide all code toggle
code-link: true       # Hyperlink function calls to docs
```

**Categories**: Use consistent category names across posts for filtering (check existing posts for taxonomy)

**Images**: Store in post directory if using `post-name/index.qmd` structure; reference relatively

## Common Patterns

**Cross-post references**: Use Quarto cross-references or direct URLs to `site-url/YYYY/MM/post-slug/`

**Data files**: Can be stored at repository root (e.g., `dem_women.csv`) or in post directories

**R scripts**: Standalone scripts (e.g., `disqus2giscus.R`) at root for utilities not part of posts

**Interactive widgets**: Use R packages like `plotly`, `DT`, `leaflet` - they render automatically in Quarto HTML output
