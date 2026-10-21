# ============================================================================
# FULL BAYESIAN 29-N · 15: DAG del modelo conjunto en SVG
#
# Dibuja el grafo de modelo_conjunto.stan con la notación de los modelos
# gráficos: círculo gris = dato observado, blanco = parámetro, doble =
# cantidad determinista; placas = índices. Las dependencias salen del Stan:
#
#   theta[c,1]  ~ N(voto anterior, eta1_sd)                (nota, no nodo)
#   sigma[c,g]  = sigma_g * exp(tau * z)                   (SGT -> SC)
#   theta[c,t]  = theta[c,t-1] + omega, sd sigma[c,g] e^(kappa * semana)
#                                                          (stub punteado)
#   beta[c,h]   ~ N(0, sigma_casa) con suma cero            (SCA -> B)
#   y_i         ~ DirMult(phi_c * celdas(softmax(theta[c,t_i] + beta[c,h_i])))
#                                                          (nota junto a y_i)
#   v[c,T]      = softmax(theta[c,T])
#   r_c (c<=5)  ~ MVN(logit v[c,T] + mu_gen, Sigma_gen)
#   Sigma_gen   = diag(sigma_aut e^gamma) Omega diag(sigma_aut e^gamma)
#   Sigma_aut   = diag(sigma_aut)         Omega diag(sigma_aut)
#     (dos matrices distintas, en la columna compartida; comparten Omega,
#      que es un nodo con flecha a cada una; gamma solo entra en Sigma_gen.
#      sigma_aut y Omega son independientes: no hay flecha entre ellas)
#   sigma_aut   = exp(log s0 + tau_s z_s)                  (nota sobre SA)
#   delta_e     ~ MVN(0, Sigma_aut)
#   z_j         ~ N(mu_aut + delta_e + b_h, sigma_enc)      (S -> Z)
#   b_h         ~ N(0, sigma_b)                             (SB -> BH)
#   eps_2026    ~ MVN(mu_gen, Sigma_gen);  v_29N = f(v[6,T], eps_2026)
#                                                          (sub-placa c = 6)
#
# Es un DAG simplificado, no un traslado literal del Stan: los efectos casa
# se centran dentro de cada campaña (suman cero), así que los beta[c,h] no
# son independientes entre sí y aquí se dibujan como si lo fueran.
#
# Van como notas (no como nodos) phi_c, theta[c,1], kappa y s0/tau_s;
# theta[c,t-1] es un stub punteado. Las escalas de casa (sigma^casa,
# sigma^b, sigma^enc) son nodos separados en su panel: son parámetros
# distintos del Stan.
#
# Salida: 2026/10/full-bayesian/dag_modelo.svg
# Uso (desde la raíz): Rscript 2026/10/full-bayesian/15_dag.R
# ============================================================================

suppressMessages(library(tidyverse))

# ---- etiquetas matemáticas en SVG -------------------------------------------
# m("θ", sub = "c,t") -> símbolo en cursiva con subíndice o superíndice. Los
# desplazamientos usan dy (baseline-shift no funciona en Firefox).
m <- function(base, sub = NULL, sup = NULL, cursiva = TRUE, tam_sup = 14) {
  b <- sprintf('<tspan font-style="%s">%s</tspan>', if (cursiva) "italic" else "normal", base)
  if (!is.null(sub)) b <- paste0(b, sprintf('<tspan dx="1.5" dy="6" font-size="14">%s</tspan><tspan dy="-6">&#8203;</tspan>', sub))
  if (!is.null(sup)) b <- paste0(b, sprintf('<tspan dx="1.5" dy="-9" font-size="%d">%s</tspan><tspan dy="9">&#8203;</tspan>', tam_sup, sup))
  b
}
tupla <- function(...) paste0("(", paste(c(...), collapse = ", "), ")")

# ---- nodos ------------------------------------------------------------------
# tipo: obs (gris), lat (blanco), det (doble). r: radio en px.
nodos <- tribble(
  ~id,     ~x,   ~y,  ~r, ~tipo, ~lab,
  # A. generales
  "SCA",   420,  80, 30, "lat", m("σ", sup = "casa"),
  "Y",     300, 180, 26, "obs", m("y", sub = "i"),
  "B",     420, 180, 28, "lat", m("β", sub = "c,h"),
  "TH",    300, 295, 30, "lat", m("θ", sub = "c,t"),
  "V",     500, 295, 28, "det", m("v", sub = "c,T"),
  "V29",   615, 295, 32, "det", m("v", sup = "29-N", tam_sup = 12),
  "R",     500, 400, 26, "obs", m("r", sub = "c"),
  "SC",    300, 405, 28, "lat", m("σ", sub = "c,g"),
  "SGT",   300, 540, 36, "lat", tupla(m("σ", sub = "g"), m("τ")),
  "MG",    570, 545, 28, "lat", m("μ", sup = "gen"),
  "EPS",   600, 385, 28, "lat", m("ε", sup = "2026"),
  # compartidos
  "SA",    750, 140, 28, "lat", m("σ", sup = "aut"),
  "OM",    890, 140, 26, "lat", m("Ω", cursiva = FALSE),
  "SGEN",  790, 480, 28, "det", m("Σ", sup = "gen", cursiva = FALSE),
  "SAUT",  890, 480, 28, "det", m("Σ", sup = "aut", cursiva = FALSE),
  # B. autonómicas
  "D",    1060, 230, 26, "lat", m("δ", sub = "e"),
  "Z",    1130, 385, 26, "obs", m("z", sub = "j"),
  "BH",   1258, 385, 26, "lat", m("b", sub = "h"),
  "MA",   1252, 235, 28, "lat", m("μ", sup = "aut"),
  "SB",   1258, 550, 26, "lat", m("σ", sup = "b"),
  "S",    1150, 560, 26, "lat", m("σ", sup = "enc")
)

# ---- flechas (de -> a) -------------------------------------------------------
# punteada = la cadena del paseo aleatorio, sin dibujar todas las semanas
flechas <- tribble(
  ~de,    ~a,      ~punteada,
  "SCA",  "B",     FALSE,
  "B",    "Y",     FALSE,
  "TH",   "Y",     FALSE,
  "SC",   "TH",    FALSE,
  "SGT",  "SC",    FALSE,
  "TH",   "V",     FALSE,
  "V",    "R",     FALSE,
  "MG",   "R",     FALSE,
  "SGEN", "R",     FALSE,
  "MG",   "EPS",   FALSE,
  "SGEN", "EPS",   FALSE,
  "V",    "V29",   FALSE,
  "EPS",  "V29",   FALSE,
  "SA",   "SGEN",  FALSE,
  "SA",   "SAUT",  FALSE,
  "OM",   "SGEN",  FALSE,
  "OM",   "SAUT",  FALSE,
  "SAUT", "D",     FALSE,
  "D",    "Z",     FALSE,
  "BH",   "Z",     FALSE,
  "MA",   "Z",     FALSE,
  "SB",   "BH",    FALSE,
  "S",    "Z",     FALSE
)

# ---- placas y paneles ---------------------------------------------------------
# pos: dónde va la etiqueta (esquina sin flechas)
placas <- tribble(
  ~x0,  ~y0, ~x1,  ~y1, ~lab,                          ~pos,
  30,   115, 550,  450, "campañas c = 1, …, 6",         "abajo-izq",
  240,  125, 368,  235, "encuestas i",                  "arriba-izq",
  380,  125, 480,  235, "casas h",                      "abajo-der",
  263,  228, 340,  348, "semanas t",                    "abajo-izq",
  450,  355, 548,  445, "c ≤ 5",                        "arriba-der",
  1000, 165, 1215, 460, "autonómicas e",                "arriba-der",
  1015, 320, 1200, 450, "encuestas j",                  "abajo-izq",
  1222, 320, 1292, 450, "casas h",                      "arriba-der"
)
paneles <- tribble(
  ~x0,  ~y0, ~x1,  ~y1, ~lab,                    ~fondo,
  10,   20,  660,  660, "A · Generales",          "#f3f6fa",
  670,  20,  980,  660, "Compartidos",            "#fbf6ee",
  990,  20,  1298, 660, "B · Autonómicas",        "#f4f7f1"
)

# ---- SVG --------------------------------------------------------------------
ancho <- 1308
alto <- 670
fuente <- "'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif"
fuente_mat <- "'STIX Two Text', 'Cambria Math', 'Times New Roman', serif"
trazo <- "#222222"

svg_panel <- with(paneles, sprintf(
  '<rect x="%d" y="%d" width="%d" height="%d" rx="10" fill="%s"/>\n<text x="%d" y="%d" font-family="%s" font-size="18" font-weight="600" fill="#555">%s</text>',
  x0, y0, x1 - x0, y1 - y0, fondo, x0 + 12, y0 + 24, fuente, lab
))

texto_placa <- function(x0, y0, x1, y1, lab, pos) {
  dcha <- str_detect(pos, "der")
  arriba <- str_detect(pos, "arriba")
  sprintf(
    '<text x="%d" y="%d" text-anchor="%s" font-family="%s" font-size="14" fill="#555">%s</text>',
    if (dcha) x1 - 6 else x0 + 6, if (arriba) y0 + 17 else y1 - 7,
    if (dcha) "end" else "start", fuente, lab
  )
}
svg_placas <- c(
  with(placas, sprintf('<rect x="%d" y="%d" width="%d" height="%d" rx="4" fill="none" stroke="#777" stroke-width="1.2"/>',
                       x0, y0, x1 - x0, y1 - y0)),
  pmap_chr(placas, texto_placa),
  # sub-placa de la predicción 2026: punteada, como el stub de las semanas
  sprintf('<rect x="%d" y="%d" width="%d" height="%d" rx="4" fill="none" stroke="#777" stroke-width="1.2" stroke-dasharray="5 4"/>',
          555, 240, 672 - 555, 420 - 240),
  nota_txt(666, 257, "c = 6", "end", 13)
)

doble <- 5  # separación del segundo círculo en los nodos deterministas
seg <- flechas |>
  left_join(nodos |> select(de = id, xa = x, ya = y, ra = r, ta = tipo), by = "de") |>
  left_join(nodos |> select(a = id, xb = x, yb = y, rb = r, tb = tipo), by = "a") |>
  mutate(
    l = sqrt((xb - xa)^2 + (yb - ya)^2),
    ra = ra + if_else(ta == "det", doble, 0) + 2,
    rb = rb + if_else(tb == "det", doble, 0) + 3,
    x0 = xa + (xb - xa) * ra / l, y0 = ya + (yb - ya) * ra / l,
    x1 = xb - (xb - xa) * rb / l, y1 = yb - (yb - ya) * rb / l
  )
svg_flechas <- with(seg, sprintf(
  '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.6"%s marker-end="url(#punta)"/>',
  x0, y0, x1, y1, trazo, if_else(punteada, ' stroke-dasharray="5 4"', "")
))

svg_nodos <- pmap_chr(nodos, function(id, x, y, r, tipo, lab) {
  relleno <- if (tipo == "obs") "#c9c9c9" else "#ffffff"
  c1 <- sprintf('<circle cx="%d" cy="%d" r="%d" fill="%s" stroke="%s" stroke-width="1.6"/>', x, y, r, relleno, trazo)
  c2 <- if (tipo == "det") {
    sprintf('<circle cx="%d" cy="%d" r="%d" fill="none" stroke="%s" stroke-width="1.6"/>', x, y, r + doble, trazo)
  } else ""
  tx <- sprintf(
    '<text x="%d" y="%d" text-anchor="middle" font-family="%s" font-size="%d" fill="#111">%s</text>',
    x, y + 6, fuente_mat, if (r > 32) 18 else 21, lab
  )
  paste(c(c2, c1, tx), collapse = "\n")
})

# notas: van en huecos vacíos, sin tocar nodos, flechas ni bordes de placa.
# El stub punteado trae theta[c,t-1] (sin círculo: la cadena de semanas va
# implícita) y la nota de debajo recuerda el kappa de campaña.
nota_txt <- function(x, y, txt, anchor = "start", size = 13) {
  sprintf('<text x="%d" y="%d" text-anchor="%s" font-family="%s" font-size="%d" fill="#555">%s</text>',
          x, y, anchor, fuente, size, txt)
}
stub_flecha <- sprintf(
  '<line x1="215" y1="295" x2="266" y2="295" stroke="%s" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#punta)"/>',
  trazo
)
nota <- c(
  stub_flecha,
  nota_txt(255, 281, "θ[c,t−1]", "end", 13),
  nota_txt(255, 316, "sd × e^(κ · semana de campaña)", "end", 12),
  nota_txt(30, 258, "θ[c,1] ~ N(voto anterior)", "start", 13),
  nota_txt(232, 218, "y ~ DirMult(φ_c · celdas(·))", "end", 12),
  nota_txt(360, 171, "h(i)", "middle"),
  nota_txt(307, 247, "t(i)"),
  # sigma_aut: de dónde sale
  nota_txt(755, 105, "σ^aut = exp(log s₀ + τ_s z)", "middle", 11),
  # las dos Sigma: su fórmula (Omega es nodo y comparte flecha con ambas)
  nota_txt(790, 528, "diag(σ^aut e^γ) Ω diag(σ^aut e^γ)", "middle", 12),
  nota_txt(885, 550, "diag(σ^aut) Ω diag(σ^aut)", "middle", 12)
)

# leyenda, abajo a la izquierda del panel A
ley_y <- 640
leyenda <- c(
  sprintf('<circle cx="34" cy="%d" r="9" fill="#c9c9c9" stroke="%s" stroke-width="1.4"/>', ley_y, trazo),
  nota_txt(48, ley_y + 5, "dato", "start", 14),
  sprintf('<circle cx="100" cy="%d" r="9" fill="#fff" stroke="%s" stroke-width="1.4"/>', ley_y, trazo),
  nota_txt(114, ley_y + 5, "parámetro", "start", 14),
  sprintf('<circle cx="212" cy="%d" r="9" fill="#fff" stroke="%s" stroke-width="1.4"/><circle cx="212" cy="%d" r="13" fill="none" stroke="%s" stroke-width="1.4"/>',
          ley_y, trazo, ley_y, trazo),
  nota_txt(230, ley_y + 5, "determinista", "start", 14),
  sprintf('<line x1="332" y1="%d" x2="358" y2="%d" stroke="%s" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#punta)"/>',
          ley_y, ley_y, trazo),
  nota_txt(364, ley_y + 5, "semanas omitidas", "start", 14)
)

svg <- c(
  sprintf('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" aria-label="DAG del modelo conjunto">',
          ancho, alto, ancho, alto),
  '<defs><marker id="punta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker></defs>',
  sprintf('<rect width="%d" height="%d" fill="#ffffff"/>', ancho, alto),
  svg_panel, svg_placas, svg_flechas, svg_nodos, nota, leyenda,
  "</svg>"
)
writeLines(svg, here::here("2026/10/full-bayesian/dag_modelo.svg"))
cat("dag_modelo.svg escrito\n")
