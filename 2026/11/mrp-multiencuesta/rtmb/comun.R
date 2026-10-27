# Piezas comunes de los modelos RTMB (mrp_voto_rtmb.R, mrp_part_rtmb.R).
# Se carga con source() desde la raíz del repo.

library(RTMB)

dir_proy <- "2026/11/mrp-multiencuesta"
dir_datos <- file.path(dir_proy, "datos")

# Hilos: TMB con OpenMP (autopar) y postestratificación con mclapply
configura_hilos <- function() {
  hilos <- as.integer(Sys.getenv("RTMB_HILOS", "1"))
  if (hilos > 1) invisible(TMB::openmp(hilos, autopar = TRUE, DLL = "RTMB"))
  cat("hilos:", hilos, "\n")
  hilos
}

# Datos: los de 05_stan.R (priors "nuevo"), como en 05b_separado.R
carga_datos <- function(modo) {
  f_datos <- tempfile(fileext = ".rds")
  st <- system2(
    "Rscript",
    c(file.path(dir_proy, "05_stan.R"), modo, "nuevo"),
    env = paste0("SOLO_DATOS=", f_datos),
    stdout = FALSE
  )
  stopifnot(st == 0, file.exists(f_datos))
  x <- readRDS(f_datos)
  stopifnot(x$datos$sd_exponencial == 0)
  x
}

# Base ortonormal del subespacio de suma 0 (n x (n-1)): x = Q u con u ~ N(0, I)
# da la misma densidad que sum_to_zero_vector x ~ std_normal() en Stan
q_suma0 <- function(n) {
  M <- diag(n)[, -n, drop = FALSE]
  M[n, ] <- -1
  qr.Q(qr(M))
}

# Ajuste: moda de las sd (fijos) con los aleatorios integrados por Laplace
ajusta <- function(nll, par0, aleatorios) {
  t0 <- Sys.time()
  obj <- MakeADFun(nll, par0, random = aleatorios, silent = TRUE)
  opt <- nlminb(obj$par, obj$fn, obj$gr,
                control = list(eval.max = 1000, iter.max = 500, trace = 10))
  t_opt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat("optimización:", round(t_opt, 1), "s  convergencia:", opt$convergence,
      " (", opt$message, ")  gradiente máx:", signif(max(abs(obj$gr(opt$par))), 3), "\n")
  list(obj = obj, opt = opt, t_opt = t_opt)
}

# Muestras: con las sd fijas en su moda, los aleatorios siguen (Laplace) una
# normal con media en la moda condicional y precisión = Hessiano de nll en
# ellos. Es lo que daría sdreport(getJointPrecision = TRUE) restringido a los
# aleatorios, sin calcular el Hessiano de las sd (337 s en la prueba del voto).
# Devuelve una lista de parámetros por muestra.
muestras_laplace <- function(obj, opt, n_draws, seed = 2026) {
  invisible(obj$fn(opt$par))                   # deja last.par en la moda
  moda <- obj$env$last.par.best
  al <- seq_along(moda) %in% obj$env$random
  H <- obj$env$spHess(moda, random = TRUE)     # disperso, aleatorios x aleatorios
  L <- Matrix::Cholesky(H, LDL = FALSE)
  set.seed(seed)
  Z <- matrix(rnorm(sum(al) * n_draws), sum(al))
  # x = moda + P' L'^{-1} z, con P H P' = L L'
  dx <- Matrix::solve(L, Matrix::solve(L, Z, system = "Lt"), system = "Pt")
  draws <- matrix(moda, length(moda), n_draws)
  draws[al, ] <- draws[al, ] + as.matrix(dx)
  lapply(seq_len(n_draws), function(k) obj$env$parList(par = draws[, k]))
}

# Apila una lista (una entrada por muestra): matrices -> array draws x filas x
# cols; vectores -> matriz draws x n; escalares -> vector
apila <- function(l) {
  a <- simplify2array(lapply(l, function(x) if (is.matrix(x) && ncol(x) == 1) c(x) else x))
  if (is.null(dim(a))) a else if (length(dim(a)) == 2) t(a) else aperm(a, c(3, 1, 2))
}
