#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

zipfile = args[1]
output_path = paste0(args[2],'.html')

fichero_qmd <- 'metricas.qmd'


# Generar informe automático

tmp_dir <- tempdir()
tmp <- tempfile()

unzip(zipfile = zipfile, exdir = tmp_dir )

fichero_json = paste0(tmp_dir, "/experimental/modelDetails.json")


quarto::quarto_render(
  input = fichero_qmd, 
  execute_params = list(fichero_json = fichero_json), 
  output_file = output_path
    
)