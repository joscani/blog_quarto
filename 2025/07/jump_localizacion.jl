using JuMP
using DataFrames
using HiGHS
using LinearAlgebra
using CSV
using Plots
#using DataStructures
#using BenchmarkTools



ANDALUCIA = ["ALMERIA", "CADIZ",
 "CORDOBA","GRANADA",
"HUELVA", "JAEN",
"MALAGA", 
"SEVILLA"]

cod_postales_raw = DataFrame(CSV.File("cod_postales.csv"))
sedes_raw = DataFrame(CSV.File("sedes.csv"))



#cod_postales = subset(cod_postales_raw, :provincia => ByRow(x -> x == "MADRID" || x == "GRANADA"))
#cod_postales = cod_postales_raw[[i ∈ ANDALUCIA for i in cod_postales_raw.provincia ], :]

#cod_postales = subset(cod_postales_raw, :provincia => ByRow(x -> x ∈ ANDALUCIA))
cod_postales =  cod_postales_raw


# sedes = sedes_raw[[i ∈ ANDALUCIA for i in sedes_raw.provincia ], :]
#sedes = subset(sedes_raw, :provincia => ByRow(x -> x ∈ ANDALUCIA))
sedes = sedes_raw


sort!(sedes, :tipo, rev = true)
#cod_postales = copy(cod_postales_raw)
#sedes = copy(sedes_raw)
#sort!(sedes, :tipo, rev = true)

"""
    haversine(lat1, long1, lat2, long2, r = 6372.8)

Compute the haversine distance between two points on a sphere of radius `r`,
where the points are given by the latitude/longitude pairs `lat1/long1` and
`lat2/long2` (in degrees).
"""
function haversine(lat1, long1, lat2, long2, r = 6372.8)
    lat1, long1 = deg2rad(lat1), deg2rad(long1)
    lat2, long2 = deg2rad(lat2), deg2rad(long2)
    hav(a, b) = sin((b - a) / 2)^2
    inner_term = hav(lat1, lat2) + cos(lat1) * cos(lat2) * hav(long1, long2)
    d = 2 * r * asin(sqrt(inner_term))
    # Round distance to nearest kilometer.
    return round( d, digits = 2)
end

# number of clients
n = size(cod_postales,1)
# number of  comerciales (facilities)
m = size(sedes, 1)

# esto es como en python, una list comprehension

c  = [haversine( 
            cod_postales.centroide_latitud[i], cod_postales.centroide_longitud[i],
            sedes.centroide_latitud[j], sedes.centroide_longitud[j]) 
            for i in 1:n, j in 1:m]



# utilizamos operador ternario 
# Si eres jefe te pongo distancia 1.1, si no 0.9
for j in 1:m
    c[:,j] = sedes.tipo[j] == "jefe" ? 1.1 * c[:, j] : 0.9 * c[:,j]
end

n_jefes = sum(sedes.tipo .=="jefe")
n_indios = m - n_jefes
       
# Fixed costs
f = ones(m);


ufl = Model(HiGHS.Optimizer)

@variable(ufl, x[1:n, 1:m], Bin);
@variable(ufl, y[1:m], Bin);

# esta restriccion es poner que  a  cada  cod postal solo puede ir un comecial, (sea indio o jefe)

@constraint(ufl, client_service[i in 1:n], sum(x[i, j] for j in 1:m) == 1);

# siguiente restricción es poner que un cod postales solo puede ir a una que esté abierta
# esto es, si y[j] = 0 significa sede cerrada y no se puede asignar nadie ahí

@constraint(ufl, open_facility[i in 1:n, j in 1:m], x[i, j] <= y[j]);
 
# restricciones para los indios, jefes  
@constraint(ufl, tot_por_indio[j in (n_jefes+1):m], 3 <= sum(x[i, j] for i in 1:n) <= 100);
@constraint(ufl, tot_por_jefe[j ∈ 1:n_jefes], 1<=sum(x[i, j] for i in 1:n) <= 10);
#@constraint(ufl, comparacion[j=1:n_jefes, k=(n_jefes+1):m],  sum(x[i, j] for i in 1:n) <= sum(x[i, k] for i in 1:n) );

@objective(ufl, Min, f'y + sum(c .* x));

optimize!(ufl)

println("Optimal value: ", objective_value(ufl))

x_ = value.(x) .> 1 - 1e-5
y_ = value.(y) .> 1 - 1e-5

#value.(comparacion)
value.(tot_por_indio)
value.(tot_por_jefe)

p = Plots.scatter(
    cod_postales.centroide_longitud,
    cod_postales.centroide_latitud;
    markershape = :circle,
    markercolor = :blue,
    label = nothing,
)


mc = [(y_[j] ? :red : :white) for j in 1:m]
Plots.scatter!(
    sedes.centroide_longitud,
    sedes.centroide_latitud;
    markershape = :square,
    markercolor = mc,
    markersize = 4,
    markerstrokecolor = :red,
    markerstrokewidth = 2,
    label = nothing,
)


for i in 1:n
    for j in 1:m
        if x_[i, j] == 1
            Plots.plot!(
                [cod_postales.centroide_longitud[i], sedes.centroide_longitud[j]],
                [cod_postales.centroide_latitud[i], sedes.centroide_latitud[j]];
                color = :black,
                label = nothing,
            )
        end
    end
end

p

# 
x_collect= collect(x_)
sum(collect(x_), dims=1)
sum(x_, dims = 1)
sum(x_, dims = 2)

[x for x in 1:10 if x % 2 == 0]
# comprehension lista 
asignaciones  = Dict()
#= for j in 1:m
    asignaciones("clave" => j, "valor" =>
        [index for (index, fila) 
        in enumerate(x_collect[:,j]) 
        if fila .== 1])
end  =#

asignaciones  = Dict()
for j in 1:m
    asignaciones[j] = 
    [index for (index, fila) in enumerate(x_collect[:,j]) if fila .== 1]
end 

for j in 1:m
    asignaciones[j] = 
    cod_postales[ [index for (index, fila) in enumerate(x_collect[:,j]) if fila .== 1], :cod_postal]
end 

for j in 1:m
    asignaciones[j] = 
    cod_postales[ [index for (index, fila) in enumerate(x_collect[:,j]) if fila .== 1], [:cod_postal, :provincia]]
end 


for j in 1:m
    println((j, size(asignaciones[j])[1]))
end 
