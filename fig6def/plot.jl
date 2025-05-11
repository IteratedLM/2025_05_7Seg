
using CSV, DataFrames,Gadfly
import Cairo, Fontconfig
include("../plotting/plotting.jl")

df = CSV.File("results.csv") |> DataFrame

function makeMatrix(df::DataFrame,type::String)
    cut_df = filter(row -> row.generation <= 100,df)
    filtered_df = filter(row -> row.type == type, cut_df)

    max_generation = maximum(filtered_df.generation)
    max_trial = maximum(filtered_df.trial)


    matrix = fill(NaN, max_generation, max_trial)  # or use zeros() if you prefer

    
    for row in eachrow(filtered_df)
        matrix[row.generation, row.trial] = row.value
    end

    matrix

end



autMin=30
autMax=180
autStep=5

autV=collect(autMin:autStep:autMax)

expressMatrix=makeMatrix(df,"x")
composeMatrix=makeMatrix(df,"c")
stableMatrix=makeMatrix(df,"s")

#generationN=filter(row-> row.propertyType=="m",df)[1,:generation]

generationN=80

generations=collect(0:generationN-1)

#filename="oilm"
filename="fig6def"

#plotPropertyLines(expressMatrix,filename*"_express.png",colorant"blue","x")
#plotPropertyLines(composeMatrix,filename*"_compose.png",colorant"orange","c")
#plotPropertyLines(stableMatrix, filename*"_stable.png" ,colorant"purple","s")


plotPropertyLinesPLoS(expressMatrix,filename*"_express",colorant"blue","x")
plotPropertyLinesPLoS(composeMatrix,filename*"_compose",colorant"orange","c")
plotPropertyLinesPLoS(stableMatrix, filename*"_stable" ,colorant"purple","s")
