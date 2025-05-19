using Pkg
using HDF5
using Flux
using Statistics: mean
using StatsBase
using Flux.Optimise: Descent
using Random
using Printf
using Combinatorics
using ColorTypes: Gray
using Random
using FileIO

using ColorTypes: RGB, N0f8
using ImageCore: clamp01nan
using FileIO: save
import Base.Filesystem: mkpath, dirname
import Base: joinpath

include("SevenSeg.jl")

using .SevenSeg



# ============ command line argument ================

if length(ARGS) < 1
    error("Please supply a 'trial' argument!")
end

trial = parse(Int, ARGS[1])   # Convert the first arg to an Int

folder = ARGS[2]

data_file= ARGS[3]

filename=folder*"/ilm_$trial.csv"

println(filename)

open(folder*"/header.txt", "w") do header
        write(header,"trial,generation,epoch,type,value\n")
end

# ============== parameters =========================

nLatents = 12

struct Parameters
    n1::Int64
    n2::Int64
    n3::Int64
    n4::Int64
    n5::Int64

end

n1 = 784
n2 = 128
n3 = nLatents+4
n4 = nLatents+2
n5 = nLatents

parameters=Parameters(n1,n2,n3,n4,n5)


learningA0 =5.0
learningED0=5.0

learningRateA=learningA0
learningRateED=learningED0

learningLambda=0.95

bottleN=40
examplesN=10*bottleN
autoN=3*examplesN
genN = 70
epochN =   15
reflectN = 15

#comment="noise save_noisy7seg_h5(filename,n_per_digit=100,mu=0.015,sigma=2.0,rho=0.025,nSub=10)\n"

#comment="no noise save_noisy7seg_h5(filename,n_per_digit=100,mu=0.005,sigma=0.25,rho=0.025,nSub=10)\n"

comment=""

open(folder*"/parameters.txt", "w") do parameter_file
    write(parameter_file,"parameters=$parameters\n")
    write(parameter_file,"learningA=$learningA0\n")
    write(parameter_file,"learningED=$learningED0\n")
    write(parameter_file,"learningLambda=$learningLambda\n")
    write(parameter_file,"bottleN=$bottleN\n")
    write(parameter_file,"examplesN=$examplesN\n")
    write(parameter_file,"autoN=$autoN\n")
    write(parameter_file,"epochN=$epochN\n")
    write(parameter_file,"reflectN=$reflectN\n")
    write(parameter_file,comment)
end

# =========== load data ========================



function load_noisy7seg_h5(path::String)
    h5open(path, "r") do f
        imgs   = read(f, "imgs")
        labels = read(f, "labels")
        return (imgs=imgs, labels=labels)
    end
end

dataFile=data_file*".h5"

data=load_noisy7seg_h5(dataFile)

imgs=data.imgs
labels=data.labels


dataFile=data_file*"_base.h5"

data=load_noisy7seg_h5(dataFile)

base_imgs=data.imgs
base_labels=data.labels

nDigits=128

# ========== flatten =============

nImg = size(imgs)[3]
x = Array{Float32}(undef, 784, nImg)

for i in 1:nImg
    vec_784 = vec(imgs[:,:,i])
    x[:, i] = vec_784
end

imageNumbers=[1:nImg...]

nBaseImg = size(base_imgs)[3]
base_x = Array{Float32}(undef,784, nBaseImg)


for i in 1:nBaseImg
    vec_784 = vec(base_imgs[:,:,i])
    base_x[:, i] = vec_784
end



# =========== some tools ======================


function perm_margin_score(A::AbstractMatrix{<:Integer})
    @assert size(A) == (7,7) "perm_margin_score expects a 7x7 matrix"

    # Row and column sums
    rs = sum(A, dims=2)[:]   # row sums (7-element vector)
    cs = sum(A, dims=1)'[:]  # col sums (7-element vector)

    # Compute contributions
    row_score = sum(r > 0 ? 1/r : 0 for r in rs)
    col_score = sum(c > 0 ? 1/c : 0 for c in cs)

    # Normalize by the maximum 14
    return (row_score + col_score) / 14
end

function best_margin_subset(A::AbstractMatrix{<:Integer}, k::Int)
    n, m = size(A)
    @assert k <= m "k must be <= number of columns"
    best_score = -Inf
    best_cols  = Int[]
    for cols in combinations(1:m, k)
        A_sub = A[:, cols]                       # nxk submatrix
        score = perm_margin_score(A_sub)        # from previous cell
        if score > best_score
            best_score = score
            best_cols  = collect(cols)
        end
    end
    return best_score
end


 
function calcC2(imgEncoder,worEncoder,nLatents)
    zMatrix=zeros(Int,7,nLatents)
    for i in 1:7
        img=SevenSeg.make_seg(i)
        z = Int32.(discretizeLatent(worEncoder(imgEncoder(vec(img)))))
        if mean(z)>0.5
            z.=1 .-z
        end
        zMatrix[i,:]=z
        println(z)        
    end
    best_margin_subset(zMatrix,7)
end



function calcC(imgEncoder,worEncoder,nLatents)
    zMatrix=zeros(Int,7,nLatents)
    for i in 1:7
        img=SevenSeg.make_seg(i)
        z = Int32.(discretizeLatent(worEncoder(imgEncoder(vec(img)))))
        zMatrix[i,:]=z
    end
    for i in 1:nLatents
        if mean(zMatrix[:,i])>0.5
            zMatrix[:,i].=1 .-zMatrix[:,i]
        end
    end
    best_margin_subset(zMatrix,7)
end




function calc_bgdC(nLatents,bgdTrialN)
    c=0.0
    
    for _ in 1:bgdTrialN
        zMatrix=zeros(Int,7,nLatents)
        for i in 1:7
            z = rand(0:1,nLatents)
            zMatrix[i,:]=z
        end
        for i in 1:nLatents
            if mean(zMatrix[:,i])>0.5
                zMatrix[:,i].=1 .-zMatrix[:,i]
            end
        end
        c+=best_margin_subset(zMatrix,7)
    end         
    c/bgdTrialN
end

function calc_bgdX(nLatents,bgdTrialN,nBaseImg)

    x=0.0
    z=zeros(Int64,nLatents)
    testZ=[copy(z) for _ in 1:nBaseImg]
    
    for _ in 1:bgdTrialN
        matching=0.0
        for i in 1:nBaseImg
            z=rand(0:1,nLatents)
            if z==testZ[i]
                matching+=1
            end
            testZ[i]=z
        end
        x+=length(unique(testZ))/nBaseImg
    end
    x/bgdTrialN
end


bgdTrialN=1000
bgdC=calc_bgdC(nLatents,bgdTrialN)
bgdX=calc_bgdX(nLatents,bgdTrialN,nBaseImg)

println(bgdC)
println(bgdX)




# ========= 0. some functions ==========



function shufflePairs(pairs)
    
    thisPairs = copy(pairs)

    shuffle!(thisPairs)
    
    ([p[1] for p in thisPairs], [p[2] for p in thisPairs])

end

function discretizeLatent(z::Vector{Float32})
    z_disc=Float32.(z .> 0.5f0)
    return z_disc
end

# ========== 5. Define a Simple MLP Autoencoder in Flux ==========
    
function makeModel(parameters)

    n1=parameters.n1
    n2=parameters.n2
    n3=parameters.n3
    n4=parameters.n4
    n5=parameters.n5

    
    imgEncoder = Chain(
        Dense(n1, n2, sigmoid),
        Dense(n2, n3, sigmoid),
    )
    
    worEncoder = Chain(
        Dense(n3, n4, sigmoid),
        Dense(n4, n5, sigmoid)
    )

    worDecoder = Chain(
        Dense(n5, n4, sigmoid),
        Dense(n4, n3, sigmoid),
    )

    
    imgDecoder = Chain(
        Dense(n3, n2, sigmoid),
        Dense(n2, n1, sigmoid)
    )
    
    return (imgEncoder,worEncoder,worDecoder,imgDecoder)

end
    
# ========= 5a. losses / optimizer ================

mutable struct TotalLoss
    total::Float64
end


lossMSE(nn, x,y)= Flux.mse(nn(x), y)


function makeLoss(totalLoss::TotalLoss,lossMSE)
    function(nn,x,y)
        loss=lossMSE(nn,x,y)
        totalLoss.total+=loss
        loss
    end
end


function makeLossWordEncoder(totalLoss::TotalLoss,imgEncoder)
    function(nn,x,y)
        loss=Flux.mse(nn(imgEncoder(x)),y)
        totalLoss.total+=loss
        loss
    end
end


function makeLossWordDecoder(totalLoss::TotalLoss,imgDecoder)
    function(nn,x,y)
        loss=Flux.mse(imgDecoder(nn(x)),y)
        totalLoss.total+=loss
        loss
    end
end


# ========= make testZ

testZ=[zeros(Int64,nLatents) for _ in 1:nBaseImg]

# ========== 5b. make first pupil =============

shuffleDigits=shuffle(collect(1:nDigits))

bottleDigits=shuffleDigits[1:bottleN]

newImageNumbers=Vector{Int}()

for i in 1:length(imageNumbers)
    if labels[i] in bottleDigits
        push!(newImageNumbers,imageNumbers[i])
    end
end

supervised=sample(newImageNumbers, examplesN; replace=false)
#supervised=newImageNumbers

pairs = Vector{Tuple{Int, Vector{Float32}}}(undef, length(supervised))
for i in 1:length(supervised)
    idx = supervised[i]
    lat = rand(Bool, nLatents) .|> Float32
    pairs[i] = (idx, lat)
end

(supervised1,parentSignal1)=shufflePairs(pairs)
(supervised2,parentSignal2)=shufflePairs(pairs)
(supervised3,parentSignal3)=shufflePairs(pairs)

# =========== seg key ==================

for i in 1:7
    img=SevenSeg.make_seg(i)
    filename="imagesSeg/key$(i).png"
    SevenSeg.save_img(img,filename,20)
end

# ========== 6. generation loop ==========

# I need to switch to a version where the indices are all that gets shuffled so
# I can shuffle for each epoch

nGlyphs=20
zMatrixOne=zeros(Float64, nGlyphs,nLatents)
zMatrixFive=zeros(Float64, nGlyphs,nLatents)
zMatrixEight=zeros(Float64, nGlyphs,nLatents)
recordingStart=50

for generation in 1:genN

    local bottleDigits
    global nGlyphs,zMatrixOne,zMatrixFive,zMatrixEight,recordingStart
    global shuffleDigits,cBgd,xBgd
    global learningRateA,learningRateED,newImageNumbers
    global file,pairs,supervised,supervised1,parentSignal1, supervised2,parentSignal2, supervised3,parentSignal3
    
    println("generation=$generation")
    println(shuffleDigits[1:bottleN])
    
    auto=sample(imageNumbers, autoN; replace=false)

    (imgEncoder,worEncoder,worDecoder,imgDecoder)=makeModel(parameters)
    outerEncoder = Chain(imgEncoder,imgDecoder)
    innerEncoder = Chain(worEncoder,worDecoder)

    encoder = Chain(imgEncoder,worEncoder)
    decoder = Chain(worDecoder,imgDecoder)

    learningRateA=learningA0
    learningRateED=learningED0
    
    for epoch in 1:epochN


        optimizerI=Flux.Optimise.Descent(learningRateA)
        optimizerO=Flux.Optimise.Descent(learningRateA)
        optimizerE=Flux.Optimise.Descent(learningRateED)
        optimizerD=Flux.Optimise.Descent(learningRateED)

        learningRateA*=learningLambda
        learningRateED*=learningLambda
        
        totalLossD=TotalLoss(0.0)
        totalLossE=TotalLoss(0.0)
        totalLossI=TotalLoss(0.0)
        totalLossO=TotalLoss(0.0)
        
        thisLossD=makeLoss(totalLossD,lossMSE)
        thisLossE=makeLoss(totalLossE,lossMSE)
        thisLossI=makeLoss(totalLossI,lossMSE)
        thisLossO=makeLoss(totalLossO,lossMSE)
        
        for batch in 1:length(supervised)
            
             # ======== encoder ==========
            image=x[:,supervised1[batch]]
            signal=parentSignal1[batch]
            dataE=[(image,signal)]
            
            Flux.train!(thisLossE, encoder, dataE, optimizerE)

            # ======== decoder ==========
            image=x[:,supervised2[batch]]
            signal=parentSignal2[batch]
            dataD=[(signal,image)]
                
            Flux.train!(thisLossD, decoder, dataD, optimizerD)
                
            # ========= autoencoder =====
            
            for _ in 1:reflectN
                            
                autoIndex = rand(1:autoN)
                image = x[:, auto[autoIndex]]
                
                dataO=[(image,image)]
                Flux.train!(thisLossO, outerEncoder, dataO, optimizerO)

                signal=imgEncoder(image)
                dataI=[(signal,signal)]
                Flux.train!(thisLossI, innerEncoder, dataI, optimizerI)
                
                
            end            

        end

        
        
        totalD=totalLossD.total/examplesN
        totalE=totalLossE.total/examplesN
        totalI=totalLossI.total/(reflectN*examplesN)
        totalO=totalLossO.total/(reflectN*examplesN)
        

        function printTotal(total, epoch, type)
            print("$type$epoch $( @sprintf("%.5f", total) ) ")
        end

        printTotal(totalD,epoch,"D")
        printTotal(totalE,epoch,"E")
        printTotal(totalI,epoch,"I")
        printTotal(totalO,epoch,"O")
        println()
        
        
        open(filename, "a") do file
            write(file,"$trial,$generation,$epoch,tD,$totalD\n")
            write(file,"$trial,$generation,$epoch,tE,$totalE\n")
            write(file,"$trial,$generation,$epoch,tI,$totalI\n")
            write(file,"$trial,$generation,$epoch,tO,$totalO\n")
            flush(file)
        end

    end
   
    # various metrics things

    matching=0.0
    for i in 1:nBaseImg
        z=Int32.(discretizeLatent(encoder(base_x[:,i])))
        if z==testZ[i]
            matching+=1
        end
        testZ[i]=z
    end
        
    stable=matching/nBaseImg
    express=length(unique(testZ))/nBaseImg
    compress=calcC(imgEncoder,worEncoder,nLatents)

    compress=(compress-bgdC)/(1.0-bgdC)
    
    println("stability=",stable)
    println("expressivity=",express)
    println("compositionality=",compress)

    open(filename, "a") do file
        write(file,"$trial,$generation,-1,s,$stable\n")
        write(file,"$trial,$generation,-1,x,$express\n")
        write(file,"$trial,$generation,-1,c,$compress\n")
        flush(file)
    end

    # make the input for the supervised learning
    
    shuffleDigits=shuffle(collect(1:nDigits))

    bottleDigits=shuffleDigits[1:bottleN]

    j=1
    for i in 1:length(imageNumbers)
        if labels[i] in bottleDigits
            newImageNumbers[j]=imageNumbers[i]
            j+=1
        end
    end

    supervised=sample(newImageNumbers, examplesN; replace=false)
#    supervised=newImageNumbers

    pairs = Vector{Tuple{Int, Vector{Float32}}}(undef, length(supervised))
    for i in 1:length(supervised)
        idx = supervised[i]
        lat = discretizeLatent(worEncoder(imgEncoder(x[:, idx])))
        pairs[i] = (idx, lat)
    end

    (supervised1,parentSignal1)=shufflePairs(pairs)
    (supervised2,parentSignal2)=shufflePairs(pairs)
    (supervised3,parentSignal3)=shufflePairs(pairs)

    
    # study latents


    function save_red_latents(zMatrix::AbstractMatrix{<:Real},
                          trial::Int, generation::Int,key::String;
                              scaleX::Int=4, scaleY::Int=10,
                              outdir::AbstractString="images")
        # 1) clamp into [0,1] and upsample
        sm   = clamp01nan.(zMatrix)                             # raw in [0,1]
        bigv = kron(sm, ones(scaleX, scaleY))                   # Float64 array
        R,  C = size(bigv)
        nL     = size(zMatrix, 2)
        C2     = C + (nL - 1)                                   # extra white cols
        period = scaleY + 1
        
        # 2) build RGB{N0f8} output, inserting white lines
        out = Array{RGB{N0f8}}(undef, R, C2)
        for j in 1:C2
        if j % period == 0
            @inbounds out[:, j] .= RGB{N0f8}(1,1,1)         # 1-pixel white divider
        else
            orig_j = j - div(j, period)
            @inbounds for i in 1:R
                v = bigv[i, orig_j]                         # in [0,1]
                out[i,j] = RGB{N0f8}(v, 0, 0)               # red channel only
            end
        end
        end

    # 3) make sure directory exists & save
        mkpath(outdir)
        path = joinpath(outdir, "latents_$(key)_$(trial)_$(generation).png")
        save(path, out)
    end
    
    #nFives=count(==(5), labels)


    scaleX=5
    scaleY=12


    #Fives
    
    zMatrix=zeros(Float64, nGlyphs,nLatents)
    
    j=1
    for i in 1:nImg
        if labels[i]==5 && j<=nGlyphs
            z=encoder(x[:,i])
            zMatrix[j,:]=z
            j+=1
        end
    end

    img_gray = Gray.(zMatrix)
    
    big_img = kron(img_gray, fill(1, scaleX, scaleY))

    save_red_latents(zMatrix, trial, generation, "five";
                 scaleX=scaleX, scaleY=scaleY, outdir="imagesFives")


    #nEights
    
    zMatrix=zeros(Float64, nGlyphs,nLatents)
    
    j=1
    for i in 1:nImg
        if labels[i]==95 && j<=nGlyphs
            z=encoder(x[:,i])
            zMatrix[j,:]=z
            j+=1
        end
    end

    img_gray = Gray.(zMatrix)
    
    big_img = kron(img_gray, fill(1, scaleX, scaleY))

    save_red_latents(zMatrix, trial, generation, "eight";
                 scaleX=scaleX, scaleY=scaleY, outdir="imagesEights")

    
    #nOnes
    
    zMatrix=zeros(Float64, nGlyphs,nLatents)
    
    j=1
    for i in 1:nImg
        if labels[i]==52 && j<=nGlyphs
            z=encoder(x[:,i])
            zMatrix[j,:]=z
            j+=1
        end
    end

    img_gray = Gray.(zMatrix)
    
    big_img = kron(img_gray, fill(1, scaleX, scaleY))

    save_red_latents(zMatrix, trial, generation, "one";
                 scaleX=scaleX, scaleY=scaleY, outdir="imagesOnes")


    #seg image

    zMatrix=zeros(Int,7,nLatents)

    for i in 1:7
        img=SevenSeg.make_seg(i)
        z = Int32.(discretizeLatent(worEncoder(imgEncoder(vec(img)))))
        zMatrix[i,:]=z
    end

    
    img_gray = Gray.(zMatrix)
    
    big_img = kron(img_gray, fill(1, scaleX, scaleY))

    save_red_latents(zMatrix, trial, generation, "imagesSeg";
                 scaleX=scaleX, scaleY=scaleY, outdir="imagesSeg")

    #generation image

    if generation>=recordingStart && generation<recordingStart+nGlyphs
        z=encoder(base_x[:,5])
        zMatrixFive[generation-recordingStart+1,:]=z
        z=encoder(base_x[:,52])
        zMatrixOne[generation-recordingStart+1,:]=z
        z=encoder(base_x[:,95])
        zMatrixEight[generation-recordingStart+1,:]=z
    end

    if generation==recordingStart+nGlyphs
        save_red_latents(zMatrixEight, trial, generation, "eightG";
                         scaleX=scaleX, scaleY=scaleY, outdir="imagesEights")
        save_red_latents(zMatrixOne, trial, generation, "oneG";
                         scaleX=scaleX, scaleY=scaleY, outdir="imagesOnes")
        save_red_latents(zMatrixFive, trial, generation, "fiveG";
                         scaleX=scaleX, scaleY=scaleY, outdir="imagesFives")
    end
    
end
