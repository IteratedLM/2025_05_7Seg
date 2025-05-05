using Pkg
using HDF5
using Flux
using Statistics: mean
using StatsBase
using Flux.Optimise: Descent
using Random
using Printf
using Images, FileIO, ImageTransformations

using Random

include("SevenSeg.jl")

using .SevenSeg


# ============ command line argument ================

if length(ARGS) < 1
    error("Please supply a 'trial' argument!")
end

trial = parse(Int, ARGS[1])   # Convert the first arg to an Int


filename="results/ilm_epoch_$trial.csv"

println(filename)

# =========== load data ========================



function load_noisy7seg_h5(path::String)
    h5open(path, "r") do f
        imgs   = read(f, "imgs")
        labels = read(f, "labels")
        return (imgs=imgs, labels=labels)
    end
end

dataFile="digits.h5"

data=load_noisy7seg_h5(dataFile)

imgs=data.imgs
labels=data.labels

nDigits=128

# ========== flatten =============

nImg = size(imgs)[3]
x = Array{Float32}(undef, 784, nImg)

for i in 1:nImg
    vec_784 = vec(imgs[:,:,i])
    x[:, i] = vec_784
end

imageNumbers=[1:nImg...]


# ========= 0. some functions ==========



function shufflePairs(pairs)
    
    thisPairs = copy(pairs)

    shuffle!(thisPairs)
    
    ([p[1] for p in thisPairs], [p[2] for p in thisPairs])

end

function discretizeLatent(z::Vector{Float32})
    return Float32.(z .> 0.5f0)
end

# ========== 5. Define a Simple MLP Autoencoder in Flux ==========

struct Parameters
    n1::Int64
    n2::Int64
    n3::Int64
    n4::Int64
    n5::Int64
    
end

nLatent = 10

n1 = 784
n2 = 128
n3 = nLatent
n4 = nLatent
n5 = nLatent

parameters=Parameters(n1,n2,n3,n4,n5)
    
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

# =========== test set for stability ======

labelCount=zeros(Int,nDigits)

testSet=[]
testSetN=5

for i in 1:length(imageNumbers)
    if labelCount[labels[i]]<testSetN
        push!(testSet,imageNumbers[i])
        labelCount[labels[i]]+=1
    end
end

# ========== some parameters ===========

learningA0 =5
learningED0=5

learningRateA=learningA0
learningRateED=learningED0

learningLambda=0.95

bottleN=40
examplesN=10*bottleN
autoN=3*examplesN
genN = 200
epochN = 15
reflectN = 15


# ========= make testZ

testZ=[zeros(Int,nLatent) for _ in 1:length(testSet)]

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
    lat = rand(Bool, nLatent) .|> Float32
    pairs[i] = (idx, lat)
end

(supervised1,parentSignal1)=shufflePairs(pairs)
(supervised2,parentSignal2)=shufflePairs(pairs)
(supervised3,parentSignal3)=shufflePairs(pairs)

# =========== filestuff ==================

file=open(filename, "a")                   

# ========== 6. generation loop ==========

# I need to switch to a version where the indices are all that gets shuffled so
# I can shuffle for each epoch


for generation in 1:genN

    local bottleDigits

    global shuffleDigits
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
        
        #=
        write(file,"$trial,$generation,$epoch,tD,$totalD\n")
        write(file,"$trial,$generation,$epoch,tE,$totalE\n")
        write(file,"$trial,$generation,$epoch,tA,$totalA\n")
        flush(file)
        =#
        
    end
   

    # make the input for the supervised learning

    matching=0.0
    for i in 1:length(testSet)
        z=discretizeLatent(encoder(x[:,testSet[i]]))
        if z==testZ[i]
            matching+=1
        end
        testZ[i]=z
    end

    println("stability=",matching/length(testSet))
    println("expressivity=",length(unique(testZ))/nDigits)
    
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

    avgActs = zeros(Float32, nDigits, nLatent)
    local counts = fill(0, nDigits)
    
    for i in 1:nImg
        d = labels[i]           # digit label in [0..9]
        # Forward pass to get latent vector shape (nLatent,)
        z = vec(encoder(x[:, i:i]))  # or x[:, i], but be consistent
        avgActs[d, :] .+= z
        counts[d] += 1
    end

    # Now divide each row by the count
    for d in 1:nDigits
        if counts[d] > 0
            avgActs[d, :] ./= counts[d]
        end
    end

    img_gray = Gray.(avgActs)

    # Scale it up by a factor, e.g., 20

    scaleX = 200
    scaleY =  50

    big_img = kron(img_gray, fill(1, scale, scale))

    
    save("latent_means_$(trial)_$(generation).png", big_img)

     
    imgIn=reshape(x[:,supervised1[1]], 28, 28)
    filename="example_$(trial)_$(generation)_In.png"
    SevenSeg.save_img(imgIn,filename,20)

    out=decoder(encoder(x[:,supervised1[1]]))
    
    imgOut=reshape(out, 28, 28)
    filename="example_$(trial)_$(generation)_Out.png"
    SevenSeg.save_img(imgOut,filename,20)

    z=discretizeLatent(encoder(x[:, supervised1[1]]))
    imgDis=reshape(decoder(z), 28, 28)
    filename="example_$(trial)_$(generation)_z.png"
    SevenSeg.save_img(imgDis,filename,20)

    #=
    
    # study latents

    =#
    
    avgActs = zeros(Float32, 7, nLatent)

    for i in 1:7
        img=SevenSeg.make_seg(i)
        z = worEncoder(imgEncoder(vec(img)))
        for latent in 1:nLatent
            avgActs[i,latent]=z[latent]
        end
    end
       
    img_gray = Gray.(avgActs)

    scale = 200
    
    big_img = kron(img_gray, fill(1, scale, scale))

    save("segs_$(trial)_$(generation).png", big_img)

    #=

    for i in 1:nLatent
        z=zeros(Float32,nLatent)
        z[i]=1.0
        img=reshape(decoder(z),28,28)
        img_gray= Gray.(img)
        scale=20
        big_img = kron(img_gray, fill(1, scale, scale))

        save("atom_$(trial)_$(generation)_$(i).png", big_img)
    end

    =#


    
end
