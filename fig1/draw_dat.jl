using Pkg
using HDF5
using Flux
using Statistics: mean
using StatsBase
using Flux.Optimise: Descent
using Random
using Printf
using Combinatorics

using Random

include("SevenSeg.jl")

using .SevenSeg

data_file="digits_noisy"



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

for d in 1:nDigits

    i=1
    img=imgs[:,:,i]
    while labels[i]!=d
        i+=1
        img=imgs[:,:,i]
    end
    
    filename="noise_images/noise_$(d).png"
    SevenSeg.save_img_color(img,filename,20)

    
    filename="base_images/base_$(d).png"
    SevenSeg.save_img_color(base_imgs[:,:,d],filename,20)
    
    
end
