using Colors,Gadfly,Compose
import Cairo, Fontconfig
using Statistics,DataFrames

function getLightColor(color,lightValue=0.3)

    r=red(color)
    g=green(color)
    b=blue(color)
    
    RGB(r+(1-r)*lightValue,g+(1-g)*lightValue,b+(1-b)*lightValue)

end


function plotProperty(propertyMatrix,xAxis,filename,color,yLabel)
    mu=vec(mean(propertyMatrix, dims=2))
    stdDev = vec(std(propertyMatrix, dims=2))
    
    min = mu .- stdDev
    max = mu .+ stdDev

    df = DataFrame(x=xAxis, y=mu, yMin=min,yMax=max)
       
    #alphaValue=0.3
    #lightColor = RGBA(color, alphaValue)
    
    plt=plot(
        layer(df,x=:x,y=:y, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df,x=:x,ymin=:yMin,ymax=:yMax,Geom.ribbon,style(default_color=color)),
        Theme(background_color=colorant"white"),
        Guide.xlabel("bottleneck"),
        Guide.ylabel(yLabel),
        Coord.Cartesian(ymin=0.0,ymax=1.0)

    )

    draw(PNG(filename, 2.5inch, 2inch),plt)

end

function plotProperty(propertyMatrix0,propertyMatrix1,xAxis,filename,color,yLabel)
    plotProperty(propertyMatrix0,propertyMatrix1,xAxis,filename,color,yLabel,"bottleneck")
end

function plotProperty(propertyMatrix0,propertyMatrix1,xAxis,filename,color,yLabel,xLabel)


    function muMinMax(propertyMatrix)
    
        mu=vec(mean(propertyMatrix, dims=2))
        stdDev = vec(std(propertyMatrix, dims=2))
    
        (mu,mu .- stdDev,mu .+ stdDev)
    end

    (mu0,min0,max0)=muMinMax(propertyMatrix0)
    (mu1,min1,max1)=muMinMax(propertyMatrix1)
    
    df0 = DataFrame(x=xAxis, y=mu0, yMin=min0,yMax=max0)
    df1 = DataFrame(x=xAxis, y=mu1, yMin=min1,yMax=max1)
       
    #alphaValue=0.3
    #lightColor = RGBA(color, alphaValue)
    
    plt=plot(
        layer(df1,x=:x,y=:y, Geom.line,style(line_width=3pt,default_color=color,line_style=[:dash])),
        layer(df1,x=:x,y=:y, Geom.line,style(line_width=1pt,default_color=colorant"black",line_style=[:dash])),
        layer(df0,x=:x,y=:y, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df0,x=:x,ymin=:yMin,ymax=:yMax,Geom.ribbon,style(default_color=color)),
        Theme(plot_padding=[0mm,2mm,0mm,0mm],background_color=colorant"white"),
        Guide.xlabel(xLabel),
        Guide.ylabel(yLabel,orientation=:vertical),
        Coord.Cartesian(ymin=0.0,ymax=1.0,xmax=maximum(df1.x))

    )

    draw(PNG(filename, 2.5inch, 2inch),plt)
    draw(PNG("big_"*filename, 5inch, 4inch),plt)

end



function plotPropertyPLoS(propertyMatrix0,propertyMatrix1,xAxis,filename,color,yLabel,xLabel)


    function muMinMax(propertyMatrix)
    
        mu=vec(mean(propertyMatrix, dims=2))
        stdDev = vec(std(propertyMatrix, dims=2))
    
        (mu,mu .- stdDev,mu .+ stdDev)
    end

    (mu0,min0,max0)=muMinMax(propertyMatrix0)
    (mu1,min1,max1)=muMinMax(propertyMatrix1)
    
    df0 = DataFrame(x=xAxis, y=mu0, yMin=min0,yMax=max0)
    df1 = DataFrame(x=xAxis, y=mu1, yMin=min1,yMax=max1)
       
    #alphaValue=0.3
    #lightColor = RGBA(color, alphaValue)
    
    plt=plot(
        layer(df1,x=:x,y=:y, Geom.line,style(line_width=3pt,default_color=color,line_style=[:dash])),
        layer(df1,x=:x,y=:y, Geom.line,style(line_width=1pt,default_color=colorant"black",line_style=[:dash])),
        layer(df0,x=:x,y=:y, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df0,x=:x,ymin=:yMin,ymax=:yMax,Geom.ribbon,style(default_color=color)),
        Theme(plot_padding=[0mm,2mm,0mm,0mm],background_color=colorant"white",
              key_label_font_size=12pt,
              major_label_font="Arial",
              major_label_font_size=10pt,
              minor_label_font="Arial",
              minor_label_font_size=10pt
              ),
        Guide.xlabel(xLabel),
        Guide.ylabel(yLabel,orientation=:vertical),
        Coord.Cartesian(ymin=0.0,ymax=1.0,xmax=maximum(df1.x))

    )

    draw(PDF(filename*".pdf", 2.5inch, 2inch),plt)
    draw(PS(filename*".eps", 2.5inch, 2inch),plt)


end


function plotPropertyLines(propertyMatrix,filename,color,yLabel)

    
    numTimePoints = size(propertyMatrix, 1)
    numTrials = size(propertyMatrix, 2)

    df = DataFrame()
    df.time = repeat(1:numTimePoints, outer=numTrials)
    df.trial = repeat(1:numTrials, inner=numTimePoints)
    df.performance = vec(propertyMatrix)

    avgPerformance = mean(propertyMatrix, dims=2)[2:end] |> vec
    avgDf = DataFrame(time=2:numTimePoints, avgPerformance=avgPerformance)

       
    alphaValue=0.3
    lightColor = getLightColor(color, alphaValue)
    
    plt=plot(
        layer(avgDf, x=:time, y=:avgPerformance, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df, x=:time, y=:performance, group=:trial, Geom.line,style(line_width=0.5pt,default_color=lightColor)),
        Theme(background_color=colorant"white"),
        Guide.xlabel("generations"),
        Guide.ylabel(yLabel),
        Coord.Cartesian(ymin=0.0,ymax=1.0)
     )

    draw(PNG(filename, 2.5inch, 2inch),plt)
    draw(PNG("big_"*filename, 5inch, 4inch),plt)
end

        
function plotPropertyLines(propertyMatrix,filename,color,yLabel)
    plotPropertyLinesScaled(propertyMatrix,filename,color,yLabel,1.0)
end

function plotPropertyLinesScaled(propertyMatrix,filename,color,yLabel,scale)

    numTimePoints = size(propertyMatrix, 1)
    numTrials = size(propertyMatrix, 2)

    df = DataFrame()
    df.time = repeat(1:numTimePoints, outer=numTrials)
    df.trial = repeat(1:numTrials, inner=numTimePoints)
    df.performance = vec(propertyMatrix)

    offset=2
    avgPerformance = mean(propertyMatrix, dims=2)[offset:end] |> vec
    avgDf = DataFrame(time=offset:numTimePoints, avgPerformance=avgPerformance)

       
    alphaValue=0.7
    lightColor = getLightColor(color, alphaValue)

    plt=plot(
        layer(avgDf, x=:time, y=:avgPerformance, Geom.line, Theme(line_width=3pt,default_color=color)),
        layer(df, x=:time, y=:performance, group=:trial,Geom.line,Theme(line_width=0.5pt,default_color=lightColor)),
        Theme(plot_padding=[0mm,0mm,0mm,0mm],background_color=colorant"white"),
        #Guide.xlabel(" "),
        Guide.xlabel("generations"),
        Guide.ylabel(yLabel,orientation=:vertical),
        Coord.Cartesian(ymin=0.0,ymax=1.0)
     )
    draw(PNG(filename, scale*2.5inch, scale*2inch),plt)
    #draw(PNG("big_"*filename, scale*5inch, scale*4inch),plt)
end

        
function plotPropertyLinesPLoS(propertyMatrix,filename,color,yLabel)
    plotPropertyLinesScaledPLoS(propertyMatrix,filename,color,yLabel,1.0)
end

function plotPropertyLinesScaledPLoS(propertyMatrix,filename,color,yLabel,scale)

    numTimePoints = size(propertyMatrix, 1)
    numTrials = size(propertyMatrix, 2)

    df = DataFrame()
    df.time = repeat(1:numTimePoints, outer=numTrials)
    df.trial = repeat(1:numTrials, inner=numTimePoints)
    df.performance = vec(propertyMatrix)

    offset=2
    avgPerformance = mean(propertyMatrix, dims=2)[offset:end] |> vec
    avgDf = DataFrame(time=offset:numTimePoints, avgPerformance=avgPerformance)

       
    alphaValue=0.7
    lightColor = getLightColor(color, alphaValue)

    plt=plot(
        layer(avgDf, x=:time, y=:avgPerformance, Geom.line, Theme(line_width=3pt,default_color=color)),
        layer(df, x=:time, y=:performance, group=:trial,Geom.line,Theme(line_width=0.5pt,default_color=lightColor)),
        Theme(plot_padding=[0mm,2mm,0mm,0mm],background_color=colorant"white"
              ,major_label_font="Arial",
               major_label_font_size=10pt,
               minor_label_font="Arial",
               minor_label_font_size=10pt,
               #title_font="Arial",
              #title_font_size=12pt
              ),
        #Guide.xlabel(" "),
        Guide.xlabel("generations"),
        Guide.ylabel(yLabel,orientation=:vertical),
        Coord.Cartesian(ymin=0.0,ymax=1.0)
     )
    draw(PDF(filename*".pdf", scale*2.5inch, scale*2inch),plt)
    draw(PS(filename*".eps", scale*2.5inch, scale*2inch),plt)

end


        
function plotPropertyLines(propertyMatrix,filename,color,yLabel,xRange)

    numTimePoints = size(propertyMatrix, 1)
    numTrials = size(propertyMatrix, 2)

    df = DataFrame()
    df.time = repeat(1:numTimePoints, outer=numTrials)
    df.trial = repeat(1:numTrials, inner=numTimePoints)
    df.performance = vec(propertyMatrix)

    avgPerformance = mean(propertyMatrix, dims=2)[2:end] |> vec
    avgDf = DataFrame(time=2:numTimePoints, avgPerformance=avgPerformance)



    alphaValue=0.2
    lightColor = getLightColor(color, alphaValue)



    plt=plot(
        layer(avgDf, x=:time, y=:avgPerformance, Geom.line,style(line_width=1pt,default_color=colorant"black")),
        layer(avgDf, x=:time, y=:avgPerformance, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df, x=:time, y=:performance, group=:trial, Geom.line,style(line_width=0.5pt,default_color=lightColor)),
        Theme(plot_padding=[0mm,0mm,0mm,0mm],background_color=colorant"white"),
        Guide.xlabel("generations"),
        Guide.ylabel(yLabel,orientation=:vertical),
        Coord.Cartesian(ymin=0.0,ymax=1.0,xmin=0,xmax=xRange)
     )

    draw(PNG(filename, 2.5inch, 2inch),plt)
    #draw(PNG("big_"*filename, 5inch, 4inch),plt)
end


        
function longPlotPropertyLines(propertyMatrix1,propertyMatrix2,propertyMatrix3,filename,colors,yLabels,xRange)
    
    numTimePoints = size(propertyMatrix1, 1)
    numTrials = size(propertyMatrix1, 2)

    function makeDF(propertyMatrix)
    
        numTimePoints = size(propertyMatrix, 1)
        numTrials = size(propertyMatrix, 2)

        df = DataFrame()
        df.time = repeat(1:numTimePoints, outer=numTrials)
        df.trial = repeat(1:numTrials, inner=numTimePoints)
        df.performance = vec(propertyMatrix)

        avgPerformance = mean(propertyMatrix, dims=2)[2:end] |> vec
        DataFrame(time=2:numTimePoints, avgPerformance=avgPerformance)
    end

    df1=makeDF(propertyMatrix1)
    df2=makeDF(propertyMatrix2)
    df3=makeDF(propertyMatrix3)

    df1[!,:property].=yLabels[1]
    df2[!,:property].=yLabels[2]
    df3[!,:property].=yLabels[3]

    df=vcat(df1,df2,df3)


plt = plot(
    layer(df, x=:time, y=:avgPerformance, color=:property, Geom.line,
          style(line_width=3pt)),
    Scale.color_discrete_manual(colors...),  # Apply custom colors
    Theme(
        plot_padding=[0mm, 2mm, 0mm, 0mm],
        background_color=colorant"white",
        key_label_font_size=12pt
    ),
    Guide.xlabel("generations"),
    Guide.ylabel("x/c/s", orientation=:vertical),
    Coord.Cartesian(ymin=0.0, ymax=1.0, xmin=0, xmax=xRange),
    Guide.colorkey(title="",pos=[0.05w,-0.4h])         
)

draw(PNG(filename, 9inch, 2.5inch), plt)

end

function longPlotPropertyLinesPLoS(propertyMatrix1,propertyMatrix2,propertyMatrix3,filename,colors,yLabels,xRange)
    
    numTimePoints = size(propertyMatrix1, 1)
    numTrials = size(propertyMatrix1, 2)

    function makeDF(propertyMatrix)
    
        numTimePoints = size(propertyMatrix, 1)
        numTrials = size(propertyMatrix, 2)

        df = DataFrame()
        df.time = repeat(1:numTimePoints, outer=numTrials)
        df.trial = repeat(1:numTrials, inner=numTimePoints)
        df.performance = vec(propertyMatrix)

        avgPerformance = mean(propertyMatrix, dims=2)[2:end] |> vec
        DataFrame(time=2:numTimePoints, avgPerformance=avgPerformance)
    end

    df1=makeDF(propertyMatrix1)
    df2=makeDF(propertyMatrix2)
    df3=makeDF(propertyMatrix3)

    df1[!,:property].=yLabels[1]
    df2[!,:property].=yLabels[2]
    df3[!,:property].=yLabels[3]

    df=vcat(df1,df2,df3)


plt = plot(
    layer(df, x=:time, y=:avgPerformance, color=:property, Geom.line,
          style(line_width=3pt)),
    Scale.color_discrete_manual(colors...),  # Apply custom colors
    Theme(
        plot_padding=[0mm, 2mm, 0mm, 0mm],
        background_color=colorant"white",
        key_label_font_size=12pt,
        major_label_font="Arial",
        major_label_font_size=10pt,
        minor_label_font="Arial",
        minor_label_font_size=10pt
    ),
    Guide.xlabel("generations"),
    Guide.ylabel("x/c/s", orientation=:vertical),
    Coord.Cartesian(ymin=0.0, ymax=1.0, xmin=0, xmax=xRange),
    Guide.colorkey(title="",pos=[0.05w,-0.4h])         
)

    draw(PS(filename*".eps", 9inch, 2.5inch), plt)
    draw(PDF(filename*".pdf", 9inch, 2.5inch), plt)

end


