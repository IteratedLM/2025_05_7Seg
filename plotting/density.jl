using DataFrames, Gadfly, KernelDensity

function densityPlot(df::DataFrame,color,targetGeneration,targetPropertyType,property::String,filename::String)

    filteredDataFrame = filter(row -> row[:generation] == targetGeneration && row[:propertyType] == targetPropertyType, df)

    println(filteredDataFrame)

    # Compute the kernel density estimate
    kdeResult = kde(filteredDataFrame.property)

    
    # Adjust the KDE to consider the support [0,1]
    #kdeResult.density = [density * (0 <= x <= 1.02) for (x, density) in zip(kdeResult.x, kdeResult.density)]

    #kdeResult.density = kdeResult.density / sum(kdeResult.density)  # Normalize

# Plot the kernel density estimate
    plt=plot(x=kdeResult.x, y=kdeResult.density, Geom.line, Theme(default_color=color),style(line_width=3pt,default_color=color),
             Guide.xlabel(property), Guide.ylabel("density"),Theme(background_color=colorant"white"),Coord.cartesian(xmin=0, xmax=1.0))
    draw(PNG(filename, 2.5inch, 2inch),plt)
    draw(PNG("big_"*filename, 5inch, 4inch),plt)


end
