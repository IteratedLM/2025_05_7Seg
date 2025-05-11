
using Gadfly,Colors

function plot_best(df::DataFrame,filename)

    plt1=plot(df, x=:n, y=:bottle, Geom.point,Theme(background_color=colorant"white",default_color="red",plot_padding=[0mm,0mm,0mm,0mm]), Geom.smooth(method=:lm))
    draw(PNG(filename*"_bottle.png", 5inch, 2inch),plt1)
    draw(PNG(filename*"_bottle_big.png", 8inch, 4inch),plt1)

    plt2=plot(df, x=:n, y=:generation, Geom.point,Theme(background_color=colorant"white",default_color="red",plot_padding=[0mm,0mm,0mm,0mm]),Geom.line)
    draw(PNG(filename*"_gen.png", 2.5inch, 2inch),plt2)
    draw(PNG(filename*"_gen_big.png", 5inch, 4inch),plt2)

    
end


function plot_best(df::DataFrame,dfM::DataFrame,dfP::DataFrame,filename)

    plt1=plot(df, x=:n, y=:bottle, Geom.point,Theme(background_color=colorant"white",default_color="red",plot_padding=[0mm,0mm,0mm,0mm]), Geom.smooth(method=:lm))
    draw(PNG(filename*"_bottle.png", 5inch, 2inch),plt1)


    
rename!(dfP, :generation => :generation_P)
rename!(dfM, :generation => :generation_M)
  
rename!(dfP, :bottle => :bottle_P)
rename!(dfM, :bottle => :bottle_M)

    combined_df = outerjoin(dfP, dfM, on=:n)

# Create a new column for the average or fallback value
combined_df[!,:avg_generation] = ifelse.(ismissing.(combined_df[!,:generation_P]),
                                       combined_df[!,:generation_M],  # Use dfM value if dfP is missing
                                       ifelse.(ismissing.(combined_df[!,:generation_M]),
                                               combined_df[!,:generation_P],  # Use dfP value if dfM is missing
                                               (combined_df[!,:generation_P] .+ combined_df[!,:generation_M]) ./ 2)  # Average if both present
                                      )

# Plot the original df (minGen)
plt2 = plot(df, 
    x=:n, 
    y=:generation, 
    Geom.point, 
    Theme(background_color=colorant"white", 
          default_color="red", 
          plot_padding=[0mm, 0mm, 0mm, 0mm]
    ), 
    Geom.line,
    
    # Add the average or fallback generation values as a new line and points in a light red-like color
    layer(combined_df, x=:n, y=:avg_generation, Geom.point, Geom.line, Theme(default_color=colorant"lightcoral"))
)


 
    draw(PNG(filename*"_gen_PM.png", 2.5inch, 2inch),plt2)


    
end



function plot_bestPLoS(df::DataFrame,dfM::DataFrame,dfP::DataFrame,filename)

    plt1=plot(df, x=:n, y=:bottle, Geom.point,Theme(background_color=colorant"white",default_color="red",plot_padding=[0mm,0mm,0mm,0mm]
,major_label_font="Arial",
               major_label_font_size=10pt,
               minor_label_font="Arial",
               minor_label_font_size=10pt
                                                    ), Geom.smooth(method=:lm))
    draw(PDF(filename*"_bottle.pdf", 5inch, 2inch),plt1)
    draw(PS(filename*"_bottle.eps", 5inch, 2inch),plt1)


    
rename!(dfP, :generation => :generation_P)
rename!(dfM, :generation => :generation_M)
  
rename!(dfP, :bottle => :bottle_P)
rename!(dfM, :bottle => :bottle_M)

    combined_df = outerjoin(dfP, dfM, on=:n)

# Create a new column for the average or fallback value
combined_df[!,:avg_generation] = ifelse.(ismissing.(combined_df[!,:generation_P]),
                                       combined_df[!,:generation_M],  # Use dfM value if dfP is missing
                                       ifelse.(ismissing.(combined_df[!,:generation_M]),
                                               combined_df[!,:generation_P],  # Use dfP value if dfM is missing
                                               (combined_df[!,:generation_P] .+ combined_df[!,:generation_M]) ./ 2)  # Average if both present
                                      )

# Plot the original df (minGen)
plt2 = plot(df, 
    x=:n, 
    y=:generation, 
    Geom.point, 
    Theme(background_color=colorant"white", 
          default_color="red", 
          plot_padding=[0mm, 0mm, 0mm, 0mm]
          ,major_label_font="Arial",
               major_label_font_size=10pt,
               minor_label_font="Arial",
               minor_label_font_size=10pt
    ), 
    Geom.line,
    
    # Add the average or fallback generation values as a new line and points in a light red-like color
    layer(combined_df, x=:n, y=:avg_generation, Geom.point, Geom.line, Theme(default_color=colorant"magenta"))
)


 
    draw(PDF(filename*"_gen.pdf", 2.5inch, 2inch),plt2)
    draw(PS(filename*"_gen.eps", 2.5inch, 2inch),plt2)


    
end



function plot_best(dfA::DataFrame,dfB::DataFrame,filename)

    plt1=plot(layer(dfA, x=:n, y=:bottle, Geom.point, Geom.smooth(method=:lm),Theme(default_color="red")),
              layer(dfB, x=:n, y=:bottle, Geom.point, Geom.smooth(method=:lm),Theme(default_color="blue")),
              Theme(background_color=colorant"white"))
    draw(PNG(filename*"_bottle_both.png", 5inch, 2inch),plt1)
    draw(PNG("big_"*filename*"_bottle_both.png", 8inch, 4inch),plt1)

    plt2=plot(
    layer(dfA, x=:n, y=:generation, Geom.point,Theme(default_color="red"),Geom.line),
    layer(dfB, x=:n, y=:generation, Geom.point,Theme(default_color="blue"),Geom.line),
    Theme(plot_padding=[0mm,0mm,0mm,0mm],background_color=colorant"white"),)
    draw(PNG(filename*"_gen_both.png", 2.5inch, 2inch),plt2)
    draw(PNG("big_"*filename*"_gen_both.png", 5inch, 4inch),plt2)

    
end



function plot_best(df::DataFrame,filename,interscept,slope,range)

    line1(x) = interscept+slope*x-range
    line2(x) = interscept+slope*x+range
    
    xrange = minimum(df.n):1:maximum(df.n) 


    line1_df = DataFrame(x=xrange, y=line1.(xrange))
    line2_df = DataFrame(x=xrange, y=line2.(xrange))

    
    plt1=plot(
        layer(df, x=:n, y=:bottle,
              Geom.point,Theme(default_color="red"),
              Geom.smooth(method=:lm)),
    layer(line1_df, x=:x, y=:y, Geom.line, Theme(default_color=colorant"blue")), 
        layer(line2_df, x=:x, y=:y, Geom.line, Theme(default_color=colorant"blue")),
        Theme(background_color=colorant"white")
    )
    draw(PNG(filename*"_bottle_gutters.png", 8inch, 4inch),plt1)

    
end
