using Images, FileIO, Plots, Plots.Measures

# Read filenames from text file
filenames = readlines("filenames.txt")

# Define labels and their positions
labels = ["1", "2", "3", "4", "5", "6", "7"]
label_cols = [6, 2, 3, 1, 4, 9, 5]
col_width = 13

# Loop through each file
for filename in filenames
    # Load the image
    img = load(filename)

    # Get image dimensions
    height, width = size(img)

    # Start plot
    plot(;
        xlim=(0, width),
        ylim=(0, height),
        framestyle=:none,
        axis=nothing,
        legend=false,
        aspect_ratio=:equal,
    )

    # Display image
    plot!(img, yflip=true)

    # Annotate columns
    for (lbl, col_idx) in zip(labels, label_cols)
        x_pos = (col_idx - 0.5) * col_width 
        annotate!(x_pos, -4, text(lbl, :black, :center, 14))
    end

    # Set margin only on the top
    plot!(top_margin= 20Plots.px)

    # Save the annotated image with a modified filename
    savefig("annotated_" * filename)
end
