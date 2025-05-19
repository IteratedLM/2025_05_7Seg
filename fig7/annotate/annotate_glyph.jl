using Images, FileIO, Plots

# Load the image (glyph "8")
img = load("glyph_eight.png")

# Get dimensions of the image
height, width = size(img)

# Set up the plot
plot(;
    xlim=(0, width),
    ylim=(0, height),
    framestyle=:none,
    axis=nothing,
    legend=false,
    aspect_ratio=:equal,
)

# Display the image as background, flipped vertically
plot!(img, yflip=true)

# Segment numbering and initial coordinates
x_center = width / 2
x_left = width * 0.145
x_right = width * 0.855

y_top = height * 0.145
y_tm  = height * 0.325
y_mid = height * 0.5
y_bm  = height * 0.68
y_bot = height * 0.869

# Coordinates for each segment following constraints
segment_positions = Dict(
    1 => (x_center, y_top),
    2 => (x_left, y_tm),
    3 => (x_right, y_tm),
    4 => (x_center, y_mid),
    5 => (x_left, y_bm),
    6 => (x_right, y_bm),
    7 => (x_center, y_bot),
)

# Annotate segments
for (seg, (x, y)) in segment_positions
    annotate!(x, y, text("$seg", :white, :center, 24, "sans-serif"; fontweight="bold"))
end

# Adjust margins if needed

# Save annotated figure
savefig("annotated_glyph_8.png")
