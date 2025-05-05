module SevenSeg

using Images, FileIO

export make_7seg, save_7seg

"""
    make_7seg(d; size=28, margin=2, thickness=4)

Return a `size×size` Float32 matrix (values 0.0/1.0) of the 7‑segment display for digit `d` (0–9).
You can override the `margin` and `thickness` of the segments.
"""
function make_7seg(d::Integer; size::Int=28, margin::Int=2, thickness::Int=4)
    @assert 0 ≤ d ≤ 9 "Digit must be in 0–9"
    # Compute vertical segment height
    h = Int((size - 2*margin - 3*thickness) ÷ 2)
    @assert h > 0 "Choose smaller margin/thickness or larger size"

    # Precompute segment rectangles as (row_range, col_range)
    s = size; m = margin; t = thickness
    # horizontal segments
    segs = Dict{Int,Tuple{UnitRange,UnitRange}}(
      # 1 = top
      1 => ( (m+1)           :(m+t),       (m+t+1)       :(s-m-t)      ),
      # 4 = middle
      4 => ( (m+t+h+1)       :(m+2*t+h),  (m+t+1)       :(s-m-t)      ),
      # 7 = bottom
      7 => ( (m+2*t+2*h+1)   :(m+3*t+2*h),(m+t+1)       :(s-m-t)      ),
      # vertical segments
      # 2 = upper‑left
      2 => ( (m+t+1) :(m+t+h),   (m+1)   :(m+t)    ),
      # 3 = upper‑right
      3 => ( (m+t+1) :(m+t+h),   (s-m-t+1):(s-m) ),
      # 5 = lower‑left
      5 => ( (m+2*t+h+1) :(m+2*t+2*h), (m+1)  :(m+t)    ),
      # 6 = lower‑right
      6 => ( (m+2*t+h+1) :(m+2*t+2*h), (s-m-t+1):(s-m) )
    )

    # which segments are lit for each digit
    digit_map = Dict(
      0 => [1,2,3,5,6,7],
      1 => [3,6],
      2 => [1,3,4,5,7],
      3 => [1,3,4,6,7],
      4 => [2,3,4,6],
      5 => [1,2,4,6,7],
      6 => [1,2,4,5,6,7],
      7 => [1,3,6],
      8 => [1,2,3,4,5,6,7],
      9 => [1,2,3,4,6,7]
    )

    # build the image
    img = zeros(Float32, size, size)
    for seg in digit_map[d]
      (rs, cs) = segs[seg]
      @inbounds img[rs, cs] .= 1f0
    end
    return img
end

function make_seg(seg::Integer; size::Int=28, margin::Int=2, thickness::Int=4)
    h = Int((size - 2*margin - 3*thickness) ÷ 2)
    @assert h > 0 "Choose smaller margin/thickness or larger size"

    # Precompute segment rectangles as (row_range, col_range)
    s = size; m = margin; t = thickness
    # horizontal segments
    segs = Dict{Int,Tuple{UnitRange,UnitRange}}(
      # 1 = top
      1 => ( (m+1)           :(m+t),       (m+t+1)       :(s-m-t)      ),
      # 4 = middle
      4 => ( (m+t+h+1)       :(m+2*t+h),  (m+t+1)       :(s-m-t)      ),
      # 7 = bottom
      7 => ( (m+2*t+2*h+1)   :(m+3*t+2*h),(m+t+1)       :(s-m-t)      ),
      # vertical segments
      # 2 = upper‑left
      2 => ( (m+t+1) :(m+t+h),   (m+1)   :(m+t)    ),
      # 3 = upper‑right
      3 => ( (m+t+1) :(m+t+h),   (s-m-t+1):(s-m) ),
      # 5 = lower‑left
      5 => ( (m+2*t+h+1) :(m+2*t+2*h), (m+1)  :(m+t)    ),
      # 6 = lower‑right
      6 => ( (m+2*t+h+1) :(m+2*t+2*h), (s-m-t+1):(s-m) )
    )

    # build the image
    img = zeros(Float32, size, size)
    (rs, cs) = segs[seg]
    @inbounds img[rs, cs] .= 1f0
    return img
end



"""
    save_7seg(d; filename="digit_d.png", scale=1)

Generate the 7‑segment image for digit `d` and save it as PNG.
If `scale>1`, each original pixel is expanded to a `scale×scale` block.
"""
function save_7seg(d::Integer; filename::AbstractString = "digit_$(d).png", scale::Int=1)
    img = make_7seg(d)
    # optionally block‑repeat via kron
    if scale != 1
        img = kron(img, fill(1f0, scale, scale))
    end
    # convert to grayscale image and save
    save(filename, Gray.(img))
end

function save_img(img, filename, scale)
    # optionally block‑repeat via kron
    if scale != 1
        img = kron(img, fill(1f0, scale, scale))
    end
    # convert to grayscale image and save
    save(filename, Gray.(img))
end


end # module
