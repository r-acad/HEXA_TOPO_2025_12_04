module MeshUtilities 
 
export inside_sphere, inside_box, element_centroid,
       check_element_quality, fix_inverted_elements!, 
       calculate_element_quality 
 
using LinearAlgebra 
 
""" 
    element_centroid(e, nodes, elements) 
 
Computes the centroid of element `e` given the node coordinates. 
""" 
function element_centroid(e::Int, nodes::Matrix{Float32}, elements::Matrix{Int}) 
    conn = view(elements, e, :) 
    # Unroll loop for slight efficiency on Hex8
    @inbounds begin
        c1 = nodes[conn[1], :]
        c2 = nodes[conn[2], :]
        c3 = nodes[conn[3], :]
        c4 = nodes[conn[4], :]
        c5 = nodes[conn[5], :]
        c6 = nodes[conn[6], :]
        c7 = nodes[conn[7], :]
        c8 = nodes[conn[8], :]
    end
    return (c1 .+ c2 .+ c3 .+ c4 .+ c5 .+ c6 .+ c7 .+ c8) ./ 8.0f0  
end 

""" 
    inside_sphere(pt, center, diam) 
Return true if point `pt` is inside a sphere of diameter `diam` at `center`. 
""" 
function inside_sphere(pt::AbstractVector, center::Tuple{Float32,Float32,Float32}, diam::Float32) 
    r = diam / 2f0 
    return norm(pt .- collect(center)) <= r 
end 

""" 
    inside_box(pt, center, side) 
Return true if point `pt` is inside a cube of side `side` centered at `center`. 
""" 
function inside_box(pt::AbstractVector, center::Tuple{Float32,Float32,Float32}, side::Float32) 
    half = side / 2f0 
    return abs(pt[1] - center[1]) <= half && 
           abs(pt[2] - center[2]) <= half && 
           abs(pt[3] - center[3]) <= half 
end 
 
""" 
    check_element_quality(nodes, elements) -> poor_elements 
Mark which elements are degenerate, etc. (Placeholder for future expansion)
""" 
function check_element_quality(nodes::Matrix{Float32}, elements::Matrix{Int}) 
    nElem = size(elements,1) 
    poor_elements = Int[] 
    # Future implementation: check scaled Jacobian
    return poor_elements 
end 
 
""" 
    fix_inverted_elements!(nodes, elements) -> (fixed_count, warning_count) 
Swap node ordering to fix negative Jacobians. (Placeholder)
""" 
function fix_inverted_elements!(nodes::Matrix{Float32}, elements::Matrix{Int}) 
    return (0, 0) 
end 
 
""" 
    calculate_element_quality(nodes, elements) 
Returns (aspect_ratios, min_jacobians) (Placeholder)
""" 
function calculate_element_quality(nodes::Matrix{Float32}, elements::Matrix{Int}) 
    nElem = size(elements, 1)
    return zeros(Float32, nElem), zeros(Float32, nElem) 
end 
 
end