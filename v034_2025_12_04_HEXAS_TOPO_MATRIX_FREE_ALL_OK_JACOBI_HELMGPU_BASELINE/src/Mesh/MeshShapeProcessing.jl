module MeshShapeProcessing 
 
export apply_geometric_modifiers!
 
using LinearAlgebra 
using Base.Threads
using ..MeshUtilities    
 
""" 
    apply_geometric_modifiers!(density, nodes, elements, shapes, target_value)

Iterates over elements and applies the `target_value` to the density field 
if the element centroid is inside the defined shape.
""" 
function apply_geometric_modifiers!(density::Vector{Float32}, 
                                    nodes::Matrix{Float32}, 
                                    elements::Matrix{Int}, 
                                    shapes::Vector{Any},
                                    target_value::Float32)
    
    if isempty(shapes)
        return
    end

    nElem = size(elements, 1)
    
    # We parallelize this loop for performance on large meshes
    Threads.@threads for e in 1:nElem
        # Use centralized centroid calculation
        centroid = MeshUtilities.element_centroid(e, nodes, elements)
        
        for shape in shapes
            shape_type = lowercase(get(shape, "type", ""))
            is_inside = false

            if shape_type == "sphere"
                if haskey(shape, "center") && haskey(shape, "diameter")
                    center = tuple(Float32.(shape["center"])...)
                    diam   = Float32(shape["diameter"])
                    is_inside = MeshUtilities.inside_sphere(centroid, center, diam)
                end
            elseif shape_type == "box"
                if haskey(shape, "center") && haskey(shape, "side")
                    center = tuple(Float32.(shape["center"])...)
                    side   = Float32(shape["side"])
                    is_inside = MeshUtilities.inside_box(centroid, center, side)
                end
            end

            if is_inside
                density[e] = target_value
                # If we are inside one shape, we apply and move to next element
                # (Last shape in list wins if they overlap, or use 'break' to prioritize first)
                break 
            end
        end
    end
end 
 
end