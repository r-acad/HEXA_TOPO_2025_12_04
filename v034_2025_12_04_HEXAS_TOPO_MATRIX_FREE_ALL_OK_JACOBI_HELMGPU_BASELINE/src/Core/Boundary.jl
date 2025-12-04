module Boundary 

using JSON 
using SparseArrays 
using LinearAlgebra 

export get_bc_indicator, reduce_system, apply_external_forces! 

""" 
    get_affected_nodes(spec, nodes) 

Returns an array of *final* node indices affected by this BC specification `spec`. 
`spec` can have: 
  - "node" => a direct 1-based node index (or array of indices) 
  - "location" => a 3-element array describing x, y, z selection.
""" 
function get_affected_nodes(spec::AbstractDict, nodes::Matrix{Float32}) 
      
    nNodes = size(nodes, 1) 

    # 1) If user gave "node" 
    if haskey(spec, "node") 
        raw = spec["node"] 
        if isa(raw, Integer) 
            idx = clamp(raw, 1, nNodes) 
            return [idx] 
        elseif isa(raw, AbstractVector) 
            node_list = Int[] 
            for r in raw 
                push!(node_list, clamp(r, 1, nNodes)) 
            end 
            return unique(node_list) 
        else 
            error("'node' must be an integer or an array of integers") 
        end 
    end 

    # 2) If user gave "location" 
    if haskey(spec, "location") 
        loc_array = spec["location"] 
        if length(loc_array) < 3 
            error("Location specification must have at least 3 components (x,y,z)") 
        end 
        return get_nodes_by_location(loc_array, nodes) 
    end 

    error("Specification must include either 'node' or 'location'") 
end 

""" 
    get_nodes_by_location(loc_array, nodes) 

Find nodes whose (x,y,z) coordinates match the "location" pattern. 
""" 
function get_nodes_by_location(loc_array::AbstractVector, nodes::Matrix{Float32}) 
    xvals = @view nodes[:, 1] 
    yvals = @view nodes[:, 2] 
    zvals = @view nodes[:, 3] 

    xmin, xmax = extrema(xvals) 
    ymin, ymax = extrema(yvals) 
    zmin, zmax = extrema(zvals) 

    xspec = loc_array[1] 
    yspec = loc_array[2] 
    zspec = loc_array[3] 

    xmask = interpret_location_component(xspec, xvals, xmin, xmax) 
    ymask = interpret_location_component(yspec, yvals, ymin, ymax) 
    zmask = interpret_location_component(zspec, zvals, zmin, zmax) 

    return findall(xmask .& ymask .& zmask) 
end 

""" 
    interpret_location_component(spec, coords, cmin, cmax) 

Returns a Bool array indicating which nodes match 'spec' in this dimension. 
""" 
function interpret_location_component(spec, 
                                      coords::AbstractVector{Float32}, 
                                      cmin::Float32, cmax::Float32) 
    nNodes = length(coords) 
    mask = falses(nNodes) 
      
    # Dynamic tolerance based on domain size
    tol = Float32(1e-4) * max(Float32(1.0), abs(cmax - cmin)) 

    if spec == ":" 
        return trues(nNodes) 
          
    elseif isa(spec, Number) 
        val = resolve_coordinate_value(spec, cmin, cmax)
          
        @inbounds for i in 1:nNodes 
            if abs(coords[i] - val) <= tol 
                mask[i] = true 
            end 
        end 

    elseif isa(spec, String) && endswith(spec, "%") 
        # e.g. "50%" => fraction 0.5 
        frac = parse(Float32, replace(spec, "%"=>"")) / Float32(100.0) 
        frac = clamp(frac, Float32(0.0), Float32(1.0)) 
        val = Float32(cmin + frac*(cmax - cmin)) 
          
        @inbounds for i in 1:nNodes 
            if abs(coords[i] - val) <= tol 
                mask[i] = true 
            end 
        end 
          
    else 
        error("Invalid location component: $spec") 
    end 

    return mask 
end 

"""
    resolve_coordinate_value(spec, cmin, cmax)

Helper to convert a spec (number or percentage) into an absolute float coordinate.
"""
function resolve_coordinate_value(spec, cmin::Float32, cmax::Float32)
    if isa(spec, Number)
        # If the number is between 0 and 1, assume it's a normalized coordinate? 
        # The logic here assumes if it's explicitly a float 0.0-1.0 it might be a ratio, 
        # BUT standard FEM coordinates often fall in this range. 
        # The safer assumption in this codebase (based on context) is:
        # If it's a string with %, it's relative. If it's a number, it's absolute.
        # However, to maintain existing logic if user passes 0.5 as a float meaning 50%:
        if spec >= Float32(0.0) && spec <= Float32(1.0) 
            return Float32(cmin + spec*(cmax - cmin)) 
        else 
            return Float32(spec) 
        end
    elseif isa(spec, String) && endswith(spec, "%")
        frac = parse(Float32, replace(spec, "%"=>"")) / Float32(100.0) 
        frac = clamp(frac, Float32(0.0), Float32(1.0)) 
        return Float32(cmin + frac*(cmax - cmin))
    end
    return Float32(cmin) 
end

""" 
    get_bc_indicator(nNodes, nodes, bc_data; T=Float32) 

Generates an (nNodes x 3) matrix with 1.0 where a boundary condition is applied.
""" 
function get_bc_indicator(nNodes::Int, 
                          nodes::Matrix{Float32}, 
                          bc_data::Vector{Any};  
                          T::Type{<:AbstractFloat} = Float32) 

    bc_indicator = zeros(T, nNodes, 3) 
      
    for bc in bc_data 
        dofs = bc["DoFs"] 
          
        for dof in dofs 
            if dof < 1 || dof > 3 
                error("Invalid DoF index: $dof (must be 1..3).") 
            end 
        end 

        affected = get_affected_nodes(bc, nodes) 
        for nd in affected 
            for d in dofs 
                bc_indicator[nd, d] = one(T) 
            end 
        end 
    end 

    return bc_indicator 
end 

""" 
    reduce_system(K, F, bc_data, nodes, elements) 

Use bc_data + final node array to mark constrained DOFs, 
then zero them out of the stiffness/force system. 
""" 
function reduce_system(K::SparseMatrixCSC{Float32,Int}, 
                       F::Vector{Float32}, 
                       bc_data::Vector{Any},  
                       nodes::Matrix{Float32}, 
                       elements::Matrix{Int}) 

    nNodes = size(nodes, 1) 
    ndof   = 3*nNodes 
    constrained = falses(ndof) 

    for bc in bc_data 
        dofs = bc["DoFs"] 
        affected = get_affected_nodes(bc, nodes) 
          
        for nd in affected 
            for d in dofs 
                gdof = 3*(nd-1) + d 
                constrained[gdof] = true 
                F[gdof] = Float32(0.0)  
            end 
        end 
    end 

    free_indices = findall(!, constrained) 
    K_reduced = K[free_indices, free_indices] 
    F_reduced = F[free_indices] 
      
    return K_reduced, F_reduced, free_indices 
end 

""" 
    find_nearest_node(target_coords, nodes)

Finds the index of the node closest to `target_coords`.
Tie-breaker: Largest X, then Largest Y, then Largest Z.
"""
function find_nearest_node(target_coords::Vector{Float32}, nodes::Matrix{Float32})
    nNodes = size(nodes, 1)
    best_idx = -1
    min_dist_sq = Inf32

    @inbounds for i in 1:nNodes
        dx = nodes[i, 1] - target_coords[1]
        dy = nodes[i, 2] - target_coords[2]
        dz = nodes[i, 3] - target_coords[3]
        dist_sq = dx*dx + dy*dy + dz*dz

        if dist_sq < (min_dist_sq - 1e-9)
            min_dist_sq = dist_sq
            best_idx = i
        elseif abs(dist_sq - min_dist_sq) <= 1e-9
            # Tie-breaker: "Largest coordinates"
            if nodes[i, 1] > nodes[best_idx, 1]
                best_idx = i
            elseif nodes[i, 1] == nodes[best_idx, 1]
                if nodes[i, 2] > nodes[best_idx, 2]
                    best_idx = i
                elseif nodes[i, 2] == nodes[best_idx, 2]
                    if nodes[i, 3] > nodes[best_idx, 3]
                        best_idx = i
                    end
                end
            end
        end
    end
    return best_idx
end

""" 
    apply_external_forces!(F, forces_data, nodes, elements) 

Look up affected nodes by coordinate or direct node index, 
then apply the force. 
If strict location matching fails, it finds the NEAREST node (closest distance, 
breaking ties with largest coordinates).
""" 
function apply_external_forces!(F::Vector{T}, 
                                 forces_data::Vector{Any},  
                                 nodes::Matrix{Float32}, 
                                 elements::Matrix{Int}) where T<:AbstractFloat 

    x_bounds = extrema(view(nodes, :, 1))
    y_bounds = extrema(view(nodes, :, 2))
    z_bounds = extrema(view(nodes, :, 3))

    for force in forces_data 
        affected_nodes = get_affected_nodes(force, nodes) 

        # If strict matching failed, try finding nearest node for location-based forces
        if isempty(affected_nodes) && haskey(force, "location")
            loc = force["location"]
            
            # Check if this is a point specification (no wildcards/ranges)
            is_point_spec = all(x -> x != ":", loc)
            
            if is_point_spec
                tx = resolve_coordinate_value(loc[1], x_bounds[1], x_bounds[2])
                ty = resolve_coordinate_value(loc[2], y_bounds[1], y_bounds[2])
                tz = resolve_coordinate_value(loc[3], z_bounds[1], z_bounds[2])
                target = Float32[tx, ty, tz]

                # @warn "No exact node found for Force at $(loc) -> $target. Searching for nearest node..."
                nearest_idx = find_nearest_node(target, nodes)
                
                if nearest_idx != -1
                    affected_nodes = [nearest_idx]
                    # found_node = nodes[nearest_idx, :]
                    # @info "  -> Applied force to nearest node #$nearest_idx at $found_node"
                end
            end
        end
        
        if isempty(affected_nodes) 
            # @warn "No nodes found for force specification (even after nearest search); skipping this force: $(force)" 
            continue 
        end 
          
        f_raw = force["F"] 
        f_arr = zeros(T, 3) 
        len_to_copy = min(length(f_raw), 3) 
        f_arr[1:len_to_copy] = T.(f_raw[1:len_to_copy])  

        # If user gave "location", we spread the total force among the matched nodes 
        scale_factor = haskey(force, "location") ? (one(T) / length(affected_nodes)) : one(T) 

        for nd in affected_nodes 
            for i in 1:3 
                global_dof = 3*(nd-1) + i 
                F[global_dof] += scale_factor * f_arr[i] 
            end 
        end 
    end 

    return F 
end 

end