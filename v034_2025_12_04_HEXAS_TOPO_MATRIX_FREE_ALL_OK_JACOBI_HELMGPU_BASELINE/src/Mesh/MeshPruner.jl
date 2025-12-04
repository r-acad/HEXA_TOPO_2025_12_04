// # FILE: .\MeshPruner.jl";
module MeshPruner

using LinearAlgebra
using SparseArrays

export prune_system, reconstruct_full_solution

"""
    prune_system(nodes, elements, density, threshold, bc_indicator, F)

Removes elements with density < threshold. 
Removes nodes that are no longer attached to any active element.
Remaps Boundary Conditions and Forces to the new reduced indices.

Returns a NamedTuple containing the reduced system and the mapping arrays.
"""
function prune_system(nodes::Matrix{Float32}, 
                      elements::Matrix{Int}, 
                      density::Vector{Float32}, 
                      threshold::Float32, 
                      bc_indicator::Matrix{Float32}, 
                      F::Vector{Float32})

    nElem = size(elements, 1)
    nNodes = size(nodes, 1)

    # 1. Identify Active Elements
    # Elements with density strictly greater than threshold are kept.
    # Usually threshold is slightly higher than min_density (e.g. 1.1 * min_density)
    active_mask = density .> threshold
    active_element_indices = findall(active_mask)
    nActiveElem = length(active_element_indices)

    if nActiveElem == 0
        error("MeshPruner: No active elements found (Threshold: $threshold). System is empty.")
    end

    # 2. Identify Active Nodes (Nodes used by at least one active element)
    active_nodes_mask = falses(nNodes)
    
    # Only iterate through the active elements to find active nodes
    for e in active_element_indices
        for i in 1:8
            node_idx = elements[e, i]
            active_nodes_mask[node_idx] = true
        end
    end

    # 3. Create Node Mappings
    # old_to_new: maps Global ID -> Reduced ID (0 if removed)
    # new_to_old: maps Reduced ID -> Global ID
    old_to_new_node_map = zeros(Int, nNodes)
    new_to_old_node_map = Int[]
    
    current_new_id = 1
    for i in 1:nNodes
        if active_nodes_mask[i]
            old_to_new_node_map[i] = current_new_id
            push!(new_to_old_node_map, i)
            current_new_id += 1
        end
    end
    
    nActiveNodes = length(new_to_old_node_map)

    # 4. Build Reduced Node Array
    reduced_nodes = nodes[new_to_old_node_map, :]

    # 5. Build Reduced Element Connectivity
    reduced_elements = Matrix{Int}(undef, nActiveElem, 8)
    for (i, old_e_idx) in enumerate(active_element_indices)
        for j in 1:8
            old_node = elements[old_e_idx, j]
            new_node = old_to_new_node_map[old_node]
            reduced_elements[i, j] = new_node
        end
    end

    # 6. Build Reduced BC Indicator
    reduced_bc = bc_indicator[new_to_old_node_map, :]

    # 7. Build Reduced Force Vector
    reduced_ndof = nActiveNodes * 3
    reduced_F = zeros(Float32, reduced_ndof)
    
    # Map forces: Global Force -> Reduced Force
    # We loop over the NEW nodes to pull data from the OLD arrays
    for (new_idx, old_idx) in enumerate(new_to_old_node_map)
        base_old = 3 * (old_idx - 1)
        base_new = 3 * (new_idx - 1)
        reduced_F[base_new+1] = F[base_old+1]
        reduced_F[base_new+2] = F[base_old+2]
        reduced_F[base_new+3] = F[base_old+3]
    end

    # 8. Filter Density Array (needed for stiffness assembly)
    reduced_density = density[active_element_indices]

    return (
        nodes = reduced_nodes,
        elements = reduced_elements,
        bc_indicator = reduced_bc,
        F = reduced_F,
        density = reduced_density,
        old_to_new_node_map = old_to_new_node_map,
        new_to_old_node_map = new_to_old_node_map,
        active_element_indices = active_element_indices,
        n_original_nodes = nNodes,
        n_original_elems = nElem
    )
end

"""
    reconstruct_full_solution(u_reduced, new_to_old_node_map, n_full_nodes)

Maps the displacement vector from the reduced system back to the full system size.
Nodes that were removed (voids) will have 0.0 displacement.
"""
function reconstruct_full_solution(u_reduced::Vector{Float32}, 
                                   new_to_old_node_map::Vector{Int}, 
                                   n_full_nodes::Int)
    
    ndof_full = n_full_nodes * 3
    u_full = zeros(Float32, ndof_full)

    # Map: Reduced U -> Full U
    for (new_node_idx, old_node_idx) in enumerate(new_to_old_node_map)
        base_new = 3 * (new_node_idx - 1)
        base_old = 3 * (old_node_idx - 1)

        u_full[base_old+1] = u_reduced[base_new+1]
        u_full[base_old+2] = u_reduced[base_new+2]
        u_full[base_old+3] = u_reduced[base_new+3]
    end

    return u_full
end

end