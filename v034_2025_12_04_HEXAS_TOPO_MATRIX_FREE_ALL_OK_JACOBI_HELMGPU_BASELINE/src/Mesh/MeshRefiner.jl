// # FILE: .\MeshRefiner.jl";
module MeshRefiner

using LinearAlgebra
using Printf
using ..Mesh
using ..Helpers

export refine_mesh_and_fields

"""
    refine_mesh_and_fields(nodes, elements, density, current_dims, 
                           target_active_count, domain_bounds)

Generates a NEW mesh from scratch to match a target active element count.
It does NOT subdivide existing elements. Instead, it calculates a new global
dx, dy, dz and interpolates the old density field onto the new centroids.

Returns: (new_nodes, new_elements, new_density, new_dims)
"""
function refine_mesh_and_fields(nodes::Matrix{Float32}, 
                                elements::Matrix{Int}, 
                                density::Vector{Float32}, 
                                current_dims::Tuple{Int, Int, Int},
                                target_active_count::Int,
                                domain_bounds::NamedTuple) # (min_pt, len_x, len_y, len_z)

    # 1. Calculate current active ratio
    # We need to know what % of the volume is solid to estimate the required total mesh size.
    # If we want `target_active_count` solids, and `active_ratio` is 0.2, 
    # we need a total mesh size of `target_active_count / 0.2`.
    
    n_total_old = length(density)
    n_active_old = count(d -> d > 0.001f0, density)
    active_ratio = max(0.01, n_active_old / n_total_old) # Avoid divide by zero
    
    # Estimate required total elements to hit the target active count
    required_total_elements = round(Int, target_active_count / active_ratio)
    
    # Safety clamp: Don't let it grow wildly beyond 5x the previous size in one step
    required_total_elements = min(required_total_elements, n_total_old * 8)
    required_total_elements = max(required_total_elements, n_total_old) # Don't shrink

    println("\n[MeshRefiner] Re-meshing...")
    println("  Target Active: $target_active_count")
    println("  Active Ratio:  $(round(active_ratio*100, digits=1))%")
    println("  New Total Est: $required_total_elements")

    # 2. Calculate New Dimensions (Nx, Ny, Nz)
    len_x, len_y, len_z = domain_bounds.len_x, domain_bounds.len_y, domain_bounds.len_z
    
    # Use the same logic as initial mesh generation to distribute elements
    new_nx, new_ny, new_nz, new_dx, new_dy, new_dz, actual_count = 
        Helpers.calculate_element_distribution(len_x, len_y, len_z, required_total_elements)
        
    # 3. Generate New Mesh
    new_nodes, new_elements, new_dims = Mesh.generate_mesh(
        new_nx, new_ny, new_nz;
        dx=new_dx, dy=new_dy, dz=new_dz
    )
    
    # Offset nodes to absolute coordinates
    min_pt = domain_bounds.min_pt
    new_nodes[:, 1] .+= min_pt[1]
    new_nodes[:, 2] .+= min_pt[2]
    new_nodes[:, 3] .+= min_pt[3]
    
    println("  New Grid: $(new_nx)x$(new_ny)x$(new_nz) = $actual_count elements")
    println("  New Resolution: $(new_dx) x $(new_dy) x $(new_dz)")

    # 4. Interpolate Density Field (Coordinate Mapping)
    # We iterate over the NEW elements, find their centroid, and look up the value
    # from the OLD mesh.
    
    n_new_total = size(new_elements, 1)
    new_density = zeros(Float32, n_new_total)
    
    # Old grid parameters for lookup
    old_nx = current_dims[1] - 1
    old_ny = current_dims[2] - 1
    old_nz = current_dims[3] - 1
    
    # To map quickly, we need the bounds and old spacing.
    # Assuming the domain is filled from min_pt to max_pt.
    # Ideally, we pass old_dx, old_dy... but we can infer:
    old_dx = len_x / old_nx
    old_dy = len_y / old_ny
    old_dz = len_z / old_nz

    # Pre-calculate 3D structure of density for fast lookup
    # Since density is a linear vector, we treat it as 3D array logic conceptually
    
    Threads.@threads for e_new in 1:n_new_total
        # Calculate centroid of new element e_new
        # We can do this analytically without accessing nodes array for speed
        # Index -> (ix, iy, iz)
        # e = ix + (iy-1)*nx + (iz-1)*nx*ny
        
        iz = div(e_new - 1, new_nx * new_ny) + 1
        rem_z = (e_new - 1) % (new_nx * new_ny)
        iy = div(rem_z, new_nx) + 1
        ix = rem_z % new_nx + 1
        
        # Centroid coordinate (local to box 0,0,0)
        cx = (ix - 0.5f0) * new_dx
        cy = (iy - 0.5f0) * new_dy
        cz = (iz - 0.5f0) * new_dz
        
        # Map to Old Grid Indices
        # old_idx = floor(cx / old_dx) + 1
        old_ix = clamp(floor(Int, cx / old_dx) + 1, 1, old_nx)
        old_iy = clamp(floor(Int, cy / old_dy) + 1, 1, old_ny)
        old_iz = clamp(floor(Int, cz / old_dz) + 1, 1, old_nz)
        
        # Convert 3D old index to linear old index
        old_linear = old_ix + (old_iy - 1)*old_nx + (old_iz - 1)*old_nx*old_ny
        
        # Assign value (Nearest Neighbor)
        # Note: For smoother results, trilinear could be used, but NN is safer for topology
        new_density[e_new] = density[old_linear]
    end
    
    return new_nodes, new_elements, new_density, new_dims
end

end