
module TopologyOptimization 

using LinearAlgebra
using SparseArrays
using Printf  
using Statistics 
using SuiteSparse 
using CUDA
using ..Element
using ..Mesh
using ..GPUHelmholtz # NEW MODULE

export update_density!, reset_filter_cache!

# Cache for CPU solver (legacy/fallback)
mutable struct FilterCache
    is_initialized::Bool
    radius::Float32
    K_filter::SuiteSparse.CHOLMOD.Factor{Float64} 
    FilterCache() = new(false, 0.0f0)
end

const GLOBAL_FILTER_CACHE = FilterCache()

function reset_filter_cache!()
    GLOBAL_FILTER_CACHE.is_initialized = false
end

# --- CPU Assemby (Legacy) ---
function assemble_helmholtz_system(nElem_x, nElem_y, nElem_z, dx, dy, dz, R)
    nNodes = (nElem_x + 1) * (nElem_y + 1) * (nElem_z + 1)
    nElem = nElem_x * nElem_y * nElem_z
    Ke_local, Me_local = Element.get_scalar_canonical_matrices(dx, dy, dz)
    
    entries_per_elem = 64 
    total_entries = nElem * entries_per_elem
    I_vec = Vector{Int32}(undef, total_entries)
    J_vec = Vector{Int32}(undef, total_entries)
    V_vec = Vector{Float64}(undef, total_entries)
    
    idx_counter = 0
    nx, ny = nElem_x + 1, nElem_y + 1
    Re_local = (R^2) .* Ke_local .+ Me_local
    
    for k in 1:nElem_z, j in 1:nElem_y, i in 1:nElem_x
        n1 = i       + (j-1)*nx       + (k-1)*nx*ny
        n2 = (i+1) + (j-1)*nx       + (k-1)*nx*ny
        n3 = (i+1) + j*nx           + (k-1)*nx*ny
        n4 = i       + j*nx           + (k-1)*nx*ny
        n5 = i       + (j-1)*nx       + k*nx*ny
        n6 = (i+1) + (j-1)*nx       + k*nx*ny
        n7 = (i+1) + j*nx           + k*nx*ny
        n8 = i       + j*nx           + k*nx*ny
        nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
        
        for r in 1:8
            row = nodes[r]
            for c in 1:8
                col = nodes[c]
                idx_counter += 1
                I_vec[idx_counter] = row
                J_vec[idx_counter] = col
                V_vec[idx_counter] = Float64(Re_local[r, c])
            end
        end
    end
    K_global = sparse(I_vec, J_vec, V_vec, nNodes, nNodes)
    n = size(K_global, 1)
    K_global = K_global + sparse(1:n, 1:n, fill(1e-9, n), n, n) # Regularization
    return cholesky(K_global)
end

function apply_helmholtz_filter_cpu(field_elem::Vector{Float32}, F_fact, nElem_x, nElem_y, nElem_z, dx, dy, dz)
    nx, ny = nElem_x + 1, nElem_y + 1
    nNodes = (nElem_x + 1) * (nElem_y + 1) * (nElem_z + 1)
    elem_vol = dx * dy * dz
    nodal_weight = elem_vol / 8.0f0
    RHS = zeros(Float64, nNodes)
    
    idx_e = 1
    for k in 1:nElem_z, j in 1:nElem_y, i in 1:nElem_x
        val = Float64(field_elem[idx_e] * nodal_weight)
        n1 = i + (j-1)*nx + (k-1)*nx*ny
        n2 = n1 + 1
        n3 = n1 + nx + 1
        n4 = n1 + nx
        n5 = n1 + nx*ny
        n6 = n2 + nx*ny
        n7 = n3 + nx*ny
        n8 = n4 + nx*ny
        
        RHS[n1] += val; RHS[n2] += val; RHS[n3] += val; RHS[n4] += val;
        RHS[n5] += val; RHS[n6] += val; RHS[n7] += val; RHS[n8] += val;
        idx_e += 1
    end
    
    nodal_filtered = F_fact \ RHS
    filtered_elem = zeros(Float32, length(field_elem))
    
    idx_e = 1
    for k in 1:nElem_z, j in 1:nElem_y, i in 1:nElem_x
        n1 = i + (j-1)*nx + (k-1)*nx*ny
        # ... (Same indexing as above)
        n2 = n1 + 1; n3 = n1 + nx + 1; n4 = n1 + nx;
        n5 = n1 + nx*ny; n6 = n2 + nx*ny; n7 = n3 + nx*ny; n8 = n4 + nx*ny;
        
        sum_nodes = nodal_filtered[n1] + nodal_filtered[n2] + nodal_filtered[n3] + nodal_filtered[n4] +
                    nodal_filtered[n5] + nodal_filtered[n6] + nodal_filtered[n7] + nodal_filtered[n8]
        filtered_elem[idx_e] = Float32(sum_nodes / 8.0)
        idx_e += 1
    end
    return filtered_elem
end

# --- MAIN UPDATE FUNCTION ---

function update_density!(density::Vector{Float32}, 
                         l1_stress_norm_field::Vector{Float32}, 
                         protected_elements_mask::BitVector, 
                         E::Float32, 
                         l1_stress_allowable::Float32, 
                         iter::Int, 
                         number_of_iterations::Int, 
                         original_density::Vector{Float32}, 
                         min_density::Float32,  
                         max_density::Float32, 
                         config::Dict,
                         elements::Matrix{Int}, # Added Argument
                         is_annealing::Bool=false) 

    nElem = length(density)
    max_change = 0.0f0 
    
    opt_params = config["optimization_parameters"]
    geom_params = config["geometry"]

    # --- Radius Calculation ---
    R_init_perc = Float32(get(opt_params, "filter_R_init_perc", 0.0f0))
    R_interm_perc = Float32(get(opt_params, "filter_R_interm_perc", 0.0f0))
    R_final_perc = Float32(get(opt_params, "filter_R_final_perc", 0.0f0))
    R_interm_iter_perc = Float32(get(opt_params, "filter_R_interm_iter_perc", 50.0f0)) / 100.0f0
    
    R_manual = Float32(get(opt_params, "filter_radius", 0.0f0))
    R_length = Float32(R_manual) 
    
    if R_init_perc > 1e-6
        max_domain_dim = geom_params["max_domain_dim"]
        R_init_length = R_init_perc / 100.0f0 * max_domain_dim
        R_interm_length = R_interm_perc / 100.0f0 * max_domain_dim
        R_final_length = R_final_perc / 100.0f0 * max_domain_dim

        iter_interm = max(1, round(Int, R_interm_iter_perc * number_of_iterations))
        calc_iter = min(iter, number_of_iterations)

        if calc_iter <= iter_interm
            t = (iter_interm > 1) ? (calc_iter - 1) / (iter_interm - 1) : 0.0
            R_length = R_init_length * (1 - t) + R_interm_length * t
        else 
            t = (number_of_iterations > iter_interm) ? (calc_iter - iter_interm) / (number_of_iterations - iter_interm) : 0.0
            R_length = R_interm_length * (1 - t) + R_final_length * t
        end
        R_length = R_length / 2.5f0
    end
    
    # --- Filtering ---
    filtered_l1_stress = l1_stress_norm_field
    
    if R_length > 1e-4
        nElem_x = Int(geom_params["nElem_x_computed"]) 
        nElem_y = Int(geom_params["nElem_y_computed"])
        nElem_z = Int(geom_params["nElem_z_computed"])
        dx = Float32(geom_params["dx_computed"])
        dy = Float32(geom_params["dy_computed"])
        dz = Float32(geom_params["dz_computed"])

        # GPU Dispatch Decision
        if CUDA.functional()
            # USE GPU MATRIX-FREE FILTER
            filtered_l1_stress = GPUHelmholtz.apply_gpu_filter!(
                l1_stress_norm_field, 
                nElem_x, nElem_y, nElem_z, 
                dx, dy, dz, R_length, elements
            )
        else
            # USE CPU DIRECT SOLVER (Fallback)
            if !GLOBAL_FILTER_CACHE.is_initialized || abs(GLOBAL_FILTER_CACHE.radius - R_length) > 1e-5
                fact = assemble_helmholtz_system(nElem_x, nElem_y, nElem_z, dx, dy, dz, R_length)
                GLOBAL_FILTER_CACHE.K_filter = fact
                GLOBAL_FILTER_CACHE.radius = R_length
                GLOBAL_FILTER_CACHE.is_initialized = true
            end
            
            filtered_l1_stress = apply_helmholtz_filter_cpu(
                l1_stress_norm_field, 
                GLOBAL_FILTER_CACHE.K_filter, 
                nElem_x, nElem_y, nElem_z, 
                dx, dy, dz
            )
        end
    end
    
    # --- Update Density ---
    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e] 
            current_l1_stress = filtered_l1_stress[e]
            
            new_density_value = (current_l1_stress / l1_stress_allowable) / E
            
            old_val = density[e]
            new_val = clamp(new_density_value, min_density, max_density)
            
            density[e] = new_val
            
            diff = abs(new_val - old_val)
            # Use atomic max for thread safety if strictness needed, but loose max is usually fine here
            if diff > max_change
                max_change = diff
            end
        end
    end
    
    # --- Heaviside Projection (Annealing) ---
    current_threshold = 0.0f0
    if iter > number_of_iterations
        current_threshold = 0.95f0
    else
        progress = Float32(iter) / Float32(number_of_iterations)
        current_threshold = Float32(0.95) * progress
    end
    
    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e]
            if density[e] < current_threshold
                density[e] = min_density 
            end
        end
        
        # Enforce protected status
        if protected_elements_mask[e]
            density[e] = original_density[e]
        end
    end
    
    return max_change, R_length, current_threshold
end

end