module DirectSolver

using LinearAlgebra, SparseArrays, Base.Threads, Printf
using ..Element
using ..Boundary
using ..Mesh

export solve_system

"""
    assemble_global_stiffness_parallel_optimized(nodes, elements, E, nu, density, min_stiffness_threshold)

Pre-allocating assembler for CPU Direct solver.
"""
function assemble_global_stiffness_parallel_optimized(nodes::Matrix{Float32},
                                                      elements::Matrix{Int},
                                                      E::Float32,
                                                      nu::Float32,
                                                      density::Vector{Float32},
                                                      min_stiffness_threshold::Float32) 
                                                     
    nElem = size(elements, 1)
    ndof = size(nodes, 1) * 3

    # 1. Active Elements
    active_indices = findall(d -> d >= min_stiffness_threshold, density)
    nActive = length(active_indices)
    
    if nActive == 0; error("No active elements."); end

    # 2. Canonical Matrix
    n1, n2, n4, n5 = nodes[elements[1,1], :], nodes[elements[1,2], :], nodes[elements[1,4], :], nodes[elements[1,5], :]
    dx, dy, dz = norm(n2-n1), norm(n4-n1), norm(n5-n1)
    Ke_base = Element.get_canonical_stiffness(dx, dy, dz, nu)

    # 3. Pre-allocation
    entries_per_elem = 576
    total_entries = nActive * entries_per_elem
    I_vec = Vector{Int32}(undef, total_entries)
    J_vec = Vector{Int32}(undef, total_entries)
    V_vec = Vector{Float32}(undef, total_entries)

    # 4. Fill
    Threads.@threads for t_idx in 1:length(active_indices)
        e = active_indices[t_idx]
        offset = (t_idx - 1) * entries_per_elem
        factor = E * density[e]
        conn = view(elements, e, :)
        
        cnt = 0
        @inbounds for i in 1:8
            row_idx = 3*(conn[i]-1)
            for r in 1:3
                g_row = row_idx + r
                for j in 1:8
                    col_idx = 3*(conn[j]-1)
                    for c in 1:3
                        g_col = col_idx + c
                        cnt += 1
                        
                        I_vec[offset+cnt] = Int32(g_row)
                        J_vec[offset+cnt] = Int32(g_col)
                        V_vec[offset+cnt] = Ke_base[3*(i-1)+r, 3*(j-1)+c] * factor
                    end
                end
            end
        end
    end

    K_global = sparse(I_vec, J_vec, V_vec, ndof, ndof)
    # Enforce symmetry
    return (K_global + K_global') / 2.0f0
end

"""
    solve_system(...)
"""
function solve_system(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                      bc_indicator::Matrix{T}, f::Vector{T};
                      density::Vector{T}=nothing,
                      shift_factor::T=Float32(1.0e-6),
                      min_stiffness_threshold::T=Float32(1.0e-3)) where T    
                        
    nElem = size(elements,1)
    if density === nothing
        density = ones(T, nElem)
    end

    nNodes = size(nodes, 1)
    ndof   = nNodes * 3

    constrained = falses(ndof)
    for i in 1:nNodes
        if bc_indicator[i,1]>0; constrained[3*(i-1)+1]=true; end
        if bc_indicator[i,2]>0; constrained[3*(i-1)+2]=true; end
        if bc_indicator[i,3]>0; constrained[3*(i-1)+3]=true; end
    end
    free_dofs = findall(!, constrained)

    # Use optimized assembler
    K_global = assemble_global_stiffness_parallel_optimized(nodes, elements, E, nu, density, min_stiffness_threshold)

    K_reduced = K_global[free_dofs, free_dofs]
    F_reduced = f[free_dofs]

    try
        max_diag = maximum(abs.(diag(K_reduced)))
        shift = shift_factor * max_diag
        println("DirectSolver: Applying diagonal shift: $shift (Factor: $shift_factor)")
        K_reduced = K_reduced + shift * I
    catch e
        @warn "Could not apply diagonal shift: $e"
    end
    
    println("Solving linear system via LU factorization (CPU Direct).")
    U_reduced = K_reduced \ F_reduced

    U_full = zeros(T, ndof)
    U_full[free_dofs] = U_reduced

    return U_full
end

end