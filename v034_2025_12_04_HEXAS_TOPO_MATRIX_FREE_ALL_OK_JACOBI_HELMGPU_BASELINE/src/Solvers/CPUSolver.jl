module CPUSolver

using LinearAlgebra, SparseArrays, Base.Threads, Printf
using ..Element

export MatrixFreeSystem, solve_system_cpu

struct MatrixFreeSystem{T}
    nodes::Matrix{T}
    elements::Matrix{Int}
    E::T
    nu::T
    bc_indicator::Matrix{T}
    free_dofs::Vector{Int}
    constrained_dofs::Vector{Int}
    density::Vector{T}
    min_stiffness_threshold::T 
    # Store canonical matrix to speed up iteratons
    canonical_ke::Matrix{T}
end


"""
    MatrixFreeSystem(...)
"""
function MatrixFreeSystem(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, density::Vector{T}=nothing,
                          min_stiffness_threshold::T=Float32(1.0e-3)) where T
                            
    nElem = size(elements, 1)
    if density === nothing; density = ones(T, nElem); end

    nNodes = size(nodes, 1)
    ndof   = nNodes * 3
    constrained = falses(ndof)
    @inbounds for i in 1:nNodes
        if bc_indicator[i,1]>0; constrained[3*(i-1)+1]=true; end
        if bc_indicator[i,2]>0; constrained[3*(i-1)+2]=true; end
        if bc_indicator[i,3]>0; constrained[3*(i-1)+3]=true; end
    end

    free_dofs        = findall(!, constrained)
    constrained_dofs = findall(x->x, constrained)

    # Calculate canonical stiffness once
    # Check first element dimensions
    n1, n2, n4, n5 = nodes[elements[1,1], :], nodes[elements[1,2], :], nodes[elements[1,4], :], nodes[elements[1,5], :]
    dx, dy, dz = norm(n2-n1), norm(n4-n1), norm(n5-n1)
    canonical_ke = Element.get_canonical_stiffness(dx, dy, dz, nu)

    return MatrixFreeSystem(nodes, elements, E, nu, bc_indicator,
                            free_dofs, constrained_dofs, density,
                            min_stiffness_threshold, canonical_ke) 
end

"""
    apply_stiffness(system, x)

Matrix-free multiplication K*x using canonical stiffness.
"""
function apply_stiffness(system::MatrixFreeSystem{T}, x::Vector{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes * 3
    nElem  = size(system.elements, 1)

    result = zeros(T, ndof)
    result_local = [zeros(T, ndof) for _ in 1:Threads.nthreads()]
    
    Ke_base = system.canonical_ke

    @threads for e in 1:nElem
        
        # Optimization: Skip soft elements
        dens = system.density[e]
        if dens < system.min_stiffness_threshold
            continue
        end
        
        tid  = Threads.threadid()
        conn = view(system.elements, e, :)
        
        # Material Factor
        factor = system.E * dens

        # Gather local displacement
        u_elem = zeros(T, 24)
        for i in 1:8
            node_id = conn[i]
            base = 3*(node_id-1)
            u_elem[3*(i-1)+1] = x[base+1]
            u_elem[3*(i-1)+2] = x[base+2]
            u_elem[3*(i-1)+3] = x[base+3]
        end

        # Local matrix-vector mult: f = (factor * Ke_base) * u
        f_elem = (Ke_base * u_elem) .* factor

        # Scatter
        for i in 1:8
            node_id = conn[i]
            base = 3*(node_id-1)
            result_local[tid][base+1] += f_elem[3*(i-1)+1]
            result_local[tid][base+2] += f_elem[3*(i-1)+2]
            result_local[tid][base+3] += f_elem[3*(i-1)+3]
        end
    end

    for r in result_local
        result .+= r
    end
    return result
end

function apply_system(system::MatrixFreeSystem{T}, x::Vector{T}) where T
    return apply_stiffness(system, x)
end

function apply_system_free_dofs(system::MatrixFreeSystem{T}, x_free::Vector{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes * 3
    x_full = zeros(T, ndof)
    x_full[system.free_dofs] = x_free
    result_full = apply_system(system, x_full)
    return result_full[system.free_dofs]
end


"""
    compute_diagonal_preconditioner(system)
"""
function compute_diagonal_preconditioner(system::MatrixFreeSystem{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes*3
    nElem  = size(system.elements, 1)
    
    diag_vec = zeros(T, ndof)
    diag_local = [zeros(T, ndof) for _ in 1:Threads.nthreads()]
    
    Ke_base = system.canonical_ke

    @threads for e in 1:nElem
        dens = system.density[e]
        if dens < system.min_stiffness_threshold
            continue
        end
        
        tid  = Threads.threadid()
        conn = view(system.elements, e, :)
        factor = system.E * dens

        for i in 1:8
            node_id = conn[i]
            base_dof = 3*(i-1)
            # Diagonal contributions
            diag_val_x = Ke_base[base_dof+1, base_dof+1] * factor
            diag_val_y = Ke_base[base_dof+2, base_dof+2] * factor
            diag_val_z = Ke_base[base_dof+3, base_dof+3] * factor
            
            idx = 3*(node_id-1)
            diag_local[tid][idx+1] += diag_val_x
            diag_local[tid][idx+2] += diag_val_y
            diag_local[tid][idx+3] += diag_val_z
        end
    end

    for d in diag_local
        diag_vec .+= d
    end
    return diag_vec
end


function matrix_free_cg_solve(system::MatrixFreeSystem{T}, f::Vector{T};
                              max_iter=1000, tol=1e-6, use_precond=true,
                              shift_factor::T=Float32(1.0e-6)) where T  
    f_free = f[system.free_dofs]
    n_free = length(system.free_dofs)
    x_free = zeros(T, n_free)

    diag_full = compute_diagonal_preconditioner(system)
    diag_free = diag_full[system.free_dofs]

    shift = T(0.0)
    try
        max_diag = maximum(diag_free)
        shift = shift_factor * max_diag
        println("CPUSolver: Applying diagonal shift: $shift (Factor: $shift_factor)")
    catch e
        @warn "Could not calculate diagonal shift: $e"
    end
    
    r = copy(f_free)
    
    # Regularize conditioner
    diag_free[diag_free .<= shift] .= shift
    
    z = use_precond ? r ./ diag_free : copy(r)
    p = copy(z)
    rz_old = dot(r, z)

    println("Starting matrix-free CG solve with $(n_free) unknowns...")
    total_time = 0.0
    
    norm_f = norm(f_free)
    if norm_f == 0
        return zeros(T, length(f))
    end

    for iter in 1:max_iter
        iter_start = time()
        
        Ap = apply_system_free_dofs(system, p) .+ (shift .* p)
        
        alpha = rz_old / dot(p, Ap)
        x_free .+= alpha .* p
        r .-= alpha .* Ap
        
        res_norm = norm(r) / norm_f
        total_time += (time() - iter_start)

        if res_norm < tol
            println("CG converged in $iter iterations, residual = $res_norm, total time = $total_time sec")
            break
        end

        diag_free[diag_free .<= shift] .= shift
        z = use_precond ? r ./ diag_free : copy(r)
        
        rz_new = dot(r, z)
        beta = rz_new / rz_old
        p .= z .+ beta .* p
        rz_old = rz_new
    end

    x_full = zeros(T, length(f))
    x_full[system.free_dofs] = x_free
    return x_full
end

function solve_system_cpu(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, f::Vector{T};
                          max_iter=1000, tol=1e-6, use_precond=true,
                          density::Vector{T}=nothing,
                          shift_factor::T=Float32(1.0e-6),
                          min_stiffness_threshold::T=Float32(1.0e-3)) where T   
                            
    system = MatrixFreeSystem(nodes, elements, E, nu, bc_indicator, density, min_stiffness_threshold)
    
    solve_start = time()
    solution = matrix_free_cg_solve(system, f, max_iter=max_iter, tol=tol, 
                                    use_precond=use_precond, shift_factor=shift_factor)
    solve_end = time()
    @printf("Total solution time (matrix-free CPU): %.6f sec\n", solve_end - solve_start)
    return solution
end

end