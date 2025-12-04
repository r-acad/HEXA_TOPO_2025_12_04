module Solver 
 
using CUDA 
using ..Helpers  
using ..DirectSolver: solve_system as solve_system_direct 
using ..IterativeSolver: solve_system_iterative 
using ..MeshPruner 
 
export solve_system 
 
function choose_solver(nNodes, nElem, config) 
    solver_params = config["solver_parameters"] 
    configured_type = Symbol(lowercase(get(solver_params, "solver_type", "direct"))) 
 
    if configured_type == :direct 
        if nElem > 100_000 
            @warn "Direct solver requested for large mesh ($(nElem) elements). Switching to Matrix-Free iterative." 
            return :matrix_free 
        end 
        return :direct 
    elseif configured_type == :gpu 
        # MODIFIED: Check memory assuming MATRIX-FREE usage (true)
        if CUDA.functional() && Helpers.has_enough_gpu_memory(nNodes, nElem, true) 
            return :gpu 
        else 
            @warn "Not enough GPU memory even for matrix-free. Falling back to CPU."
            return :matrix_free 
        end 
    elseif configured_type == :matrix_free 
        return :matrix_free 
    else 
        @warn "Unknown solver_type: $(configured_type). Defaulting to matrix_free." 
        return :matrix_free 
    end 
end 
 
""" 
    solve_system(...) 
 
Solves K*u = F. 
If `prune_voids` is true, it creates a temporary reduced mesh of only active elements,
solves that smaller system, and maps the results back to the full mesh size.
""" 
function solve_system(nodes::Matrix{Float32}, 
                      elements::Matrix{Int}, 
                      E::Float32, 
                      nu::Float32, 
                      bc_indicator::Matrix{Float32}, 
                      F::Vector{Float32}; 
                      density::Vector{Float32}=nothing, 
                      config::Dict, 
                      min_stiffness_threshold::Float32=Float32(1.0e-3),
                      prune_voids::Bool=true) 
                         
    active_system = nothing
     
    if prune_voids && density !== nothing
        # Threshold slightly above zero to catch elements that are basically void
        prune_threshold = min_stiffness_threshold * 1.01f0 
         
        nElem_total = size(elements, 1)
         
        nActive = count(d -> d > prune_threshold, density)
         
        # Only prune if we are removing > 1% of elements, otherwise overhead isn't worth it
        if nActive < (nElem_total * 0.99)
             
            active_system = MeshPruner.prune_system(nodes, elements, density, prune_threshold, bc_indicator, F)
             
            solve_nodes = active_system.nodes
            solve_elements = active_system.elements
            solve_bc = active_system.bc_indicator
            solve_F = active_system.F
            solve_density = active_system.density
        else
            solve_nodes = nodes
            solve_elements = elements
            solve_bc = bc_indicator
            solve_F = F
            solve_density = density
        end
    else
        solve_nodes = nodes
        solve_elements = elements
        solve_bc = bc_indicator
        solve_F = F
        solve_density = density
    end

    nNodes_solve = size(solve_nodes, 1) 
    nElem_solve = size(solve_elements, 1) 
      
    solver_params = config["solver_parameters"] 
    solver_type = choose_solver(nNodes_solve, nElem_solve, config) 
      
    tol = Float32(get(solver_params, "tolerance", 1.0e-6)) 
    max_iter = Int(get(solver_params, "max_iterations", 1000)) 
    shift_factor = Float32(get(solver_params, "diagonal_shift_factor", 1.0e-6)) 
      
    use_precond = true 
      
    U_solved = if solver_type == :direct 
        solve_system_direct(solve_nodes, solve_elements, E, nu, solve_bc, solve_F; 
                            density=solve_density, 
                            shift_factor=shift_factor, 
                            min_stiffness_threshold=min_stiffness_threshold) 
                             
    elseif solver_type == :gpu 
        gpu_method = Symbol(lowercase(get(solver_params, "gpu_method", "krylov"))) 
        krylov_solver = Symbol(lowercase(get(solver_params, "krylov_solver", "cg"))) 
 
        solve_system_iterative(solve_nodes, solve_elements, E, nu, solve_bc, solve_F; 
                             solver_type=:gpu, max_iter=max_iter, tol=tol, 
                             density=solve_density, 
                             use_precond=use_precond,  
                             gpu_method=gpu_method, krylov_solver=krylov_solver, 
                             shift_factor=shift_factor, 
                             min_stiffness_threshold=min_stiffness_threshold) 
                             
    else  
        solve_system_iterative(solve_nodes, solve_elements, E, nu, solve_bc, solve_F; 
                             solver_type=:matrix_free, max_iter=max_iter, tol=tol, 
                             use_precond=use_precond, 
                             density=solve_density, 
                             shift_factor=shift_factor, 
                             min_stiffness_threshold=min_stiffness_threshold) 
    end 
 
    if active_system !== nothing
        # Map solution back to full domain (void nodes get 0 displacement)
        U_full = MeshPruner.reconstruct_full_solution(U_solved, active_system.new_to_old_node_map, size(nodes, 1))
        return U_full
    else
        return U_solved
    end
end 
 
end