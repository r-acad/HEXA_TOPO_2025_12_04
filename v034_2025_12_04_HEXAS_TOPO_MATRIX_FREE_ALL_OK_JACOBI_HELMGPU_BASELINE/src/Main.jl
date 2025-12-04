

println(">>> SCRIPT START: Loading Modules...")

module HEXA

using LinearAlgebra
using SparseArrays
using Printf
using Base.Threads
using JSON
using Dates
using Statistics 
using CUDA
using YAML

# --- UTILITIES ---
include("Utils/Diagnostics.jl")
include("Utils/Helpers.jl")

using .Diagnostics
using .Helpers

# --- CORE PHYSICS ---
include("Core/Element.jl")
include("Core/Boundary.jl")
include("Core/Stress.jl")

using .Element
using .Boundary
using .Stress

# --- MESHING ---
include("Mesh/Mesh.jl")
include("Mesh/MeshUtilities.jl")
include("Mesh/MeshPruner.jl") 
include("Mesh/MeshRefiner.jl") 
include("Mesh/MeshShapeProcessing.jl") 

using .Mesh
using .MeshUtilities
using .MeshPruner 
using .MeshRefiner 
using .MeshShapeProcessing

# --- SOLVERS ---
include("Solvers/CPUSolver.jl")
include("Solvers/GPUSolver.jl")
include("Solvers/DirectSolver.jl")
include("Solvers/IterativeSolver.jl")
include("Solvers/Solver.jl") 

using .CPUSolver
using .GPUSolver
using .DirectSolver
using .IterativeSolver
using .Solver

# --- IO & OPTIMIZATION ---
include("IO/Configuration.jl")
include("IO/ExportVTK.jl")
include("IO/Postprocessing.jl")
include("Optimization/GPUHelmholtz.jl") # NEW: Include before TopOpt
include("Optimization/TopOpt.jl") 

using .Configuration
using .ExportVTK
using .Postprocessing
using .TopologyOptimization 

function __init__()
    Diagnostics.log_status("HEXA Finite Element Solver initialized")
    Helpers.clear_gpu_memory()
end

function run_main(config_file=nothing)
    try
        _run_safe(config_file)
    catch e
        println("\n" * "!"^60)
        println("!!! FATAL ERROR DETECTED !!!")
        println("!"^60)
        showerror(stderr, e, catch_backtrace())
        
        open("crash_log.txt", "w") do io
            write(io, "Fatal Error at $(now())\n")
            showerror(io, e, catch_backtrace())
        end
        println("\nError details written to 'crash_log.txt'.")
    end
end

function _run_safe(config_file)
    
    if config_file === nothing
        config_file = joinpath(@__DIR__, "..", "configs", "default.yaml")
    end
    
    println("Loading configuration from: $config_file")
    if !isfile(config_file)
        error("Configuration file not found: $config_file")
    end

    config = load_configuration(config_file)
    
    out_settings = get(config, "output_settings", Dict())
    export_freq = get(out_settings, "export_frequency", 5)
    log_filename = get(out_settings, "log_filename", "results/simulation_log.txt")
    
    mkpath(dirname(log_filename))
    Diagnostics.init_log_file(log_filename, config)

    geom = setup_geometry(config)
    
    nodes, elements, dims = generate_mesh(
        geom.nElem_x, geom.nElem_y, geom.nElem_z;
        dx = geom.dx, dy = geom.dy, dz = geom.dz
    )
    
    initial_target_count = size(elements, 1)
    println(">> Initial Target Active Elements: $initial_target_count")

    domain_bounds = (
        min_pt = [0.0f0, 0.0f0, 0.0f0],
        len_x = geom.dx * geom.nElem_x,
        len_y = geom.dy * geom.nElem_y,
        len_z = geom.dz * geom.nElem_z
    )

    current_dx = geom.dx
    current_dy = geom.dy
    current_dz = geom.dz
    
    config["geometry"]["nElem_x_computed"] = geom.nElem_x
    config["geometry"]["nElem_y_computed"] = geom.nElem_y
    config["geometry"]["nElem_z_computed"] = geom.nElem_z
    config["geometry"]["dx_computed"] = current_dx
    config["geometry"]["dy_computed"] = current_dy
    config["geometry"]["dz_computed"] = current_dz
    config["geometry"]["max_domain_dim"] = geom.max_domain_dim
    
    nNodes = size(nodes, 1)
    
    bc_data = config["boundary_conditions"]
    bc_indicator = get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
    
    E = Float32(config["material"]["E"])
    nu = Float32(config["material"]["nu"])
    
    ndof = nNodes * 3
    F = zeros(Float32, ndof)
    forces_data = config["external_forces"]
    apply_external_forces!(F, Vector{Any}(forces_data), nodes, elements)
    
    density, original_density, protected_elements_mask = 
        initialize_density_field(nodes, elements, geom.shapes_to_add, geom.shapes_to_remove, config)
    
    opt_params = config["optimization_parameters"]
    min_density = Float32(get(opt_params, "min_density", 1.0e-3))
    max_density_clamp = Float32(get(opt_params, "density_clamp_max", 1.0))

    base_name = splitext(basename(config_file))[1]
    RESULTS_DIR = "results"
    mkpath(RESULTS_DIR)
    
    number_of_iterations = get(config, "number_of_iterations", 0)
    l1_stress_allowable = Float32(get(config, "l1_stress_allowable", 1.0))
    if l1_stress_allowable == 0.0f0; l1_stress_allowable = 1.0f0; end

    U_full = zeros(Float32, ndof)
    
    max_change = 1.0f0
    filter_R = 0.0f0
    curr_threshold = 0.0f0
    
    iter = 1
    keep_running = true
    is_annealing = false
    max_annealing_iters = 100 
    convergence_threshold = 0.01 

    max_gpu_elems = Helpers.get_max_feasible_elements()
    
    while keep_running
        iter_start_time = time()
        refine_status = "No"
        
        active_count = count(d -> d > min_density * 1.1, density)
        is_too_coarse = active_count < (initial_target_count * 0.6)
        fits_in_gpu = (active_count * 4) < max_gpu_elems
        
        should_refine = (iter > 5) && (!is_annealing) && is_too_coarse && fits_in_gpu
        
        if should_refine
            refine_status = "YES"
            nodes, elements, density, dims = MeshRefiner.refine_mesh_and_fields(
                nodes, elements, density, dims, initial_target_count, domain_bounds
            )
            GC.gc()
            
            nElem_x_new, nElem_y_new, nElem_z_new = dims[1]-1, dims[2]-1, dims[3]-1
            current_dx = domain_bounds.len_x / nElem_x_new
            current_dy = domain_bounds.len_y / nElem_y_new
            current_dz = domain_bounds.len_z / nElem_z_new
            
            config["geometry"]["nElem_x_computed"] = nElem_x_new
            config["geometry"]["nElem_y_computed"] = nElem_y_new
            config["geometry"]["nElem_z_computed"] = nElem_z_new
            config["geometry"]["dx_computed"] = current_dx
            config["geometry"]["dy_computed"] = current_dy
            config["geometry"]["dz_computed"] = current_dz
            
            geom = (
                nElem_x = nElem_x_new, nElem_y = nElem_y_new, nElem_z = nElem_z_new,
                dx = current_dx, dy = current_dy, dz = current_dz,
                shapes_to_add = geom.shapes_to_add, shapes_to_remove = geom.shapes_to_remove,
                actual_elem_count = size(elements, 1),
                max_domain_dim = geom.max_domain_dim
            )

            nNodes = size(nodes, 1)
            ndof = nNodes * 3
            bc_indicator = get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
            F = zeros(Float32, ndof)
            apply_external_forces!(F, Vector{Any}(forces_data), nodes, elements)
            
            _, original_density, protected_elements_mask = 
                initialize_density_field(nodes, elements, geom.shapes_to_add, geom.shapes_to_remove, config)
            
            U_full = zeros(Float32, ndof)
            TopologyOptimization.reset_filter_cache!()
            GC.gc()
        end

        if number_of_iterations > 0 && iter > number_of_iterations
            is_annealing = true
            annealing_idx = iter - number_of_iterations
            if annealing_idx > max_annealing_iters
                Diagnostics.log_status("Max annealing iterations reached.")
                break
            end
        end

        if iter > 1
            Threads.@threads for e in 1:size(elements, 1)
                if protected_elements_mask[e]
                    density[e] = original_density[e]
                end
            end
        end

        U_full = Solver.solve_system(
            nodes, elements, E, nu, bc_indicator, F;
            density=density,
            config=config,
            min_stiffness_threshold=min_density,
            prune_voids=true 
        )
        
        compliance = dot(F, U_full)
        strain_energy = 0.5 * compliance
        
        principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field =
            compute_stress_field(nodes, elements, U_full, E, nu, density)
        
        avg_l1_stress = mean(l1_stress_norm_field)
        
        vol_total = length(density)
        active_non_soft = count(d -> d > min_density, density)
        vol_frac = sum(density) / vol_total
        
        if number_of_iterations > 0
            # FIXED: Pass 'elements' to update_density! for GPU filter
            res_tuple = update_density!(
                density, l1_stress_norm_field, protected_elements_mask,
                E, l1_stress_allowable, iter, number_of_iterations,
                original_density,
                min_density, max_density_clamp,
                config,
                elements, 
                is_annealing
            )
            max_change = res_tuple[1]
            filter_R = res_tuple[2]
            curr_threshold = res_tuple[3]
        end
        
        iter_time = time() - iter_start_time

        cur_dims_str = "$(config["geometry"]["nElem_x_computed"])x$(config["geometry"]["nElem_y_computed"])x$(config["geometry"]["nElem_z_computed"])"
        
        Diagnostics.write_iteration_log(
            log_filename, iter, cur_dims_str, vol_total, active_non_soft, 
            filter_R, curr_threshold, 
            compliance, strain_energy, avg_l1_stress, vol_frac, max_change, 
            refine_status, iter_time
        )

        should_export = (iter == 1) || (iter % export_freq == 0) || is_annealing
        
        if should_export
            export_iteration_results(
                iter, base_name, RESULTS_DIR, nodes, elements,
                U_full, F, bc_indicator, principal_field,
                vonmises_field, full_stress_voigt,
                l1_stress_norm_field, density, E,
                geom 
            )
        end
        
        if is_annealing && max_change < convergence_threshold
            Diagnostics.log_status("CONVERGED: Change < 1.0%")
            keep_running = false
        elseif number_of_iterations == 0
             keep_running = false
        end

        if CUDA.functional()
            Helpers.clear_gpu_memory()
        end
        
        iter += 1
        GC.gc() 
    end

    Diagnostics.log_status("Finished.")
    return nothing
end

end

using .HEXA

if length(ARGS) > 0 && isfile(ARGS[1])
    println(">>> Using provided config: $(ARGS[1])")
    HEXA.run_main(ARGS[1])
else
    default_config = joinpath(@__DIR__, "..", "configs", "default.yaml")
    if isfile(default_config)
        println(">>> Using default config: $default_config")
        HEXA.run_main(default_config)
    else
        println("\n!!! ERROR: No config file provided and default not found.")
        println("Usage: julia src/Main.jl path/to/config.yaml")
    end
end