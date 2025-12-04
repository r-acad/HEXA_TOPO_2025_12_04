module Configuration 
 
using YAML  
using ..Mesh  
using ..Helpers 
using ..MeshShapeProcessing # Now uses the refactored module
 
export load_configuration, setup_geometry, initialize_density_field 
 
""" 
    load_configuration(filename::String) 
 
Load and parse a JSON/YAML configuration file. 
""" 
function load_configuration(filename::String) 
    if !isfile(filename) 
        error("Configuration file '$(filename)' not found") 
    end 
      
    return YAML.load_file(filename) 
end 
 
""" 
    setup_geometry(config) 
 
Process the geometry configuration and return parameters for mesh generation. 
""" 
function setup_geometry(config) 
      
    length_x = config["geometry"]["length_x"] 
    length_y = config["geometry"]["length_y"] 
    length_z = config["geometry"]["length_z"] 
    target_elem_count = config["geometry"]["target_elem_count"] 
      
    println("Domain dimensions:") 
    println("  X: 0 to $(length_x)") 
    println("  Y: 0 to $(length_y)") 
    println("  Z: 0 to $(length_z)") 
      
    shapes_add = Any[] 
    shapes_remove = Any[] 
    
    for (key, shape) in config["geometry"] 
        if key in ["length_x", "length_y", "length_z", "target_elem_count", "shape_notes"] 
            continue 
        end 
          
        if haskey(shape, "type") 
            action = lowercase(get(shape, "action", "remove")) 
            if action == "add" 
                push!(shapes_add, shape) 
            elseif action == "remove" 
                push!(shapes_remove, shape) 
            else 
                push!(shapes_remove, shape) 
            end 
        end 
    end 
 
    println("Found $(length(shapes_add)) shapes to add and $(length(shapes_remove)) shapes to remove.") 
      
    nElem_x, nElem_y, nElem_z, dx, dy, dz, actual_elem_count = 
        Helpers.calculate_element_distribution(length_x, length_y, length_z, target_elem_count) 
      
    println("Mesh parameters:") 
    println("  Domain: $(length_x) x $(length_y) x $(length_z) meters") 
    println("  Elements: $(nElem_x) x $(nElem_y) x $(nElem_z) = $(actual_elem_count)") 
    println("  Element sizes: $(dx) x $(dy) x $(dz)") 
      
    max_domain_dim = max(length_x, length_y, length_z) 
 
    return ( 
        nElem_x = nElem_x,  
        nElem_y = nElem_y,  
        nElem_z = nElem_z, 
        dx = dx, 
        dy = dy, 
        dz = dz, 
        shapes_to_add = shapes_add, 
        shapes_to_remove = shapes_remove, 
        actual_elem_count = actual_elem_count, 
        max_domain_dim = Float32(max_domain_dim)  
    ) 
end 
 
""" 
    initialize_density_field(nodes, elements, shapes_to_add, shapes_to_remove, config) 
 
Processes geometric shapes to set the initial density array. 
Returns `density`, `original_density`, and `protected_elements_mask`. 
""" 
function initialize_density_field(nodes::Matrix{Float32}, 
                                  elements::Matrix{Int}, 
                                  shapes_to_add::Vector{Any}, 
                                  shapes_to_remove::Vector{Any}, 
                                  config::Dict) 
      
    min_density = Float32(get(config["optimization_parameters"], "min_density", 1e-3)) 
    max_density_add = Float32(get(config["optimization_parameters"], "max_density_initial_add", 10.0)) 
 
    nElem = size(elements, 1) 
    println("Processing geometric density modifications...") 
    
    # Start with uniform density of 1.0
    density = ones(Float32, nElem) 
      
    # 1. Apply ADD shapes (Hard regions)
    # This uses the refactored module instead of manual loops
    MeshShapeProcessing.apply_geometric_modifiers!(density, nodes, elements, shapes_to_add, max_density_add)

    # 2. Apply REMOVE shapes (Soft/Void regions)
    MeshShapeProcessing.apply_geometric_modifiers!(density, nodes, elements, shapes_to_remove, min_density)
      
    println("Element density processing complete. Min Density floor: $(min_density)") 
 
    original_density = copy(density) 
      
    # Protected elements are those explicitly modified (not 1.0)
    protected_elements_mask = (original_density .!= 1.0f0) 
    num_protected = sum(protected_elements_mask) 
    println("Found $(num_protected) protected elements (voids/rigid) that will not be iterated.") 
 
    return density, original_density, protected_elements_mask 
end 
 
end