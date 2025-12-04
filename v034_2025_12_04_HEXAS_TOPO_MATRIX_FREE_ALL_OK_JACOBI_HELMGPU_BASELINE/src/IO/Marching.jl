using JSON3 
using Statistics 
using Printf 
import MarchingCubes: MC, march 
 
const LX = 60.0f0 
const LY = 20.0f0 
const LZ = 0.1f0 
 
const JSON_FILE_NAME = "iter_20_config_element_data.json" 
const JSON_FILE_PATH = joinpath(@__DIR__, "RESULTS", JSON_FILE_NAME)  
 
const ISOSURFACE_DENSITY = 0.8f0 
const STL_OUTPUT_FILE = "topopt_isosurface_80.stl" 
 
function infer_mesh_and_load_density(filepath::String) 
     
    println("1. Loading element data and inferring mesh structure from: $filepath") 
     
    if !isfile(filepath) 
        error("File not found at path: $filepath. Ensure the optimization run completed successfully and the 'RESULTS' folder is correctly placed relative to this script.") 
    end 
 
    data = JSON3.read(read(filepath, String)) 
     
    n_total_el_from_file = length(data) 
 
    if n_total_el_from_file < 2 
        error("Not enough elements in the result file to infer mesh size.") 
    end 
 
    density_field = Dict{Int, Float32}() 
     
    first_el_centroid = data[1]["centroid"] 
    dx, dy, dz = 0.0f0, 0.0f0, 0.0f0 
     
    if length(data) >= 2 
        second_el_centroid = data[2]["centroid"] 
        dx = abs(second_el_centroid[1] - first_el_centroid[1]) 
    end 
 
    cy1 = first_el_centroid[2] 
    for i = 2:length(data) 
        cy_i = data[i]["centroid"][2] 
        if abs(cy_i - cy1) > 1e-6  
            dy = abs(cy_i - cy1) 
            break 
        end 
    end 
     
    cz1 = first_el_centroid[3] 
    for i = 2:length(data) 
        cz_i = data[i]["centroid"][3] 
        if abs(cz_i - cz1) > 1e-6  
            dz = abs(cz_i - cz1) 
            break 
        end 
    end 
 
    if dx < 1e-6 
        dx_final = LX 
        NX = 1 
    else 
        NX = round(Int, LX / dx) 
        dx_final = LX / NX  
    end 
 
    if dy < 1e-6 
        dy_final = LY 
        NY = 1 
    else 
        NY = round(Int, LY / dy) 
        dy_final = LY / NY 
    end 
     
    if dz < 1e-6 
        dz_final = LZ 
        NZ = 1 
    else 
        NZ = round(Int, LZ / dz) 
        dz_final = LZ / NZ 
    end 
 
    inferred_n_total_el = NX * NY * NZ 
 
    if inferred_n_total_el != length(data) 
        @warn "Inferred element count ($inferred_n_total_el) does not match file count ($(length(data))). Using file count for array size." 
    end 
     
    println("   Inferred mesh dimensions: $(NX) x $(NY) x $(NZ) elements.") 
    println("   Element size (dx, dy, dz): $(@sprintf("%.6f", dx_final)), $(@sprintf("%.6f", dy_final)), $(@sprintf("%.6f", dz_final))") 
 
    for elem in data 
        density = elem["young_modulus"] 
        density_field[elem["element_id"]] = Float32(density) 
    end 
     
    sorted_densities = [density_field[i] for i in 1:length(data)] 
 
    x_coords = collect(Float32, range(0.0f0, stop=LX, length=NX + 1)) 
    y_coords = collect(Float32, range(0.0f0, stop=LY, length=NY + 1)) 
    z_coords = collect(Float32, range(0.0f0, stop=LZ, length=NZ + 1)) 
 
    return sorted_densities, NX, NY, NZ, x_coords, y_coords, z_coords 
end 
 
function prepare_density_grid(densities::Vector{Float32}, NX::Int, NY::Int, NZ::Int) 
    println("2. Preparing 3D density grid...") 
     
    node_densities = zeros(Float32, NX + 1, NY + 1, NZ + 1) 
     
    cell_densities = reshape(densities, NX, NY, NZ) 
     
    for k in 1:NZ 
        for j in 1:NY 
            for i in 1:NX 
                node_densities[i, j, k] = cell_densities[i, j, k] 
            end 
        end 
    end 
 
    println("   Node grid of dimensions $(NX+1)x$(NY+1)x$(NZ+1) created from cell data.") 
    return node_densities 
end 
 
function run_marching_cubes(density_grid::Array{Float32, 3}, x_coords::Vector{Float32}, y_coords::Vector{Float32}, z_coords::Vector{Float32}) 
    println("3. Running Marching Cubes (Isosurface Extraction) at threshold $(ISOSURFACE_DENSITY)...") 
     
    mc_struct = MC( 
        density_grid,  
        Int;  
        normal_sign=1, 
        x=x_coords,  
        y=y_coords,  
        z=z_coords 
    ) 
 
    march(mc_struct, ISOSURFACE_DENSITY) 
 
    vertices = mc_struct.vertices 
    triangles = mc_struct.triangles 
     
    num_vertices = length(vertices) 
    num_faces = length(triangles) 
     
    @printf("   Surface extracted: %d vertices, %d faces.\n", num_vertices, num_faces) 
     
    return (vertices, triangles) 
end 
 
function export_to_stl(result::Tuple) 
    vertices = result[1] 
    faces = result[2] 
     
    println("4. Exporting surface to ASCII STL file: $STL_OUTPUT_FILE") 
     
    open(STL_OUTPUT_FILE, "w") do io 
        write(io, "solid topopt_model\n") 
         
        for face in faces 
            v1 = vertices[face[1]] 
            v2 = vertices[face[2]] 
            v3 = vertices[face[3]] 
             
            e1 = (v2[1]-v1[1], v2[2]-v1[2], v2[3]-v1[3]) 
            e2 = (v3[1]-v1[1], v3[2]-v1[2], v3[3]-v1[3]) 
             
            nx = e1[2]*e2[3] - e1[3]*e2[2] 
            ny = e1[3]*e2[1] - e1[1]*e2[3] 
            nz = e1[1]*e2[2] - e1[2]*e2[1] 
             
            mag = sqrt(nx^2 + ny^2 + nz^2) 
            if mag > 1e-12 
                nx /= mag 
                ny /= mag 
                nz /= mag 
            else 
                nx, ny, nz = 0.0, 0.0, 0.0 
            end 
             
            @printf(io, "  facet normal %e %e %e\n", nx, ny, nz) 
            write(io, "    outer loop\n") 
             
            @printf(io, "      vertex %e %e %e\n", v1[1], v1[2], v1[3]) 
            @printf(io, "      vertex %e %e %e\n", v2[1], v2[2], v2[3]) 
            @printf(io, "      vertex %e %e %e\n", v3[1], v3[2], v3[3]) 
             
            write(io, "    endloop\n") 
            write(io, "  endfacet\n") 
        end 
         
        write(io, "endsolid topopt_model\n") 
    end 
     
    println("   Export complete. The file is ready for your CAD/viewer software.") 
end 
 
 
function main_postprocess() 
    try 
        densities, NX, NY, NZ, x_coords, y_coords, z_coords = infer_mesh_and_load_density(JSON_FILE_PATH) 
         
        density_grid = prepare_density_grid(densities, NX, NY, NZ) 
         
        surface_result = run_marching_cubes(density_grid, x_coords, y_coords, z_coords) 
         
        export_to_stl(surface_result) 
         
    catch e 
        @error "An error occurred during post-processing." exception=(e, catch_backtrace()) 
    end 
end 
 
main_postprocess() 
