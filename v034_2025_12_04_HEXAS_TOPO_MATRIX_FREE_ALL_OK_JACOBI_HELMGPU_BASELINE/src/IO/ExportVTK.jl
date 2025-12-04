
module ExportVTK 

using Printf 

export export_mesh, export_solution 

function export_mesh(nodes::Matrix{Float32}, 
                     elements::Matrix{Int}; 
                     bc_indicator=nothing, 
                     filename::String="mesh_output.vtu") 
     
    if !endswith(lowercase(filename), ".vtk") && !endswith(lowercase(filename), ".vtu") 
        filename *= ".vtk" 
    end 
     
    if any(isnan, nodes) || any(isinf, nodes) 
        @warn "Found NaN or Inf values in node coordinates. Replacing with zeros." 
        nodes = replace(nodes, NaN => Float32(0.0), Inf => Float32(0.0), -Inf => Float32(0.0)) 
    end 
     
    max_coord = maximum(abs.(nodes)) 
    if max_coord > Float32(1.0e10) 
        @warn "Very large coordinate values detected (maximum absolute value: $max_coord). Clamping to reasonable range." 
        nodes = clamp.(nodes, Float32(-1.0e10), Float32(1.0e10)) 
    end 

    nElem  = size(elements, 1) 
    nNodes = size(nodes, 1) 

    valid_elements = Int[] 
    for e = 1:nElem 
        elem_nodes = elements[e, :] 
        if any(n -> n < 1 || n > nNodes, elem_nodes) 
            @warn "Element $e has invalid node indices; skipping it entirely." 
        else 
            push!(valid_elements, e) 
        end 
    end 

    if isempty(valid_elements) 
        @warn "No valid elements found. Skipping VTK export." 
        return 
    end 

    nElem_valid = length(valid_elements) 
     
    try 
        open(filename, "w") do file 
            write(file, "# vtk DataFile Version 3.0\n") 
            write(file, "HEXA FEM Mesh (BINARY)\n") 
            write(file, "BINARY\n") 
            write(file, "DATASET UNSTRUCTURED_GRID\n") 
             
            write(file, "POINTS $(nNodes) float\n") 
            coords_flat = zeros(Float32, nNodes * 3) 
            for i in 1:nNodes 
                coords_flat[3*(i-1)+1] = nodes[i, 1] 
                coords_flat[3*(i-1)+2] = nodes[i, 2] 
                coords_flat[3*(i-1)+3] = nodes[i, 3] 
            end 
            write(file, hton.(coords_flat)) 
             
            write(file, "\nCELLS $(nElem_valid) $(nElem_valid * 9)\n") 
            cell_data = zeros(Int32, nElem_valid * 9) 
            for (idx_out, idx_in) in enumerate(valid_elements) 
                offset = (idx_out - 1) * 9 
                cell_data[offset + 1] = Int32(8) 
                for j in 1:8 
                    cell_data[offset + 1 + j] = Int32(elements[idx_in, j] - 1) 
                end 
            end 
            write(file, hton.(cell_data)) 
             
            write(file, "\nCELL_TYPES $(nElem_valid)\n") 
            cell_types = fill(Int32(12), nElem_valid) 
            write(file, hton.(cell_types)) 
             
            if bc_indicator !== nothing && size(bc_indicator, 1) == nNodes 
                write(file, "\nPOINT_DATA $(nNodes)\n") 
                ncols_bc = min(size(bc_indicator, 2), 3) 
                 
                if ncols_bc >= 2 
                    write(file, "VECTORS BC float\n") 
                    bc_data = zeros(Float32, nNodes * 3) 
                    for i in 1:nNodes 
                        if ncols_bc == 2 
                            bc_data[3*(i-1)+1] = bc_indicator[i, 1] 
                            bc_data[3*(i-1)+2] = bc_indicator[i, 2] 
                            bc_data[3*(i-1)+3] = Float32(0.0) 
                        else 
                            bc_data[3*(i-1)+1] = bc_indicator[i, 1] 
                            bc_data[3*(i-1)+2] = bc_indicator[i, 2] 
                            bc_data[3*(i-1)+3] = bc_indicator[i, 3] 
                        end 
                    end 
                    write(file, hton.(bc_data)) 
                else 
                    write(file, "SCALARS BCx float 1\n") 
                    write(file, "LOOKUP_TABLE default\n") 
                    bc_data = [bc_indicator[i, 1] for i in 1:nNodes] 
                    write(file, hton.(bc_data)) 
                end 
            end 
        end 
         
        println("Successfully exported mesh to $filename (BINARY format)") 
    catch e 
        @error "Failed to save VTK file: $e" 
        println("Error details: ", e) 
    end 
end 

function export_solution(nodes::Matrix{Float32}, 
                         elements::Matrix{Int}, 
                         U_full::Vector{Float32}, 
                         F::Vector{Float32}, 
                         bc_indicator::Matrix{Float32}, 
                         principal_field::Matrix{Float32}, 
                         vonmises_field::Vector{Float32}, 
                         full_stress_voigt::Matrix{Float32}, 
                         l1_stress_norm_field::Vector{Float32}; 
                         density::Union{Vector{Float32}, Nothing}=nothing,
                         scale::Float32=Float32(1.0), 
                         filename::String="solution_output.vtu") 

    function sanitize_data(data) 
        data = replace(data, NaN => Float32(0.0), Inf => Float32(0.0), -Inf => Float32(0.0)) 
        max_val = maximum(abs.(data)) 
        if max_val > Float32(1.0e10) 
            @warn "Very large values detected (max abs: $max_val). Clamping to prevent ParaView crashes." 
            return clamp.(data, Float32(-1.0e10), Float32(1.0e10)) 
        end 
        return data 
    end 
     
    U_full = sanitize_data(U_full) 
    F = sanitize_data(F) 
    nodes = sanitize_data(nodes) 
    principal_field = sanitize_data(principal_field) 
    vonmises_field = sanitize_data(vonmises_field) 
    full_stress_voigt = sanitize_data(full_stress_voigt) 
    l1_stress_norm_field = sanitize_data(l1_stress_norm_field) 

    nNodes = size(nodes, 1) 
    nElem  = size(elements, 1) 

    valid_elements = Int[] 
    for e = 1:nElem 
        elem_nodes = elements[e, :] 
        if any(n -> n < 1 || n > nNodes, elem_nodes) 
            @warn "Element $e has invalid node indices; skipping it." 
        else 
            push!(valid_elements, e) 
        end 
    end 

    nElem_valid = length(valid_elements) 
    if nElem_valid == 0 
        @warn "No valid elements remain. Skipping solution export." 
        return 
    end 

    function ensure_array_size(arr, expected_size, pad_value=Float32(0.0)) 
        if length(arr) < expected_size 
            return vcat(arr, fill(pad_value, expected_size - length(arr))) 
        elseif length(arr) > expected_size 
            return arr[1:expected_size] 
        else 
            return arr 
        end 
    end 
     
    U_full = ensure_array_size(U_full, 3*nNodes) 
    F = ensure_array_size(F, 3*nNodes) 
     
    displacement = zeros(Float32, nNodes, 3) 
    forces = zeros(Float32, nNodes, 3) 
     
    for i in 1:nNodes 
        base_idx = 3*(i-1) 
        if base_idx + 3 <= length(U_full) 
            displacement[i, 1] = U_full[base_idx + 1] 
            displacement[i, 2] = U_full[base_idx + 2] 
            displacement[i, 3] = U_full[base_idx + 3] 
        end 
         
        if base_idx + 3 <= length(F) 
            forces[i, 1] = F[base_idx + 1] 
            forces[i, 2] = F[base_idx + 2] 
            forces[i, 3] = F[base_idx + 3] 
        end 
    end 
     
    disp_mag = sqrt.(sum(displacement.^2, dims=2))[:,1]   

    max_disp = maximum(abs.(displacement)) 
    if max_disp > 0 
        max_dim = maximum([ 
            maximum(nodes[:,1]) - minimum(nodes[:,1]), 
            maximum(nodes[:,2]) - minimum(nodes[:,2]), 
            maximum(nodes[:,3]) - minimum(nodes[:,3]) 
        ]) 
        if scale * max_disp > max_dim * 5 
            @warn "Scale factor causes very large deformation => auto reducing." 
            scale = Float32(0.5) * max_dim / max_disp 
        end 
    end 

    deformed_nodes = copy(nodes) 
    @inbounds for i in 1:nNodes 
        deformed_nodes[i,1] += scale*displacement[i,1] 
        deformed_nodes[i,2] += scale*displacement[i,2] 
        deformed_nodes[i,3] += scale*displacement[i,3] 
    end 
     
    deformed_nodes = sanitize_data(deformed_nodes) 

    if size(principal_field, 2) < nElem 
        principal_field = hcat(principal_field, zeros(Float32, 3, nElem - size(principal_field, 2))) 
    end 
     
    vonmises_field = ensure_array_size(vonmises_field, nElem) 
    l1_stress_norm_field = ensure_array_size(l1_stress_norm_field, nElem) 
     
    if size(full_stress_voigt, 2) < nElem 
        full_stress_voigt = hcat(full_stress_voigt, zeros(Float32, 6, nElem - size(full_stress_voigt, 2))) 
    end 
     
    principal_field_valid = principal_field[:, valid_elements] 
    vonmises_field_valid = vonmises_field[valid_elements] 
    l1_stress_norm_field_valid = l1_stress_norm_field[valid_elements] 
    full_stress_voigt_valid = full_stress_voigt[:, valid_elements] 

    # SINGLE FILE OUTPUT - Modified to output only one file with all data
    if endswith(lowercase(filename), ".vtk") || 
       endswith(lowercase(filename), ".vtu") 
        base_filename = filename[1:end-4] 
    else 
        base_filename = filename 
    end 
     
    # Use only one filename for the combined output
    combined_filename = base_filename * "_combined.vtk"

    try 
        open(combined_filename, "w") do file 
            write(file, "# vtk DataFile Version 3.0\n") 
            write(file, "HEXA FEM Solution (Combined Data - BINARY)\n") 
            write(file, "BINARY\n") 
            write(file, "DATASET UNSTRUCTURED_GRID\n") 
             
            # Write original mesh coordinates
            write(file, "POINTS $(nNodes) float\n") 
            coords_flat = zeros(Float32, nNodes * 3) 
            for i in 1:nNodes 
                coords_flat[3*(i-1)+1] = nodes[i, 1] 
                coords_flat[3*(i-1)+2] = nodes[i, 2] 
                coords_flat[3*(i-1)+3] = nodes[i, 3] 
            end 
            write(file, hton.(coords_flat)) 
             
            write(file, "\nCELLS $(nElem_valid) $(nElem_valid * 9)\n") 
            cell_data = zeros(Int32, nElem_valid * 9) 
            for (idx_out, idx_in) in enumerate(valid_elements) 
                offset = (idx_out - 1) * 9 
                cell_data[offset + 1] = Int32(8) 
                for j in 1:8 
                    cell_data[offset + 1 + j] = Int32(elements[idx_in, j] - 1) 
                end 
            end 
            write(file, hton.(cell_data)) 
             
            write(file, "\nCELL_TYPES $(nElem_valid)\n") 
            cell_types = fill(Int32(12), nElem_valid) 
            write(file, hton.(cell_types)) 
             
            # POINT DATA SECTION
            write(file, "\nPOINT_DATA $(nNodes)\n") 
             
            # Displacement vectors
            write(file, "VECTORS Displacement float\n") 
            disp_flat = zeros(Float32, nNodes * 3) 
            for i in 1:nNodes 
                disp_flat[3*(i-1)+1] = displacement[i, 1] 
                disp_flat[3*(i-1)+2] = displacement[i, 2] 
                disp_flat[3*(i-1)+3] = displacement[i, 3] 
            end 
            write(file, hton.(disp_flat)) 
             
            # Displacement magnitude
            write(file, "\nSCALARS Displacement_Magnitude float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(disp_mag)) 
             
            # Force vectors
            write(file, "\nVECTORS Force float\n") 
            force_flat = zeros(Float32, nNodes * 3) 
            for i in 1:nNodes 
                force_flat[3*(i-1)+1] = forces[i, 1] 
                force_flat[3*(i-1)+2] = forces[i, 2] 
                force_flat[3*(i-1)+3] = forces[i, 3] 
            end 
            write(file, hton.(force_flat)) 
             
            # Boundary conditions
            if size(bc_indicator, 1) == nNodes 
                ncols_bc = min(size(bc_indicator, 2), 3) 
                if ncols_bc >= 2 
                    write(file, "\nVECTORS BC float\n") 
                    bc_data = zeros(Float32, nNodes * 3) 
                    for i in 1:nNodes 
                        if ncols_bc == 2 
                            bc_data[3*(i-1)+1] = bc_indicator[i, 1] 
                            bc_data[3*(i-1)+2] = bc_indicator[i, 2] 
                            bc_data[3*(i-1)+3] = Float32(0.0) 
                        else 
                            bc_data[3*(i-1)+1] = bc_indicator[i, 1] 
                            bc_data[3*(i-1)+2] = bc_indicator[i, 2] 
                            bc_data[3*(i-1)+3] = bc_indicator[i, 3] 
                        end 
                    end 
                    write(file, hton.(bc_data)) 
                else 
                    write(file, "\nSCALARS BC float 1\n") 
                    write(file, "LOOKUP_TABLE default\n") 
                    bc_data = [bc_indicator[i, 1] for i in 1:nNodes] 
                    write(file, hton.(bc_data)) 
                end 
            end 
             
            # CELL DATA SECTION
            write(file, "\nCELL_DATA $(nElem_valid)\n") 
             
            # Von Mises Stress
            write(file, "SCALARS Von_Mises_Stress float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(vonmises_field_valid)) 
             
            # L1 Stress Norm
            write(file, "\nSCALARS l1_stress_norm float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(l1_stress_norm_field_valid)) 
             
            # Principal Stress vectors
            write(file, "\nVECTORS Principal_Stress float\n") 
            principal_flat = zeros(Float32, nElem_valid * 3) 
            for i in 1:nElem_valid 
                principal_flat[3*(i-1)+1] = principal_field_valid[1, i] 
                principal_flat[3*(i-1)+2] = principal_field_valid[2, i] 
                principal_flat[3*(i-1)+3] = principal_field_valid[3, i] 
            end 
            write(file, hton.(principal_flat)) 
             
            # Individual stress components
            stress_names = ["Stress_XX", "Stress_YY", "Stress_ZZ", "Stress_XY", "Stress_YZ", "Stress_XZ"] 
            for idx in 1:6 
                write(file, "\nSCALARS $(stress_names[idx]) float 1\n") 
                write(file, "LOOKUP_TABLE default\n") 
                stress_component = [full_stress_voigt_valid[idx, i] for i in 1:nElem_valid] 
                write(file, hton.(stress_component)) 
            end 

            # Element Density (if provided)
            if density !== nothing
                write(file, "\nSCALARS Element_Density float 1\n")
                write(file, "LOOKUP_TABLE default\n")
                density_valid = density[valid_elements]
                write(file, hton.(density_valid))
            end
        end 
         
        println("Successfully exported combined solution to $combined_filename (BINARY format)") 
        println("File contains: displacement, forces, BCs, stress tensors, and element density")
        
    catch e 
        @error "Failed to save combined VTK file: $e" 
        println("Error details: ", e) 
    end 

    return nothing 
end 

end