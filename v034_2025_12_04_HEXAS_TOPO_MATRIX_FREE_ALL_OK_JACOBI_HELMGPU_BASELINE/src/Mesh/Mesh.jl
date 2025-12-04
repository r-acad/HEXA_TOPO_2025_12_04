module Mesh 
 
export node_index, generate_mesh 
# Note: Geometric primitives (centroid, sphere, box) are now exclusively in MeshUtilities.jl
 
using LinearAlgebra, Printf 
 
""" 
    node_index(i, j, k, nNodes_x, nNodes_y) 
 
Converts 3D indices (i, j, k) into a linear node index (column‑major ordering). 
""" 
function node_index(i, j, k, nNodes_x, nNodes_y) 
    return i + (j-1)*nNodes_x + (k-1)*nNodes_x*nNodes_y 
end 
 
""" 
    generate_mesh(nElem_x, nElem_y, nElem_z; 
                  dx=1.0f0, dy=1.0f0, dz=1.0f0) 
 
Generates a structured (prismatic) hexahedral mesh. 
""" 
function generate_mesh(nElem_x::Int, nElem_y::Int, nElem_z::Int; 
                       dx::Float32=Float32(1.0),  
                       dy::Float32=Float32(1.0),  
                       dz::Float32=Float32(1.0))  
      
    nNodes_x = nElem_x + 1 
    nNodes_y = nElem_y + 1 
    nNodes_z = nElem_z + 1 
    dims = (nNodes_x, nNodes_y, nNodes_z) 
        
    nNodes = nNodes_x * nNodes_y * nNodes_z 
    nodes = zeros(Float32, nNodes, 3) 
    idx = 1 
    for k in 1:nNodes_z, j in 1:nNodes_y, i in 1:nNodes_x 
        nodes[idx, :] = [(i-1)*dx, (j-1)*dy, (k-1)*dz] 
        idx += 1 
    end 
      
    nElem = (nNodes_x - 1) * (nNodes_y - 1) * (nNodes_z - 1) 
    elements = Matrix{Int}(undef, nElem, 8) 
    elem_idx = 1 
    for k in 1:(nNodes_z-1), j in 1:(nNodes_y-1), i in 1:(nNodes_x-1) 
        n1 = node_index(i, j, k, nNodes_x, nNodes_y) 
        n2 = node_index(i+1, j, k, nNodes_x, nNodes_y) 
        n3 = node_index(i+1, j+1, k, nNodes_x, nNodes_y) 
        n4 = node_index(i, j+1, k, nNodes_x, nNodes_y) 
        n5 = node_index(i, j, k+1, nNodes_x, nNodes_y) 
        n6 = node_index(i+1, j, k+1, nNodes_x, nNodes_y) 
        n7 = node_index(i+1, j+1, k+1, nNodes_x, nNodes_y) 
        n8 = node_index(i, j+1, k+1, nNodes_x, nNodes_y) 
        elements[elem_idx, :] = [n1, n2, n3, n4, n5, n6, n7, n8] 
        elem_idx += 1 
    end 
 
    println("Generated structured mesh: $(nElem) elements, $(nNodes) nodes.") 
 
    return nodes, elements, dims 
end 
 
end