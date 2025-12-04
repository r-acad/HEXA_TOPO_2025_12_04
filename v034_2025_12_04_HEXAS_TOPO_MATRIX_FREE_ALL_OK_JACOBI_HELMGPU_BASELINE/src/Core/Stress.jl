module Stress 
 
using LinearAlgebra 
using ..Element 
export compute_stress_field 
 
""" 
    compute_element_stress(element_nodes, element_disp, E, nu) 
 
Computes the 3x3 stress tensor at the center of a hexahedral element via B*U => strain => stress. 
""" 
function compute_element_stress(element_nodes::Array{Float32,2}, 
                                 element_disp::Array{Float32,1}, 
                                 E::Float32, nu::Float32) 
    D = Element.material_matrix(E, nu) 
     
    xi, eta, zeta = Float32(0.0), Float32(0.0), Float32(0.0)  
    _, dN_dxi = Element.shape_functions(xi, eta,  
zeta) 
    J = transpose(dN_dxi)*element_nodes 
    detJ = det(J) 
    if detJ <= Float32(0.0)  
        error("Non-positive Jacobian!") 
    end 
    invJ = inv(J) 
    dN_dx = dN_dxi * transpose(invJ) 
 
     
    B = zeros(Float32, 6, 24) 
    for i in 1:8 
        idx = 3*(i-1)+1 
        dN_i = dN_dx[i, :] 
 
         
     
        B[1, idx]   = dN_i[1] 
        B[2, idx+1] = dN_i[2] 
        B[3, idx+2] = dN_i[3] 
 
         
        B[4, idx]   = dN_i[2]   
        B[4, idx+1] = dN_i[1] 
        B[5, idx+1] = dN_i[3]   
        B[5, idx+2] = dN_i[2] 
         
        B[6, idx]   = dN_i[3]   
        B[6, idx+2] = dN_i[1] 
    end 
 
    strain = B * element_disp 
    stress_voigt = D * strain 
 
    σ = zeros(Float32, 3, 3) 
    σ[1,1] = stress_voigt[1]   
    σ[2,2] = stress_voigt[2]   
    σ[3,3] = stress_voigt[3]   
    σ[1,2] = stress_voigt[4]; σ[2,1] = stress_voigt[4]   
    σ[2,3] = stress_voigt[5]; σ[3,2] = stress_voigt[5]   
    σ[1,3] = stress_voigt[6]; σ[3,1] = stress_voigt[6]   
    return σ 
end 
 
""" 
    compute_principal_and_vonmises(σ) 
 
Given a 3x3 stress tensor, returns (principal_stresses, von_mises). 
Principal are sorted descending. Von Mises uses standard formula. 
""" 
function compute_principal_and_vonmises(σ::Matrix{Float32}) 
    eigvals = eigen(σ).values 
     
    principal_stresses = sort(eigvals, rev=true) 
 
    σxx = σ[1,1] 
    σyy = σ[2,2] 
    σzz = σ[3,3] 
    σxy = σ[1,2] 
    σyz = σ[2,3] 
    σxz = σ[1,3] 
 
    vm = sqrt(Float32(0.5) * ((σxx-σyy)^2 + (σyy-σzz)^2 + (σzz-σxx)^2) + 
              Float32(3.0)*(σxy^2 + σyz^2 + σxz^2))  
 
    return principal_stresses, vm 
end 
 
""" 
    compute_stress_field(nodes, elements, U, E, nu, density) 
 
Loop over elements, compute: 
  - principal stresses (3 x nElem) 
  - von Mises (1 x nElem) 
  - full stress in Voigt form (6 x nElem) 
  - l1 stress norm (1 x nElem) <-- MODIFIED 
 
Returns (principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field). <-- MODIFIED 
""" 
function compute_stress_field(nodes, elements, U, E::Float32, nu::Float32, density::Vector{Float32}) 
    nElem = size(elements, 1) 
    principal_field     = zeros(Float32, 3, nElem) 
    vonmises_field    = zeros(Float32, nElem) 
    full_stress_voigt = zeros(Float32, 6,  
nElem) 
    l1_stress_norm_field = zeros(Float32, nElem)  
 
    for e in 1:nElem 
        conn = elements[e, :] 
        element_nodes = nodes[conn, :] 
 
         
        element_disp = zeros(Float32, 24) 
        for i in 1:8 
            global_node = conn[i] 
            element_disp[3*(i-1)+1 : 3*i] = U[3*(global_node-1)+1 : 3*global_node] 
         
        end 
 
        # Use the scaled Young's Modulus for this element 
        E_local = E * density[e] 
         
        σ = compute_element_stress(element_nodes, element_disp, E_local, nu) 
        (principal, vm) = compute_principal_and_vonmises(σ) 
 
         
        l1_norm = abs(principal[1]) + abs(principal[2]) + abs(principal[3]) 
         
 
        principal_field[:, e] = principal 
        vonmises_field[e]       = vm 
        l1_stress_norm_field[e] = l1_norm  
 
         
        full_stress_voigt[:, e] .= (σ[1,1], σ[2,2], σ[3,3], σ[1,2], σ[2,3], σ[1,3]) 
    end 
 
    return principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field  
end 
 
end 
