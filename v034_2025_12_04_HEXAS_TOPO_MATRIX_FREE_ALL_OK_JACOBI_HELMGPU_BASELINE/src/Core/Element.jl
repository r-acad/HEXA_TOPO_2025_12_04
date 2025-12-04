# FILE: .\Element.jl
module Element

using LinearAlgebra
export NAT_COORDS, shape_functions, material_matrix, hex_element_stiffness, get_canonical_stiffness, get_scalar_canonical_matrices

const NAT_COORDS = Float32[
    -1 -1 -1;
     1 -1 -1;
     1  1 -1;
    -1  1 -1;
    -1 -1  1;
     1 -1  1;
     1  1  1;
    -1  1  1
]

"""
    shape_functions(xi, eta, zeta)

Computes the trilinear shape functions and their derivatives at (xi, eta, zeta).
Returns (N, dN) with N=8 shape values, dN=8x3 derivative matrix.
"""
function shape_functions(xi, eta, zeta)
    N  = zeros(Float32, 8)
    dN = zeros(Float32, 8, 3)
    
    # 1/8
    p1 = Float32(0.125)
    
    @inbounds for i in 1:8
        xi_i, eta_i, zeta_i = NAT_COORDS[i,1], NAT_COORDS[i,2], NAT_COORDS[i,3]
        
        term_xi   = (1.0f0 + xi*xi_i)
        term_eta  = (1.0f0 + eta*eta_i)
        term_zeta = (1.0f0 + zeta*zeta_i)
        
        N[i] = p1 * term_xi * term_eta * term_zeta
        
        dN[i,1] = p1 * xi_i * term_eta * term_zeta
        dN[i,2] = p1 * term_xi * eta_i * term_zeta
        dN[i,3] = p1 * term_xi * term_eta * zeta_i
    end
    return N, dN
end

"""
    material_matrix(E, nu)

Constructs the 6x6 isotropic material matrix for 3D elasticity.
"""
function material_matrix(E::Float32, nu::Float32)
    inv_den = 1.0f0 / ((1.0f0 + nu) * (1.0f0 - 2.0f0 * nu))
    factor = E * inv_den
    
    c1 = (1.0f0 - nu) * factor
    c2 = nu * factor
    c3 = ((1.0f0 - 2.0f0 * nu) / 2.0f0) * factor
    
    # Voigt notation: xx, yy, zz, xy, yz, xz
    D = zeros(Float32, 6, 6)
    
    D[1,1] = c1; D[1,2] = c2; D[1,3] = c2
    D[2,1] = c2; D[2,2] = c1; D[2,3] = c2
    D[3,1] = c2; D[3,2] = c2; D[3,3] = c1
    
    D[4,4] = c3
    D[5,5] = c3
    D[6,6] = c3
    
    return D
end

"""
    hex_element_stiffness(nodes, E, nu)

Computes the 24x24 stiffness for a hex element.
"""
function hex_element_stiffness(nodes::AbstractMatrix{Float32}, E::Float32, nu::Float32)
    D = material_matrix(E, nu)
    ke = zeros(Float32, 24, 24)

    a = 1.0f0 / sqrt(3.0f0)
    gauss_pts = Float32[-a, a]
    
    # Pre-allocate B matrix
    B = zeros(Float32, 6, 24)

    @inbounds for xi in gauss_pts, eta in gauss_pts, zeta in gauss_pts
        # Shape functions derivatives
        _, dN_dxi = shape_functions(xi, eta, zeta)
        J    = transpose(dN_dxi) * nodes
        detJ = det(J)
        
        if detJ <= 0.0f0 
            # In topology opt, elements can become distorted or inverted.
            # A rigorous solver handles this, but here we error or warn.
            # For robust production code, return a penalty stiffness or error out.
            error("Non-positive Jacobian detected in element stiffness calculation.")
        end
        
        invJ = inv(J)
        dN_dx = dN_dxi * transpose(invJ)

        fill!(B, 0.0f0)
        
        for i in 1:8
            idx = 3*(i-1)
            dN_i = view(dN_dx, i, :)

            B[1, idx+1] = dN_i[1]
            B[2, idx+2] = dN_i[2]
            B[3, idx+3] = dN_i[3]

            B[4, idx+1] = dN_i[2]
            B[4, idx+2] = dN_i[1]
            
            B[5, idx+2] = dN_i[3]
            B[5, idx+3] = dN_i[2]
            
            B[6, idx+1] = dN_i[3]
            B[6, idx+3] = dN_i[1]
        end

        # ke += B^T * D * B * detJ * weight (weight=1.0)
        ke .+= transpose(B) * D * B * detJ
    end

    return ke
end

"""
    get_canonical_stiffness(dx, dy, dz, nu)

Computes a "base" stiffness matrix for a standard rectangular element of size dx*dy*dz
with Young's Modulus E=1.0. 
This optimization significantly speeds up assembly for structured meshes.
"""
function get_canonical_stiffness(dx::Float32, dy::Float32, dz::Float32, nu::Float32)
    # Centered at origin
    nodes = Float32[
        0.0 0.0 0.0;
        dx  0.0 0.0;
        dx  dy  0.0;
        0.0 dy  0.0;
        0.0 0.0 dz;
        dx  0.0 dz;
        dx  dy  dz;
        0.0 dy  dz
    ]
    
    nodes .-= [dx/2 dy/2 dz/2]
    
    return hex_element_stiffness(nodes, 1.0f0, nu)
end

"""
    get_scalar_canonical_matrices(dx, dy, dz)

Computes the 8x8 element Mass matrix (Me) and Stiffness/Laplacian matrix (Ke) 
for a scalar field (like density or temperature) on a rectangular Hex8 element.
Used for Helmholtz filtering.
"""
function get_scalar_canonical_matrices(dx::Float32, dy::Float32, dz::Float32)
    # Nodes for a canonical element centered at 0
    nodes = Float32[
        -dx/2 -dy/2 -dz/2;
         dx/2 -dy/2 -dz/2;
         dx/2  dy/2 -dz/2;
        -dx/2  dy/2 -dz/2;
        -dx/2 -dy/2  dz/2;
         dx/2 -dy/2  dz/2;
         dx/2  dy/2  dz/2;
        -dx/2  dy/2  dz/2
    ]

    Ke = zeros(Float32, 8, 8)
    Me = zeros(Float32, 8, 8)

    # 2-point Gauss quadrature
    a = 1.0f0 / sqrt(3.0f0)
    gauss_pts = Float32[-a, a]

    @inbounds for xi in gauss_pts, eta in gauss_pts, zeta in gauss_pts
        N, dN_dxi = shape_functions(xi, eta, zeta)
        
        # Jacobian
        J = transpose(dN_dxi) * nodes
        detJ = det(J)
        invJ = inv(J)
        
        # Derivatives w.r.t physical coordinates
        dN_dx = dN_dxi * transpose(invJ)
        
        weight = detJ # Gauss weight is 1.0 for 2-point rule, so just detJ
        
        # Scalar Laplacian (Conductivity) k = integral(gradN * gradN^T)
        Ke .+= (dN_dx * transpose(dN_dx)) .* weight
        
        # Scalar Mass m = integral(N * N^T)
        Me .+= (N * transpose(N)) .* weight
    end

    return Ke, Me
end

end