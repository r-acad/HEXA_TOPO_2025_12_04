
module GPUHelmholtz

using CUDA
using LinearAlgebra
using Printf
using ..Element

export HelmholtzWorkspace, setup_helmholtz_workspace, apply_gpu_filter!

# --- Data Structure to hold GPU Memory ---
mutable struct HelmholtzWorkspace{T}
    is_initialized::Bool
    radius::T
    
    # Mesh Data on GPU
    elements::CuVector{Int32} # Flattened connectivity (nElem * 8)
    Ae_base::CuMatrix{T}      # 8x8 Elemental Matrix (R^2*Ke + Me)
    inv_diag::CuVector{T}     # Jacobi Preconditioner
    
    # Solver Vectors
    r::CuVector{T}
    p::CuVector{T}
    z::CuVector{T}
    Ap::CuVector{T}
    x::CuVector{T} # Solution vector (nodal)
    b::CuVector{T} # RHS vector (nodal)
    
    # Connectivity Helpers
    nNodes::Int
    nElem::Int
    
    HelmholtzWorkspace{T}() where T = new{T}(false, T(0))
end

const GLOBAL_HELMHOLTZ_CACHE = HelmholtzWorkspace{Float32}()

# --- KERNELS ---

"""
    compute_rhs_kernel!(b, density, elem_vol_over_8, nElem)
    
Maps element density to nodal RHS vector: b = M * rho.
For a lumped mass matrix on a uniform grid, this is just adding rho*vol/8 to corners.
"""
function compute_rhs_kernel!(b, density, elements, val_scale, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        val = density[e] * val_scale
        
        # Flattened access: elements is length nElem * 8
        base_idx = (e - 1) * 8
        
        @inbounds for i in 1:8
            node = elements[base_idx + i]
            CUDA.atomic_add!(pointer(b, node), val)
        end
    end
    return nothing
end

"""
    matvec_kernel!(y, x, elements, Ae, nElem)
    
Computes y = A * x where A is the global Helmholtz operator.
Operation: y += Ae_local * x_local
"""
function matvec_kernel!(y, x, elements, Ae, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        
        # 1. Load X (nodal values) into registers
        x_loc_1 = x[elements[base_idx + 1]]
        x_loc_2 = x[elements[base_idx + 2]]
        x_loc_3 = x[elements[base_idx + 3]]
        x_loc_4 = x[elements[base_idx + 4]]
        x_loc_5 = x[elements[base_idx + 5]]
        x_loc_6 = x[elements[base_idx + 6]]
        x_loc_7 = x[elements[base_idx + 7]]
        x_loc_8 = x[elements[base_idx + 8]]
        
        # 2. Compute Y local = Ae * X local
        # We unroll the 8x8 matrix-vector multiplication
        @inbounds for r in 1:8
            val = Ae[r,1]*x_loc_1 + Ae[r,2]*x_loc_2 + Ae[r,3]*x_loc_3 + Ae[r,4]*x_loc_4 +
                  Ae[r,5]*x_loc_5 + Ae[r,6]*x_loc_6 + Ae[r,7]*x_loc_7 + Ae[r,8]*x_loc_8
            
            # 3. Atomic Add to global Y
            node = elements[base_idx + r]
            CUDA.atomic_add!(pointer(y, node), val)
        end
    end
    return nothing
end

"""
    extract_solution_kernel!(filtered_density, x, elements, nElem)
    
Maps nodal filtered values back to element centroids (simple averaging).
"""
function extract_solution_kernel!(filtered_density, x, elements, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        sum_val = 0.0f0
        @inbounds for i in 1:8
            sum_val += x[elements[base_idx + i]]
        end
        filtered_density[e] = sum_val / 8.0f0
    end
    return nothing
end

function compute_diagonal_kernel!(diag, elements, Ae_diag, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        @inbounds for i in 1:8
            node = elements[base_idx + i]
            val = Ae_diag[i]
            CUDA.atomic_add!(pointer(diag, node), val)
        end
    end
    return nothing
end

# --- HOST FUNCTIONS ---

function setup_helmholtz_workspace(elements_cpu::Matrix{Int}, 
                                   dx::T, dy::T, dz::T, radius::T) where T
    
    ws = GLOBAL_HELMHOLTZ_CACHE
    nElem = size(elements_cpu, 1)
    nNodes = maximum(elements_cpu)
    
    # Check if we need to re-initialize
    if !ws.is_initialized || ws.nElem != nElem || abs(ws.radius - radius) > 1e-5
        
        # 1. Compute Element Matrices (CPU)
        Ke, Me = Element.get_scalar_canonical_matrices(dx, dy, dz)
        Ae_cpu = (radius^2) .* Ke .+ Me
        
        # 2. Allocate/Upload to GPU
        # Flatten elements for faster access: [e1_n1, e1_n2... e2_n1...]
        elements_flat = vec(elements_cpu') # Transpose to group by element
        ws.elements = CuArray(Int32.(elements_flat))
        ws.Ae_base = CuArray(Ae_cpu)
        
        # 3. Compute Diagonal for Preconditioner
        diag_vec = CUDA.zeros(T, nNodes)
        Ae_diag_gpu = CuArray(diag(Ae_cpu))
        
        threads = 256
        blocks = cld(nElem, threads)
        @cuda threads=threads blocks=blocks compute_diagonal_kernel!(diag_vec, ws.elements, Ae_diag_gpu, nElem)
        
        # Jacobi Preconditioner (Inverse Diagonal)
        ws.inv_diag = 1.0f0 ./ diag_vec
        
        # 4. Allocate Solver Vectors
        ws.r  = CUDA.zeros(T, nNodes)
        ws.p  = CUDA.zeros(T, nNodes)
        ws.z  = CUDA.zeros(T, nNodes)
        ws.Ap = CUDA.zeros(T, nNodes)
        ws.x  = CUDA.zeros(T, nNodes)
        ws.b  = CUDA.zeros(T, nNodes)
        
        ws.nNodes = nNodes
        ws.nElem = nElem
        ws.radius = radius
        ws.is_initialized = true
        
        # println("  [GPU Filter] Workspace initialized. Memory: $(sizeof(ws.elements)/1024^2) MB")
    end
    return ws
end

function apply_gpu_filter!(density_cpu::Vector{T}, nElem_x, nElem_y, nElem_z, dx, dy, dz, radius, elements_cpu) where T
    
    # 1. Setup / Retrieve Cache
    ws = setup_helmholtz_workspace(elements_cpu, T(dx), T(dy), T(dz), T(radius))
    
    density_gpu = CuArray(density_cpu) # Upload current density
    filtered_gpu = CUDA.zeros(T, ws.nElem)
    
    threads = 256
    blocks = cld(ws.nElem, threads)
    
    # 2. Construct RHS (b)
    fill!(ws.b, 0.0f0)
    elem_vol = dx * dy * dz
    val_scale = elem_vol / 8.0f0
    @cuda threads=threads blocks=blocks compute_rhs_kernel!(ws.b, density_gpu, ws.elements, val_scale, ws.nElem)
    
    # 3. CG Solve: Ax = b
    # Initial Guess x = 0
    fill!(ws.x, 0.0f0) 
    ws.r .= ws.b
    ws.z .= ws.r .* ws.inv_diag # Apply Preconditioner
    ws.p .= ws.z
    
    rho_old = dot(ws.r, ws.z)
    
    tol = 1e-5
    max_iter = 200 # Filter doesn't need to be perfect
    
    for iter in 1:max_iter
        # Ap = A * p
        fill!(ws.Ap, 0.0f0)
        @cuda threads=threads blocks=blocks matvec_kernel!(ws.Ap, ws.p, ws.elements, ws.Ae_base, ws.nElem)
        
        alpha = rho_old / dot(ws.p, ws.Ap)
        
        ws.x .+= alpha .* ws.p
        ws.r .-= alpha .* ws.Ap
        
        if norm(ws.r) < tol * norm(ws.b)
            break
        end
        
        ws.z .= ws.r .* ws.inv_diag # Apply Preconditioner
        
        rho_new = dot(ws.r, ws.z)
        beta = rho_new / rho_old
        ws.p .= ws.z .+ beta .* ws.p
        
        rho_old = rho_new
    end
    
    # 4. Extract back to Elements
    @cuda threads=threads blocks=blocks extract_solution_kernel!(filtered_gpu, ws.x, ws.elements, ws.nElem)
    
    return Array(filtered_gpu)
end

end