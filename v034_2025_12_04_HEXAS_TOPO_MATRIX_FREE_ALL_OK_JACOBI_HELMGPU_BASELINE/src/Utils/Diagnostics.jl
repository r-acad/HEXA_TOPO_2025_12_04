# // FILE: src/Utils/Diagnostics.jl
module Diagnostics

using CUDA
using Printf
using Dates

export log_status, check_memory, init_log_file, write_iteration_log

const LOG_HEADER = """
| Iter | Mesh Size | Total El | Active El | Radius | Cutoff | Compliance | Strain Energy | Avg L1 Stress | Vol Frac | Delta Rho | Refine? | Time (s) | Wall Time | VRAM |
|------|-----------|----------|-----------|--------|--------|------------|---------------|---------------|----------|-----------|---------|----------|-----------|------|
"""

function log_status(msg::String)
    timestamp = Dates.format(now(), "HH:MM:SS")
    println("[$timestamp] $msg")
    flush(stdout) 
end

function check_memory()
    if CUDA.functional()
        free_gpu, total_gpu = CUDA.available_memory(), CUDA.total_memory()
        return free_gpu
    end
    return 0
end

function format_memory_str()
    if CUDA.functional()
        free_gpu, total_gpu = CUDA.available_memory(), CUDA.total_memory()
        used_gb = (total_gpu - free_gpu) / 1024^3
        return @sprintf("%.1fG", used_gb)
    end
    return "CPU"
end

function init_log_file(filename::String, config::Dict)
    open(filename, "w") do io
        write(io, "HEXA FEM TOPOLOGY OPTIMIZATION LOG\n")
        write(io, "Start Date: $(now())\n")
        write(io, "Config Geometry: $(config["geometry"])\n")
        write(io, "="^180 * "\n")
        write(io, LOG_HEADER)
    end
    println("\n" * LOG_HEADER) 
end

function write_iteration_log(filename::String, iter, mesh_dims_str, nTotal, nActive, 
                             filter_R, threshold, compliance, strain_energy, avg_l1, 
                             vol_frac, delta_rho, refine_status, time_sec)
    
    vram_str = format_memory_str()
    wall_time = Dates.format(now(), "HH:MM:SS")
    
    line = @sprintf("| %4d | %9s | %8d | %9d | %6.3f | %6.3f | %10.3e | %13.3e | %13.3e | %8.4f | %8.2f%% | %7s | %8.2f | %9s | %4s |",
                    iter, mesh_dims_str, nTotal, nActive, filter_R, threshold,
                    compliance, strain_energy, avg_l1, vol_frac, 
                    delta_rho*100, refine_status, time_sec, wall_time, vram_str)
    
    open(filename, "a") do io
        println(io, line)
    end
    
    println(line)
    flush(stdout)
end

end