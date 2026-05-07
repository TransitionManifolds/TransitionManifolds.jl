using Test
using Random: seed!
using TransitionManifolds
using Distances
using LinearAlgebra

# Base package
include("tests_utils.jl")
include("tests_td_gaussian_mmd.jl")
include("tests_em_diffusion_maps.jl")

# Extensions
include("tests_td_kernel_mmd.jl")
include("tests_graphs.jl")
