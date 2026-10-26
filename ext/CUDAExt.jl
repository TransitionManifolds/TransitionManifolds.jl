module CUDAExt

using TransitionManifolds
using CUDA
using LinearAlgebra: mul!
using ProgressMeter: Progress, next!

# VStat + Contiguous, on the GPU: blockwise computation using CUBLAS + broadcasting
# instead of the CPU implementation in td_gaussian_mmd.jl
function TransitionManifolds.compute_kernel_matrix_buffered(
    data::CuArray{T,3}, alg::GaussianVStatMMD; progress::Bool=false
)::Matrix{T} where {T<:AbstractFloat}
    dim, n_samples, n_anchors = size(data)
    inv_sigma_sq = T(-1.0 / (alg.bandwidth * alg.bandwidth))

    blocksize = min(alg.blocksize, n_anchors)
    if n_anchors % blocksize != 0
        @info "blocksize=$blocksize does not evenly divide n_anchors=$n_anchors"
    end
    block_ranges = [i:min(i + blocksize - 1, n_anchors) for i in 1:blocksize:n_anchors]
    n_iter = length(block_ranges)

    # allocations: sized for the full-size block, a small buffer is
    # allocated for the last block.
    kernel_matrix = similar(data, n_anchors, n_anchors)
    max_buffer = similar(data, n_samples * blocksize, n_samples * blocksize)
    max_buffer2 = similar(data, 1, blocksize, 1, blocksize)

    # We need ||x||^2 for the distance expansion ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    # compute norm once for entire dataset
    data_flat = reshape(data, (dim, n_samples * n_anchors))
    norms_sq = sum(abs2, data_flat; dims=1) # (1, total_points)

    pbar = Progress(
        binomial(n_iter, 2) + n_iter;
        enabled=progress,
        showspeed=true,
        desc="Computing Kernel Matrix:",
    )

    for (i_pos, i_range) in enumerate(block_ranges)
        bi = length(i_range)
        idx_i = ((first(i_range) - 1) * n_samples + 1):(last(i_range) * n_samples)

        Xi = view(data_flat, :, idx_i)
        norms_i = view(norms_sq, :, idx_i)

        for j_range in block_ranges[i_pos:end]
            bj = length(j_range)
            idx_j = ((first(j_range) - 1) * n_samples + 1):(last(j_range) * n_samples)

            Xj = view(data_flat, :, idx_j)
            norms_j = view(norms_sq, :, idx_j)

            full_size = (bi == blocksize) && (bj == blocksize)
            buffer = full_size ? max_buffer : similar(data, n_samples * bi, n_samples * bj)
            buffer2 = full_size ? max_buffer2 : similar(data, 1, bi, 1, bj)

            # (Samples*bi x dim) * (dim x Samples*bj) => (Samples*bi) x (Samples*bj)
            mul!(buffer, Xi', Xj)

            # ||x||^2 + ||y||^2 - 2<x,y> and kernel
            buffer .*= T(-2)
            buffer .+= norms_j .+ norms_i'
            buffer .= exp.(buffer .* inv_sigma_sq)

            # reshape to separate anchors from samples
            K = reshape(buffer, (n_samples, bi, n_samples, bj))

            # sum sub-blocks of size (samples x samples)
            sum!(buffer2, K)
            kernel_matrix[i_range, j_range] .= reshape(buffer2, (bi, bj))

            next!(pbar; step=1, showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])
        end
    end

    kernel_matrix ./= n_samples^2
    # we don't have to worry about handling CuArray down the line.
    return Array(kernel_matrix)
end

"""
    cu(prob::TransitionDistanceProblem{T,W,Contiguous}) -> TransitionDistanceProblem

Move the `data` (and `weights`, if present) of `prob` onto the GPU as `CuArray`s.

Only the [`Contiguous`](@ref) layout is supported, since that is the layout used
by the GPU kernel implementations, e.g. in [`GaussianVStatMMD`](@ref).
"""
function CUDA.cu(
    prob::TransitionDistanceProblem{T,W,Contiguous}
)::TransitionDistanceProblem where {T,W}
    weights = isnothing(prob.weights) ? nothing : cu(prob.weights)
    return TransitionDistanceProblem(cu(prob.data), weights)
end

"""
    cu(pres::PreprocessResult) -> PreprocessResult

Move the `TransitionDistanceProblem` held by `pres` onto the GPU, see [`cu(::TransitionDistanceProblem)`](@ref).
"""
function CUDA.cu(pres::PreprocessResult)::PreprocessResult
    return PreprocessResult(cu(pres.prob), pres.info)
end

end # module
