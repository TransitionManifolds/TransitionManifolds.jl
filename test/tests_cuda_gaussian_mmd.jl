using CUDA
using LinearAlgebra: triu

if CUDA.functional()
    @testset "CUDAExt: GaussianVStatMMD" begin
        @testset "compute_kernel_matrix_buffered matches CPU" begin
            # compute_kernel_matrix_buffered only fills the upper triangle (both
            # CPU and GPU); the lower triangle is left as uninitialized memory
            # from `similar`, so only the upper triangle is meaningful to compare.
            x = rand(Float64, 2, 5, 20)

            @testset "$T, blocksize $b" for T in (Float32, Float64), b in (1, 3, 20)
                alg = GaussianVStatMMD(; bandwidth=0.123, blocksize=b)
                xt = T.(x)
                expected = TransitionManifolds.compute_kernel_matrix_buffered(xt, alg)
                if b == 3 #logging for uneven blocksize
                    actual = @test_logs (:info, r"does not evenly divide") TransitionManifolds.compute_kernel_matrix_buffered(
                        cu(xt), alg
                    )
                else
                    actual = TransitionManifolds.compute_kernel_matrix_buffered(cu(xt), alg)
                end
                @test triu(actual) ≈ triu(expected)
            end
        end

        @testset "compute_distances end-to-end" begin
            x = rand(Float32, 2, 5, 20)
            prob_cpu = TransitionDistanceProblem(x)
            prob_gpu = TransitionDistanceProblem(cu(x))

            @testset "fixed bandwidth" begin
                alg = GaussianVStatMMD(; bandwidth=0.123)
                res_cpu = compute_distances(prob_cpu, alg)
                res_gpu = compute_distances(prob_gpu, alg)
                @test res_gpu.distances ≈ res_cpu.distances
            end

            @testset "automatic bandwidth" begin
                alg = GaussianVStatMMD()
                res_cpu = compute_distances(prob_cpu, alg)
                res_gpu = compute_distances(prob_gpu, alg)
                @test res_gpu.distances ≈ res_cpu.distances
            end
        end
    end
else
    @info "Skipping CUDAExt tests: no functional GPU detected"
end
