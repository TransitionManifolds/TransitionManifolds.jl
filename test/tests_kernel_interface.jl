@testset "compute_kernel_matrix" begin
    alg = GaussianDStatMMD(sqrt(2))
    seed!(1234)
    x = [rand(2, 4), rand(2, 3), rand(2, 2)]

    kmat = TransitionManifolds.compute_kernel_matrix(x, alg)
    @test size(kmat) == (3, 3)
    @test kmat[1, 1] == TransitionManifolds.kernel_eval(x[1], alg)
    @test kmat[2, 2] == TransitionManifolds.kernel_eval(x[2], alg)
    @test kmat[3, 3] == TransitionManifolds.kernel_eval(x[3], alg)
    @test kmat[1, 2] == TransitionManifolds.kernel_eval(x[1], x[2], alg)
    @test kmat[1, 3] == TransitionManifolds.kernel_eval(x[1], x[3], alg)
    @test kmat[2, 3] == TransitionManifolds.kernel_eval(x[2], x[3], alg)
end

@testset "convert_kernel_to_distance_matrix" begin
    # Dij = Kii + Kjj - 2Kij.
    # K is assmumed to be symmetric and only the upper triangular is used
    K = [3.0 2 1; 0 4 2; 0 0 5]
    D = [0 3.0 6; 3 0 5; 6 5 0]

    TransitionManifolds.convert_kernel_to_distance_matrix!(K)
    @test K == D
end
