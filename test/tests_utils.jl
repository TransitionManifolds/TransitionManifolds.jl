@testset "convert_contiguous_to_jagged" begin
    x = rand(Int32, 2, 3, 4)
    w = rand(2, 3, 4)

    @testset "no weights" begin
        p = TransitionDistanceProblem(x)
        p_j = TransitionManifolds.convert_contiguous_to_jagged(p)

        @test p_j isa TransitionDistanceProblem{Int32,Nothing,Jagged}
        for i in 1:4
            @test p.data[:, :, i] == p_j.data[i]
        end
    end

    @testset "weights" begin
        p = TransitionDistanceProblem(x, w)
        p_j = TransitionManifolds.convert_contiguous_to_jagged(p)

        @test p_j isa TransitionDistanceProblem{Int32,Float64,Jagged}
        for i in 1:4
            @test p.data[:, :, i] == p_j.data[i]
            @test p.weights[:, :, i] == p_j.weights[i]
        end
    end
end

@testset "convert_kernel_to_distance_matrix" begin
    # Dij = Kii + Kjj - 2Kij.
    # K is assmumed to be symmetric and only the upper triangular is used
    K = [3.0 2 1; 0 4 2; 0 0 5]
    D = [0 3.0 6; 3 0 5; 6 5 0]

    TransitionManifolds.convert_kernel_to_distance_matrix!(K)
    @test K == D
end

@testset "subsamples_from_jagged" begin
    @testset "sufficient data" begin
        data = [rand(3, 6), rand(3, 4)]
        subsamples = TransitionManifolds.subsamples_from_jagged(data, 4)
        @test size(subsamples) == (3, 4)
    end

    @testset "insufficient data" begin
        data = [rand(3, 2), rand(3, 3)]
        subsamples = TransitionManifolds.subsamples_from_jagged(data, 10)
        @test size(subsamples) == (3, 5)
    end
end

@testset "tune_bandwidth_gaussian" begin
    quant = 0.95
    val_at_quant = 0.01
    metric = SqEuclidean()
    seed!(123)
    data = rand(2, 1000)
    # bandwidth is tuned so that at the 95th percentile of distances 
    # it produces a similarity of 0.01.
    bandwidth = TransitionManifolds.tune_bandwidth_gaussian(
        data; quant=quant, val_at_quant=val_at_quant
    )
    @test bandwidth > 0
    kernel_evals = [
        exp(-metric(data[:, i], data[:, j]) / bandwidth^2) for i in axes(data, 2) for
        j in 1:(i - 1)
    ]
    # check that we get `val_at_quant` at the `quant` quantile
    sort!(kernel_evals)
    idx = Int(round(length(kernel_evals) * (1 - quant)))
    @test kernel_evals[idx] ≈ val_at_quant atol = 1e-4
end

@testset "normalize_cloud" begin
    # 5 points: (1,1,1), (2,2,2), ..., (5,5,5)
    data = range(1, 5) .* ones(3)'
    # `normalize_cloud` should rotate these points onto a single axis
    # with center at the mean and scaled so that the largest value is 1
    @testset "q=0.99" begin
        # all points are inliers
        # centered to (-2,-2,-2), ..., (2,2,2)
        # rotated and scaled to (-1,0,0), (-0.5,0,0), ..., (1,0,0)
        expected = zeros(5, 3)
        expected[:, 1] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        data_n = normalize_cloud(data; quant=0.99)
        # could be rotated either way
        @test data_n ≈ expected || data_n ≈ -expected
    end

    @testset "q=0.6" begin
        # the two outer points (1,1,1) and (5,5,5) are outliers
        # so they get scaled to larger values
        expected = zeros(5, 3)
        expected[:, 1] = [-2.0, -1.0, 0.0, 1.0, 2.0]
        data_n = normalize_cloud(data; quant=0.6)
        # could be rotated either way
        @test data_n ≈ expected || data_n ≈ -expected
    end
end
