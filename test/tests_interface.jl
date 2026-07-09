@testset "TransitionDistanceProblem" begin
    @testset "Contiguous" begin
        x = rand(4, 3, 2)
        p = TransitionDistanceProblem(x)
        @test p isa TransitionDistanceProblem{Float64,Nothing,Contiguous}
        @test p.data === x
        @test isnothing(p.weights)
        @test n_anchors(p) == 2
        @test n_samples(p) == 3
        @test dimension(p) == 4
        @test layout(p) === Contiguous
    end

    @testset "Jagged" begin
        x = [rand(4, 3), rand(4, 5)]
        p = TransitionDistanceProblem(x)
        @test p isa TransitionDistanceProblem{Float64,Nothing,Jagged}
        @test p.data === x
        @test isnothing(p.weights)
        @test n_anchors(p) == 2
        @test n_samples(p) == [3, 5]
        @test dimension(p) == 4
        @test layout(p) === Jagged
    end

    @testset "weights Contiguous" begin
        x = rand(4, 3, 2)
        w = rand(3, 2)
        p = TransitionDistanceProblem(x, w)
        @test p isa TransitionDistanceProblem{Float64,Float64,Contiguous}
        @test p.data === x
        @test p.weights === w
        @test n_anchors(p) == 2
        @test n_samples(p) == 3
        @test dimension(p) == 4
        @test layout(p) === Contiguous
    end

    @testset "weights Jagged" begin
        x = [rand(4, 3), rand(4, 5)]
        w = [rand(3), rand(5)]
        p = TransitionDistanceProblem(x, w)
        @test p isa TransitionDistanceProblem{Float64,Float64,Jagged}
        @test p.data === x
        @test p.weights === w
        @test n_anchors(p) == 2
        @test n_samples(p) == [3, 5]
        @test dimension(p) == 4
        @test layout(p) === Jagged
    end

    @testset "T and W Contiguous" begin
        x = rand(Int32, 4, 3, 2)
        w = rand(Float32, 3, 2)
        p = TransitionDistanceProblem(x, w)
        @test p isa TransitionDistanceProblem{Int32,Float32,Contiguous}
        @test p.data === x
        @test p.weights === w
    end

    @testset "T and W Jagged" begin
        x = [rand(Int32, 4, 3), rand(Int32, 4, 5)]
        w = [rand(Float32, 3), rand(Float32, 5)]
        p = TransitionDistanceProblem(x, w)
        @test p isa TransitionDistanceProblem{Int32,Float32,Jagged}
        @test p.data === x
        @test p.weights === w
    end

    @testset "Jagged different d" begin
        x = [rand(4, 3), rand(3, 3)]
        @test_throws ArgumentError TransitionDistanceProblem(x)
    end

    @testset "data and weights dont match" begin
        xc = rand(4, 3, 2)
        wc = rand(3, 2)
        wc_wrong = rand(2, 2)
        xj = [rand(4, 3), rand(4, 5)]
        wj = [rand(3), rand(5)]
        wj_wrong = [rand(3), rand(4)]

        @test_throws ArgumentError TransitionDistanceProblem(xc, wc_wrong)
        @test_throws ArgumentError TransitionDistanceProblem(xj, wj_wrong)
        @test_throws ArgumentError TransitionDistanceProblem(xj, wc)
        @test_throws ArgumentError TransitionDistanceProblem(xc, wj)
    end

    @testset "equality and hash" begin
        p1 = TransitionDistanceProblem(ones(2, 3, 4))
        p2 = TransitionDistanceProblem(ones(2, 3, 4))
        p3 = TransitionDistanceProblem(zeros(2, 3, 4))
        p4 = TransitionDistanceProblem(ones(2, 3, 4), ones(3, 4))
        p5 = TransitionDistanceProblem(ones(2, 3, 4), ones(3, 4))
        p6 = TransitionDistanceProblem(ones(2, 3, 4), 2 .* ones(3, 4))
        p7 = TransitionDistanceProblem([ones(2, 3) for _ in 1:4])
        p8 = TransitionDistanceProblem([ones(2, 3) for _ in 1:4])
        p9 = TransitionDistanceProblem([ones(2, 3) for _ in 1:4], [ones(3) for _ in 1:4])
        p10 = TransitionDistanceProblem([ones(2, 3) for _ in 1:4], [ones(3) for _ in 1:4])
        p11 = TransitionDistanceProblem(
            [ones(2, 3) for _ in 1:4], [2 * ones(3) for _ in 1:4]
        )
        ps = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]
        equal_pairs = [(1, 2), (4, 5), (7, 8), (9, 10)]

        @testset "$i vs $j" for i in 1:11, j in i:11
            if i == j || (i, j) ∈ equal_pairs
                @test ps[i] == ps[j]
                @test hash(ps[i]) == hash(ps[j])

            else
                @test ps[i] != ps[j]
                # inequality of hashes is not guaranteed
                # but works almost always
                @test hash(ps[i]) != hash(ps[j])
            end
        end
    end

    @testset "cat anchors" begin
        @testset "no weights" begin
            p1 = TransitionDistanceProblem(rand(4, 3, 2))
            p2 = TransitionDistanceProblem(rand(4, 3, 5))
            p3 = TransitionDistanceProblem(rand(4, 3, 1))
            p = cat_anchors(p1, p2, p3)
            @test p isa TransitionDistanceProblem{Float64,Nothing,Contiguous}
            @test size(p.data) == (4, 3, 8)
            @test p.data[:, :, 1:2] == p1.data
            @test p.data[:, :, 3:7] == p2.data
            @test p.data[:, :, 8:8] == p3.data
        end

        @testset "weights" begin
            p1 = TransitionDistanceProblem(rand(4, 3, 2), rand(3, 2))
            p2 = TransitionDistanceProblem(rand(4, 3, 5), rand(3, 5))
            p = cat_anchors(p1, p2)
            @test p isa TransitionDistanceProblem{Float64,Float64,Contiguous}
            @test size(p.data) == (4, 3, 7)
            @test size(p.weights) == (3, 7)
            @test p.data[:, :, 1:2] == p1.data
            @test p.weights[:, 1:2] == p1.weights
            @test p.data[:, :, 3:7] == p2.data
            @test p.weights[:, 3:7] == p2.weights
        end
    end

    @testset "append anchors" begin
        @testset "no weights" begin
            p1_data = [rand(4, 3), rand(4, 2)]
            p1 = TransitionDistanceProblem(copy(p1_data))
            p2 = TransitionDistanceProblem([rand(4, 4), rand(4, 3), rand(4, 2)])
            p3 = TransitionDistanceProblem([rand(4, 3)])
            append_anchors!(p1, p2, p3)
            @test length(p1.data) == 6
            @test p1.data[1:2] == p1_data
            @test p1.data[3:5] == p2.data
            @test p1.data[6:6] == p3.data
        end

        @testset "weights" begin
            p1_data = [rand(4, 3), rand(4, 2)]
            p1_weights = [rand(3), rand(2)]
            p1 = TransitionDistanceProblem(copy(p1_data), copy(p1_weights))
            p2 = TransitionDistanceProblem(
                [rand(4, 4), rand(4, 3), rand(4, 2)], [rand(4), rand(3), rand(2)]
            )
            append_anchors!(p1, p2)
            @test length(p1.data) == 5
            @test length(p1.weights) == 5
            @test p1.data[1:2] == p1_data
            @test p1.weights[1:2] == p1_weights
            @test p1.data[3:5] == p2.data
            @test p1.weights[3:5] == p2.weights
        end
    end
end
