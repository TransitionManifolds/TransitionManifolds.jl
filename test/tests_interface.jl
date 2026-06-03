@testset "TransitionDistanceProblem" begin
    @testset "Contiguous" begin
        x = rand(4, 3, 2)
        p = TransitionDistanceProblem(x)
        @test p isa TransitionDistanceProblem{Float64,Nothing,Contiguous}
        @test p.data === x
        @test isnothing(p.weights)
    end

    @testset "Jagged" begin
        x = [rand(4, 3), rand(4, 5)]
        p = TransitionDistanceProblem(x)
        @test p isa TransitionDistanceProblem{Float64,Nothing,Jagged}
        @test p.data === x
        @test isnothing(p.weights)
    end

    @testset "weights Contiguous" begin
        x = rand(4, 3, 2)
        w = rand(3, 2)
        p = TransitionDistanceProblem(x, w)
        @test p isa TransitionDistanceProblem{Float64,Float64,Contiguous}
        @test p.data === x
        @test p.weights === w
    end

    @testset "weights Jagged" begin
        x = [rand(4, 3), rand(4, 5)]
        w = [rand(3), rand(5)]
        p = TransitionDistanceProblem(x, w)
        @test p isa TransitionDistanceProblem{Float64,Float64,Jagged}
        @test p.data === x
        @test p.weights === w
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
end
