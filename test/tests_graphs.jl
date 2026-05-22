using Graphs

@testset "preprocess graphs" begin
    g1 = cycle_graph(4)
    g2 = complete_graph(4)
    g3 = star_graph(4)
    data = [g1 g2 g3; g3 g1 g2]  # n_samples = 2, n_anchors = 3

    @testset "mode adjacency_matrix" begin
        @testset "data" begin
            flat_adj1 = [1, 0, 1, 1, 0, 1]
            flat_adj2 = [1, 1, 1, 1, 1, 1]
            flat_adj3 = [1, 1, 1, 0, 0, 0]
            expected = cat(
                [flat_adj1 flat_adj3], [flat_adj2 flat_adj1], [flat_adj3 flat_adj2]; dims=3
            )
            pres = preprocess(data, :adjacency_matrix)
            @test pres.prob isa TransitionDistanceProblem{Float32,Nothing,Contiguous}
            @test pres.prob.data == expected
        end

        @testset "throws when different number of nodes" begin
            g_other = cycle_graph(5)
            data_invalid = [g1 g2 g3; g3 g_other g2]
            @test_throws ArgumentError preprocess(data_invalid, :adjacency_matrix)
        end
    end
end
