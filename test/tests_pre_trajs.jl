@testset "Trajectories" begin
    @testset "constructor" begin
        t1 = rand(3, 10)
        t2 = rand(3, 5)

        @testset "multiple trajs" begin
            trajs = Trajectories([t1, t2])
            @test trajs.trajs == [t1, t2]
            @test trajs.n_trajs == 2
            @test trajs.d == 3
            @test trajs.n_points == 15
        end

        @testset "single traj" begin
            trajs = Trajectories(t1)
            @test trajs.trajs == [t1]
            @test trajs.n_trajs == 1
            @test trajs.d == 3
            @test trajs.n_points == 10
        end

        @testset "err empty trajs" begin
            @test_throws ArgumentError Trajectories(Matrix{Float64}[])
        end

        @testset "err dimension mismatch" begin
            t = rand(2, 5)
            @test_throws ArgumentError Trajectories([t1, t])
        end

        @testset "err too short" begin
            t = rand(2, 1)
            @test_throws ArgumentError Trajectories([t1, t])
        end
    end

    @testset "iteration and length" begin
        t1 = rand(3, 10)
        t2 = rand(3, 5)
        trajs = Trajectories([t1, t2])

        @test length(trajs) == 15
        points = [p for p in trajs]
        @test points[1:10] == eachcol(t1)
        @test points[11:15] == eachcol(t2)
    end

    @testset "indexing" begin
        t1 = rand(3, 3)
        t2 = rand(3, 5)
        t3 = rand(3, 4)
        trajs = Trajectories([t1, t2, t3])

        @test trajs.offsets == [1, 4, 9, 13]
        @test trajs[begin] == t1[:, 1]
        @test trajs[end] == t3[:, 4]

        for i in 1:3
            @test trajs[i] == t1[:, i]
        end
        for i in 4:8
            @test trajs[i] == t2[:, i - 3]
        end
        for i in 9:12
            @test trajs[i] == t3[:, i - 8]
        end

        @test trajs[4:8] == eachcol(t2)
        @test trajs[[1, 5, 11]] == [t1[:, 1], t2[:, 2], t3[:, 3]]

        @test_throws BoundsError trajs[13]
    end

    @testset "sample_points" begin
        t1 = rand(3, 10)
        t2 = rand(3, 5)
        trajs = Trajectories([t1, t2])
        samples = TransitionManifolds.sample_points(trajs, 6)
        @test size(samples) == (3, 6)

        all = vcat(eachcol(t1), eachcol(t2))
        for s in eachcol(samples)
            @test s in all
            # end points are not allowed
            @test s != all[10]
            @test s != all[15]
        end
    end

    @testset "mean_jump_dist" begin
        t1 = hcat([0, 1], [0, 0], [0.5, 0])
        t2 = hcat([0, 0], [0, -3])
        trajs = Trajectories([t1, t2])
        expected = (1.0 + 0.5 + 3.0) / 3
        @test TransitionManifolds.mean_jump_dist(trajs, Euclidean()) ≈ expected
    end

    @testset "preprocess" begin
        @testset "correct samples" begin
            trajs = Trajectories(
                hcat([0, 2], [0, 1], [0, 0.25], [0, 0], [0.5, 0], [1.5, 0])
            )
            anchors = hcat([0, 1.5], [0, 0])

            @testset "default" begin
                res = preprocess(trajs; anchors=anchors, max_dist=1)
                samples = res.prob.data
                @test length(samples) == 2

                # first anchor:
                # close start points are [0, 2] and [0, 1]
                # -> samples are [0, 1] and [0, 0.25]
                s1 = samples[1]
                @test size(s1) == (2, 2)
                @test [0, 1] in eachcol(s1)
                @test [0, 0.25] in eachcol(s1)

                # second anchor:
                # close start points are [0, 1], [0, 0.25], [0, 0], [0.5, 0]
                # -> samples are [0, 0.25], [0, 0], [0.5, 0], [1.5, 0]
                s2 = samples[2]
                @test size(s2) == (2, 4)
                @test [0, 0.25] in eachcol(s2)
                @test [0, 0] in eachcol(s2)
                @test [0.5, 0] in eachcol(s2)
                @test [1.5, 0] in eachcol(s2)
            end

            @testset "max_samples" begin
                res = preprocess(trajs; anchors=anchors, max_dist=1, max_samples=3)
                samples = res.prob.data
                @test length(samples) == 2

                # first anchor:
                # close start points are [0, 2] and [0, 1]
                # -> samples are [0, 1] and [0, 0.25]
                s1 = samples[1]
                @test size(s1) == (2, 2)
                @test [0, 1] in eachcol(s1)
                @test [0, 0.25] in eachcol(s1)

                # second anchor:
                # close start points are [0, 1], [0, 0.25], [0, 0], [0.5, 0]
                # -> samples are [0, 0.25], [0, 0], [0.5, 0], [1.5, 0]
                # max_samples = 3: the startpoint [0, 1] is the farthest and should be deleted 
                s2 = samples[2]
                @test size(s2) == (2, 3)
                @test [0, 0] in eachcol(s2)
                @test [0.5, 0] in eachcol(s2)
                @test [1.5, 0] in eachcol(s2)
            end

            @testset "min_samples" begin
                res = @test_warn "" preprocess(
                    trajs; anchors=anchors, max_dist=1, min_samples=3
                )
                samples = res.prob.data
                @test length(samples) == 1

                # first anchor:
                # close start points are [0, 2] and [0, 1]
                # less than `min_samples`, so gets removed

                # second anchor:
                # close start points are [0, 1], [0, 0.25], [0, 0], [0.5, 0]
                # -> samples are [0, 0.25], [0, 0], [0.5, 0], [1.5, 0]
                s2 = samples[1]
                @test size(s2) == (2, 4)
                @test [0, 0.25] in eachcol(s2)
                @test [0, 0] in eachcol(s2)
                @test [0.5, 0] in eachcol(s2)
                @test [1.5, 0] in eachcol(s2)
            end
        end

        @testset "default kwargs" begin
            traj = rand(2, 500)
            data = Trajectories(traj)
            res = preprocess(data)
            @test size(res.info["anchors"]) == (2, 5)
            @test res.info["max_dist"] ≈
                0.5 * TransitionManifolds.mean_jump_dist(data, Euclidean())
        end

        @testset "all empty" begin
            traj = rand(2, 20)
            data = Trajectories(traj)
            anchors = rand(2, 3) .+ 100

            @test_throws ErrorException preprocess(data; anchors=anchors, max_dist=1)
        end

        @testset "some empty" begin
            traj = rand(2, 20)
            data = Trajectories(traj)
            anchors = rand(2, 4)
            anchors[:, 1] .+= 1000
            anchors[:, 3] .+= 1000

            # for anchors 2 and 4, all samples match
            # for anchors 1 and 3, no samples match and they should be removed
            res = @test_warn "" preprocess(data; anchors=anchors, max_dist=10)

            @test res.info["anchors"] == anchors[:, [2, 4]]
            out = res.prob.data
            @test length(out) == 2
            @test out[1] == traj[:, 2:end]
            @test out[2] == traj[:, 2:end]
        end

        @testset "max_dist = large" begin
            traj1 = rand(2, 20)
            traj2 = rand(2, 10)
            data = Trajectories([traj1, traj2])
            anchors = rand(2, 3)

            # for large max_dist, all points match
            res = preprocess(data; anchors=anchors, max_dist=100)
            out = res.prob.data

            @test length(out) == 3
            expected_out = hcat(traj1[:, 2:end], traj2[:, 2:end])
            for i in 1:3
                @test out[i] == expected_out
            end
        end

        @testset "max_dist = 0" begin
            traj1 = rand(2, 20)
            traj2 = rand(2, 10)
            data = Trajectories([traj1, traj2])
            anchors = stack([traj1[:, 5], traj2[:, 8], traj1[:, 17], traj2[:, 1]])

            # for max_dist=0, only the anchors themselves match
            # and the output should be exactly their successors
            res = preprocess(data; anchors=anchors, max_dist=0)
            out = res.prob.data
            @test length(out) == 4
            @test out[1] == traj1[:, 6:6]
            @test out[2] == traj2[:, 9:9]
            @test out[3] == traj1[:, 18:18]
            @test out[4] == traj2[:, 2:2]
        end
    end
end
