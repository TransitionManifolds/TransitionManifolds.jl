using BenchmarkTools

const SUITE = BenchmarkGroup()

include("benchmarks_td.jl")
SUITE["td"] = BenchmarkTD.SUITE

include("benchmarks_trajs.jl")
SUITE["trajs"] = BenchmarkTrajs.SUITE

run_benchmarks() = run(SUITE; verbose=true, seconds=5)
