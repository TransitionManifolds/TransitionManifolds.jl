using BenchmarkTools

const SUITE = BenchmarkGroup()

include("benchmarks_td.jl")
SUITE["td"] = BenchmarkTD.SUITE

run_benchmarks() = run(SUITE; verbose=true, seconds=5)
