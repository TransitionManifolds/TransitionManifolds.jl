using Documenter
using Changelog
using TransitionManifolds

# trigger extension
using Graphs
using KernelFunctions

Changelog.generate(
    Changelog.Documenter(),                 # output type
    joinpath(@__DIR__, "../CHANGELOG.md"),  # input file
    joinpath(@__DIR__, "src/CHANGELOG.md"); # output file
    repo="TransitionManifolds/TransitionManifolds.jl",
)

makedocs(;
    sitename="TransitionManifolds.jl",
    checkdocs=:exports,
    modules=[
        TransitionManifolds,
        Base.get_extension(TransitionManifolds, :GraphsExt),
        Base.get_extension(TransitionManifolds, :KernelMMDExt),
    ],
)

deploydocs(; repo="github.com/TransitionManifolds/TransitionManifolds.jl.git")
