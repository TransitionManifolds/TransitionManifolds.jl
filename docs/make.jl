using Documenter
using TransitionManifolds

# trigger extension
using Graphs
using KernelFunctions

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
