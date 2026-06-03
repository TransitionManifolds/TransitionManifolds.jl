module GraphsExt

using TransitionManifolds
using Graphs

"""
    preprocess(data::Array{Graph, 2}, mode=:adjacency_matrix) -> PreprocessResult

Convert an array of graphs of shape `(n_samples, n_anchors)` to an array of numbers
by replacing each graph with a vector representation.

Supports the following modes for the conversion:

  - `mode=:adjacency_matrix`: replace each graph with its flattened adjacency matrix.
    All graphs must have the same number of nodes.
"""
function TransitionManifolds.preprocess(
    data::AbstractArray{<:Graph,2}, mode::Symbol=:adjacency_matrix
)::PreprocessResult
    if mode == :adjacency_matrix
        return _preprocess_mode_adjacency_matrix(data)
    end

    error("unknown mode $mode")
end

function _preprocess_mode_adjacency_matrix(data::AbstractArray{<:Graph,2})::PreprocessResult
    # check that all graphs have the same number of nodes
    n_nodes = nv(data[begin])
    for g in data
        if nv(g) != n_nodes
            throw(ArgumentError("all graphs must have the same number of nodes"))
        end
    end

    out = zeros(Float32, binomial(n_nodes, 2), size(data)...)
    for i in CartesianIndices(data)
        flat_lower_triangular!(view(out, :, i), data[i])
    end
    return PreprocessResult(TransitionDistanceProblem(out), Dict{String,Any}())
end

function flat_lower_triangular!(flat_lt::AbstractVector, graph::Graph)
    adj = adjacency_matrix(graph)
    n = size(adj, 1)
    k = 1
    for (i, col) in enumerate(eachcol(adj))
        if i != n
            @views flat_lt[k:(k + n - i - 1)] = col[(i + 1):end]
            k += n - i
        end
    end
end

end
