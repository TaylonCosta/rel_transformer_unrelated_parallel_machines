using Flux

struct JobFeatureEncoder
    projection
    p_ref::Float32
end

Flux.@layer JobFeatureEncoder

function JobFeatureEncoder(
    m::Int,
    d::Int;
    p_ref::Real = 100.0
)
    @assert m > 0
    @assert d > 0
    @assert p_ref > 0

    projection = Dense(m => d)

    return JobFeatureEncoder(
        projection,
        Float32(p_ref)
    )
end

function (encoder::JobFeatureEncoder)(
    p::AbstractMatrix{<:Real}
)
    p_normalized = normalize_processing_times(
        p,
        encoder.p_ref
    )

    # Input: m × n
    # Output: n × m, one row per job
    X = permutedims(p_normalized)

    # Dense layers in Flux expect features × observations.
    # Therefore, transpose X before applying the layer.
    H = encoder.projection(permutedims(X))

    # Return one embedding per row.
    return permutedims(H)
end