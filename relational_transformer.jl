using Flux

struct RelationalTransformerBlock
    attention
    norm1
    ff1
    ff2
    norm2
end

Flux.@layer RelationalTransformerBlock

function RelationalTransformerBlock(
    d::Int;
    nheads::Int = 4,
    ff_multiplier::Int = 4,
    setup_ref::Real = 50.0
)
    @assert d > 0
    @assert nheads > 0
    @assert d % nheads == 0
    @assert ff_multiplier > 0

    attention = RelationalSelfAttention(
        d;
        nheads = nheads,
        setup_ref = setup_ref
    )

    norm1 = LayerNorm(d)
    ff1 = Dense(d => ff_multiplier * d, relu)
    ff2 = Dense(ff_multiplier * d => d)
    norm2 = LayerNorm(d)

    return RelationalTransformerBlock(
        attention,
        norm1,
        ff1,
        ff2,
        norm2
    )
end

function (block::RelationalTransformerBlock)(
    H::AbstractMatrix,
    S::AbstractArray{<:Real, 3}
)
    # First sublayer: relational self-attention
    A = block.attention(H, S)

    H1 = block.norm1(
        permutedims(H + A)
    ) |> permutedims

    # Second sublayer: position-wise feed-forward network
    F = block.ff2(
        block.ff1(permutedims(H1))
    ) |> permutedims

    H_out = block.norm2(
        permutedims(H1 + F)
    ) |> permutedims

    return H_out
end