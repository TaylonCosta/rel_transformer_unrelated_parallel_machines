using Flux

struct RelationalTransformer
    encoder
    blocks
end

Flux.@layer RelationalTransformer

function RelationalTransformer(
    m::Int;
    d::Int = 64,
    nheads::Int = 4,
    num_blocks::Int = 2,
    ff_multiplier::Int = 4,
    p_ref::Real = 100.0,
    setup_ref::Real = 50.0
)
    @assert m > 0
    @assert d > 0
    @assert nheads > 0
    @assert d % nheads == 0
    @assert num_blocks > 0

    encoder = JobFeatureEncoder(
        m,
        d;
        p_ref = p_ref
    )

    blocks = [
        RelationalTransformerBlock(
            d;
            nheads = nheads,
            ff_multiplier = ff_multiplier,
            setup_ref = setup_ref
        )
        for _ in 1:num_blocks
    ]

    return RelationalTransformer(encoder, blocks)
end

function (model::RelationalTransformer)(
    p::AbstractMatrix{<:Real},
    S::AbstractArray{<:Real, 3}
)
    H = model.encoder(p)

    for block in model.blocks
        H = block(H, S)
    end

    return H
end