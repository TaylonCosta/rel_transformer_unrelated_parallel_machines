using Flux
using LinearAlgebra

struct RelationalSelfAttention
    Wq
    Wk
    Wv
    Wo
    setup_scale::Vector{Float32}
    setup_ref::Float32
    nheads::Int
end

Flux.@layer RelationalSelfAttention trainable=(Wq, Wk, Wv, Wo, setup_scale)

function RelationalSelfAttention(
    d::Int;
    nheads::Int = 4,
    setup_ref::Real = 50.0
)
    @assert d > 0
    @assert nheads > 0
    @assert d % nheads == 0
    @assert setup_ref > 0

    return RelationalSelfAttention(
        Dense(d => d),
        Dense(d => d),
        Dense(d => d),
        Dense(d => d),
        zeros(Float32, nheads),
        Float32(setup_ref),
        nheads
    )
end

function (layer::RelationalSelfAttention)(
    H::AbstractMatrix,
    S::AbstractArray{<:Real, 3}
)
    n, d = size(H)
    m = size(S, 1)

    @assert size(S, 2) == n && size(S, 3) == n
    @assert d % layer.nheads == 0
    @assert m > 0

    dh = d ÷ layer.nheads

    # Fixed scaling for setup times.
    S_norm = Float32.(S) ./ layer.setup_ref

    # Flux Dense layers expect features × observations.
    X = permutedims(H)

    Q = layer.Wq(X)
    K = layer.Wk(X)
    V = layer.Wv(X)

    # Convert to observations × features, then split into heads.
    Qh = reshape(permutedims(Q), n, dh, layer.nheads)
    Kh = reshape(permutedims(K), n, dh, layer.nheads)
    Vh = reshape(permutedims(V), n, dh, layer.nheads)

    # Compute one machine-conditioned attention output per machine.
    machine_outputs = map(1:m) do i

        # Compute one output matrix per attention head.
        head_outputs = map(1:layer.nheads) do h

            Q_i = Qh[:, :, h]
            K_i = Kh[:, :, h]
            V_i = Vh[:, :, h]

            # Row j, column k represents the relation j -> k.
            scores = (Q_i * K_i') / sqrt(Float32(dh)) .+
                     layer.setup_scale[h] .* S_norm[i, :, :]

            # Normalise over candidate successor jobs.
            attention = Flux.softmax(scores; dims=2)

            # Aggregate the value embeddings.
            attention * V_i
        end

        # Concatenate heads: n × d.
        hcat(head_outputs...)
    end

    # Average the machine-conditioned representations.
    Z = reduce(+, machine_outputs) ./ Float32(m)

    # Output projection.
    return permutedims(layer.Wo(permutedims(Z)))
end