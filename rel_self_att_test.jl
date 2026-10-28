instance = generate_instance(
    3,
    5;
    seed = 42,
    processing_range = (1, 100),
    setup_range = (0, 50)
)

encoder = JobFeatureEncoder(
    instance.m,
    64;
    p_ref = 100.0
)

H = encoder(instance.p)

attention = RelationalSelfAttention(
    64;
    nheads = 4,
    setup_ref = 50.0
)

Z = attention(H, instance.S)

println("Input embedding dimensions: ", size(H))
println("Setup tensor dimensions: ", size(instance.S))
println("Output embedding dimensions: ", size(Z))

@assert size(H) == (5, 64)
@assert size(instance.S) == (3, 5, 5)
@assert size(Z) == (5, 64)
@assert all(isfinite, Z)

println("Relational self-attention test passed.")