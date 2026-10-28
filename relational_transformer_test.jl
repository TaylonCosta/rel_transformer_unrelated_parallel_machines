instance = generate_instance(
    3, 5;
    seed = 42,
    processing_range = (1, 100),
    setup_range = (0, 50)
)

d = 64

encoder = JobFeatureEncoder(
    instance.m,
    d;
    p_ref = 100.0
)

block = RelationalTransformerBlock(
    d;
    nheads = 4,
    ff_multiplier = 4,
    setup_ref = 50.0
)

H = encoder(instance.p)
H_out = block(H, instance.S)

@assert size(H) == (instance.n, d)
@assert size(H_out) == (instance.n, d)
@assert all(isfinite, H_out)

println("Input embedding size: ", size(H))
println("Output embedding size: ", size(H_out))
println("Transformer block test passed.")