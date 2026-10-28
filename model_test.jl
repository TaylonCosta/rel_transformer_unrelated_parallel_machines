instance = generate_instance(
    3, 5;
    seed = 42,
    processing_range = (1, 100),
    setup_range = (0, 50)
)

model = RelationalTransformer(
    instance.m;
    d = 64,
    nheads = 4,
    num_blocks = 2,
    ff_multiplier = 4,
    p_ref = 100.0,
    setup_ref = 50.0
)

H = model(instance.p, instance.S)

@assert size(H) == (instance.n, 64)
@assert all(isfinite, H)

println("Final embedding size: ", size(H))
println("Number of Transformer blocks: ", length(model.blocks))
println("Full model test passed.")