instance = generate_instance(
    3,
    5;
    seed = 42,
    processing_range = (1, 100)
)

encoder = JobFeatureEncoder(
    instance.m,
    64;
    p_ref = 100.0
)

H = encoder(instance.p)

println("Processing-time dimensions: ", size(instance.p))
println("Embedding dimensions: ", size(H))

@assert size(H) == (instance.n, 64)
@assert all(isfinite, H)

println("Job feature encoder test passed.")