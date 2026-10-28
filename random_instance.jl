using Random

function generate_instance(
    m::Int,
    n::Int;
    seed::Int = 1234,
    processing_range::Tuple{Int, Int} = (1, 100),
    setup_range::Tuple{Int, Int} = (0, 50)
)
    @assert m > 0 "The number of machines must be positive."
    @assert n > 0 "The number of jobs must be positive."

    p_min, p_max = processing_range
    s_min, s_max = setup_range

    @assert 0 < p_min <= p_max
    @assert 0 <= s_min <= s_max

    rng = Xoshiro(seed)

    p = rand(rng, p_min:p_max, m, n)
    S = rand(rng, s_min:s_max, m, n, n)

    return (; m, n, p, S)
end