using JuMP
using HiGHS
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


instance = generate_instance(2, 5; seed = 42)

m = instance.m
n = instance.n

M = 1:m
N = 1:n
N0 = 0:n

p = Float64.(instance.p)
S = Float64.(instance.S)

big_M = 1000.0

model = Model(HiGHS.Optimizer)

@variable(model, X[i in M, j in N0, k in N], Bin)
@variable(model, C[i in M, j in N0] >= 0)
@variable(model, c_max >= 0)

@objective(model, Min, c_max)

@constraint(model, [k in N], sum(sum(X[i,j,k] for j in N0 if j!=k) for i in M) == 1)
@constraint(model, [j in N], sum(sum(X[i,j,k] for k in N if j!=k) for i in M) <=1)
@constraint(model, [i in M], sum(X[i,0,k] for k in N) <= 1)
@constraint(model, [i in M, j in N, k in N; j != k], sum(X[i,h,j] for h in N0 if h!=k && h!=j) >= X[i,j,k])

@constraint(model, [i in M, j in N, k in N; j!=k], C[i,k] + big_M * (1 - X[i,j,k]) >= C[i,j]+S[i,j,k]+p[i,k])
@constraint(model, [i in M, k in N], C[i, k] >= p[i, k] - big_M * (1 - X[i, 0, k]))

@constraint(model, [i in M], C[i,0] >= 0)
@constraint(model, [j in N, i in M], C[i,j] >= 0)
@constraint(model, [j in N, i in M], c_max >= C[i,j])

print(model)

optimize!(model)

print(value(X))
