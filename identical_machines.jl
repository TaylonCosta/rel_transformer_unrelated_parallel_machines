using JuMP
using HiGHS
using Plots

n = 8;
m = 3;
t = [1, 2, 3, 4, 5, 6, 7, 8]

model = Model(HiGHS.Optimizer)

@variable(model, y >= 0)
@variable(model, x[i=1:n, j=1:m], Bin)

@objective(model, Min, y)

@constraint(model, [j = 1:m], y-sum(t[i] * x[i,j] for i in 1:n) >= 0)

@constraint(model, [i = 1:n], sum(x[i,j] for j in 1:m) == 1)

print(model)

optimize!(model)

for i in 1:n
    for j in 1:m
        print(value(t[i]*x[i,j]))
        print("  ")
    end
    println(" ")
end