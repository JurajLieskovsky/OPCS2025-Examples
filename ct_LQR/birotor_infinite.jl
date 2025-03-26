using DifferentialEquations
using Plots

# System's dynamics
mass = 1
moi = 1
a = 0.3
g = 9.81

f(x, u) = [
    x[4],
    x[5],
    x[6],
    -sin(x[3]) / mass * (u[1] + u[2]),
    cos(x[3]) / mass * (u[1] + u[2]) - mass * g,
    a / moi * (u[1] - u[2])
]

# Simulation
tspan = (0.0, 10.0)
u_eq = mass * g / 2 * ones(2)

fun = ODEFunction((x, _, _) -> f(x, u_eq))
prob = ODEProblem(fun, zeros(6), tspan)
sol = solve(prob)

plot(sol)
