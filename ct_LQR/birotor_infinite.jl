using Base: Forward
using DifferentialEquations
using Plots
using MeshCatBenchmarkMechanisms
using LinearAlgebra
using ForwardDiff
using MatrixEquations
using ComponentArrays

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

# LQR
x_eq = [2.0, 1, 0, 0, 0, 0]
u_eq = mass * g / 2 * ones(2)

Q = diagm([10, 10, 10, 1, 1, 1])
R = I(2)

A = ForwardDiff.jacobian(x -> f(x, u_eq), x_eq)
B = ForwardDiff.jacobian(u -> f(x_eq, u), u_eq)

S, _ = arec(A, B, R, Q)
K = R \ B' * S

# DAE definition
y0 = ComponentArray(
    x=ones(6),
    u=zeros(2)
)

dae_M = diagm(y0)

function dae_f(dy, y, _, _)
    dy.x = f(y.x,y.u)
    dy.u = u_eq - K * (y.x - x_eq) - y.u
    return nothing
end

# Simulation
tspan = (0.0, 10.0)

x0 = zeros(6)
y0.x = x0

fun = ODEFunction(dae_f, mass_matrix=dae_M)
prob = ODEProblem(fun, y0, tspan)
sol = solve(prob, Rodas5P())

plt = plot(layout=(2,1))
plot!(plt, sol, idxs=1:6, subplot=1)
plot!(plt, sol, idxs=7:8, subplot=2)
display(plt)

# Visualization
to3D(x) = vcat(0, x[1:2], [cos(x[3] / 2), sin(x[3] / 2), 0, 0])

(@isdefined vis) || (vis = Visualizer())
render(vis)

## quadrotor and target
MeshCatBenchmarkMechanisms.set_quadrotor!(vis, 2 * a, 0.12, 0.25)
MeshCatBenchmarkMechanisms.set_target!(vis, 0.12)

## initial configuration
MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, to3D(x0))
MeshCatBenchmarkMechanisms.set_target_position!(vis, to3D(x_eq)[1:3])

## animation
fps = 100
anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=fps)
for (i, t) in enumerate(tspan[1]:1/fps:tspan[2])
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, to3D(sol(t)))
    end
end
setanimation!(vis, anim, play=false);
