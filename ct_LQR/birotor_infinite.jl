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

A = ForwardDiff.jacobian(x -> f(x, u_eq), x_eq)
B = ForwardDiff.jacobian(u -> f(x_eq, u), u_eq)

## infinite horizon
inf_Q = diagm([10, 10, 10, 1, 1, 1])
inf_R = I(2)

inf_S, _ = arec(A, B, inf_R, inf_Q)
inf_K = inf_R \ B' * inf_S

## finite horizon
fin_Φ = diagm([10, 10, 10, 1, 1, 1])
fin_Q = zeros(6,6)
fin_R = I(2)

bwd_y0 = ComponentArray(
    S=ones(6,6),
    K=zeros(2,6)
)

bwd_M = diagm(bwd_y0)

function bwd_f(dy,y, _, _)
    dy.K = fin_R \ B' * y.S - y.K
    dy.S = -(fin_Q - y.K' * fin_R * y.K + y.S * A + A' * y.S)
    return nothing
end

bwd_y0.S = fin_Φ
bwd_fun = ODEFunction(bwd_f, mass_matrix=bwd_M)
bwd_prob = ODEProblem(bwd_fun, bwd_y0, (10.0, 0.0))
bwd_sol = solve(bwd_prob, Rodas5P())

# DAE definition
fwd_y0 = ComponentArray(
    x=ones(6),
    u=zeros(2)
)

fwd_M = diagm(fwd_y0)

function fwd_f(dy, y, _, t)
    dy.x = f(y.x,y.u)
    dy.u = u_eq - bwd_sol(t).K * (y.x - x_eq) - y.u
    return nothing
end

# Simulation
tspan = (0.0, 10.0)

x0 = zeros(6)
fwd_y0.x = x0

fwd_fun = ODEFunction(fwd_f, mass_matrix=fwd_M)
fwd_prob = ODEProblem(fwd_fun, fwd_y0, tspan)
fwd_sol= solve(fwd_prob, Rodas5P())

plt = plot(layout=(2,1))
plot!(plt, fwd_sol, idxs=1:6, subplot=1)
plot!(plt, fwd_sol, idxs=7:8, subplot=2)
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
        MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, to3D(fwd_sol(t)))
    end
end
setanimation!(vis, anim, play=false);
