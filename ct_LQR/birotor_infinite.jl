using DifferentialEquations
using Plots
using MeshCatBenchmarkMechanisms

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
x_eq = [2.0, 1, 0, 0, 0, 0]
u_eq = mass * g / 2 * ones(2)

x0 = zeros(6)

fun = ODEFunction((x, _, _) -> f(x, 2 * u_eq))
prob = ODEProblem(fun, x0, tspan)
sol = solve(prob)

plot(sol)

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
