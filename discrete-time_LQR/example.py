# nonlinear ct dynamics
#    state: [x, y, \theta, \dot{x}, \dot{y}, \dot{\theta}]
#    input: [u_1, u_2]
# dt dynamics
# linearized dt dynamics
# define LQR problem
# design controller
# apply controller to nonlinear system

import numpy as np
from scipy.linalg import solve_discrete_are, solve
from scipy.integrate import ode
import matplotlib.pyplot as plt

import meshcat
from meshcat.animation import Animation
import birotor_visualizer

m = 1
I = 1
g = 9.81
a = 0.1


# nonlinear continuous-time dynamics
def f(t, x, u):
    return np.array(
        [
            x[3],
            x[4],
            x[5],
            -np.sin(x[2]) * (u[0] + u[1]) / m,
            np.cos(x[2]) * (u[0] + u[1]) / m - g,
            a * (u[0] - u[1]) / I,
        ]
    )


# linearized discrete-time dynamics
h = 1e-2

A = np.eye(6) + h * np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, -g, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

B = h * np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1 / m, 1 / m],
        [a / I, -a / I],
    ]
)

# simulation
N = 500

x0 = np.zeros(6)

x_eq = np.array([2, 1, 0, 0, 0, 0])
u_eq = m * g / 2 * np.ones(2)

xs = np.zeros((6, N + 1))
us = np.zeros((2, N))

#  infinite horizon
#   problem
Q = np.diag(np.array([10, 10, 10, 1, 1, 1]))
R = 1e1 * np.diag(np.ones(2))

#   controller
S = solve_discrete_are(A, B, Q, R)
K = solve(R + B.T @ S @ B, B.T @ S @ A)

#   simulation
solver = ode(f).set_integrator("dopri5")

xs[:, 0] = x0

for k in range(N):
    solver.set_initial_value(xs[:, k])  # reset initial conditions to last state
    us[:, k] = u_eq - K @ (xs[:, k] - x_eq)  # calculate control input
    solver.set_f_params(us[:, k])  # set control input in solver
    solver.integrate(h)  # integrate a single step
    xs[:, k + 1] = solver.y  # save result to states

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1)

for i in range(3):
    ax1.plot(xs[i, :], label=f"x{i} - inf")

for i in range(2):
    ax2.plot(us[i, :], label=f"u{i} - inf")

ax1.legend()
ax2.legend()
plt.show(block=False)

#  finite horizon
#   problem
Q_N = 1e4 * np.diag(np.array([10, 10, 10, 1, 1, 1]))
Q = np.zeros((6, 6))
R = 1e-1 * np.diag(np.ones(2))

#   controlller
S = np.zeros((6, 6, N + 1))
K = np.zeros((2, 6, N))

S[:, :, N] = Q_N
for k in reversed(range(N)):
    S[:, :, k] = (
        Q
        + A.T @ S[:, :, k + 1] @ A
        - A.T
        @ S[:, :, k + 1]
        @ B
        @ np.linalg.inv(B.T @ S[:, :, k + 1] @ B + R)
        @ B.T
        @ S[:, :, k + 1]
        @ A
    )
    S[:, :, k] = 0.5 * (S[:, :, k] + S[:, :, k].T)

    K[:, :, k] = solve(R + B.T @ S[:, :, k + 1] @ B, B.T @ S[:, :, k + 1] @ A)

#   simulation
solver = ode(f).set_integrator("dopri5")

xs[:, 0] = x0

for k in range(N):
    us[:, k] = u_eq - K[:, :, k] @ (xs[:, k] - x_eq)  # calculate control input
    solver.set_initial_value(xs[:, k])  # reset initial conditions to last state
    solver.set_f_params(us[:, k])  # set control input in solver
    solver.integrate(h)  # integrate a single step
    xs[:, k + 1] = solver.y  # save result to states

# Plotting
for i in range(3):
    ax1.plot(xs[i, :], label=f"x{i} - fin")

for i in range(2):
    ax2.plot(us[i, :], label=f"u{i} - fin")

ax1.legend()
ax2.legend()
plt.show(block=False)

#  animation
vis = meshcat.Visualizer()

birotor_visualizer.set_birotor(vis, 2 * a, 0.04, 0.09)

anim = Animation(default_framerate=1 / h)
for i in range(N + 1):
    with anim.at_frame(vis, i) as frame:
        birotor_visualizer.set_birotor_state(frame, xs[:, i])

vis.set_animation(anim, play=False)

input("Press Enter to continue...")
