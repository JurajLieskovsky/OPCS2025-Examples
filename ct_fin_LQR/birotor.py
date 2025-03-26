import numpy as np
from scipy.linalg import solve_continuous_are, solve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import meshcat
from meshcat.animation import Animation
import birotor_visualizer

# system parameters
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

A = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, -g, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

B = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1 / m, 1 / m],
        [a / I, -a / I],
    ]
)

# finite horizon problem
T = 5

Qf = 1e6 * np.diag(np.array([10, 10, 10, 1, 1, 1]))
Q = np.zeros((6, 6))
R = 1e1 * np.diag(np.ones(2))


# LQR
def riccati(t, P):
    return (
        -0.5 * Q @ np.inverse(P).T
        + 0.5 * P @ P.T @ B @ np.inverse(R) @ B.T @ P
        - A.T @ P
    )


riccati_sol = solve_ivp(riccati, [T, 0], Qf, dense_output=True)
S = solve_continuous_are(A, B, Q, R)
K = solve(R, B.T @ S)

# simulation
x0 = np.zeros(6)

x_eq = np.array([2, 1, 0, 0, 0, 0])
u_eq = m * g / 2 * np.ones(2)

sol = solve_ivp(
    lambda t, x: f(
        t,
        x,
        u_eq - solve(R, B.T @ riccati_sol.sol(t) @ riccati_sol.sol(t).T) @ (x - x_eq),
    ),
    [0, T],
    x0,
    dense_output=True,
)

# timespan
nstep = 500
tspan = np.linspace(0, 5, nstep)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1)

for i in range(3):
    ax1.plot(tspan, sol.sol(tspan)[i, :], label=f"x{i} - inf")

ax1.legend()
ax2.legend()
plt.show(block=False)

#  animation
vis = meshcat.Visualizer()

birotor_visualizer.set_birotor(vis, 2 * a, 0.04, 0.09)

anim = Animation(default_framerate=nstep / T)
for i, x in enumerate(sol.sol(tspan).T):
    with anim.at_frame(vis, i) as frame:
        birotor_visualizer.set_birotor_state(frame, x)

vis.set_animation(anim, play=False)

input("Press Enter to continue...")
