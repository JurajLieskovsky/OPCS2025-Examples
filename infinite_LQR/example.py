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

B = h * np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1 / m, 1 / m], [a / I, -a / I]])

# infinite horizon LQR problem
Q = np.diag(np.array([10, 10, 10, 1, 1, 1]))
R = 1e0 * np.diag(np.ones(2))

# controller
S = solve_discrete_are(A, B, Q, R)
K = solve(R + B.T @ S @ B, B.T @ S @ A)

# simulation
#  problem
N = 500

x0 = np.array([1, 0, 0, 0, 0, 0])

x_eq = np.array([0, 0, 0, 0, 0, 0])
u_eq = m * g / 2 * np.ones(2)

#  solution
xs = np.zeros((6, N + 1))
us = np.zeros((2, N))

xs[:, 0] = x0

solver = ode(f).set_integrator("dopri5")

for k in range(N):
    solver.set_initial_value(xs[:, k])  # reset initial conditions to last state
    us[:, k] = u_eq - K @ (xs[:, k] - x_eq)  # calculate control input
    solver.set_f_params(us[:, k])  # set control input in solver
    solver.integrate(h)  # integrate a single step
    xs[:, k + 1] = solver.y  # save result to states


#  plotting
fig, (ax1, ax2) = plt.subplots(2, 1)

for i in range(3):
    ax1.plot(xs[i, :].T, label=f"x{i}")

for i in range(2):
    ax2.plot(us[i, :].T, label=f"u{i}")

ax1.legend()
ax2.legend()
plt.show()
