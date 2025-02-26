from sympy import symbols, Function, Matrix, sin, cos, diff, latex, simplify
from lagrangian2eom import massMatrix, velocityTerms, potentialTerms, wrenchMatrix

t = symbols("t")

x = Function("x")(t)
y = Function("y")(t)
theta = Function("theta")(t)

g, mass, moi, a = symbols("g m I a")

# vector of generalized coordinates
q = Matrix([x, y, theta])

# kinetic and potential energy
kin = 0.5 * mass * (diff(x, t) ** 2 + diff(y, t) ** 2) + 0.5 * moi * diff(theta, t) ** 2
pot = g * mass * y

# input matrix
f = Matrix([-sin(theta), cos(theta)])  # thrust direction vector

r_1 = Matrix([x + a * cos(theta), y + a * sin(theta)])
r_2 = Matrix([x - a * cos(theta), y - a * sin(theta)])

input_matrix = Matrix.hstack(wrenchMatrix(r_1, q) * f, wrenchMatrix(r_2, q) * f)

# Quantities
mass_matrix = massMatrix(kin, q, t)
velocity_terms = velocityTerms(kin, q, t)
potential_terms = potentialTerms(pot, q)

# Printout
print("T &=", latex(simplify(kin)), "\\\\")
print("V &=", latex(simplify(pot)))
print("M &=", latex(simplify(mass_matrix)), "\\\\")
print("c &=", latex(simplify(velocity_terms)), "\\\\")
print("\\tau_p &=", latex(simplify(potential_terms)), "\\\\")
print("B &=", latex(simplify(input_matrix)))
