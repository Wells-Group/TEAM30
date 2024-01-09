import basix.ufl
from ufl import (FunctionSpace, Mesh, TestFunction, TrialFunction,
                 Coefficient, dx, inner, Constant, curl)

element = basix.ufl.element("N1curl", "tetrahedron", 1)
domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3, )))

V = FunctionSpace(domain, element)
DG0 = FunctionSpace(domain, basix.ufl.element("DG", "tetrahedron", 0))

v = TestFunction(V)
u = TrialFunction(V)

mu_R = Coefficient(DG0)
sigma = Coefficient(DG0)
density = Coefficient(DG0)

dt = Constant(domain)
mu_0 = Constant(domain)  # Relative permability of air [H/m]=[kg m/(s^2 A^2)]
J0z = Coefficient(DG0)

u0 = Coefficient(V)

# -- Weak Form -- #


a = dt * (1 / mu_R) * inner(curl(u), curl(v)) * dx
a += mu_0 * sigma * inner(u, v) * dx

L = dt * mu_0 * J0z * v[2] * dx
L += sigma * mu_0 * inner(u0, v) * dx
