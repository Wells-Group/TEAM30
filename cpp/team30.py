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


# -- Expressions -- #
A_out = Coefficient(V)
B_3D = curl(u0)

family = basix.finite_element.string_to_family("Lagrange", "tetrahedron")
basix_cell = basix.cell.string_to_type("tetrahedron")
b_element = basix.create_element(family, basix_cell, 1, basix.LagrangeVariant.gll_warped, discontinuous=True)
interpolation_points = b_element.points

forms = [a, L]
expressions = [(B_3D, interpolation_points)]