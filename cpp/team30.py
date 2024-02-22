import basix.ufl
from ufl import (Coefficient, Constant, FunctionSpace, Mesh, TestFunction,
                 TrialFunction, curl, div, dx, grad, inner)

element = basix.ufl.element("N1curl", "tetrahedron", 1)
domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
V_A = FunctionSpace(domain, element)
V_V = FunctionSpace(domain, basix.ufl.element("Lagrange", "tetrahedron", 1))

DG0 = FunctionSpace(domain, basix.ufl.element("DG", "tetrahedron", 0))

v_A = TestFunction(V_A)  # Test function for the vector potential
u_A = TrialFunction(V_A)  # Trial function for the vector potential

v_V = TestFunction(V_V)  # Test function for the scalar potential
u_V = TrialFunction(V_V)  # Trial function for the scalar potential

mu_R = Coefficient(DG0)
sigma = Coefficient(DG0)
density = Coefficient(DG0)

dt = Constant(domain)
mu_0 = Constant(domain)  # Relative permability of air [H/m]=[kg m/(s^2 A^2)]
zero = Constant(domain)
J0z = Coefficient(DG0)

u_A0 = Coefficient(V_A)

# -- Weak Form -- #


a00 = dt * (1 / mu_R) * inner(curl(u_A), curl(v_A)) * dx
a00 += mu_0 * sigma * inner(u_A, v_A) * dx

a01 = mu_0 * sigma * inner(v_A, grad(u_V)) * dx
a11 = mu_0 * sigma * inner(grad(u_V), grad(v_V)) * dx
a10 = zero * div(u_A) * v_V * dx

L0 = dt * mu_0 * J0z * v_A[2] * dx
L0 += sigma * mu_0 * inner(u_A0, v_A) * dx
L1 = zero * v_V * dx


forms = [a00, a01, a10, a11, L0, L1]


# -- Expressions -- #
A_out = Coefficient(V_A)
B_3D = curl(A_out)

family = basix.finite_element.string_to_family("Lagrange", "tetrahedron")
basix_cell = basix.cell.string_to_type("tetrahedron")
b_element = basix.create_element(family, basix_cell, 1, basix.LagrangeVariant.gll_warped, discontinuous=True)
interpolation_points = b_element.points

expressions = [(B_3D, interpolation_points)]