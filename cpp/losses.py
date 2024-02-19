import basix
import ufl
from ufl import Coefficient, Constant, FunctionSpace, Mesh, dx

element = basix.ufl.element("N1curl", "tetrahedron", 1)
domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3, )))

V = FunctionSpace(domain, element)
DG0 = FunctionSpace(domain, basix.ufl.element("DG", "tetrahedron", 0))

sigma = Coefficient(DG0)
dt = Constant(domain)
A = Coefficient(V)
An = Coefficient(V)

E = -(A - An) / dt
q = sigma * ufl.inner(E, E) * dx
