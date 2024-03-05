import basix
import ufl
from ufl import Coefficient, Constant, FunctionSpace, Measure, Mesh

element = basix.ufl.element("N1curl", "tetrahedron", 1)
domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))

domains = {
    "Cu": (7, 8, 9, 10, 11, 12),
    "Stator": (6,),
    "Rotor": (5,),
    "Al": (4,),
    "AirGap": (2, 3),
    "Air": (1,),
}

V = FunctionSpace(domain, element)
DG0 = FunctionSpace(domain, basix.ufl.element("DG", "tetrahedron", 0))

sigma = Coefficient(DG0)
dt = Constant(domain)
A = Coefficient(V)
An = Coefficient(V)

dx = Measure("dx", domain=domain)

E = -(A - An) / dt

# Add loss for each domain
q = sigma * ufl.inner(E, E) * dx(domains["Rotor"])

forms = [q]
