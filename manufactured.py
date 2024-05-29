#%%
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
from basix.ufl import element
from dolfinx import fem, io, la
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import Function, form, locate_dofs_topological, petsc, Expression
from dolfinx.mesh import locate_entities_boundary
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    curl,
    div,
    grad,
    inner,
    cross,
    split,
    sin,
    cos,
    as_vector
)
from generate_team30_meshes_3D import domain_parameters, model_parameters
from utils import update_current_density
import numpy as np

# Example usage:
# python3 generate_team30_meshes_3D.py --res 0.005 --three
# python3 solve_3D_time.py


# -- Parameters -- #

num_phases = 3
steps_per_phase = 100
freq = model_parameters["freq"]
T = num_phases * 1 / freq
dt_ = 1.0 / steps_per_phase * 1 / freq

mu_0 = model_parameters["mu_0"]
omega_J = 2 * np.pi * freq

# TODO FIXME
single_phase = False
mesh_dir = "meshes"
ext = "single" if single_phase else "three"
fname = f"{mesh_dir}/{ext}_phase3D"

output = True
write_stats = True

domains, currents = domain_parameters(single_phase)
degree = 1


# -- Load Mesh -- #

with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")

print(mesh.topology.index_map(tdim).size_global)

# -- Functions and Spaces -- #

x = SpatialCoordinate(mesh)
cell = mesh.ufl_cell()
dt = fem.Constant(mesh, dt_)

DG0 = fem.functionspace(mesh, ("DG", 0))
mu_R = fem.Function(DG0)
sigma = fem.Function(DG0)
density = fem.Function(DG0)

for material, domain in domains.items():
    for marker in domain:
        cells = ct.find(marker)
        mu_R.x.array[cells] = model_parameters["mu_r"][material]
        sigma.x.array[cells] = model_parameters["sigma"][material]
        density.x.array[cells] = model_parameters["densities"][material]

Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

dx = Measure("dx", domain=mesh, subdomain_data=ct)

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
A_space = fem.functionspace(mesh, nedelec_elem)

# Electric Scalar potential
element = element("Lagrange", mesh.basix_cell(), degree)
V = fem.functionspace(mesh, element)

Z = fem.functionspace(mesh, nedelec_elem*element)

A,S = split(TrialFunction(Z))   #A - Mag Vec Potential,  v - Test Function of A, 
v,q = split(TestFunction(Z))    #S - Elec Scalar Potential, q - Test Function of S

A_prev = fem.Function(A_space)
J0z = fem.Function(DG0)
zero = fem.Constant(mesh, PETSc.ScalarType(0))  # type: ignore

# print(A_prev.x.array.size)
# print(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

ndofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs
# print(f"Number of dofs: {ndofs}")

#a00 - Trial and test function of Magnetic Vector Potential only,  
#a01 - Coupling of A and S using test function of A
#a11 - Involving only Electric Scalar Potential

#TO DO: When AMS is fixed start changing the area start isolating the Omega_n. Currently Air is still conductive which is why everything is Omega_n +Omega_c

a00 = dt * (1 / mu_R) * inner(curl(A), curl(v)) * dx(Omega_n) #Magnetic Vector potential with edge basis
a00 += dt * (1 / mu_R) * inner(curl(A), curl(v)) * dx(Omega_c)
a00 += sigma * mu_0 * inner(A, v) * dx(Omega_c)

a01 = mu_0 * sigma * inner(grad(S),v) * dx(Omega_c + Omega_n) # Coupling of Elec Scalar Potential(S) with the test function of A (Magnetic Vector Potentail)
a10 = mu_0 * sigma * inner(grad(q), A) * dx # Coupling of A with the test function of the Electric Scalar Potential
a11 = mu_0 * sigma * inner(grad(S), grad(q)) * dx #Lagrange Test and Trial

a = form(a00+ a01 + a10 + a11)

L0 = dt * mu_0 * J0z * v[2] * dx(Omega_n)
L0 += sigma * mu_0 * inner(A_prev, v) * dx(Omega_c + Omega_n)

# L1 = inner(fem.Constant(mesh, PETSc.ScalarType(0)), q) * dx  # type: ignore
L = form(L0)

# -- Create boundary conditions -- #
def boundary_marker(x):
    return np.full(x.shape[1], True)

# Manufactured solutions

x = SpatialCoordinate(mesh)
A_ex = as_vector((sin(np.pi*x[0]), sin(np.pi*x[1]), sin(np.pi*x[2])))

V_ex = as_vector((sin(np.pi*x[0]), sin(np.pi*x[1]), sin(np.pi*x[2])))

f_ex = dt*curl(mu_0* curl(A_ex)) + sigma*A_ex + sigma * V_ex



uex = source(x)
u_bc_expr = Expression(uex, V.element.interpolation_points())
u_bc = Function(V)
u_bc.interpolate(u_bc_expr)
bc = dirichletbc(u_bc, dofs)



mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
bdofs0 = locate_dofs_topological(A_space, entity_dim=tdim - 1, entities=boundary_facets)

zeroA = fem.Function(A_space)
zeroA.x.array[:] = 0
bc0 = fem.dirichletbc(zeroA, bdofs0)

bdofs1 = locate_dofs_topological(V, entity_dim=tdim - 1, entities=boundary_facets)
bc1 = fem.dirichletbc(fem.Constant(mesh, PETSc.ScalarType(0)), bdofs1, V)  # type: ignore

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]
A_matrix = assemble_matrix(a, bcs = bcs)
A_matrix.assemble()

update_current_density(J0z, omega_J, 0.5, ct, currents)

b = petsc.assemble_vector(L)

petsc.apply_lifting(b, [a], bcs=[bcs])

A_out, S_out = Function(A_space), Function(V)

x = fem.Function(Z)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A_matrix)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.setFromOptions()
ksp.view()


#%%
ksp.solve(b, x.vector)



shape = (mesh.geometry.dim,)
W1 = fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,)))

if output:
    B_output = Function(W1)
    B_vtx = io.VTXWriter(mesh.comm, "output_3D_B.bp", [B_output], engine="BP4")

t = 0
results = []

for i in range(num_phases * steps_per_phase):
    print(f"Step = {i}")

    A_out.x.array[:] = 0
    t += dt_

    # Update Current and Re-assemble LHS
    update_current_density(J0z, omega_J, t, ct, currents)
    # with b.localForm() as loc_b:
    #     loc_b.set(0)
    # petsc.assemble_vector(b, L)
    # FIXME Don't create new
    b = petsc.assemble_vector(L)

    petsc.apply_lifting(b, [a], bcs=[bcs])
    max_b = max(b.array)

    x = fem.Function(Z)

    ksp.solve(b, x.vector)

    # Compute B
    el_B = ("DG", max(degree - 1, 1), shape)
    VB = fem.functionspace(mesh, el_B)
    B = fem.Function(VB)
    B_3D = curl(A_out)
    Bexpr = fem.Expression(B_3D, VB.element.interpolation_points())
    B.interpolate(Bexpr)

    # Compute F
    E = - (A_out - A_prev) / dt
    f = cross(sigma * E, B)
    F = fem.Function(VB)
    fexpr = fem.Expression(f, VB.element.interpolation_points())
    F.interpolate(fexpr)
    A_prev.x.array[:] = A_out.x.array  # Set A_prev

    # Write B
    if output:
        B_output_1 = Function(W1)
        B_output_1.interpolate(B)
        B_output.x.array[:] = B_output_1.x.array[:]
        B_vtx.write(t)
    
#%%