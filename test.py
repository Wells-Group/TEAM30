#%%
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pandas as pd
from basix.ufl import element
from dolfinx import fem, io
from dolfinx.common import Timer, timing
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import Function, form, locate_dofs_topological, petsc
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import VTXWriter
from dolfinx.mesh import locate_entities_boundary
from ufl import Measure, SpatialCoordinate, TestFunction, TrialFunction, cross, curl, inner

from generate_team30_meshes_3D import domain_parameters, model_parameters
from utils import update_current_density

# Example usage:
# python3 generate_team30_meshes_3D.py --res 0.005 --three
# python3 solve_3D_time.py


# -- Parameters -- #

num_phases = 3
steps_per_phase = 10
freq = model_parameters["freq"]
T = num_phases * 1 / freq
dt_ = 1.0 / steps_per_phase * 1 / freq

mu_0 = model_parameters["mu_0"]
omega_J = 2 * np.pi * freq

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
        p = model_parameters["sigma"][material]
        sigma.x.array[cells] = model_parameters["sigma"][material]
        density.x.array[cells] = model_parameters["densities"][material]

Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

dx = Measure("dx", domain=mesh, subdomain_data=ct)

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
A_space = fem.functionspace(mesh, nedelec_elem)

A = TrialFunction(A_space)
v = TestFunction(A_space)

element = element("Lagrange", mesh.basix_cell(), degree)
V = fem.functionspace(mesh, element)

S = TrialFunction(V) 
q = TestFunction(V)

A_prev = fem.Function(A_space)
S_prev = fem.Function(V)
J0z = fem.Function(DG0)

ndofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs

print(f"Number of dofs: {ndofs}")

# -- BCs and Assembly -- #

def boundary_marker(x):
    return np.full(x.shape[1], True)

mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
boundary_dofs_A = locate_dofs_topological(A_space, entity_dim=tdim - 1, entities=boundary_facets)

zeroA = fem.Function(A_space)
zeroA.x.array[:] = 0
bc = fem.dirichletbc(zeroA, boundary_dofs_A)

mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
boundary_dofs_V = locate_dofs_topological(A_space, entity_dim=tdim - 1, entities=boundary_facets)


zeroA = fem.Function(A_space)
zeroA.x.array[:] = 0
bc = fem.dirichletbc(zeroA, boundary_dofs_V)

bcs = [[bc]]


# -- Weak Form -- #

a = dt * 1 / mu_R * inner(curl(A), curl(v)) * dx(Omega_c + Omega_n)
a += sigma * mu_0 * inner(A, v) * dx(Omega_c + Omega_n)
a = form(a)

L = dt * mu_0 * J0z * v[2] * dx(Omega_n)
L += sigma * mu_0 * inner(A_prev, v) * dx(Omega_c + Omega_n)
L = form(L)


#%%
A_out = Function(A_space)
A = assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = fem.petsc.create_vector(L)

# -- AMS Solver Setup -- #

ksp = PETSc.KSP().create(mesh.comm)  # type: ignore
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
ksp.setOperators(A)
pc = ksp.getPC()
opts = PETSc.Options()  # type: ignore

ams_options = {
    "pc_hypre_ams_cycle_type": 1,
    "pc_hypre_ams_tol": 1e-8,
    "ksp_atol": 1e-10,
    "ksp_rtol": 1e-8,
    "ksp_initial_guess_nonzero": True,
    "ksp_type": "gmres",
    "ksp_norm_type": "unpreconditioned",
}

pc.setType("hypre")
pc.setHYPREType("ams")

option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for option, value in ams_options.items():
    opts[option] = value
opts.prefixPop()

W = fem.functionspace(mesh, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, A_space._cpp_object)
G.assemble()

shape = (mesh.geometry.dim,)
Q = fem.functionspace(mesh, ("Lagrange", degree, shape))
Pi = interpolation_matrix(Q._cpp_object, A_space._cpp_object)
Pi.assemble()

pc.setHYPREDiscreteGradient(G)
pc.setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)

ksp.setFromOptions()
pc.setUp()
ksp.setUp()

# -- Time simulation -- #

shape = (mesh.geometry.dim,)
W1 = fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,)))

if output:
    B_output = Function(W1)
    B_vtx = VTXWriter(mesh.comm, "output_3D_B.bp", [B_output], engine="BP4")

t = 0
results = []

for i in range(10):
    A_out.x.array[:] = 0
    t += dt_

    # Update Current and Re-assemble LHS
    update_current_density(J0z, omega_J, t, ct, currents)
    with b.localForm() as loc_b:
        loc_b.set(0)
    b = petsc.assemble_vector(b, L)

    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    petsc.set_bc(b, [bc])
    max_b = max(b.array)

    # Solve
    with Timer("solve"):
        ksp.solve(b, A_out.vector)
        A_out.x.scatter_forward()

    # Compute B
    el_B = ("DG", max(degree - 1, 1), shape)
    VB = fem.functionspace(mesh, el_B)
    B = fem.Function(VB)
    B_3D = curl(A_out)
    Bexpr = fem.Expression(B_3D, VB.element.interpolation_points())
    B.interpolate(Bexpr)

    # Compute F
    E = -(A_out - A_prev) / dt
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

    print("A_out after solve ", max(B_output.x.array))
    min_cond = model_parameters["sigma"]["Cu"]

