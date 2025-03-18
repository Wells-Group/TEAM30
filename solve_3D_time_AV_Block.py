#%%
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl
from basix.ufl import element
from dolfinx import default_scalar_type, fem, io
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import Function, form, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.mesh import locate_entities_boundary
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    curl,
    grad,
    inner,
)

from generate_team30_meshes_3D import domain_parameters, model_parameters
from utils import update_current_density

# Example usage:
# python3 generate_team30_meshes_3D.py --res 0.005 --three
# python3 solve_3D_time.py

# -- Parameters -- #

def compute_loss(A_out, A_prev, dt):
    E = -(A_out - A_prev) / dt
    q = sigma * ufl.inner(E, E)
    al = q * dx(domains["Al"])  # Loss in rotor
    steel = q * dx(domains["Rotor"])  # Loss in only steel
    loss_al = fem.form(al)
    loss_steel = fem.form(steel)

    comm = MPI.COMM_WORLD
    al = comm.allreduce(fem.assemble_scalar(loss_al), op=MPI.SUM)
    steel = comm.allreduce(fem.assemble_scalar(loss_steel), op=MPI.SUM)

    return al, steel


num_phases = 3
steps_per_phase = 10
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

solver = "direct"

# -- Load Mesh -- #

with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")

# print(mesh.topology.index_map(tdim).size_global)

# -- Functions and Spaces -- #

x = SpatialCoordinate(mesh)
cell = mesh.ufl_cell()
dt = fem.Constant(mesh, dt_)

DG0 = fem.functionspace(mesh, ("DG", 0))
mu_R = fem.Function(DG0)
sigma = fem.Function(DG0)
density = fem.Function(DG0)
nu = fem.Function(DG0)

for material, domain in domains.items():
    for marker in domain:
        cells = ct.find(marker)
        mu_R.x.array[cells] = model_parameters["mu_r"][material]
        sigma.x.array[cells] = model_parameters["sigma"][material]
        density.x.array[cells] = model_parameters["densities"][material]
        nu.x.array[cells] = model_parameters["nu"][material]

Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

dx = Measure("dx", domain=mesh, subdomain_data=ct)

# Scalar potential

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
V = fem.functionspace(mesh, nedelec_elem)
lagrange_elem = element("Lagrange", mesh.basix_cell(), degree)
V1 = fem.functionspace(mesh, lagrange_elem)

u_n = Function(V)
u_n1 = Function(V1)

u = TrialFunction(V)
v = TestFunction(V)

u1 = TrialFunction(V1)
v1 = TestFunction(V1)

J0z = fem.Function(DG0)

a00 = dt * inner((1 / mu_R) * curl(u), curl(v)) * dx + inner((u * sigma), v) * dx

a01 = dt * inner(sigma * grad(u1), v) * dx
a10 = inner(sigma * u, grad(v1)) * dx

a11 = dt * inner(sigma * grad(u1), grad(v1)) * dx

a = form([[a00, a01], [a10, a11]])

L0 = dt * J0z * v[2] * dx(Omega_c + Omega_n)
L0 += inner(sigma * u_n, v) * dx(Omega_c + Omega_n)

L1 = inner(sigma * grad(u_n1), grad(v1))* dx(Omega_c + Omega_n)
L = form([L0, L1])

# -- Create boundary conditions -- #

def boundary_marker(x):
    return np.full(x.shape[1], True)

mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
bdofs0 = locate_dofs_topological(V, entity_dim=tdim - 1, entities=boundary_facets)

zeroA = fem.Function(V)
zeroA.x.array[:] = 0
bc0 = fem.dirichletbc(zeroA, bdofs0)

zeroV = fem.Constant(mesh, PETSc.ScalarType(0.0))

#%%

tags = np.unique(ft.values)
boundary_entities = np.concatenate([ft.find(tag) for tag in tags])
bdofs_conductive = locate_dofs_topological(V1, entity_dim=tdim - 1, entities=boundary_entities)
bc_conductive = fem.dirichletbc(zeroV, bdofs_conductive, V1)

bdofs1 = locate_dofs_topological(V1, entity_dim=tdim - 1, entities=boundary_facets)
bc1 = fem.dirichletbc(zeroV, bdofs1, V1)

# Collect Dirichlet boundary conditions
bc = [bc0, bc1, bc_conductive]

# Assemble block matrix operators

A_mat = assemble_matrix_block(a, bcs = bc)
A_mat.assemble()
print("norm of A ", A_mat.norm())

L = form([L0, L1])
b = assemble_vector_block(L, a, bcs=bc)

a_p = form([[a00, None], [None, a11]])

P = assemble_matrix_block(a_p, bcs = bc)
P.assemble()

u_map = V.dofmap.index_map
u1_map = V1.dofmap.index_map

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + u1_map.local_range[0]
offset_u1 = offset_u + u_map.size_local * V.dofmap.index_map_bs

is_u = PETSc.IS().createStride(
    u_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF
)
is_u1 = PETSc.IS().createStride(
    u1_map.size_local, offset_u1, 1, comm=PETSc.COMM_SELF
)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A_mat, P)
ksp.setType("fgmres")
ksp.setTolerances(rtol=1e-10)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(("u", is_u), ("u1", is_u1))
ksp_u, ksp_u1 = ksp.getPC().getFieldSplitSubKSP()

ksp_u.setType("preonly")
pc0 = ksp_u.getPC()
pc0.setType("hypre")
pc0.setHYPREType("ams")

V_CG = fem.functionspace(mesh, ("CG", degree))._cpp_object
G = discrete_gradient(V_CG, V._cpp_object)
G.assemble()
pc0.setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(V)
    cvec_0.interpolate(
        lambda x: np.vstack(
            (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_1 = Function(V)
    cvec_1.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_2 = Function(V)
    cvec_2.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
        )
    )
    pc0.setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )
else:
    Vec_CG = fem.functionspace(mesh, ("CG", degree, (mesh.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc0.setHYPRESetInterpolations(mesh.geometry.dim, None, None, Pi, None)

opts = PETSc.Options()
opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 7
# opts[f"{ksp_u.prefix}pc_hypre_ams_tol"] = 0
opts[f"{ksp_u.prefix}pc_hypre_ams_max_iter"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
opts[f"{ksp_u.prefix}pc_hypre_ams_print_level"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_alpha_options"] = "10,1,3"
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,3"
opts[f"{ksp_u.prefix}pc_hypre_ams_print_level"] = 0

ksp_u.setFromOptions()

# Preconditioner for u1
ksp_u1.setType("preonly")
pc1 = ksp_u1.getPC()
pc1.setType("gamg")

ksp.setUp()
pc0.setUp()
pc1.setUp()

u_n_prev = u_n.copy()

uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

sol = A_mat.createVecRight()

da_dt = (u_n - u_n_prev) / dt
E = -grad(u_n1) - da_dt
B = curl(u_n)
J = sigma * E

#%%

shape = (mesh.geometry.dim,)
W1 = fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,)))

if output:
    B_output = Function(W1)
    B_vtx = io.VTXWriter(mesh.comm, "output_3D_B.bp", [B_output], engine="BP4")

t = 0
results = []

#Initial Conditions
u_n.x.array[:] = 0
u_n1.x.array[:] = 0

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

num_steps = 2

# num_steps = num_phases * steps_per_phase

total_loss = np.zeros(num_steps + 1, dtype=default_scalar_type)

#%%
for i in range(num_steps):
    print(f"Step = {i}")

    t += dt_

    # Update Current and Re-assemble RHS
    update_current_density(J0z, omega_J, t, ct, currents)

    b = assemble_vector_block(L, a, bcs = bc)

    # Solve
    sol = A_mat.createVecRight()  #Solution Vector

    print("pre solve")
    ksp.solve(b, sol)

    residual = A_mat * sol - b
    print('residual is ', residual.norm())

    uh.x.array[:offset] = sol.array_r[:offset]
    uh1.x.array[: (len(sol.array_r) - offset)] = sol.array_r[offset:]

    uh.x.scatter_forward()
    uh1.x.scatter_forward()

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    B = curl(u_n)
    E = -grad(u_n1) - da_dt


    # Write B
    if output:
        B_output_1 = Function(W1)
        Bexpr = fem.Expression(B, W1.element.interpolation_points())
        B_output_1.interpolate(Bexpr)
        B_output.x.array[:] = B_output_1.x.array[:]
        B_vtx.write(t)

print(total_loss)
