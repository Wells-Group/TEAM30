# %%
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem, io
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import (
    Expression,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import create_submesh, locate_entities_boundary
from ufl import (
    Measure,
    MixedFunctionSpace,
    SpatialCoordinate,
    TestFunctions,
    TrialFunctions,
    curl,
    grad,
    inner,
)

from generate_team30_meshes_3D import domain_parameters, model_parameters
from utils import L2_norm, convert_facet_tags, update_current_density

comm = MPI.COMM_WORLD
degree = 1

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

domains, currents = domain_parameters(single_phase)
degree = 1

with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")

x = SpatialCoordinate(mesh)
cell = mesh.ufl_cell()
dt = fem.Constant(mesh, dt_)

DG0 = functionspace(mesh, ("DG", 0))
mu_R = Function(DG0)
sigma = Function(DG0)
density = Function(DG0)
nu = Function(DG0)

for material, domain in domains.items():
    for marker in domain:
        cells = ct.find(marker)
        mu_R.x.array[cells] = model_parameters["mu_r"][material]
        sigma.x.array[cells] = model_parameters["sigma"][material]
        density.x.array[cells] = model_parameters["densities"][material]
        nu.x.array[cells] = model_parameters["nu"][material]

Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]
whole = Omega_n + Omega_c

dx = Measure("dx", domain=mesh, subdomain_data=ct)

# with XDMFFile(mesh.comm, "box_with_sigma.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(sigma)

tdim = mesh.topology.dim
fdim = tdim - 1

target_tags = Omega_c
cell_mask = np.isin(ct.values, target_tags)
inner_cells = ct.indices[cell_mask]

submesh_inner, subdomain_inner_to_domain = create_submesh(mesh, tdim, inner_cells)[:2]

cell_imap = mesh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts

mesh_to_submesh_inner = np.full(num_cells, -1, dtype=np.int32)
mesh_to_submesh_inner[subdomain_inner_to_domain] = np.arange(len(subdomain_inner_to_domain))

entity_maps = {
    submesh_inner: mesh_to_submesh_inner,
}

dx = Measure("dx", mesh, subdomain_data=ct)

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
V = functionspace(mesh, nedelec_elem)
lagrange_elem = element("Lagrange", submesh_inner.basix_cell(), degree)
V1 = functionspace(submesh_inner, lagrange_elem)

W = MixedFunctionSpace(V, V1)

J0z = Function(DG0)

u, u1 = TrialFunctions(W)
v, v1 = TestFunctions(W)

u_n = Function(V)
u_n1 = Function(V1)

a = dt * inner(nu * curl(u), curl(v)) * dx(whole) + inner((u * sigma), v) * dx(Omega_c)

a += dt * inner(sigma * grad(u1), v) * dx(Omega_c)
a += inner(sigma * u, grad(v1)) * dx(Omega_c)

a += dt * inner(sigma * grad(u1), grad(v1)) * dx(Omega_c)

a = form(ufl.extract_blocks(a), entity_maps=entity_maps)


L0 = dt * J0z * v[2] * dx(domains["Cu"]) + inner(sigma * u_n, v) * dx(whole)
L0 += inner(grad(v1), sigma * u_n) * dx(Omega_c)

# L = form([L0, L1], entity_maps=entity_maps)

L = form(ufl.extract_blocks(L0), entity_maps=entity_maps)


# Boundary conditions

surface_map = {
    "Exterior": 1,
    "MidAir": 2,
    "LowerRotor": 3,
    "RotorInterface": 4,
    "UpperRotor": 5,
    "AlLower": 7,
    "AlOuter": 9,
    "AlUpper": 10,
}


# Bcs for outer submesh
def boundary_marker(x):
    return np.full(x.shape[1], True)

mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
bdofs0 = locate_dofs_topological(V, entity_dim=tdim - 1, entities=boundary_facets)
zeroA = Function(V)
zeroA.x.array[:] = 0

bc_outer = fem.dirichletbc(zeroA, bdofs0)

# Bcs for the inner submesh
submesh_inner.topology.create_connectivity(fdim, tdim)
ft_inner = convert_facet_tags(mesh, submesh_inner, subdomain_inner_to_domain, ft)

source_rotor = ft_inner.find(surface_map["UpperRotor"])
source_al = ft_inner.find(surface_map["AlUpper"])
source = np.concatenate([source_rotor, source_al])

highV = fem.Constant(mesh, PETSc.ScalarType(10.0))
bdofs_high = locate_dofs_topological(V1, fdim, source)
bc_high = dirichletbc(highV, bdofs_high, V1)

ground_rotor = ft_inner.find(surface_map["LowerRotor"])
ground_al = ft_inner.find(surface_map["AlLower"])
ground = np.concatenate([ground_rotor, ground_al])

groundV = fem.Constant(mesh, PETSc.ScalarType(0.0))
bdofs_ground = locate_dofs_topological(V1, fdim, ground)
bc_ground = dirichletbc(groundV, bdofs_ground, V1)

bc = [bc_outer, bc_high, bc_ground]

A = assemble_matrix_block(a, bcs=bc)
A.assemble()

b = assemble_vector_block(L, a, bcs=bc)

a_p = form([[a[0][0], None], [None, a[1][1]]], entity_maps=entity_maps)
P = assemble_matrix_block(a_p, bcs=bc)
P.assemble()

u_map = V.dofmap.index_map
u1_map = V1.dofmap.index_map

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + u1_map.local_range[0]
offset_u1 = offset_u + u_map.size_local * V.dofmap.index_map_bs

is_u = PETSc.IS().createStride(
    u_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF
)
is_u1 = PETSc.IS().createStride(u1_map.size_local, offset_u1, 1, comm=PETSc.COMM_SELF)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A, P)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-10)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(("u", is_u), ("u1", is_u1))
ksp_u, ksp_u1 = ksp.getPC().getFieldSplitSubKSP()

ksp_u.setType("preonly")
pc0 = ksp_u.getPC()
pc0.setType("hypre")
pc0.setHYPREType("ams")

V_CG = functionspace(mesh, ("CG", degree))._cpp_object
G = discrete_gradient(V_CG, V._cpp_object)
G.assemble()
pc0.setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(V)
    cvec_0.interpolate(
        lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_1 = Function(V)
    cvec_1.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_2 = Function(V)
    cvec_2.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])))
    )
    pc0.setHYPRESetEdgeConstantVectors(cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec)
else:
    Vec_CG = functionspace(mesh, ("CG", degree, (mesh.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc0.setHYPRESetInterpolations(mesh.geometry.dim, None, None, Pi, None)

opts = PETSc.Options()
opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 7
opts[f"{ksp_u.prefix}pc_hypre_ams_tol"] = 0
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

dofs_u0 = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
dofs_u1 = V1.dofmap.index_map.size_local * V1.dofmap.index_map_bs
total_dofs = dofs_u0 + dofs_u1

# Initial Conditions
u_n.x.array[:] = 0
u_n1.x.array[:] = 0
t = 0.0

# E field conductive region post pro

dt_submesh = fem.Constant(submesh_inner, dt_)

V_submesh = functionspace(submesh_inner, nedelec_elem)
u_n_submesh = Function(V_submesh)
u_n_submesh_prev = u_n_submesh.copy()

da_dt_submesh = (u_n_submesh - u_n_submesh_prev) / dt_submesh
E = -grad(u_n1) - da_dt_submesh

Submesh_DG = functionspace(submesh_inner, ("DG", degree + 1, (submesh_inner.geometry.dim,)))
E_vis = Function(Submesh_DG)
E_expr = Expression(E, Submesh_DG.element.interpolation_points)
E_vis.interpolate(E_expr)

E_file = VTXWriter(mesh.comm, "E_field_submesh.bp", E_vis, "BP4")
E_file.write(t)

# J field conductive region post pro

submesh_DG0 = functionspace(submesh_inner, ("DG",0))
sigma_submesh = Function(submesh_DG0)
sigma_submesh.x.array[:] = sigma.x.array[subdomain_inner_to_domain]

J = sigma_submesh * E
J_vis = Function(Submesh_DG)
J_expr = Expression(J, Submesh_DG.element.interpolation_points)
J_vis.interpolate(J_expr)

J_file = VTXWriter(mesh.comm, "J_field_submesh.bp", J_vis, "BP4")
J_file.write(t)

# B Field post pro

B = curl(u_n)

# Post pro for motor
target_tags = [4, 5, 6, 7, 8, 9, 10, 11, 12]
cell_mask = np.isin(ct.values, target_tags)
motor_cells = ct.indices[cell_mask]
motor_submesh, parent_cells, _, _ = create_submesh(mesh, tdim, motor_cells)

A_DG = functionspace(
    motor_submesh, ("Discontinuous Lagrange", degree + 1, (motor_submesh.geometry.dim,))
)
B_vis_motor = Function(A_DG)
B_file_motor = VTXWriter(mesh.comm, "B_field_submesh.bp", B_vis_motor, "BP4")

shape = (mesh.geometry.dim,)
el_B_motor = ("DG", max(degree - 1, 1), shape)
VB = functionspace(mesh, el_B_motor)
B_func_motor = Function(VB)
Bexpr = Expression(B, VB.element.interpolation_points)

B_vis_motor.interpolate(
    B_func_motor, cells0=parent_cells, cells1=np.arange(len(parent_cells), dtype=np.int32)
)
B_file_motor.write(t)

# Post pro whole domain

W1 = fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,)))

# B_vis_all = Function(W1)
# B_vis_all.interpolate(Bexpr)
# B_file_all = VTXWriter(mesh.comm, "Output_whole.bp", [B_vis_all], engine="BP4")
# B_file_all.write(t)

# ksp.setMonitor(lambda ksp, n, r: print(f"Step = {n}, Residual = {r}"))

target_tags_non = [2, 3, 4, 8, 9, 10, 11, 12]
cell_mask_non = np.isin(ct.values, target_tags)
outer_cells = ct.indices[cell_mask_non]

submesh_outer, subdomain_outer_to_domain = create_submesh(mesh, tdim, outer_cells)[:2]

DG_outer = functionspace(
    submesh_outer, ("Discontinuous Lagrange", degree + 1, (submesh_outer.geometry.dim,))
)

u_n1_file = VTXWriter(
    mesh.comm, "u_n1_field_submesh.bp", [u_n1], engine="BP4"
)
u_n1_file.write(t)

u_n_vis_motor = Function(A_DG)
u_n_vis_motor.interpolate(
    u_n, cells0=parent_cells, cells1=np.arange(len(parent_cells), dtype=np.int32)
)

u_n_file = VTXWriter(
    mesh.comm, "u_n_field_submesh.bp", [u_n_vis_motor], engine="BP4"
)
u_n_file.write(t)

A_DG_all = functionspace(
    mesh, ("Discontinuous Lagrange", degree + 1, (mesh.geometry.dim,))
)
u_n_vis_all = Function(A_DG_all)
u_n_vis_all.interpolate(u_n)

u_n_file_all = VTXWriter(
    mesh.comm, "u_n_field_whole.bp", [u_n_vis_all], engine="BP4"
)
u_n_file_all.write(t)

num_steps = 4

for n in range(num_steps):
    print(f"Step = {n}")
    t += dt_

    u_n_prev = u_n.copy()

    # update_current_density(J0z, omega_J, t, ct, currents)

    b = assemble_vector_block(L, a, bcs=bc)

    sol = A.createVecRight()
    ksp.solve(b, sol)

    residual = A * sol - b
    print("residual is ", residual.norm())

    uh.x.array[:offset] = sol.array_r[:offset]
    uh1.x.array[: (len(sol.array_r) - offset)] = sol.array_r[offset:]

    uh.x.scatter_forward()
    uh1.x.scatter_forward()

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    print("Max of A", max(u_n.x.array))

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    B = curl(u_n)
    E = -grad(u_n1) - (u_n_submesh - u_n_submesh_prev) / dt_submesh
    J = sigma_submesh * E

    iterations = ksp.getIterationNumber()
    print("Convergence reason", ksp.getConvergedReason())

    B_vis_motor.interpolate(
        B_func_motor, cells0=parent_cells, cells1=np.arange(len(parent_cells), dtype=np.int32)
    )
    B_file_motor.write(t)

    # B_vis_all.interpolate(Bexpr)
    # B_file_all.write(t)

    E_vis.interpolate(E_expr)
    E_file.write(t)

    J_vis.interpolate(J_expr)
    J_file.write(t)

    u_n_file.write(t)
    u_n1_file.write(t)
    u_n_file_all.write(t)


B_file_motor.close()
# B_file_all.close()
E_file.close()
J_file.close()
u_n_file.close()