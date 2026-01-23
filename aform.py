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
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_submesh, locate_entities_boundary
from ufl import Measure, SpatialCoordinate, TestFunction, TrialFunction, cross, curl, inner

from generate_team30_meshes_3D import domain_parameters, model_parameters
from utils import L2_norm, update_current_density

# Example usage:
# python3 generate_team30_meshes_3D.py --res 0.005 --three
# python3 solve_3D_time.py


# -- Parameters -- #

num_phases = 3
steps_per_phase = 100
freq = model_parameters["freq"]
T = num_phases * 1 / freq
dt_ = 1.0 / steps_per_phase * 1 / freq
t = 0.0

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
nu = fem.Function(DG0)

for material, domain in domains.items():
    for marker in domain:
        cells = ct.find(marker)
        mu_R.x.array[cells] = model_parameters["mu_r"][material]
        p = model_parameters["sigma"][material]
        sigma.x.array[cells] = model_parameters["sigma"][material]
        density.x.array[cells] = model_parameters["densities"][material]
        nu.x.array[cells] = model_parameters["nu"][material]

Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

dx = Measure("dx", domain=mesh, subdomain_data=ct)

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
A_space = fem.functionspace(mesh, nedelec_elem)


A = TrialFunction(A_space)
v = TestFunction(A_space)

A_prev = fem.Function(A_space)
J0z = fem.Function(DG0)

ndofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs

print(f"Number of dofs: {ndofs}")

# -- Weak Form -- #

a = dt * inner(nu * curl(A), curl(v)) * dx(Omega_c + Omega_n)
a += inner(sigma * A, v) * dx(Omega_c + Omega_n)
a = form(a)

L = dt * J0z * v[2] * dx(Omega_n)
L += inner(sigma * A_prev, v) * dx(Omega_c + Omega_n)
L = form(L)

# -- BCs and Assembly -- #

def boundary_marker(x):
    return np.full(x.shape[1], True)


mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
boundary_dofs = locate_dofs_topological(A_space, entity_dim=tdim - 1, entities=boundary_facets)

zeroA = fem.Function(A_space)
zeroA.x.array[:] = 0
bc = fem.dirichletbc(zeroA, boundary_dofs)

A_out = Function(A_space)
A = petsc.assemble_matrix(a, bcs=[bc])
A.assemble()

b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])


# -- AMS Solver Setup -- #

ksp = PETSc.KSP().create(mesh.comm)  # type: ignore
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
ksp.setOperators(A)
pc = ksp.getPC()
opts = PETSc.Options()  # type: ignore

ams_options = {
    "ksp_atol": 1e-10,
    "ksp_rtol": 1e-10,
    "ksp_type": "cg",
    "ksp_max_it": 50,
    "ksp_monitor_true_residual": None,
    "ksp_norm_type": "unpreconditioned",
    "pc_hypre_ams_cycle_type": 1,
    "pc_hypre_ams_tol": 0.0,  # Default is 1e-6 but we set it to 0.0 for AMS to be used as preconditioner
    "pc_hypre_ams_max_iter": 1,  # Set to 1 to use AMS as a preconditioner
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,6,6,4",
    "pc_hypre_ams_amg_beta_options": "10,1,6,6,4",
    "pc_hypre_ams_relax_type": 2,
    "pc_hypre_ams_relax_weight": 1.0,
    "pc_hypre_ams_relax_times": 1,
    "pc_hypre_ams_omega": 1.0,
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

results = []
num_steps = num_phases * steps_per_phase

# Create submeshs

target_tags = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
cell_mask = np.isin(ct.values, target_tags)
inner_cells = ct.indices[cell_mask]
inner_submesh, parent_cells, _, _ = create_submesh(mesh, tdim, inner_cells)

smsh_cell_imap = inner_submesh.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts)
parent_cells = parent_cells.sub_topology_to_topology(smsh_cells, inverse=False)

submesh_vec_vis = fem.functionspace(inner_submesh, ("DG", degree, (inner_submesh.geometry.dim,)))

vector_vis = fem.functionspace(mesh, ("Discontinuous Lagrange", degree, (mesh.geometry.dim,)))

scalar_vis = fem.functionspace(mesh, ("Discontinuous Lagrange", degree))

# B Field

B = curl(A_out)
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis = Function(vector_vis)
B_vis.interpolate(Bexpr)
B_file = VTXWriter(mesh.comm, "B_field_3D.bp", B_vis, "BP4")
B_file.write(t)

B_vis_submesh = Function(submesh_vec_vis)
B_vis_submesh.interpolate(B_vis, cells0=parent_cells, cells1=smsh_cells)

# E Field

E = -(A_out - A_prev) / dt
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis = Function(vector_vis)
E_vis.interpolate(Eexpr)

E_submesh = fem.Function(submesh_vec_vis)
E_submesh.interpolate(E_vis, cells0=parent_cells, cells1=smsh_cells)

# J Field

J_ind = sigma * E
J_ind_expr = fem.Expression(J_ind, vector_vis.element.interpolation_points)
J_ind_vis = Function(vector_vis)
J_ind_vis.interpolate(J_ind_expr)

J_ind_submesh = fem.Function(submesh_vec_vis)
J_ind_submesh.interpolate(J_ind_vis, cells0=parent_cells, cells1=smsh_cells)


J_vis = Function(scalar_vis)
J0z_expr = fem.Expression(J0z, scalar_vis.element.interpolation_points)
J_vis.interpolate(J0z_expr)


if output:
    B_file = VTXWriter(mesh.comm, "B_field_3D.bp", B_vis, "BP4")
    B_file.write(t)

    B_file_submesh = VTXWriter(mesh.comm, "B_field_3D_submesh.bp", B_vis_submesh, "BP4")
    B_file_submesh.write(t)

    J_file = VTXWriter(mesh.comm, "J0z_3D.bp", J_vis, "BP4")
    J_file.write(t)

    E_file = VTXWriter(mesh.comm, "E_field_3D.bp", E_submesh, "BP4")
    E_file.write(t)

    J_ind_file = VTXWriter(mesh.comm, "J_induced_3D.bp", J_ind_submesh, "BP4")
    J_ind_file.write(t)


for i in range(num_steps):
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
        ksp.solve(b, A_out.x.petsc_vec)
        A_out.x.scatter_forward()

    reason = ksp.getConvergedReason()
    iter_count = ksp.getIterationNumber()

    # Compute B

    B = curl(A_out)

    # Compute F
    E = -(A_out - A_prev) / dt
    f = cross(sigma * E, B)
    F = fem.Function(vector_vis)
    fexpr = fem.Expression(f, vector_vis.element.interpolation_points)
    F.interpolate(fexpr)

    # Compute J_ind
    J_ind = sigma * E

    # Write B
    if output:
        B_vis.interpolate(Bexpr)
        B_file.write(t)

        B_vis_submesh.interpolate(B_vis, cells0=parent_cells, cells1=smsh_cells)
        B_file_submesh.write(t)

        J_vis.interpolate(J0z_expr)
        J_file.write(t)

        E_vis.interpolate(Eexpr)
        E_submesh.interpolate(E_vis, cells0=parent_cells, cells1=smsh_cells)
        E_file.write(t)

        J_ind_vis.interpolate(J_ind_expr)
        J_ind_submesh.interpolate(J_ind_vis, cells0=parent_cells, cells1=smsh_cells)
        J_ind_file.write(t)

    A_prev.x.array[:] = A_out.x.array  # Set A_prev

    sigma_non_conducting = model_parameters["sigma"]["Cu"]
    stats = {
        "step": i,
        "ndofs": ndofs,
        "sigma_value": sigma_non_conducting,
        "solve_time": timing("solve")[1],
        "iterations": iter_count,
        "reason": reason,
        "norm_A": np.linalg.norm(A_out.x.array),
        "norm_B": L2_norm(B),
        "max_b": np.max(B_vis.x.array),
        "residual_norm": ksp.getResidualNorm(),
    }
    print(stats)
    results.append(stats)

    if write_stats:
        df = pd.DataFrame.from_dict(results)
        df.to_csv("output_3D_stats.csv", mode="w")
