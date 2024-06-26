#%%
from petsc4py import PETSc
from mpi4py import MPI

import numpy as np
from basix.ufl import element
from dolfinx import fem, io, la, default_scalar_type
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import Function, form, locate_dofs_topological, petsc
from dolfinx.mesh import locate_entities_boundary
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    curl,
    grad,
    inner,
    cross,
)
import ufl
from generate_team30_meshes_3D import domain_parameters, model_parameters
from utils import update_current_density
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, assemble_matrix

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

# Scalar potential
element = element("Lagrange", mesh.basix_cell(), degree)
V = fem.functionspace(mesh, element)

A = TrialFunction(A_space)
v = TestFunction(A_space)

S = TrialFunction(V)   #Scalar Potential Trial
q = TestFunction(V)

A_prev = fem.Function(A_space)
S_prev = fem.Function(V)
J0z = fem.Function(DG0)

ndofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs

# -- Weak Form -- #
# a = [[a00, a01],
#      [a10, a11]]

a00 = dt * (1 / mu_R) * inner(curl(A), curl(v)) * dx(Omega_n) #Magnetic Vector potential with edge basis
a00 += sigma * mu_0 * inner(A, v) * dx(Omega_n)
a00 += dt * (1 / mu_R) * inner(curl(A), curl(v)) * dx(Omega_c)
a00 += sigma * mu_0 * inner(A, v) * dx(Omega_c)

a01 = sigma * inner(grad(S),v) * (dx(Omega_c)+dx(Omega_n)) # Coupling of Elec Scalar Potential(S) with the test function of A (Magnetic Vector Potentail)
a10 = sigma * mu_0 * inner(grad(q), A) * (dx(Omega_c)+dx(Omega_n)) # Coupling of A with the test function of the Electric Scalar Potential

a11 = sigma * mu_0 * inner(grad(S), grad(q)) * (dx(Omega_c)+dx(Omega_n)) #Lagrange Test and Trial
# a01 = None
# a10 = None

a = form([[a00, a01], [a10, a11]])

# A block-diagonal preconditioner will be used with the iterative
# solvers for this problem:
a_p = form([[a00, None], [None, a10]])

L0 = dt * mu_0 * J0z * v[2] * dx(Omega_c + Omega_n)
L0 += sigma * mu_0 * inner(A_prev, v) * dx(Omega_c + Omega_n)

L1 = sigma * mu_0 * inner(grad(S_prev), grad(q))* dx(Omega_c + Omega_n)

L = form([L0, L1])

# -- Create boundary conditions -- #

def boundary_marker(x):
    return np.full(x.shape[1], True)

# TODO ext facet inds
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

# Assemble block matrix operators

# A00 = assemble_matrix(form(a00), bcs= [bc0])
# A00.scatter_reverse()
# print('A00 norm = ', np.sqrt(A00.squared_norm()))

#%%
A_mat = assemble_matrix_block(a, bcs = bcs)
A_mat.assemble()
print("norm of A ", A_mat.norm())

P = assemble_matrix_block(a_p, bcs = bcs)
P.assemble()

b = assemble_vector_block(L, a, bcs = bcs)

A_map = A_space.dofmap.index_map
V_map = V.dofmap.index_map

offset_A = A_map.local_range[0] * A_space.dofmap.index_map_bs + V_map.local_range[0]
offset_S = offset_A + A_map.size_local * A_space.dofmap.index_map_bs
is_A = PETSc.IS().createStride(A_map.size_local * A_space.dofmap.index_map_bs, offset_A, 1, comm=PETSc.COMM_SELF)
is_S = PETSc.IS().createStride(V_map.size_local, offset_S, 1, comm=PETSc.COMM_SELF)

# Set preconditioned parameters

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A_mat, P)
ksp.setType("gmres")
pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
pc.setFieldSplitIS(("A", is_A), ("S", is_S))
ksp_A, ksp_S = ksp.getPC().getFieldSplitSubKSP()

pc_a = ksp_A.getPC()

pc_a.setType("hypre")
pc_a.setHYPREType("ams")

W = fem.functionspace(mesh, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, A_space._cpp_object)
G.assemble()
pc_a.setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(A_space)
    cvec_0.interpolate(
        lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_1 = Function(A_space)
    cvec_1.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_2 = Function(A_space)
    cvec_2.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])))
    )
    pc_a.setHYPRESetEdgeConstantVectors(cvec_0.vector, cvec_1.vector, cvec_2.vector)
else:
    shape = (mesh.geometry.dim,)
    Q = fem.functionspace(mesh, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q._cpp_object, A_space._cpp_object)
    Pi.assemble()
    pc_a.setHYPRESetInterpolations(mesh.geometry.dim, None, None, Pi, None)


ksp_S.setType("cg")
pc_s = ksp_S.getPC()
pc_s.setType("gamg")

ksp.setUp()
pc_a.setUp()
pc_s.setUp()

ksp.view()
#%%
# -- Time simulation -- #

A_out, S_out = Function(A_space), Function(V)

shape = (mesh.geometry.dim,)
W1 = fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,)))

if output:
    B_output = Function(W1)
    B_vtx = io.VTXWriter(mesh.comm, "output_3D_B.bp", [B_output], engine="BP4")

t = 0
results = []

#Initial Conditions
A_out.x.array[:] = 0 
S_out.x.array[:] = 0

offset = A_space.dofmap.index_map.size_local * A_space.dofmap.index_map_bs
#%%
for i in range(20):  #num_phases * steps_per_phase
    print(f"Step = {i}")

    t += dt_

    # Update Current and Re-assemble RHS
    update_current_density(J0z, omega_J, t, ct, currents)

    b = assemble_vector_block(L, a, bcs = bcs)

    # Solve
    sol = A_mat.createVecRight()  #Solution Vector
    
    ksp.solve(b, sol)

    residual = A_mat * sol - b
    print('resiidual is ', residual.norm())

    A_out.x.array[:offset] = sol.array_r[:offset]
    S_out.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

    print("sol after solve ", max(A_out.x.array))

    # print("Norm of A mat ", A_mat.norm())

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

    # print(compute_loss(A_out,A_prev, dt))

    # A_prev.x.array[:offset_A] = x.array_r[:offset_A]
    # S_prev.x.array[:offset_S] = x.array_r[offset_S:]
    A_prev.x.array[:] = A_out.x.array
    S_prev.x.array[:] = S_out.x.array

print("Max B field is", max(B.x.array))

# # Write B
# if output:
#     B_output_1 = Function(W1)
#     B_output_1.interpolate(B)
#     B_output.x.array[:] = B_output_1.x.array[:]
#     B_vtx.write(t)

# print(B_output.x.array)
# from IPython import embed
# embed()

# print(max(B_output.x.array))


# num_steps = int(T / float(dt.value))
# times = np.zeros(num_steps + 1, dtype=default_scalar_type)
# num_periods = np.round(60 * T)
# last_period = np.flatnonzero(
#     np.logical_and(times > (num_periods - 1) / 60, times < num_periods / 60)
# )


# # Create Functions to split A and v
# a_sol, v_sol = Function(A_space), Function(V)
# offset = A_map.size_local * A_space.dofmap.index_map_bs
# a_sol.x.array[:offset] = x.array_r[:offset]
# v_sol.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]

# norm_u, norm_p = a_sol.x.norm(), v_sol.x.norm()
# if MPI.COMM_WORLD.rank == 0:
#     print(f"(B) Norm of velocity coefficient vector (blocked, iterative): {norm_u}")
#     print(f"(B) Norm of pressure coefficient vector (blocked, iterative): {norm_p}")


    # min_cond = model_parameters['sigma']['Cu']
    # stats = {"step": i, "ndofs": ndofs, "min_cond": min_cond, "solve_time": timing("solve")[1],
    #          "iterations": ksp.its, "reason": ksp.getConvergedReason(),
    #          "norm_A": np.linalg.norm(A_out.x.array), "max_b": max_b}
    # print(stats)
    # results.append(stats)

    # if write_stats:
    #     df = pd.DataFrame.from_dict(results)
    #     df.to_csv('output_3D_stats.csv', mode="w")

# %%