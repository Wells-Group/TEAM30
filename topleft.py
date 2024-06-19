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
from dolfinx.fem.petsc import assemble_vector


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

A = TrialFunction(A_space)
v = TestFunction(A_space)

A_prev = fem.Function(A_space)
J0z = fem.Function(DG0)

ndofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs

# -- Weak Form -- #
# a = [[a00, a01],
#      [a10, a11]]

a00 = dt * (1 / mu_R) * inner(curl(A), curl(v)) * dx(Omega_n) #Magnetic Vector potential with edge basis
a00 += sigma * mu_0 * inner(A, v) * dx(Omega_n)
a00 += dt * (1 / mu_R) * inner(curl(A), curl(v)) * dx(Omega_c)
a00 += sigma * mu_0 * inner(A, v) * dx(Omega_c)

a = form(a00)

# A block-diagonal preconditioner will be used with the iterative
# solvers for this problem:

L0 = dt * mu_0 * J0z * v[2] * dx(Omega_c + Omega_n)
L0 += sigma * mu_0 * inner(A_prev, v) * dx(Omega_c + Omega_n)

L = form(L0)

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

bcs = bc0

# Assemble block matrix operators

A00 = assemble_matrix(form(a00), bcs= [bc0])
A00.assemble()

b = assemble_vector(L)

ksp = PETSc.KSP().create(mesh.comm)

ksp.setOperators(A00)

ksp.setType("gmres")
pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

W = fem.functionspace(mesh, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, A_space._cpp_object)
G.assemble()
pc.setHYPREDiscreteGradient(G)

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
    pc.setHYPRESetEdgeConstantVectors(cvec_0.vector, cvec_1.vector, cvec_2.vector)
else:
    shape = (mesh.geometry.dim,)
    Q = fem.functionspace(mesh, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q._cpp_object, A_space._cpp_object)
    Pi.assemble()
    pc.setHYPRESetInterpolations(mesh.geometry.dim, None, None, Pi, None)

pc.setUp()
ksp.setUp()

ksp.view()

# -- Time simulation -- #

A_out = Function(A_space)
 
shape = (mesh.geometry.dim,)
W1 = fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,)))

if output:
    B_output = Function(W1)
    B_vtx = io.VTXWriter(mesh.comm, "output_3D_B.bp", [B_output], engine="BP4")

t = 0
results = []

#Initial Conditions
# A_out.x.array[:] = 0 
# #%%
# for i in range(10):  #num_phases * steps_per_phase
#     print(f"Step = {i}")

#     t += dt_

#     # Update Current and Re-assemble RHS
#     update_current_density(J0z, omega_J, t, ct, currents)

#     b = assemble_vector(L)

#     # Solve
#     sol = A00.createVecRight()  #Solution Vector
    
#     ksp.solve(b, sol)

#     residual = A00 * sol - b
#     print('resiidual is ', residual.norm())

#     A_out.x.array[:] = sol.array_r[:]

#     print("sol after solve ", max(A_out.x.array))

#     print("Norm of A mat ", A00.norm())

#     # Compute B
#     el_B = ("DG", max(degree - 1, 1), shape)
#     VB = fem.functionspace(mesh, el_B)
#     B = fem.Function(VB)
#     B_3D = curl(A_out)
#     Bexpr = fem.Expression(B_3D, VB.element.interpolation_points())
#     B.interpolate(Bexpr)

#     # Compute F
#     E = - (A_out - A_prev) / dt
#     f = cross(sigma * E, B)
#     F = fem.Function(VB)
#     fexpr = fem.Expression(f, VB.element.interpolation_points())
#     F.interpolate(fexpr)

#     # print(compute_loss(A_out,A_prev, dt))

#     # A_prev.x.array[:offset_A] = x.array_r[:offset_A]
#     # S_prev.x.array[:offset_S] = x.array_r[offset_S:]
#     A_prev.x.array[:] = A_out.x.array

# print("Max B field is", max(B.x.array))

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
#Eigenvalue calculation

from slepc4py import SLEPc

shift = 5.5
n_eigs = 1

eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
eps.setOperators(A00)
eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
eps.setFromOptions()
eps.solve()

exit()

its = eps.getIterationNumber()
eps_type = eps.getType()
n_ev, n_cv, mpd = eps.getDimensions()
tol, max_it = eps.getTolerances()
n_conv = eps.getConverged()

computed_eigenvalues = []
for i in range(min(n_conv, n_eigs)):
    lmbda = eps.getEigenvalue(i)
    computed_eigenvalues.append(np.round(np.real(lmbda), 1))

print(f"Number o iterations: {its}")
print(f"Solution method: {eps_type}")
print(f"Number of requested eigenvalues: {n_ev}")
print(f"Stopping condition: tol={tol}, maxit={max_it}")
print(f"Number of converged eigenpairs: {n_conv}")
# %%
