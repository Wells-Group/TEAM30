from petsc4py.PETSc import NormType
from basix.ufl import element
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, io
from dolfinx.cpp.fem.petsc import (discrete_gradient, interpolation_matrix)
from dolfinx.fem import (Function, form, locate_dofs_topological, petsc)
from dolfinx.mesh import locate_entities_boundary
from ufl import (TestFunction, TrialFunction, curl, grad, inner, div, SpatialCoordinate, Measure)

from utils import update_current_density
from generate_team30_meshes_3D import domain_parameters, model_parameters


# Example usage:
# python3 generate_team30_meshes_3D.py --res 0.005 --three
# python3 solve_3D_time.py


# -- Parameters -- #

num_phases = 1
steps_per_phase = 100
freq = model_parameters["freq"]
T = num_phases * 1 / freq
dt_ = 1.0 / steps_per_phase * 1 / freq

print(dt_)

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

for (material, domain) in domains.items():
    for marker in domain:
        cells = ct.find(marker)
        mu_R.x.array[cells] = model_parameters["mu_r"][material]
        sigma.x.array[cells] = model_parameters["sigma"][material]
        density.x.array[cells] = model_parameters["densities"][material]

np.set_printoptions(threshold=np.inf)

Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

dx = Measure("dx", domain=mesh, subdomain_data=ct)

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
A_space = fem.functionspace(mesh, nedelec_elem)

# Scalar potential
element = element("CG", mesh.basix_cell(), degree)
V = fem.functionspace(mesh, element)


A = TrialFunction(A_space)
v = TestFunction(A_space)

S = TrialFunction(V)
q = TestFunction(V)

A_prev = fem.Function(A_space)
J0z = fem.Function(DG0)
zero = fem.Constant(mesh, PETSc.ScalarType(0))

print(A_prev.x.array.size)
print(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

ndofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs
print(f"Number of dofs: {ndofs}")

# -- Weak Form -- #
# a = [[a00, a01],
#      [a10, a11]]

a00 = dt * 1 / mu_R * inner(curl(A), curl(v)) * dx(Omega_c + Omega_n)
a00 += sigma * mu_0 * inner(A, v) * dx(Omega_c + Omega_n)


a01 = mu_0 * sigma * inner(v, grad(S)) * dx(Omega_c + Omega_n)
a10 = zero * div(A) * q * dx
a11 = mu_0 * sigma * inner(grad(S), grad(q)) * dx

a = form([[a00, a01], [a10, a11]])

# A block-diagonal preconditioner will be used with the iterative
# solvers for this problem:
a_p = [[a00, None], [None, a11]]

L0 = dt * mu_0 * J0z * v[2] * dx(Omega_n)
L0 += sigma * mu_0 * inner(A_prev, v) * dx(Omega_c + Omega_n)

L1 = inner(fem.Constant(mesh, PETSc.ScalarType(0)), q) * dx
L = form([L0, L1])

# -- Create boundary conditions -- #


def boundary_marker(x):
    return np.full(x.shape[1], True)


mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
bdofs0 = locate_dofs_topological(A_space, entity_dim=tdim - 1, entities=boundary_facets)
print(bdofs0)

zeroA = fem.Function(A_space)
zeroA.x.array[:] = 0
bc0 = fem.dirichletbc(zeroA, bdofs0)

bdofs1 = locate_dofs_topological(V, entity_dim=tdim - 1, entities=boundary_facets)
bc1 = fem.dirichletbc(fem.Constant(mesh, PETSc.ScalarType(0)), bdofs1, V)

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]

# -- Create PETSc matrices and vectors -- #

# create RHS vector
b = fem.petsc.create_vector_nest(L)
x = fem.petsc.create_vector_nest(L)

# Assemble nested matrix operators
A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
A.assemble()

A00 = A.getNestSubMatrix(0, 0)
A10 = A.getNestSubMatrix(1, 0)
A11 = A.getNestSubMatrix(1, 1)
A01 = A.getNestSubMatrix(0, 1)


print(A11.norm(NormType.NORM_FROBENIUS))
exit()

# Extract submatrices, A11 is Positive Definite
A00 = A.getNestSubMatrix(0, 0)
A11 = A.getNestSubMatrix(1, 1)
A11.setOption(PETSc.Mat.Option.SPD, True)


# -- Create Krylov solver -- #

# Create matrix for preconditioner
P = PETSc.Mat().createNest([[A00, None], [None, A11]])
P00, P11 = P.getNestSubMatrix(0, 0), P.getNestSubMatrix(1, 1)

# Create a GMRES Krylov solver and a block-diagonal preconditioner
# using PETSc's additive fieldsplit preconditioner
ksp = PETSc.KSP().create(mesh.comm)  # type: ignore
ksp.setOperators(A, P)
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-9)
ksp.setNormType(PETSc.KSP.NormType.NORM_UNPRECONDITIONED)

# Set preconditioner parameters
pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

# Define the matrix blocks in the preconditioner with the Vector
# and scalar potential matrices index sets
nested_IS = P.getNestISs()
pc.setFieldSplitIS(("A", nested_IS[0][0]), ("S", nested_IS[1][1]))

# Set the preconditioners for each block. For the top-left
# curl-curl-type operator we use AMS. For the
# lower-right block we use a AMG preconditioner.
ksp_0, ksp_1 = pc.getFieldSplitSubKSP()
ksp_0.setType("preonly")
pc0 = ksp_0.getPC()

pc0.setType("hypre")
pc0.setHYPREType("ams")

# FIXME: How to set the AMS preconditioner options?
# ams_options = {"pc_hypre_ams_cycle_type": 1,
#                "pc_hypre_ams_tol": 1e-8,
#                }

W = fem.functionspace(mesh, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, A_space._cpp_object)
G.assemble()

shape = (mesh.geometry.dim,)
Q = fem.functionspace(mesh, ("Lagrange", degree, shape))
Pi = interpolation_matrix(Q._cpp_object, A_space._cpp_object)
Pi.assemble()

pc0.setHYPREDiscreteGradient(G)
pc0.setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)

ksp_1.setType("preonly")
pc1 = ksp_1.getPC()
pc1.setType("gamg")


# ksp.solve(b, x)
# ksp.setFromOptions()
# pc.setUp()
# ksp.setUp()

update_current_density(J0z, omega_J, 0.5, ct, currents)
petsc.assemble_vector_nest(b, L)
fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs)
fem.petsc.set_bc_nest(b, bcs0)

ksp.setUp()
A_out, S_out = Function(A_space), Function(V)
# ksp.solve(b, x)
ksp.view()

x.view()

# ksp.view()
# # -- Time simulation -- #

# shape = (mesh.geometry.dim,)
# W1 = fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,)))

# if output:
#     B_output = Function(W1)
#     B_vtx = VTXWriter(mesh.comm, "output_3D_B.bp", [B_output], engine="BP4")

# t = 0
# results = []

# for i in range(num_phases * steps_per_phase):

#     A_out.x.array[:] = 0
#     t += dt_

#     # Update Current and Re-assemble LHS
#     update_current_density(J0z, omega_J, t, ct, currents)
#     with b.localForm() as loc_b:
#         loc_b.set(0)
#     b = petsc.assemble_vector(b, L)

#     petsc.apply_lifting(b, [a], bcs=[[bc]])
#     b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
#     petsc.set_bc(b, [bc])
#     max_b = max(b.array)

#     # Solve
#     with Timer("solve"):
#         ksp.solve(b, A_out.vector)
#         A_out.x.scatter_forward()

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
#     A_prev.x.array[:] = A_out.x.array  # Set A_prev

#     # Write B
#     if output:
#         B_output_1 = Function(W1)
#         B_output_1.interpolate(B)
#         B_output.x.array[:] = B_output_1.x.array[:]
#         B_vtx.write(t)

#     min_cond = model_parameters['sigma']['Cu']
#     stats = {"step": i, "ndofs": ndofs, "min_cond": min_cond, "solve_time": timing("solve")[1],
#              "iterations": ksp.its, "reason": ksp.getConvergedReason(),
#              "norm_A": np.linalg.norm(A_out.x.array), "max_b": max_b}
#     print(stats)
#     results.append(stats)

#     if write_stats:
#         df = pd.DataFrame.from_dict(results)
#         df.to_csv('output_3D_stats.csv', mode="w")
