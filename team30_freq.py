# Copyright (C) 2021 Jørgen S. Dokken and Igor A. Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import argparse
import os

import dolfinx
import dolfinx.io
import matplotlib.pyplot as plt
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from typing import Callable

from generate_team30_meshes import (domain_parameters, model_parameters,
                                    surface_map)
from utils import (DerivedQuantities2D, MagneticFieldProjection2D, XDMFWrapper,
                   update_current_density)


def solve_team30(single_phase: bool, omega_u: np.float64, degree: np.int32,
                 form_compiler_parameters: dict = {}, jit_parameters: dict = {}):
    """
    Solve the TEAM 30 problem for a single or three phase engine.

    Parameters
    ==========
    single_phase
        If true run the single phase model, otherwise run the three phase model

    omega_u
        Angular speed of rotor (Used as initial speed if apply_torque is True)

    degree
        Degree of magnetic vector potential functions space

    form_compiler_parameters
        Parameters used in FFCx compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINx.

    jit_parameters
        Parameters used in CFFI JIT compilation of C code generated by FFCx.
        See `python/dolfinx/jit.py` for all available parameters.
        Takes priority over all other parameter values.

    """

    if not PETSc.ScalarType is np.complex128:
        raise RuntimeError("PETSc needs to be compiler with complex support.")

    mu_0 = model_parameters["mu_0"]
    omega_J = 2 * np.pi * model_parameters["freq"]

    ext = "single" if single_phase else "three"
    fname = f"meshes/{ext}_phase"

    domains, currents = domain_parameters(single_phase, True)

    # Read mesh and cell markers
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    # Read facet tag
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")

    # Create DG 0 function for mu_R and sigma
    DG0 = dolfinx.FunctionSpace(mesh, ("DG", 0))
    mu_R = dolfinx.Function(DG0)
    sigma = dolfinx.Function(DG0)
    density = dolfinx.Function(DG0)
    for (material, domain) in domains.items():
        for marker in domain:
            cells = ct.indices[ct.values == marker]
            mu_R.x.array[cells] = model_parameters["mu_r"][material]
            sigma.x.array[cells] = model_parameters["sigma"][material]
            density.x.array[cells] = model_parameters["densities"][material]

    # Define problem function space
    cell = mesh.ufl_cell()
    FE = ufl.FiniteElement("Lagrange", cell, degree)
    ME = ufl.MixedElement([FE, FE])
    VQ = dolfinx.FunctionSpace(mesh, ME)

    # Define test, trial and functions for previous timestep
    Az, V = ufl.TrialFunctions(VQ)
    vz, q = ufl.TestFunctions(VQ)
    AnVn = dolfinx.Function(VQ)
    An, _ = ufl.split(AnVn)  # Solution at previous time step
    J0z = dolfinx.Function(DG0)  # Current density

    # Create integration sets
    Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
    Omega_c = domains["Rotor"] + domains["Al"]

    # Create integration measures
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=surface_map["Exterior"])

    # Define temporal and spatial parameters
    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)

    omega = dolfinx.Constant(mesh, omega_u)

    # Define variational form
    a = 1 / mu_R * ufl.inner(ufl.grad(Az), ufl.grad(vz)) * dx(Omega_n + Omega_c)
    a += 1j * omega_J * mu_0 * sigma * ufl.inner(Az, vz) * dx(Omega_c)
    a += mu_0 * sigma * (ufl.inner(V.dx(0), q.dx(0)) + ufl.inner(V.dx(1), q.dx(1))) * dx(Omega_c)
    L = mu_0 * ufl.inner(J0z, vz) * dx(Omega_n)

    # Motion voltage term
    u = omega * ufl.as_vector((-x[1], x[0]))
    a += mu_0 * sigma * ufl.inner(ufl.dot(u, ufl.grad(Az)), vz) * dx(Omega_c)

    # Find all dofs in Omega_n for Q-space
    cells_n = np.hstack([ct.indices[ct.values == domain] for domain in Omega_n])
    Q = VQ.sub(1).collapse()
    deac_dofs = dolfinx.fem.locate_dofs_topological((VQ.sub(1), Q), tdim, cells_n)

    # Create zero condition for V in Omega_n
    zeroQ = dolfinx.Function(Q)
    zeroQ.x.array[:] = 0
    bc_Q = dolfinx.DirichletBC(zeroQ, deac_dofs, VQ.sub(1))

    # Create external boundary condition for V space
    V_ = VQ.sub(0).collapse()
    tdim = mesh.topology.dim

    def boundary(x):
        return np.full(x.shape[1], True)

    boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, boundary)
    bndry_dofs = dolfinx.fem.locate_dofs_topological((VQ.sub(0), V_), tdim - 1, boundary_facets)
    zeroV = dolfinx.Function(V_)
    zeroV.x.array[:] = 0
    bc_V = dolfinx.DirichletBC(zeroV, bndry_dofs, VQ.sub(0))
    bcs = [bc_V, bc_Q]

    # Create sparsity pattern and matrix with additional non-zeros on diagonal
    cpp_a = dolfinx.Form(a, form_compiler_parameters=form_compiler_parameters,
                         jit_parameters=jit_parameters)._cpp_object
    pattern = dolfinx.cpp.fem.create_sparsity_pattern(cpp_a)
    block_size = VQ.dofmap.index_map_bs
    deac_blocks = deac_dofs[0] // block_size
    pattern.insert_diagonal(deac_blocks)
    pattern.assemble()

    # Create matrix based on sparsity pattern
    A = dolfinx.cpp.la.create_matrix(mesh.mpi_comm(), pattern)
    A.zeroEntries()

    dolfinx.fem.assemble_matrix(A, cpp_a, bcs=bcs)
    A.assemble()

    # Create inital vector for LHS
    cpp_L = dolfinx.Form(L, form_compiler_parameters=form_compiler_parameters,
                         jit_parameters=jit_parameters)._cpp_object
    b = dolfinx.fem.create_vector(cpp_L)

    # Create solver
    solver = PETSc.KSP().create(mesh.mpi_comm())
    solver.setOperators(A)
    prefix = "AV_"
    solver.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    opts[f"{prefix}ksp_type"] = "preonly"
    opts[f"{prefix}pc_type"] = "lu"
    # opts[f"{prefix}ksp_converged_reason"] = None
    # opts[f"{prefix}ksp_monitor_true_residual"] = None
    # opts[f"{prefix}ksp_gmres_modifiedgramschmidt"] = None
    # opts[f"{prefix}ksp_diagonal_scale"] = None
    # opts[f"{prefix}ksp_gmres_restart"] = 500
    # opts[f"{prefix}ksp_rtol"] = 1e-08
    # opts[f"{prefix}ksp_max_it"] = 50000
    # opts[f"{prefix}ksp_view"] = None
    # opts[f"{prefix}ksp_monitor"] = None
    solver.setFromOptions()
    # Function for containg the solution
    AzV = dolfinx.Function(VQ)

    # Post-processing function for projecting the magnetic field potential
    post_B = MagneticFieldProjection2D(AzV)

    # Class for computing torque, losses and induced voltage
    derived = DerivedQuantities2D(AzV, AnVn, u, sigma, domains, ct, ft)

    # Create output file
    postproc = XDMFWrapper(mesh.mpi_comm(), f"results/TEAM30_{omega_u}_{ext}")
    postproc.write_mesh(mesh)
    # postproc.write_function(sigma, 0, "sigma")
    # postproc.write_function(mu_R, 0, "mu_R")

    # set current density
    for domain, values in currents.items():
        cells = ct.indices[ct.values == domain]
        J0z.x.array[cells] = model_parameters["J"] * values["alpha"] * np.exp(1j * values["beta"])

    # exit()
    # Reassemble RHS
    with b.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.assemble_vector(b, cpp_L)
    dolfinx.fem.apply_lifting(b, [cpp_a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    # Solve problem
    solver.solve(b, AzV.vector)
    AzV.x.scatter_forward()

    torque_surface = derived.torque_surface()
    torque_volume = derived.torque_volume()

    post_B.solve()

    postproc.write_function(AzV.sub(0).collapse(), 0, "Az")
    postproc.write_function(J0z, 0, "J0z")
    postproc.write_function(AzV.sub(1).collapse(), 0, "V")
    postproc.write_function(post_B.B, 0, "B")

    print(f"RMS Torque (surface): {torque_surface}")
    print(f"RMS Torque (vol): {torque_volume}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scripts to  solve the TEAM 30 problem"
        + " (http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _single = parser.add_mutually_exclusive_group(required=False)
    _single.add_argument('--single', dest='single', action='store_true',
                         help="Solve single phase problem", default=False)
    _three = parser.add_mutually_exclusive_group(required=False)
    _three.add_argument('--three', dest='three', action='store_true',
                        help="Solve three phase problem", default=False)
    _torque = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument("--omega", dest='omegaU', type=np.float64, default=0, help="Angular speed of rotor [rad/s]")
    parser.add_argument("--degree", dest='degree', type=int, default=1,
                        help="Degree of magnetic vector potential functions space")
    args = parser.parse_args()

    os.system("mkdir -p results_complex")
    if args.single:
        solve_team30(True, args.omegaU, args.degree)
    if args.three:
        solve_team30(False, args.omegaU, args.degree)
