# Copyright (C) 2021-2022 Jørgen S. Dokken and Igor A. Baratta
#
# SPDX-License-Identifier:    MIT

import argparse
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Callable, Optional, TextIO, Union

import dolfinx.fem.petsc as _petsc
import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import ufl
from dolfinx import cpp, fem, io, default_scalar_type
from dolfinx.io import VTXWriter
from mpi4py import MPI
from petsc4py import PETSc

from generate_team30_meshes import (domain_parameters, model_parameters,
                                    surface_map)
from utils import DerivedQuantities2D, MagneticField2D, update_current_density


def solve_team30(single_phase: bool, num_phases: int, omega_u: np.float64, degree: np.int32, petsc_options: dict = {},
                 form_compiler_options: dict = {}, jit_parameters: dict = {}, apply_torque: bool = False,
                 T_ext: Callable[[float], float] = lambda t: 0, outdir: Path = Path("results"),
                 steps_per_phase: int = 100, outfile: Optional[Union[TextIOWrapper, TextIO]] = sys.stdout,
                 plot: bool = False, progress: bool = False, mesh_dir: Path = Path("meshes"),
                 save_output: bool = False):
    """
    Solve the TEAM 30 problem for a single or three phase engine.

    Parameters
    ==========
    single_phase
        If true run the single phase model, otherwise run the three phase model

    num_phases
        Number of phases to run the simulation for

    omega_u
        Angular speed of rotor (Used as initial speed if apply_torque is True)

    degree
        Degree of magnetic vector potential functions space

    petsc_options
        Parameters that is passed to the linear algebra backend
        PETSc. For available choices for the 'petsc_options' kwarg,
        see the `PETSc-documentation
        <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`

    form_compiler_options
        Parameters used in FFCx compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINx.

    jit_parameters
        Parameters used in CFFI JIT compilation of C code generated by FFCx.
        See `python/dolfinx/jit.py` for all available parameters.
        Takes priority over all other parameter values.

    apply_torque
        Boolean if torque should affect rotation. If True `omega_u` is ignored and T_ext
        is used as external forcing

    T_ex
        Lambda function for describing the external forcing as a function of time

    outdir
        Directory to put results in

    steps_per_phase
        Number of time steps per phase of the induction engine

    outfile
        File to write results to. (Default is print to terminal)

    plot
        Plot torque and voltage over time

    progress
        Show progress bar for solving in time

    mesh_dir
        Directory containing mesh

    save_output
        Save output to bp-files
    """
    freq = model_parameters["freq"]
    T = num_phases * 1 / freq
    dt_ = 1 / steps_per_phase * 1 / freq
    mu_0 = model_parameters["mu_0"]
    omega_J = 2 * np.pi * freq

    ext = "single" if single_phase else "three"
    fname = mesh_dir / f"{ext}_phase"

    domains, currents = domain_parameters(single_phase)

    # Read mesh and cell markers
    with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        ct = xdmf.read_meshtags(mesh, name="Cell_markers")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        ft = xdmf.read_meshtags(mesh, name="Facet_markers")
    # Create DG 0 function for mu_R and sigma
    DG0 = fem.FunctionSpace(mesh, ("DG", 0))
    mu_R = fem.Function(DG0)
    sigma = fem.Function(DG0)
    density = fem.Function(DG0)
    for (material, domain) in domains.items():
        for marker in domain:
            cells = ct.find(marker)
            mu_R.x.array[cells] = model_parameters["mu_r"][material]
            sigma.x.array[cells] = model_parameters["sigma"][material]
            density.x.array[cells] = model_parameters["densities"][material]

    # Define problem function space
    cell = mesh.ufl_cell()
    FE = ufl.FiniteElement("Lagrange", cell, degree)
    ME = ufl.MixedElement([FE, FE])
    VQ = fem.FunctionSpace(mesh, ME)

    # Define test, trial and functions for previous timestep
    Az, V = ufl.TrialFunctions(VQ)
    vz, q = ufl.TestFunctions(VQ)
    AnVn = fem.Function(VQ)
    An, _ = ufl.split(AnVn)  # Solution at previous time step
    J0z = fem.Function(DG0)  # Current density

    # Create integration sets
    Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
    Omega_c = domains["Rotor"] + domains["Al"]

    # Create integration measures
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=surface_map["Exterior"])

    # Define temporal and spatial parameters
    n = ufl.FacetNormal(mesh)
    dt = fem.Constant(mesh, dt_)
    x = ufl.SpatialCoordinate(mesh)

    omega = fem.Constant(mesh, default_scalar_type(omega_u))

    # Define variational form
    a = dt / mu_R * ufl.inner(ufl.grad(Az), ufl.grad(vz)) * dx(Omega_n + Omega_c)
    a += dt / mu_R * vz * (n[0] * Az.dx(0) - n[1] * Az.dx(1)) * ds
    a += mu_0 * sigma * Az * vz * dx(Omega_c)
    a += dt * mu_0 * sigma * (V.dx(0) * q.dx(0) + V.dx(1) * q.dx(1)) * dx(Omega_c)
    L = dt * mu_0 * J0z * vz * dx(Omega_n)
    L += mu_0 * sigma * An * vz * dx(Omega_c)

    # Motion voltage term
    u = omega * ufl.as_vector((-x[1], x[0]))
    a += dt * mu_0 * sigma * ufl.dot(u, ufl.grad(Az)) * vz * dx(Omega_c)

    # Find all dofs in Omega_n for Q-space
    cells_n = np.hstack([ct.find(domain) for domain in Omega_n])
    Q, _ = VQ.sub(1).collapse()
    deac_dofs = fem.locate_dofs_topological((VQ.sub(1), Q), tdim, cells_n)

    # Create zero condition for V in Omega_n
    zeroQ = fem.Function(Q)
    zeroQ.x.array[:] = 0
    bc_Q = fem.dirichletbc(zeroQ, deac_dofs, VQ.sub(1))

    # Create external boundary condition for V space
    V_, _ = VQ.sub(0).collapse()
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bndry_dofs = fem.locate_dofs_topological((VQ.sub(0), V_), tdim - 1, boundary_facets)
    zeroV = fem.Function(V_)
    zeroV.x.array[:] = 0
    bc_V = fem.dirichletbc(zeroV, bndry_dofs, VQ.sub(0))
    bcs = [bc_V, bc_Q]

    # Create sparsity pattern and matrix with additional non-zeros on diagonal
    cpp_a = fem.form(a, form_compiler_options=form_compiler_options,
                     jit_options=jit_parameters)
    pattern = fem.create_sparsity_pattern(cpp_a)
    block_size = VQ.dofmap.index_map_bs
    deac_blocks = deac_dofs[0] // block_size
    pattern.insert_diagonal(deac_blocks)
    pattern.finalize()

    # Create matrix based on sparsity pattern
    A = cpp.la.petsc.create_matrix(mesh.comm, pattern)
    A.zeroEntries()
    if not apply_torque:
        A.zeroEntries()
        _petsc.assemble_matrix(A, cpp_a, bcs=bcs)  # type: ignore
        A.assemble()

    # Create inital vector for LHS
    cpp_L = fem.form(L, form_compiler_options=form_compiler_options,
                     jit_options=jit_parameters)
    b = _petsc.create_vector(cpp_L)

    # Create solver
    solver = PETSc.KSP().create(mesh.comm)  # type: ignore
    solver.setOperators(A)
    prefix = "AV_"

    # Give PETSc solver options a unique prefix
    solver_prefix = f"TEAM30_solve_{id(solver)}"
    solver.setOptionsPrefix(solver_prefix)

    # Set PETSc options
    opts = PETSc.Options()  # type: ignore
    opts.prefixPush(solver_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    solver.setFromOptions()
    solver.setOptionsPrefix(prefix)
    solver.setFromOptions()
    # Function for containg the solution
    AzV = fem.Function(VQ)
    Az_out = AzV.sub(0).collapse()

    # Post-processing function for projecting the magnetic field potential
    post_B = MagneticField2D(AzV)

    # Class for computing torque, losses and induced voltage
    derived = DerivedQuantities2D(AzV, AnVn, u, sigma, domains, ct, ft)
    Az_out.name = "Az"
    post_B.B.name = "B"
    # Create output file
    if save_output:
        Az_vtx = VTXWriter(mesh.comm, str(outdir / "Az.bp"), [Az_out])
        B_vtx = VTXWriter(mesh.comm, str(outdir / "B.bp"), [post_B.B])

    # Computations needed for adding addiitonal torque to engine
    x = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    L = 1  # Depth of domain
    I_rotor = mesh.comm.allreduce(fem.assemble_scalar(fem.form(L * r**2 * density * dx(Omega_c))))

    # Post proc variables
    num_steps = int(T / float(dt.value))
    torques = np.zeros(num_steps + 1, dtype=default_scalar_type)
    torques_vol = np.zeros(num_steps + 1, dtype=default_scalar_type)
    times = np.zeros(num_steps + 1, dtype=default_scalar_type)
    omegas = np.zeros(num_steps + 1, dtype=default_scalar_type)
    omegas[0] = omega_u
    pec_tot = np.zeros(num_steps + 1, dtype=default_scalar_type)
    pec_steel = np.zeros(num_steps + 1, dtype=default_scalar_type)
    VA = np.zeros(num_steps + 1, dtype=default_scalar_type)
    VmA = np.zeros(num_steps + 1, dtype=default_scalar_type)
    # Generate initial electric current in copper windings
    t = 0.
    update_current_density(J0z, omega_J, t, ct, currents)

    if MPI.COMM_WORLD.rank == 0 and progress:
        progressbar = tqdm.tqdm(desc="Solving time-dependent problem",
                                total=int(T / float(dt.value)))

    for i in range(num_steps):
        # Update time step and current density
        if MPI.COMM_WORLD.rank == 0 and progress:
            progressbar.update(1)
        t += float(dt.value)
        update_current_density(J0z, omega_J, t, ct, currents)

        # Reassemble LHS
        if apply_torque:
            A.zeroEntries()
            _petsc.assemble_matrix(A, cpp_a, bcs=bcs)  # type: ignore
            A.assemble()

        # Reassemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        _petsc.assemble_vector(b, cpp_L)
        _petsc.apply_lifting(b, [cpp_a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        fem.set_bc(b, bcs)

        # Solve problem
        solver.solve(b, AzV.vector)
        AzV.x.scatter_forward()

        # Compute losses, torque and induced voltage
        loss_al, loss_steel = derived.compute_loss(float(dt.value))
        pec_tot[i + 1] = float(dt.value) * (loss_al + loss_steel)
        pec_steel[i + 1] = float(dt.value) * loss_steel
        torques[i + 1] = derived.torque_surface()
        torques_vol[i + 1] = derived.torque_volume()
        vA, vmA = derived.compute_voltage(float(dt.value))
        VA[i + 1] = vA
        VmA[i + 1] = vmA
        times[i + 1] = t

        # Update previous time step
        AnVn.x.array[:] = AzV.x.array
        AnVn.x.scatter_forward()

        # Update rotational speed depending on torque
        if apply_torque:
            omega.value += float(dt.value) * (derived.torque_volume() - T_ext(t)) / I_rotor
        omegas[i + 1] = float(omega.value)

        # Write solution to file
        if save_output:
            post_B.interpolate()
            Az_out.x.array[:] = AzV.sub(0).collapse().x.array[:]
            Az_vtx.write(t)
            B_vtx.write(t)
    b.destroy()

    if save_output:
        Az_vtx.close()
        B_vtx.close()

    # Compute torque and voltage over last period only
    num_periods = np.round(60 * T)
    last_period = np.flatnonzero(np.logical_and(times > (num_periods - 1) / 60, times < num_periods / 60))
    steps = len(last_period)
    VA_p = VA[last_period]
    VmA_p = VmA[last_period]
    min_T, max_T = min(times[last_period]), max(times[last_period])
    torque_v_p = torques_vol[last_period]
    torque_p = torques[last_period]
    avg_torque = np.sum(torque_p) / steps
    avg_vol_torque = np.sum(torque_v_p) / steps

    pec_tot_p = np.sum(pec_tot[last_period]) / (max_T - min_T)
    pec_steel_p = np.sum(pec_steel[last_period]) / (max_T - min_T)
    RMS_Voltage = np.sqrt(np.dot(VA_p, VA_p) / steps) + np.sqrt(np.dot(VmA_p, VmA_p) / steps)
    # RMS_T = np.sqrt(np.dot(torque_p, torque_p) / steps)
    # RMS_T_vol = np.sqrt(np.dot(torque_v_p, torque_v_p) / steps)
    elements = mesh.topology.index_map(mesh.topology.dim).size_global
    num_dofs = VQ.dofmap.index_map.size_global * VQ.dofmap.index_map_bs
    # Print values for last period
    if mesh.comm.rank == 0:
        print(f"{omega_u}, {avg_torque}, {avg_vol_torque}, {RMS_Voltage}, {pec_tot_p}, {pec_steel_p}, "
              + f"{num_phases}, {steps_per_phase}, {freq}, {degree}, {elements}, {num_dofs}, {single_phase}",
              file=outfile)

    # Plot over all periods
    if mesh.comm.rank == 0 and plot:
        plt.figure()
        plt.plot(times, torques, "--r", label="Surface Torque")
        plt.plot(times, torques_vol, "-b", label="Volume Torque")
        plt.plot(times[last_period], torque_v_p, "--g")
        plt.grid()
        plt.legend()
        plt.savefig(outdir / f"torque_{omega_u}_{ext}.png")
        if apply_torque:
            plt.figure()
            plt.plot(times, omegas, "-ro", label="Angular velocity")
            plt.title(f"Angular velocity {omega_u}")
            plt.grid()
            plt.legend()
            plt.savefig(outdir / f"/omega_{omega_u}_{ext}.png")

        plt.figure()
        plt.plot(times, VA, "-ro", label="Phase A")
        plt.plot(times, VmA, "-bo", label="Phase -A")
        plt.title("Induced Voltage in Phase A and -A")
        plt.grid()
        plt.legend()
        plt.savefig(outdir / f"voltage_{omega_u}_{ext}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scripts to  solve the TEAM 30 problem"
        + " (http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--single', dest='single', action='store_true',
                        help="Solve single phase problem", default=False)
    parser.add_argument('--three', dest='three', action='store_true',
                        help="Solve three phase problem", default=False)
    parser.add_argument('--apply-torque', dest='apply_torque', action='store_true',
                        help="Apply external torque to engine (ignore omega)", default=False)
    parser.add_argument("--num_phases", dest='num_phases', type=int, default=6, help="Number of phases to run")
    parser.add_argument("--omega", dest='omegaU', type=np.float64, default=0, help="Angular speed of rotor [rad/s]")
    parser.add_argument("--degree", dest='degree', type=int, default=1,
                        help="Degree of magnetic vector potential functions space")
    parser.add_argument("--steps", dest='steps', type=int, default=100,
                        help="Time steps per phase of the induction engine")
    parser.add_argument('--plot', dest='plot', action='store_true',
                        help="Plot induced voltage and torque over time", default=False)
    parser.add_argument('--progress', dest='progress', action='store_true',
                        help="Show progress bar", default=False)
    parser.add_argument('--output', dest='output', action='store_true',
                        help="Save output to VTXFiles files", default=False)

    args = parser.parse_args()

    def T_ext(t):
        T = args.num_phases * 1 / 60
        if t > 0.5 * T:
            return 1
        else:
            return 0

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    # FIXME: These complex parameters inspired by the template models does not converge
    # petsc_options = {"ksp_type": "gmres", "pc_type": "bjacobi", "ksp_converged_reason": None,
    #                  "ksp_monitor_true_residual": None, "ksp_gmres_modifiedgramschmidt": None,
    #                  "ksp_diagonal_scale": None, "ksp_gmres_restart": 500,
    #                  "ksp_rtol": 1e-8, "ksp_max_it": 1000, "ksp_view": None, "ksp_monitor": None}

    if args.single:
        outdir = Path(f"TEAM30_{args.omegaU}_single")
        outdir.mkdir(exist_ok=True)

        solve_team30(True, args.num_phases, args.omegaU, args.degree, petsc_options=petsc_options,
                     apply_torque=args.apply_torque, T_ext=T_ext, outdir=outdir, steps_per_phase=args.steps,
                     plot=args.plot, progress=args.progress, save_output=args.output)
    if args.three:
        outdir = Path(f"TEAM30_{args.omegaU}_tree")
        outdir.mkdir(exist_ok=True)
        solve_team30(False, args.num_phases, args.omegaU, args.degree, petsc_options=petsc_options,
                     apply_torque=args.apply_torque, T_ext=T_ext, outdir=outdir, steps_per_phase=args.steps,
                     plot=args.plot, progress=args.progress, save_output=args.output)
