import argparse

import dolfinx
import os
import dolfinx.io
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

# Model parameters
mu_0 = 1.25663753e-6  # Relative permability of air
_mu_r = {"Cu": 1, "Strator": 30, "Rotor": 30, "Al": 1, "Air": 1}
_sigma = {"Rotor": 1.6e6, "Al": 3.72e7, "Strator": 0, "Cu": 0, "Air": 0}


# Single phase model domains:
# Copper (0 degrees): 1
# Copper (180 degrees): 2
# Steel strator: 3
# Steel rotor: 4
# Air: 5, 7, 8, 9
# Alu rotor: 6
_domains_single = {"Cu": (1, 2), "Strator": (3,), "Rotor": (4,),
                   "Al": (6,), "Air": (5, 7, 8, 9)}
# Currents on the form J_0 = (0,0, alpha*J*cos(omega*t + beta)) in domain i
_currents_single = {1: {"alpha": 1, "beta": 0}, 2: {"alpha": -1, "beta": 0}}

# Three phase model domains:
# Copper (0 degrees): 1
# Copper (60 degrees): 2
# Copper (120 degrees): 3
# Copper (180 degrees): 4
# Copper (240 degrees): 5
# Copper (300 degrees): 6
# Steel strator: 7
# Steel rotor: 8
# Air: 9, 11, 12, 13, 14, 15, 16, 17
# Alu rotor: 10
_domains_three = {"Cu": (1, 2, 3, 4, 5, 6), "Strator": (7,), "Rotor": (8,),
                  "Al": (10,), "Air": (9, 11, 12, 13, 14, 15, 16, 17)}

# Currents on the form J_0 = (0,0, alpha*J*cos(omega*t + beta)) in domain i
_currents_three = {1: {"alpha": 1, "beta": 0}, 2: {"alpha": -1, "beta": 2 * np.pi / 3},
                   3: {"alpha": 1, "beta": 4 * np.pi / 3}, 4: {"alpha": -1, "beta": 0},
                   5: {"alpha": 1, "beta": 2 * np.pi / 3}, 6: {"alpha": -1, "beta": 4 * np.pi / 3}}

J = 3.1e6  # [A/m^2] Current density of copper winding


def update_current_density(J_0, omega, t, ct, currents):
    """
    Given a DG-0 scalar field J_0, update it to be alpha*J*cos(omega*t + beta)
    in the domains with copper windings
    """
    with J_0.vector.localForm() as j0:
        j0.set(0)
        for domain, values in currents.items():
            _cells = ct.indices[ct.values == domain]
            j0.setValues(
                _cells, np.full(len(_cells), values["alpha"] * np.cos(omega * t + values["beta"])))


def solve_team30(single_phase: bool):
    """
    Solve the TEAM 30 problem for a single or three phase engine
    """
    if single_phase:
        domains = _domains_single
        currents = _currents_single
        fname = "meshes/single_phase"
    else:
        domains = _domains_three
        currents = _currents_three
        fname = "meshes/three_phase"
        #raise NotImplementedError("Three phase not implemented")

    # Read mesh and cell markers
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    # Create DG 0 function for mu_R and sigma
    DG0 = dolfinx.FunctionSpace(mesh, ("DG", 0))
    mu_R = dolfinx.Function(DG0)
    sigma = dolfinx.Function(DG0)
    with mu_R.vector.localForm() as mu_loc, sigma.vector.localForm() as sigma_loc:
        for (material, domain) in domains.items():
            for marker in domain:
                _cells = ct.indices[ct.values == marker]
                data = np.empty(len(_cells), dtype=PETSc.ScalarType)
                data[:] = _mu_r[material]
                mu_loc.setValues(_cells, data)
                data[:] = _sigma[material]
                sigma_loc.setValues(_cells, data)

    omega = 1200  # FIXME: Should be user input
    t = 0

    # Generate initial electric current in copper windings
    J_0 = dolfinx.Function(DG0)
    update_current_density(J_0, omega, t, ct, currents)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/sigma.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(sigma)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/mu_R.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(mu_R)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/J0.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(J_0, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scripts to  solve the TEAM 30 problem"
        + " (http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _single = parser.add_mutually_exclusive_group(required=False)
    _single.add_argument('--single', dest='single', action='store_true',
                         help="Generate single phase mesh", default=False)
    _three = parser.add_mutually_exclusive_group(required=False)
    _three.add_argument('--three', dest='three', action='store_true',
                        help="Generate three phase mesh", default=False)

    args = parser.parse_args()
    single = args.single
    three = args.three

    os.system("mkdir -p results")
    if single:
        solve_team30(True)
    if three:
        solve_team30(False)
