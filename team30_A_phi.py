import argparse

import dolfinx
import os
import dolfinx.io
import numpy as np
import ufl
from mpi4py import MPI

# Model parameters
mu_0 = 1.25663753e-6  # Relative permability of air

# Single phase model domains:
# Copper (0 degrees): 1
# Copper (180 degrees): 2
# Steel strator: 3
# Steel rotor: 4
# Air: 5, 7, 8, 9
# Alu rotor: 6
mu_r_single = {1: 1, 2: 1, 3: 30, 4: 30, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
sigma_single = {4: 1.6e6, 6: 3.72e7, 3: 0}

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
mu_r_three = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 30, 8: 30,
              9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1}
sigma_three = {8: 1.6e6, 10: 3.72e7, 7: 0}


def solve_team30(single_phase: bool):
    """
    Solve the TEAM 30 problem for a single or three phase engine
    """
    if single_phase:
        mu_r_dict = mu_r_single
        sigma_dict = sigma_single
    else:
        mu_r_dict = mu_r_three
        sigma_dict = sigma_three
        return NotImplementedError("Three phase not implemented")

    # Read mesh and cell markers
    fname = "meshes/single_phase" if single_phase else "meshes/three_phase"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    # Create DG 0 function for mu_R and sigma
    DG0 = dolfinx.FunctionSpace(mesh, ("DG", 0))
    mu_R = dolfinx.Function(DG0)
    sigma = dolfinx.Function(DG0)
    with mu_R.vector.localForm() as mu_loc:
        mu_loc.set(0)
        for (marker, value) in mu_r_dict.items():
            _cells = ct.indices[ct.values == marker]
            mu_loc.setValues(_cells, np.full(len(_cells), value))
    with sigma.vector.localForm() as sigma_loc:
        sigma_loc.set(0)
        for (marker, value) in sigma_dict.items():
            _cells = ct.indices[ct.values == marker]
            sigma_loc.setValues(_cells, np.full(len(_cells), value))


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
    else:
        solve_team30(False)
