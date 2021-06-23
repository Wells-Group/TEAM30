import argparse
import os

import gmsh
import numpy as np
from mpi4py import MPI


__all__ = ["model_parameters", "mesh_parameters", "domain_parameters"]

# Model parameters for the TEAM 3- model
model_parameters = {"mu_0": 1.25663753e-6,  # Relative permability of air [H/m]=[kg m/(s^2 A^2)]
                    "freq": 60,  # Frequency of excitation,
                    "J": 3.1e6 * np.sqrt(2),  # [A/m^2] Current density of copper winding
                    "mu_r": {"Cu": 1, "Stator": 30, "Rotor": 30, "Al": 1, "Air": 1},  # Relative permability
                    "sigma": {"Rotor": 1.6e6, "Al": 3.72e7, "Stator": 0, "Cu": 0, "Air": 0},  # Conductivity
                    "densities": {"Rotor": 7850, "Al": 2700, "Stator": 0, "Air": 0, "Cu": 0}  # [kg/m^3]
                    }

# The different radiuses used in domain specifications
mesh_parameters = {"r1": 0.02, "r2": 0.03, "r3": 0.032, "r4": 0.052, "r5": 0.057}


def domain_parameters(single: bool):
    """
    Domain parameters for the different mesh surfaces
    FIXME: Make this part of mesh generation script
    """
    if single:
        # Single phase model domains:
        # Copper (0 degrees): 1
        # Copper (180 degrees): 2
        # Steel Stator: 3
        # Steel rotor: 4
        # Air: 5, 6, 8, 9, 10
        # Alu rotor: 7
        _domains = {"Cu": (1, 2), "Stator": (3,), "Rotor": (4,),
                    "Al": (7,), "Air": (5, 6, 8, 9, 10)}
        # Currents on the form J_0 = (0,0, alpha*J*cos(omega*t + beta)) in domain i
        _currents = {1: {"alpha": 1, "beta": 0}, 2: {"alpha": -1, "beta": 0}}
        # Domain data for air gap between rotor and windings, MidAir is the tag an internal interface
        # AirGap is the domain markers for the domain
        _torque = {"MidAir": (11,), "restriction": "+", "AirGap": (5, 6)}
    else:
        # Three phase model domains:
        # Copper (0 degrees): 1
        # Copper (60 degrees): 2
        # Copper (120 degrees): 3
        # Copper (180 degrees): 4
        # Copper (240 degrees): 5
        # Copper (300 degrees): 6
        # Steel Stator: 7
        # Steel rotor: 8
        # Air: 9, 10, 12, 13, 14, 15, 16, 17, 18
        # Alu rotor: 11
        _domains = {"Cu": (1, 2, 3, 4, 5, 6), "Stator": (7,), "Rotor": (8,),
                    "Al": (11,), "Air": (9, 10, 12, 13, 14, 15, 16, 17, 18)}
        # Domain data for air gap between rotor and windings, MidAir is the tag an internal interface
        # AirGap is the domain markers for the domain
        _torque = {"MidAir": (19,), "restriction": "+", "AirGap": (9, 10)}
        # Currents on the form J_0 = (0,0, alpha*J*cos(omega*t + beta)) in domain i
        _currents = {1: {"alpha": 1, "beta": 0}, 2: {"alpha": -1, "beta": 2 * np.pi / 3},
                     3: {"alpha": 1, "beta": 4 * np.pi / 3}, 4: {"alpha": -1, "beta": 0},
                     5: {"alpha": 1, "beta": 2 * np.pi / 3}, 6: {"alpha": -1, "beta": 4 * np.pi / 3}}
    return _domains, _torque, _currents


# Add copper areas
def add_copper_segment(start_angle=0):
    """
    Helper function
    Add a 45 degree copper segement, r in (r3, r4) with midline at "start_angle".
    """
    copper_arch_inner = gmsh.model.occ.addCircle(
        0, 0, 0, mesh_parameters["r3"], angle1=start_angle - np.pi / 8, angle2=start_angle + np.pi / 8)
    copper_arch_outer = gmsh.model.occ.addCircle(
        0, 0, 0, mesh_parameters["r4"], angle1=start_angle - np.pi / 8, angle2=start_angle + np.pi / 8)
    gmsh.model.occ.synchronize()
    nodes_inner = gmsh.model.getBoundary([(1, copper_arch_inner)])
    nodes_outer = gmsh.model.getBoundary([(1, copper_arch_outer)])
    l0 = gmsh.model.occ.addLine(nodes_inner[0][1], nodes_outer[0][1])
    l1 = gmsh.model.occ.addLine(nodes_inner[1][1], nodes_outer[1][1])
    c_l = gmsh.model.occ.addCurveLoop([copper_arch_inner, l1, copper_arch_outer, l0])

    copper_segment = gmsh.model.occ.addPlaneSurface([c_l])
    gmsh.model.occ.synchronize()
    return copper_segment


def generate_mesh(filename: str, res: np.float64, L: np.float64, angles):
    gmsh.initialize()
    # Generate three phase induction motor
    rank = MPI.COMM_WORLD.rank
    gdim = 2  # Geometric dimension of the mesh
    if rank == 0:
        center = gmsh.model.occ.addPoint(0, 0, 0)
        air_box = gmsh.model.occ.addRectangle(-L / 2, - L / 2, 0, 2 * L / 2, 2 * L / 2)
        # Define the different circular layers
        strator_steel = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r5"])
        air_2 = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r4"])
        air = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r3"])
        air_mid = gmsh.model.occ.addCircle(0, 0, 0, 0.5 * (mesh_parameters["r2"] + mesh_parameters["r3"]))
        aluminium = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r2"])
        rotor_steel = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r1"])

        # Create out strator steel
        steel_loop = gmsh.model.occ.addCurveLoop([strator_steel])
        air_2_loop = gmsh.model.occ.addCurveLoop([air_2])
        strator_steel = gmsh.model.occ.addPlaneSurface([steel_loop, air_2_loop])

        # Create air layer
        air_loop = gmsh.model.occ.addCurveLoop([air])
        air = gmsh.model.occ.addPlaneSurface([air_2_loop, air_loop])

        domains = [(2, add_copper_segment(angle)) for angle in angles]

        # Add second air segment (in two pieces)
        air_mid_loop = gmsh.model.occ.addCurveLoop([air_mid])
        al_loop = gmsh.model.occ.addCurveLoop([aluminium])
        air_surf1 = gmsh.model.occ.addPlaneSurface([air_loop, air_mid_loop])
        air_surf2 = gmsh.model.occ.addPlaneSurface([air_mid_loop, al_loop])

        # Add aluminium segement
        rotor_loop = gmsh.model.occ.addCurveLoop([rotor_steel])
        aluminium_surf = gmsh.model.occ.addPlaneSurface([al_loop, rotor_loop])

        # Add steel rotor
        rotor_disk = gmsh.model.occ.addPlaneSurface([rotor_loop])
        gmsh.model.occ.synchronize()
        domains.extend([(2, strator_steel), (2, rotor_disk), (2, air),
                       (2, air_surf1), (2, air_surf2), (2, aluminium_surf)])
        surfaces, _ = gmsh.model.occ.fragment([(2, air_box)], domains)
        gmsh.model.occ.synchronize()

        for surface in surfaces:
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]])

        # Mark all interior and exterior boundaries
        lines = gmsh.model.getBoundary(surfaces, combined=False, oriented=False)
        lines_filtered = set([line[1] for line in lines])
        for line in lines_filtered:
            gmsh.model.addPhysicalGroup(1, [line])
        lines = gmsh.model.getBoundary(surfaces, combined=True, oriented=False)
        for line in lines_filtered:
            gmsh.model.addPhysicalGroup(1, [line])

        # Generate mesh

        # gmsh.option.setNumber("Mesh.MeshSizeMin", res)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", res)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [center])
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 25 * res)
        gmsh.model.mesh.field.setNumber(2, "DistMin", mesh_parameters["r4"])
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * mesh_parameters["r5"])
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)

        gmsh.option.setNumber("Mesh.Algorithm", 7)
        gmsh.model.mesh.generate(gdim)
        gmsh.write(f"{filename}.msh")
    gmsh.finalize()


def convert_mesh(filename, cell_type, prune_z=False, ext=None):
    """
    Given the filename of a msh file, read data and convert to XDMF file containing cells of given cell type
    """
    try:
        import meshio
    except ImportError:
        print("Meshio and h5py must be installed to convert meshes."
              + " Please run `pip3 install --no-binary=h5py h5py meshio`")
        exit(1)
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        mesh = meshio.read(f"{filename}.msh")
        cells = mesh.get_cells_type(cell_type)
        data = np.hstack([mesh.cell_data_dict["gmsh:physical"][key]
                          for key in mesh.cell_data_dict["gmsh:physical"].keys() if key == cell_type])
        pts = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=pts, cells={cell_type: cells}, cell_data={
                               "markers": [data]})
        if ext is None:
            ext = ""
        else:
            ext = "_" + ext
        meshio.write(f"{filename}{ext}.xdmf", out_mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMSH scripts to generate induction engines for"
        + "the TEAM 30 problem (http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--res", default=0.001, type=np.float64, dest="res",
                        help="Mesh resolution")
    parser.add_argument("--L", default=1, type=np.float64, dest="L",
                        help="Size of surround box with air")
    _single = parser.add_mutually_exclusive_group(required=False)
    _single.add_argument('--single', dest='single', action='store_true',
                         help="Generate single phase mesh", default=False)
    _three = parser.add_mutually_exclusive_group(required=False)
    _three.add_argument('--three', dest='three', action='store_true',
                        help="Generate three phase mesh", default=False)

    args = parser.parse_args()
    L = args.L
    res = args.res
    single = args.single
    three = args.three

    os.system("mkdir -p meshes")

    if single:
        angles = [0, np.pi]
        fname = "meshes/single_phase"
        generate_mesh(fname, res, L, angles)
        convert_mesh(fname, "triangle", prune_z=True)
        convert_mesh(fname, "line", prune_z=True, ext="facets")

    if three:
        fname = "meshes/three_phase"
        spacing = (np.pi / 4) + (np.pi / 4) / 3
        angles = np.array([spacing * i for i in range(6)])
        generate_mesh(fname, res, L, angles)
        convert_mesh(fname, "triangle", prune_z=True)
        convert_mesh(fname, "line", prune_z=True, ext="facets")
