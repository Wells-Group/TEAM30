import argparse
import os

import gmsh
import numpy as np
from mpi4py import MPI

os.system("mkdir -p meshes")
rank = MPI.COMM_WORLD.rank


# http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf
r1 = 0.02
r2 = 0.03
r_air = 0.031
r3 = 0.032
r4 = 0.052
r5 = 0.057


# Add copper areas
def add_copper_segment(start_angle=0):
    """
        Helper function
        Add a 45 degree copper segement, r in (r3, r4) with midline at "start_angle".
    """
    copper_arch_inner = gmsh.model.occ.addCircle(
        0, 0, 0, r3, angle1=start_angle - np.pi / 8, angle2=start_angle + np.pi / 8)
    copper_arch_outer = gmsh.model.occ.addCircle(
        0, 0, 0, r4, angle1=start_angle - np.pi / 8, angle2=start_angle + np.pi / 8)
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

    gdim = 2  # Geometric dimension of the mesh
    if rank == 0:
        center = gmsh.model.occ.addPoint(0, 0, 0)
        air_box = gmsh.model.occ.addRectangle(-L / 2, - L / 2, 0, 2 * L / 2, 2 * L / 2)
        # Define the different circular layers
        strator_steel = gmsh.model.occ.addCircle(0, 0, 0, r5)
        air_2 = gmsh.model.occ.addCircle(0, 0, 0, r4)
        air = gmsh.model.occ.addCircle(0, 0, 0, r3)
        air_mid = gmsh.model.occ.addCircle(0, 0, 0, r_air)
        aluminium = gmsh.model.occ.addCircle(0, 0, 0, r2)
        rotor_steel = gmsh.model.occ.addCircle(0, 0, 0, r1)

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
        gmsh.model.mesh.field.setNumber(2, "DistMin", r4)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * r5)
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
    parser.add_argument("--res", default=0.0005, type=np.float64, dest="res",
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
