# Copyright (C) 2021 Jørgen S. Dokken and Igor A. Baratta
#
# SPDX-License-Identifier:    MIT

import argparse
from pathlib import Path
from typing import Dict, Union

import dolfinx
import gmsh
import numpy as np
from mpi4py import MPI

__all__ = ["model_parameters", "mesh_parameters", "domain_parameters", "surface_map", "generate_team30_mesh"]

# Model parameters for the TEAM 3- model
model_parameters = {
    "mu_0": 1.25663753e-6,  # Relative permability of air [H/m]=[kg m/(s^2 A^2)]
    "freq": 60,  # Frequency of excitation,
    "J": 3.1e6 * np.sqrt(2),  # [A/m^2] Current density of copper winding
    "mu_r": {"Cu": 1, "Stator": 30, "Rotor": 30, "Al": 1, "Air": 1, "AirGap": 1},  # Relative permability
    "sigma": {"Rotor": 1.6e6, "Al": 3.72e7, "Stator": 0, "Cu": 0, "Air": 0, "AirGap": 0},  # Conductivity
    "densities": {"Rotor": 7850, "Al": 2700, "Stator": 0, "Air": 0, "Cu": 0, "AirGap": 0}  # [kg/m^3]
}
# Marker for facets, and restriction to use in surface integral of airgap
surface_map: Dict[str, Union[int, str]] = {"Exterior": 1, "MidAir": 2, "restriction": "+"}

# Copper wires is ordered in counter clock-wise order from angle = 0, 2*np.pi/num_segments...
_domain_map_single: Dict[str, tuple[int, ...]] = {"Cu": (7, 8), "Stator": (
    6, ), "Rotor": (5, ), "Al": (4,), "AirGap": (2, 3), "Air": (1,)}
_domain_map_three: Dict[str, tuple[int, ...]] = {"Cu": (7, 8, 9, 10, 11, 12), "Stator": (6, ), "Rotor": (5, ),
                                                 "Al": (4,), "AirGap": (2, 3), "Air": (1,)}

# Currents mapping to the domain marker sof the copper
_currents_single: Dict[int, Dict[str, int]] = {7: {"alpha": 1, "beta": 0}, 8: {"alpha": -1, "beta": 0}}
_currents_three: Dict[int, Dict[str, float]] = {7: {"alpha": 1, "beta": 0}, 8: {"alpha": -1, "beta": 2 * np.pi / 3},
                                                9: {"alpha": 1, "beta": 4 * np.pi / 3}, 10: {"alpha": -1, "beta": 0},
                                                11: {"alpha": 1, "beta": 2 * np.pi / 3},
                                                12: {"alpha": -1, "beta": 4 * np.pi / 3}}


# The different radiuses used in domain specifications
mesh_parameters: Dict[str, float] = {"r1": 0.02, "r2": 0.03, "r3": 0.032, "r4": 0.052, "r5": 0.057}


def domain_parameters(single_phase: bool):
    """
    Get domain markers and current specifications for either the single phase or three phase engine
    """
    if single_phase:
        return _domain_map_single, _currents_single
    else:
        return _domain_map_three, _currents_three


def _add_copper_segment(start_angle=0):
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


def generate_team30_mesh(filename: Path, single: bool, res: np.float64, L: np.float64):
    """
    Generate the single phase or three phase team 30 model, with a given minimal resolution, encapsilated in
    a LxL box.
    All domains are marked, while only the exterior facets and the mid air gap facets are marked
    """
    if single:
        angles = np.array([0, np.pi], dtype=np.float64)
        domain_map = _domain_map_single
    else:
        spacing = (np.pi / 4) + (np.pi / 4) / 3
        angles = np.asarray([i * spacing for i in range(6)], dtype=np.float64)
        domain_map = _domain_map_three
    assert len(domain_map["Cu"]) == len(angles)

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

        domains = [(2, _add_copper_segment(angle)) for angle in angles]

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

        # Helpers for assigning domain markers based on area of domain
        rs = [mesh_parameters[f"r{i}"] for i in range(1, 6)]
        r_mid = 0.5 * (rs[1] + rs[2])  # Radius for middle of air gap
        area_helper = (rs[3]**2 - rs[2]**2) * np.pi  # Helper function to determine area of copper and air
        frac_cu = 45 / 360
        frac_air = (360 - len(angles) * 45) / (360 * len(angles))
        _area_to_domain_map: Dict[float, str] = {rs[0]**2 * np.pi: "Rotor",
                                                 (rs[1]**2 - rs[0]**2) * np.pi: "Al",
                                                 (r_mid**2 - rs[1]**2) * np.pi: "AirGap1",
                                                 (rs[2]**2 - r_mid**2) * np.pi: "AirGap0",
                                                 area_helper * frac_cu: "Cu",
                                                 area_helper * frac_air: "Air",
                                                 (rs[4]**2 - rs[3]**2) * np.pi: "Stator",
                                                 float(L**2 - np.pi * rs[4]**2): "Air"}

        # Helper for assigning current wire tag to copper windings
        cu_points = np.asarray([[np.cos(angle), np.sin(angle)] for angle in angles])

        # Assign physical surfaces based on the mass of the segment
        # For copper wires order them counter clockwise
        other_air_markers = []
        for surface in surfaces:
            mass = gmsh.model.occ.get_mass(surface[0], surface[1])
            found_domain = False
            for _mass in _area_to_domain_map.keys():
                if np.isclose(mass, _mass):
                    domain_type = _area_to_domain_map[_mass]
                    if domain_type == "Cu":
                        com = gmsh.model.occ.get_center_of_mass(surface[0], surface[1])
                        point = np.array([com[0], com[1]]) / np.sqrt(com[0]**2 + com[1]**2)
                        index = np.flatnonzero(np.isclose(cu_points, point).all(axis=1))[0]
                        marker = domain_map[domain_type][index]
                        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], marker)
                        found_domain = True
                        break
                    elif domain_type == "AirGap0":
                        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], domain_map["AirGap"][0])
                        found_domain = True
                        break
                    elif domain_type == "AirGap1":
                        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], domain_map["AirGap"][1])
                        found_domain = True
                        break

                    elif domain_type == "Air":
                        other_air_markers.append(surface[1])
                        found_domain = True
                        break
                    else:
                        marker = domain_map[domain_type][0]
                        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], marker)
                        found_domain = True
                        break
            if not found_domain:
                raise RuntimeError(f"Domain with mass {mass} and id {surface[1]} not matching expected domains.")

        # Assign air domains
        gmsh.model.addPhysicalGroup(surface[0], other_air_markers, domain_map["Air"][0])

        # Mark air gap boundary and exterior box
        lines = gmsh.model.getBoundary(surfaces, combined=False, oriented=False)
        lines_filtered = set([line[1] for line in lines])
        air_gap_circumference = 2 * r_mid * np.pi
        for line in lines_filtered:
            length = gmsh.model.occ.get_mass(1, line)
            if np.isclose(length - air_gap_circumference, 0):
                gmsh.model.addPhysicalGroup(1, [line], surface_map["MidAir"])
        lines = gmsh.model.getBoundary(surfaces, combined=True, oriented=False)
        gmsh.model.addPhysicalGroup(1, [line[1] for line in lines], surface_map["Exterior"])

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

        # gmsh.option.setNumber("Mesh.Algorithm", 7)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.mesh.generate(gdim)
        gmsh.write(f"{filename}.msh")
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


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

    folder = Path("meshes")
    folder.mkdir(exist_ok=True)

    if single:
        fname = folder / "single_phase"
        generate_team30_mesh(fname, True, res, L)
        mesh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
            str(fname.with_suffix(".msh")), MPI.COMM_WORLD, 0, gdim=2)
        cell_markers.name = "Cell_markers"
        facet_markers.name = "Facet_markers"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname.with_suffix(".xdmf"), "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(cell_markers, mesh.geometry)
            xdmf.write_meshtags(facet_markers, mesh.geometry)

    if three:
        fname = folder / "three_phase"
        generate_team30_mesh(fname, False, res, L)
        mesh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(
            str(fname.with_suffix(".msh")), MPI.COMM_WORLD, 0, gdim=2)
        cell_markers.name = "Cell_markers"
        facet_markers.name = "Facet_markers"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname.with_suffix(".xdmf"), "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(cell_markers, mesh.geometry)
            xdmf.write_meshtags(facet_markers, mesh.geometry)
