# Copyright (C) 2021 Jørgen S. Dokken and Igor A. Baratta
#
# SPDX-License-Identifier:    MIT

from typing import Dict

import dolfinx.mesh as dmesh
import numpy as np
import ufl
from dolfinx import fem, io
from mpi4py import MPI
from petsc4py import PETSc

from generate_team30_meshes import (mesh_parameters, model_parameters,
                                    surface_map)

__all__ = ["XDMFWrapper", "DerivedQuantities2D", "update_current_density"]


def _cross_2D(A, B):
    """ Compute cross of two 2D vectors """
    return A[0] * B[1] - A[1] * B[0]


class DerivedQuantities2D():
    """
    Collection of methods for computing derived quantities used in the TEAM 30 benchmark including:
    - Torque of rotor (using classical surface calculation and Arkkio's method)
    - Loss in the rotor (steel and aluminium component separately)
    - Induced voltage in one copper winding
    """

    def __init__(self, AzV: fem.Function, AnVn: fem.Function, u, sigma: fem.Function, domains: dict,
                 ct: dmesh.MeshTags, ft: dmesh.MeshTags,
                 form_compiler_parameters: dict = {}, jit_parameters: dict = {}):
        """
        Parameters
        ==========
        AzV
            The mixed function of the magnetic vector potential Az and the Scalar electric potential V

        AnVn
            The mixed function of the magnetic vector potential Az and the Scalar electric potential V
            from the previous time step

        u
            Rotational velocity (Expressed as an ufl expression)

        sigma
            Conductivity

        domains
            dictonary were each key indicates a material in the problem. Each item is a tuple of indices relating to the
            volume tags ct and facet tags

        ct
            Meshtag containing cell indices
        ft
            Meshtag containing facet indices

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
        self.mesh = AzV.function_space.mesh
        self.comm = self.mesh.comm

        # Functions
        Az = AzV[0]
        Azn = AnVn[0]
        self.sigma = sigma

        # Constants
        self.dt = fem.Constant(self.mesh, PETSc.ScalarType(0))
        self.L = 1  # Depth of domain (for torque and voltage calculations)

        # Integration quantities
        self.x = ufl.SpatialCoordinate(self.mesh)
        self.r = ufl.sqrt(self.x[0]**2 + self.x[1]**2)
        self.domains = domains
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=ct)
        self.dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=ft)

        # Derived quantities
        B = ufl.as_vector((Az.dx(1), -Az.dx(0)))  # Electromagnetic field
        self.Bphi = ufl.inner(B, ufl.as_vector((-self.x[1], self.x[0]))) / self.r
        self.Br = ufl.inner(B, self.x) / self.r

        A_res = Az(surface_map["restriction"])
        self.B_2D_rst = ufl.as_vector((A_res.dx(1), -A_res.dx(0)))  # Restricted electromagnetic field
        self.E = -(Az - Azn) / self.dt  # NOTE: as grad(V)=dV/dz=0 in 2D (-ufl.grad(V)) is excluded
        self.Ep = self.E + _cross_2D(u, B)

        # Parameters
        self.fp = form_compiler_parameters
        self.jp = jit_parameters

        self._init_voltage()
        self._init_loss()
        self._init_torque()

    def _init_voltage(self):
        """
        Initializer for computation of induced voltage in for each the copper winding (phase A and -A)
        """
        N = 1  # Number of turns in winding
        if len(self.domains["Cu"]) == 2:
            windings = self.domains["Cu"]
        elif len(self.domains["Cu"]) == 6:
            windings = [self.domains["Cu"][0], self.domains["Cu"][2]]  # NOTE: assumption on ordering of input windings
        else:
            raise RuntimeError("Only single or three phase computations implemented")
        self._C = []
        self._voltage = []
        for winding in windings:
            self._C.append(N * self.L
                           / self.comm.allreduce(fem.assemble_scalar(1 * self.dx(winding)), op=MPI.SUM))
            self._voltage.append(fem.Form(self.E * self.dx(winding), form_compiler_parameters=self.fp,
                                          jit_parameters=self.jp))

    def compute_voltage(self, dt):
        """
        Compute induced voltage between two time steps of distance dt
        """
        self.dt.value = dt
        voltages = [self.comm.allreduce(fem.assemble_scalar(voltage)) for voltage in self._voltage]
        return [voltages[i] * self._C[i] for i in range(len(voltages))]

    def _init_loss(self):
        """
        Compute the Loss in the rotor, total and steel component.
        """
        # Induced voltage
        q = self.sigma * ufl.inner(self.Ep, self.Ep)
        al = q * self.dx(self.domains["Al"])  # Loss in rotor
        steel = q * self.dx(self.domains["Rotor"])  # Loss in only steel
        self._loss_al = fem.Form(al, form_compiler_parameters=self.fp, jit_parameters=self.jp)
        self._loss_steel = fem.Form(steel, form_compiler_parameters=self.fp, jit_parameters=self.jp)

    def compute_loss(self, dt: float) -> float:
        """
        Compute loss between two time steps of distance dt
        """
        self.dt.value = dt
        al = self.comm.allreduce(fem.assemble_scalar(self._loss_al), op=MPI.SUM)
        steel = self.comm.allreduce(fem.assemble_scalar(self._loss_steel), op=MPI.SUM)
        return (al, steel)

    def _init_torque(self):
        """
        Compute torque induced by magnetic field on the TEAM 30 engine using the surface formulation
        (with Maxwell's stress tensor) or Akkio's method.
        """
        mu_0 = model_parameters["mu_0"]

        dS_air = dS_air = self.dS(surface_map["MidAir"])

        # Create variational form for Electromagnetic torque
        dF = 1 / mu_0 * ufl.dot(self.B_2D_rst, self.x / self.r) * self.B_2D_rst
        dF -= 1 / mu_0 * 0.5 * ufl.dot(self.B_2D_rst, self.B_2D_rst) * self.x / self.r
        torque_surface = self.L * _cross_2D(self.x, dF) * dS_air
        # NOTE: Fake integration over dx to orient normals
        torque_surface += fem.Constant(self.mesh, PETSc.ScalarType(0)) * self.dx(0)
        self._surface_torque = fem.Form(
            torque_surface, form_compiler_parameters=self.fp, jit_parameters=self.jp)

        # Volume formulation of torque (Arkkio's method)
        torque_vol = (self.r * self.L / (mu_0 * (mesh_parameters["r3"] - mesh_parameters["r2"])
                                         ) * self.Br * self.Bphi) * self.dx(self.domains["AirGap"])
        self._volume_torque = fem.Form(torque_vol, form_compiler_parameters=self.fp, jit_parameters=self.jp)

    def torque_surface(self) -> float:
        """
        Compute torque using surface integration in air gap and Maxwell's stress tensor
        """
        return self.comm.allreduce(fem.assemble_scalar(self._surface_torque), op=MPI.SUM)

    def torque_volume(self) -> float:
        """
        Compute torque using Arkkio's method, derived on Page 55 of:
        "Analysis of induction motors based on the numerical solution of the magnetic field and circuit equations",
        Antero Arkkio, 1987.
        """
        return self.comm.allreduce(fem.assemble_scalar(self._volume_torque), op=MPI.SUM)


class MagneticFieldProjection2D():
    def __init__(self, AzV: fem.Function,
                 petsc_options: dict = {}, form_compiler_parameters: dict = {}, jit_parameters: dict = {}):
        """
        Class for projecting the magnetic vector potential (here as the first part of the mixed function AvZ)
        to the magnetic flux intensity B=curl(A)

        Parameters
        ==========
        AzV
            The mixed function of the magnetic vector potential Az and the Scalar electric potential V

        petsc_options
            Parameters that is passed to the linear algebra backend PETSc.
            For available choices for the 'petsc_options' kwarg, see the
            `PETSc-documentation <https://www.mcs.anl.gov/petsc/documentation/index.html>`.

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
        degree = AzV.function_space.ufl_element().degree()
        cell = AzV.function_space.ufl_cell()
        mesh = AzV.function_space.mesh

        # Create variational form for electromagnetic field B (post processing)
        el_B = ufl.VectorElement("DG", cell, degree - 1)
        VB = fem.FunctionSpace(mesh, el_B)
        self.B = fem.Function(VB)
        ub = ufl.TrialFunction(VB)
        vb = ufl.TestFunction(VB)
        self.Az = AzV[0]
        a = ufl.inner(ub, vb) * ufl.dx
        B_2D = ufl.as_vector((self.Az.dx(1), -self.Az.dx(0)))
        self._a = fem.Form(a, form_compiler_parameters=form_compiler_parameters,
                           jit_parameters=jit_parameters)
        L = ufl.inner(B_2D, vb) * ufl.dx
        self._L = fem.Form(L, form_compiler_parameters=form_compiler_parameters,
                           jit_parameters=jit_parameters)

        self.A = fem.assemble_matrix(self._a)
        self.A.assemble()
        self.b = fem.create_vector(self._L)

        self.ksp = PETSc.KSP().create(mesh.comm)
        self.ksp.setOperators(self.A)

        # Set PETSc options
        solver_prefix = "dolfinx_solve_{}".format(id(self))
        self.ksp.setOptionsPrefix(solver_prefix)

        prefix = self.ksp.getOptionsPrefix()
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self.ksp.setFromOptions()

    def solve(self):
        """
        Solve projection problem (only reassemble RHS)
        """
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.assemble_vector(self.b, self._L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.ksp.solve(self.b, self.B.vector)
        self.B.x.scatter_forward()


class XDMFWrapper(io.XDMFFile):
    """
    Post processing class adding a sligth overhead to the XDMFFile class
    """

    def __init__(self, comm: MPI.Intracomm, filename: str):
        super(XDMFWrapper, self).__init__(comm, f"{filename}.xdmf", "w")

    def write_function(self, u, t, name: str = None):
        if name is not None:
            u.name = name
        super(XDMFWrapper, self).write_function(u, t)


def update_current_density(J_0: fem.Function, omega: np.float64, t: np.float64, ct: dmesh.MeshTags,
                           currents: Dict[np.int32, Dict[str, np.float64]]):
    """
    Given a DG-0 scalar field J_0, update it to be alpha*J*cos(omega*t + beta)
    in the domains with copper windings
    """
    J_0.x.array[:] = 0
    for domain, values in currents.items():
        _cells = ct.indices[ct.values == domain]
        J_0.x.array[_cells] = np.full(len(_cells), model_parameters["J"] * values["alpha"]
                                      * np.cos(omega * t + values["beta"]))
