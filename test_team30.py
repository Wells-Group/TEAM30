from mpi4py import MPI
import pytest
import pandas
from team30_A_phi import solve_team30
from generate_team30_meshes import generate_team30_mesh, convert_mesh
import os
import tqdm
import numpy as np


@pytest.mark.parametrize("single_phase", [True, False])
@pytest.mark.parametrize("degree", [1])
def test_team30(single_phase, degree):
    steps = 5  # Number of steps per phase
    rtol = 0.05  # Tolerance for relative tolerance compared to ref data
    atol = 1e-5
    num_phases = 5

    ext = "single" if single_phase else "three"
    outdir = "test_results"
    os.system(f"mkdir -p {outdir}")
    fname = f"{outdir}/{ext}_phase"

    # Generate mesh
    generate_team30_mesh(fname, single=single_phase, res=0.001, L=1)
    convert_mesh(fname, "triangle", prune_z=True)
    convert_mesh(fname, "line", prune_z=True, ext="facets")

    # Open output file on rank 0
    output = None
    outfile = f"{outdir}/results_{ext}_{degree}.txt"
    if MPI.COMM_WORLD.rank == 0:
        output = open(outfile, "w")
        print("Speed, Torque, Torque_Arkkio, Voltage, Rotor_loss, Steel_loss, num_phases, "
              + "steps_per_phase, freq, degree, num_elements, num_dofs, single_phase", file=output)

    # Solve problem
    df = pandas.read_csv(f"ref_{ext}_phase.txt", delimiter=", ")
    speed = df["Speed"]
    progress = tqdm.tqdm(desc="Parametric sweep", total=len(speed))
    cffi_options = ["-Ofast", "-march=native"]
    jit_parameters = {"cffi_extra_compile_args": cffi_options,
                      "cffi_libraries": ["m"]}
    for omega in speed:
        solve_team30(single_phase, num_phases, omega, degree, outdir=outdir,
                     steps_per_phase=steps, outfile=output, progress=False, mesh_dir=outdir,
                     jit_parameters=jit_parameters)
        progress.update(1)
    if MPI.COMM_WORLD.rank == 0:
        # Close output file
        output.close()

    # Compare results
    df_num = pandas.read_csv(outfile, delimiter=", ")

    # Torque
    trq_ex = df["Torque"]
    trq_vol = df_num["Torque_Arkkio"]
    trq_surf = df_num["Torque"]

    # Voltage
    V_ex = df["Voltage"]
    V_num = df_num["Voltage"]

    # Loss rotor
    L_ex = df["Rotor_loss"]
    L_num = df_num["Rotor_loss"]

    # Loss steel
    Ls_ex = df["Steel_loss"]
    Ls_num = df_num["Steel_loss"]

    def comp_to_print(ex, comp):
        "Helper for printing comparison of output"
        for i, (e, c) in enumerate(zip(ex, comp)):
            close = np.isclose(e, c, atol=atol, rtol=rtol)
            if not close:
                print(f"{i}: {abs(e-c):.3e}<={atol + rtol * abs(e):.3e}")
        print()

    if MPI.COMM_WORLD.rank == 0:
        print("--------Errors-------")
        print("Torque Arkkio")
        comp_to_print(trq_ex, trq_vol)
        print("Torque Surface")
        comp_to_print(trq_ex, trq_surf)
        print("Voltage")
        comp_to_print(V_ex, V_num)
        print("Rotor loss")
        comp_to_print(L_ex, L_num)
        print("Steel loss")
        comp_to_print(Ls_ex, Ls_num)

    assert np.allclose(trq_vol, trq_ex, rtol=rtol, atol=atol)
    assert np.allclose(trq_surf, trq_ex, rtol=rtol, atol=atol)
    assert np.allclose(V_num, V_ex, rtol=rtol, atol=atol)
    assert np.allclose(L_num, L_ex, rtol=rtol, atol=atol)
    assert np.allclose(Ls_num, Ls_ex, rtol=rtol, atol=atol)
