from mpi4py import MPI
import pytest
import pandas
from team30_A_phi import solve_team30
from generate_team30_meshes import generate_team30_mesh, convert_mesh
import os
import tqdm

# @pytest.mark.parametrize("single_phase", [True, False])
# @pytest.mark.parametrize("degree", [1, 2])


@pytest.mark.parametrize("single_phase", [False])
@pytest.mark.parametrize("degree", [1])
def test_team30(single_phase, degree):
    steps = 250  # Number of steps per phase
    tol = 0.05  # Tolerance for relative tolerance compared to reference data
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
    for omega in speed:
        solve_team30(single_phase, num_phases / 60, omega, degree, outdir=outdir,
                     steps_per_phase=steps, outfile=output, progress=False, mesh_dir=outdir)
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

        r_err_arkkio = abs((trq_vol - trq_ex) / trq_ex)
        print("Torque Arkkio", r_err_arkkio)
        assert(max(r_err_arkkio) < tol)

        r_err_trq = abs((trq_surf - trq_ex) / trq_ex)
        print("Torque", r_err_trq)
        assert(max(r_err_trq) < tol)

        # Voltage
        V_ex = df["Voltage"]
        V_num = df_num["Voltage"]
        r_err_V = abs((V_num - V_ex) / V_ex)
        print("Voltage", r_err_V)
        assert(max(r_err_V) < tol)

        # Loss rotor
        L_ex = df["Rotor_loss"]
        L_num = df_num["Rotor_loss"]
        r_err_loss = abs((L_num - L_ex) / L_ex)
        print("Rotor loss", r_err_loss)
        assert(max(r_err_loss) < tol)

        # Loss steel
        Ls_ex = df["Steel_loss"]
        Ls_num = df_num["Steel_loss"]
        r_err_s_loss = abs((Ls_num - Ls_ex) / Ls_ex)
        print("steel loss", r_err_s_loss)
        assert(max(r_err_s_loss) < tol)
