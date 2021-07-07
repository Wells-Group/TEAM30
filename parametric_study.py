import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import tqdm
from mpi4py import MPI

from team30_A_phi import solve_team30


def create_plots(outdir: str, outfile: str):
    """
    Create comparsion plots of Torque of numerical data in location
    outdir/outfile with reference data from either single or three phase model
    """

    # Read reference and numerical data
    df_num = pandas.read_csv(f"{outdir}/{outfile}", delimiter=", ")
    degrees = df_num["degree"]
    degree = degrees[0]
    phase = df_num["single_phase"]
    assert(np.allclose(phase, phase[0]))
    assert(np.allclose(degrees, degree))
    elements = df_num["num_elements"]
    num_elements = elements[0]
    assert(np.allclose(elements, num_elements))

    dts = df_num["dt"]
    dt = dts[0]
    assert(np.allclose(dts, dt))

    Ts = df_num["T"]
    T = Ts[0]
    assert(np.allclose(Ts, T))

    ext = "single" if phase[0] else "three"
    df = pandas.read_csv(f"ref_{ext}_phase.txt", delimiter=", ")

    # Plot torque
    plt.figure(figsize=(12, 8))
    plt.plot(df_num["Speed"], df_num["Torque_Arkkio"], "-ro", label="Simulation (Arkkio)")
    plt.plot(df_num["Speed"], df_num["Torque"], "--gs", label="Simulation")
    plt.plot(df["Speed"], df["Torque"], "bX", label="Reference")

    plt.legend()
    plt.title(f"TEAM 30 {ext} phase using {num_elements} elements of order {degree}")
    plt.grid()
    plt.xlabel("Rotational speed")
    plt.ylabel("Torque (N/m)")
    txt = r"Torque averaged over the period $t\in[$" + f"{T - 1 / 60:.3e}, {T:.3e}" + r"] with " + f"dt={dt:.3e}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(f"{outdir}/torque_{ext}.png")

    # Plot Induced voltage
    plt.figure(figsize=(12, 8))
    plt.plot(df_num["Speed"], df_num["Voltage"], "-ro", label="Simulation")
    plt.plot(df["Speed"], df["Voltage"], "bX", label="Reference")
    plt.title(f"TEAM 30 {ext} phase using {num_elements} elements of order {degree}")
    plt.grid()
    plt.legend()
    plt.xlabel("Rotational speed")
    plt.ylabel("Voltage/turn (V/m/turn)")
    txt = r"RMS Voltage for Phase A and -A over the period $t\in[$" + \
        f"{T - 1 / 60:.3e}, {T:.3e}" + r"] with " + f"dt={dt:.3e}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(f"{outdir}/voltage_{ext}.png")

    # Plot rotor loss
    plt.figure(figsize=(12, 8))
    plt.plot(df_num["Speed"], df_num["Rotor_loss"], "-ro", label="Simulation")
    plt.plot(df["Speed"], df["Rotor_loss"], "bX", label="Reference")
    plt.title(f"TEAM 30 {ext} phase using {num_elements} elements of order {degree}")
    plt.grid()
    plt.legend()
    plt.xlabel("Rotational speed")
    plt.ylabel("Rotor Loss (W/m)")
    txt = r"Power dissipation in the rotor over the period $t\in[$" + \
        f"{T - 1 / 60:.3e}, {T:.3e}" + r"] with " + f"dt={dt:.3e}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(f"{outdir}/rotor_loss_{ext}.png")

    # Plot rotor loss steel
    plt.figure(figsize=(12, 8))
    plt.plot(df_num["Speed"], df_num["Steel_loss"], "-ro", label="Simulation")
    plt.plot(df["Speed"], df["Steel_loss"], "bX", label="Reference")
    plt.title(f"TEAM 30 {ext} phase using {num_elements} elements of order {degree}")
    plt.xlabel("Rotational speed")
    plt.ylabel("Steel Loss (W/m)")
    plt.grid()
    txt = r"Power dissipation in the steel rotor over the period $t\in[$" + \
        f"{T - 1 / 60:.3e}, {T:.3e}" + r"] with " + f"dt={dt:.3e}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.legend()
    plt.savefig(f"{outdir}/steel_loss_{ext}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scripts for creating comparisons for the Team30 problem"
        + " (http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _single = parser.add_mutually_exclusive_group(required=False)
    _single.add_argument('--single', dest='single', action='store_true',
                         help="Solve single phase problem", default=False)
    parser.add_argument("--T", dest='T', type=np.float64, default=0.13333333, help="End time of simulation")
    parser.add_argument("--degree", dest='degree', type=int, default=1,
                        help="Degree of magnetic vector potential functions space")
    parser.add_argument("--outdir", dest='outdir', type=str, default=None,
                        help="Directory for results")
    parser.add_argument("--steps", dest='steps', type=int, default=100,
                        help="Time steps per phase of the induction engine")
    parser.add_argument("--outfile", dest='outfile', type=str, default="results.txt",
                        help="File to write derived quantities to")
    args = parser.parse_args()

    T = args.T
    degree = args.degree
    outdir = args.outdir
    outfile = args.outfile

    if outdir is None:
        outdir = "results"
    os.system(f"mkdir -p {outdir}")

    # Open output file on rank 0
    output = None
    if MPI.COMM_WORLD.rank == 0:
        output = open(f"{outdir}/{outfile}", "w")
        print("Speed, Torque, Torque_Arkkio, Voltage, Rotor_loss, Steel_loss, dt, T, degree, "
              + "num_elements, num_dofs, single_phase", file=output)

    # Run all simulations
    ext = "_single" if args.single else "_three"
    df = pandas.read_csv(f"ref{ext}_phase.txt", delimiter=", ")
    speed = df["Speed"]
    progress = tqdm.tqdm(desc="Parametric sweep", total=len(speed))
    for omega in speed:
        solve_team30(args.single, T, omega, degree, outdir=outdir,
                     steps_per_phase=args.steps, outfile=output, progress=False)
        progress.update(1)
    # Close output file
    if MPI.COMM_WORLD.rank == 0:
        output.close()

    # Create plots
    create_plots(outdir, outfile)
