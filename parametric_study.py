import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpi4py import MPI

from team30_A_phi import solve_team30


def create_plots(outdir: str, outfile: str, single: bool, degree: int):
    """
    Create comparsion plots of Torque of numerical data in location
    outdir/outfile with reference data from either single or three phase model
    """

    ext = "single" if single else "three"
    # Read reference and numerical data
    df = pandas.read_csv(f"ref_{ext}_phase.txt", delimiter=", ")
    df_num = pandas.read_csv(f"{outdir}/{outfile}", delimiter=", ")

    # Plot torque
    plt.figure()
    plt.plot(df["Speed"], df["Torque"], "b", label="Reference")
    plt.plot(df_num["Speed"], df_num["Torque_Arkkio"], "-ro", label="Approximate (Arkkio)")
    plt.plot(df_num["Speed"], df_num["Torque"], "--gs", label="Approximate")
    plt.legend()
    plt.title(f"Torque for TEAM 30 {ext} model using elements of order {degree}")
    plt.grid()
    plt.savefig(f"{outdir}/torque_{ext}.png")

    # Plot Induced voltage
    plt.figure()
    plt.plot(df["Speed"], df["Voltage"], label="Reference")
    plt.plot(df_num["Speed"], df_num["Voltage"], label="Numerical")
    plt.tilte(f"RMS torque for Team 30 {ext} model using element of order {degree}")
    plt.grid()
    plt.legend()
    plt.savefig(f"{outdir}/voltage_{ext}.png")

    # Plot rotor loss
    plt.figure()
    plt.plot(df["Speed"], df["Rotor_loss"], label="Reference")
    plt.plot(df_num["Speed"], df_num["Rotor_loss"], label="Numerical")
    plt.tilte(f"Rotor loss for Team 30 {ext} model using element of order {degree}")
    plt.grid()
    plt.legend()
    plt.savefig(f"{outdir}/rotor_loss_{ext}.png")

    # Plot rotor loss steel
    plt.figure()
    plt.plot(df["Speed"], df["Steel_loss"], label="Reference")
    plt.plot(df_num["Speed"], df_num["Steel_loss"], label="Numerical")
    plt.tilte(f"Rotor loss for Team 30 {ext} model using element of order {degree}")
    plt.grid()
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
        print("Speed, Torque, Torque_Arkkio, Voltage, Rotor_loss, Steel_loss", file=output)

    # Run all simulations
    ext = "_single" if args.single else "_three"
    df = pandas.read_csv(f"ref{ext}_phase.txt", delimiter=", ")
    speed = df["Speed"]
    for omega in speed:
        solve_team30(args.single, T, omega, degree, outdir=outdir, steps_per_phase=args.steps, outfile=output)

    # Close output file
    if MPI.COMM_WORLD.rank == 0:
        output.close()

    # Create plots
    create_plots(outdir, outfile, args.single, degree)
