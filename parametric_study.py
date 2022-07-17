# Copyright (C) 2021-2022 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import argparse
import os
from io import TextIOWrapper
import matplotlib.pyplot as plt
import numpy as np
import pandas
import tqdm
from mpi4py import MPI

from team30_A_phi import solve_team30


def create_caption(dtype, app, ext):
    out_data = {}
    for var in ["degree", "num_elements"]:
        out_data[var] = app[var][0]
        assert np.allclose(app[var], app[var][0])
    degree = out_data["degree"]
    nel = out_data["num_elements"]
    caption = f"{dtype} for {ext} phase engine at various speeds. CG function space of degree"
    caption += f" {degree} on mesh with {nel} elements"
    return caption


def to_latex(data: str, single: bool = False):
    ext = "single" if single else "three"

    # Read reference
    df = pandas.read_csv(f"ref_{ext}_phase.txt", delimiter=", ")
    # Read approximate
    app = pandas.read_csv(data, delimiter=", ")

    f_format = "{:0.4f}".format
    i_format = "{:.2f}".format if single else "{: d}".format
    e_format = "{:0.4e}".format

    # Create torque table
    df2 = pandas.DataFrame(df, columns=["Speed", "Torque"])
    df2 = df2.rename(columns={"Speed": r"$\omega$", "Torque": "TEAM 30"})
    df2["Torque"] = app["Torque"]
    df2["Relative Err"] = abs((df2["TEAM 30"] - df2["Torque"]) / df2["TEAM 30"])
    df2["Torque (Arkkio)"] = app["Torque_Arkkio"]
    df2["Relative Err (Arkkio)"] = abs((df2["TEAM 30"] - df2["Torque (Arkkio)"]) / df2["TEAM 30"])

    caption = create_caption("Torque", app, ext)
    latex_table = df2.to_latex(columns=[r"$\omega$", "TEAM 30", "Torque", "Torque (Arkkio)", "Relative Err",
                                        "Relative Err (Arkkio)"],
                               index=False, escape=False,
                               formatters={r"$\omega$": i_format, "Torque": f_format,
                                           "TEAM 30": f_format, "Relative Err": e_format, "Torque (Arkkio)": f_format,
                                           "Relative Err (Arkkio)": e_format},
                               position="!ht", column_format="cccccc",
                               caption=caption,
                               label=f"tab:torque:{ext}")
    print(latex_table)
    print()
    f_format = "{:0.2f}".format

    # Create Loss table
    df2 = pandas.DataFrame(df, columns=["Speed", "Rotor_loss", "Steel_loss"])
    df2 = df2.rename(columns={"Speed": r"$\omega$", "Rotor_loss": "TEAM 30 (rotor)",
                     "Steel_loss": "TEAM 30 (steel)"})
    df2["Loss (rotor)"] = app["Rotor_loss"]
    df2["Relative Err (rotor)"] = abs((df2["TEAM 30 (rotor)"] - df2["Loss (rotor)"]) / df2["TEAM 30 (rotor)"])
    df2["Loss (steel)"] = app["Steel_loss"]
    df2["Relative Err (steel)"] = abs((df2["TEAM 30 (steel)"] - df2["Loss (steel)"]) / df2["TEAM 30 (steel)"])

    caption = create_caption("Loss", app, ext)
    latex_table = df2.to_latex(columns=[r"$\omega$", "TEAM 30 (rotor)", "Loss (rotor)", "Relative Err (rotor)",
                                        "TEAM 30 (steel)", "Loss (steel)", "Relative Err (steel)"
                                        ],
                               index=False, escape=False,
                               formatters={r"$\omega$": i_format, "Loss (rotor)": f_format,
                                           "TEAM 30 (rotor)": f_format, "Relative Err (rotor)": e_format,
                                           "Loss (steel)": f_format,
                                           "TEAM 30 (steel)": f_format, "Relative Err (steel)": e_format},
                               position="!ht", column_format="ccccccc",
                               caption=caption,
                               label=f"tab:loss:{ext}")
    print(latex_table)

    # Induced voltage table
    df2 = pandas.DataFrame(df, columns=["Speed", "Voltage"])
    df2 = df2.rename(columns={"Speed": r"$\omega$", "Voltage": "TEAM 30"})
    df2["Induced Voltage"] = app["Voltage"]
    df2["Relative Error"] = abs((df2["TEAM 30"] - df2["Induced Voltage"]) / df2["TEAM 30"])

    f_format = "{:0.4f}".format
    caption = create_caption("Induced voltage (Phase A)", app, ext)
    latex_table = df2.to_latex(columns=[r"$\omega$", "TEAM 30", "Induced Voltage", "Relative Error"],
                               index=False, escape=False,
                               formatters={r"$\omega$": i_format, "TEAM 30": f_format,
                                           "Induced Voltage": f_format, "Relative Error": e_format},
                               position="!ht", column_format="ccccccc",
                               caption=caption,
                               label=f"tab:voltage:{ext}")
    print(latex_table)


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

    num_steps = df_num["steps_per_phase"][0]
    assert(np.allclose(df_num["steps_per_phase"], num_steps))

    freq = df_num["freq"][0]
    assert(np.allclose(df_num["freq"], freq))

    ext = "single" if phase[0] else "three"
    df = pandas.read_csv(f"ref_{ext}_phase.txt", delimiter=", ")

    # Derive temporal quantities
    T_min = (num_steps - 1) * 1 / freq
    T_max = (num_steps) * 1 / freq
    dt = 1 / freq * 1 / num_steps

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

    txt = r"Torque averaged over the period $t\in[$" + f"{T_min:.3e}, {T_max:.3e}" + r"] with " + f"dt={dt:.3e}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(f"{outdir}/torque_{ext}_{num_steps}.png")

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
        f"{T_min:.3e}, {T_max:.3e}" + r"] with " + f"dt={dt:.3e}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(f"{outdir}/voltage_{ext}_{num_steps}.png")

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
        f"{T_min:.3e}, {T_max:.3e}" + r"] with " + f"dt={dt:.3e}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(f"{outdir}/rotor_loss_{ext}_{num_steps}.png")

    # Plot rotor loss steel
    plt.figure(figsize=(12, 8))
    plt.plot(df_num["Speed"], df_num["Steel_loss"], "-ro", label="Simulation")
    plt.plot(df["Speed"], df["Steel_loss"], "bX", label="Reference")
    plt.title(f"TEAM 30 {ext} phase using {num_elements} elements of order {degree}")
    plt.xlabel("Rotational speed")
    plt.ylabel("Steel Loss (W/m)")
    plt.grid()
    txt = r"Power dissipation in the steel rotor over the period $t\in[$" + \
        f"{T_min:.3e}, {T_max:.3e}" + r"] with " + f"dt={dt:.3e}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.legend()
    plt.savefig(f"{outdir}/steel_loss_{ext}_{num_steps}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scripts for creating comparisons for the Team30 problem"
        + " (http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _single = parser.add_mutually_exclusive_group(required=False)
    _single.add_argument('--single', dest='single', action='store_true',
                         help="Solve single phase problem", default=False)
    parser.add_argument("--num_phases", dest='num_phases', type=int, default=6,
                        help="Number of phases")
    parser.add_argument("--steps", dest='steps', type=int, default=100,
                        help="Time steps per phase of the induction engine")
    parser.add_argument("--degree", dest='degree', type=int, default=1,
                        help="Degree of magnetic vector potential functions space")
    parser.add_argument("--outdir", dest='outdir', type=str, default=None,
                        help="Directory for results")
    parser.add_argument("--outfile", dest='outfile', type=str, default="results.txt",
                        help="File to write derived quantities to")
    args = parser.parse_args()

    num_phases = args.num_phases
    degree = args.degree
    outdir = args.outdir
    outfile = args.outfile

    if outdir is None:
        outdir = "results"
    os.system(f"mkdir -p {outdir}")

    # Open output file on rank 0
    output: TextIOWrapper
    if MPI.COMM_WORLD.rank == 0:
        output = open(f"{outdir}/{outfile}", "w")
        print("Speed, Torque, Torque_Arkkio, Voltage, Rotor_loss, Steel_loss, num_phases, "
              + "steps_per_phase, freq, degree, num_elements, num_dofs, single_phase", file=output)
    else:
        output = None  # type: ignore

    # Run all simulations
    ext = "_single" if args.single else "_three"
    df = pandas.read_csv(f"ref{ext}_phase.txt", delimiter=", ")
    speed = df["Speed"]
    progress = tqdm.tqdm(desc="Parametric sweep", total=len(speed))
    for omega in speed:
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
        if MPI.COMM_WORLD.rank == 0:
            print(f"Running for speed {omega}", flush=True)
        solve_team30(args.single, num_phases, omega, degree, outdir=outdir,
                     steps_per_phase=args.steps, outfile=output, progress=True)
        progress.update(1)
    # Close output file
    if MPI.COMM_WORLD.rank == 0:
        output.close()
    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0:
        # Print to Latex tables
        to_latex(f"{outdir}/{outfile}", args.single)

        # Create plots
        create_plots(outdir, outfile)
