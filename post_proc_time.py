import numpy as np
import matplotlib.pyplot as plt
from team30_A_phi import solve_team30
from mpi4py import MPI
import argparse
import os

# Reference data from http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf
_speed_3 = np.array([0, 200, 400, 600, 800, 1000, 1200])
_torque_3 = np.array([3.825857, 6.505013, -3.89264, -5.75939, -3.59076, -2.70051, -2.24996])

_speed_1 = np.array([0, 39.79351, 79.58701, 119.3805, 159.174, 198.9675, 238.761, 278.5546, 318.3481, 358.1416])
_torque_1 = np.array([0, 0.052766, 0.096143, 0.14305, 0.19957, 0.2754, 0.367972, 0.442137, 0.375496, -0.0707])

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
    args = parser.parse_args()

    T = args.T
    degree = args.degree
    outdir = args.outdir
    if outdir is None:
        outdir = "results"
    os.system(f"mkdir -p {outdir}")

    if args.single:
        speed = _speed_1
        torque = _torque_1
        ext = "_single"
    else:
        speed = _speed_3
        torque = _torque_3
        ext = "_three"

    torques_vol = np.zeros(len(speed))
    torques_surf = np.zeros(len(speed))
    for i, omega in enumerate(speed):
        torques_vol[i], torques_surf[i] = solve_team30(
            args.single, T, omega, degree, outdir=outdir, steps_per_phase=args.steps)

    if MPI.COMM_WORLD.rank == 0:
        plt.figure()
        plt.plot(speed, torque, "b", label="Reference")
        plt.plot(speed, torques_vol, "-ro", label="Approximate (Arkkio)")
        plt.plot(speed, torques_surf, "--gs", label="Approximate")
        print(torque / torques_vol)
        plt.legend()
        plt.title(f"Torque for TEAM 30 model at T={T:.3f} using elements of order {degree}")
        plt.grid()
        plt.savefig(f"results/torque_comp{ext}.png")
