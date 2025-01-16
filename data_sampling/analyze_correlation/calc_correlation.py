"""
Plot scatter plot and calculate Pearson's correlation.

Examples:
    python calc_correlation.py --error_type geodesic_length --alpha 1.7 --beta 0.01 --input_csv ./qm9m.csv --visualize --loglog
    python calc_correlation.py --error_type RMSD --input_csv ./qm9m.csv --visualize --loglog
    python calc_correlation.py --error_type DMAE --input_csv ./qm9m.csv --visualize --loglog
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scipy.stats


parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha",
    type=float,
    help="coefficient alpha for the Morse scaler",
    default=None,
)
parser.add_argument(
    "--beta",
    type=float,
    help="coefficient beta for the Morse scaler",
    default=None,
)
parser.add_argument(
    "--gamma",
    type=float,
    help="coefficient gamma for the Morse scaler",
    default=0.0,
)
parser.add_argument(
    "--error_type",
    type=str,
    help="structural error type",
    choices=["DMAE", "RMSD", "q_norm", "geodesic_length"],
    required=True,
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="visualize scatter plot",
)
parser.add_argument(
    "--loglog",
    action="store_true",
    help="set log-log scale for both axes",
)
parser.add_argument("--input_csv", type=str, help="input csv filepath (e.g., qm9m.csv)")
args = parser.parse_args()
print(args)


df = pd.read_csv(args.input_csv)
df = df.iloc[-1000:]  # NOTE: 1000 samples
print(df)
# df = df[df["dE"] < 10.0]; print("filtering dE > 10.0")  # NOTE: test removing outliers
y = np.array(df["dE"])

if args.error_type in ["q_norm", "geodesic_length"]:
    x = np.array(
        df[args.error_type.lower() + f"({args.alpha},{args.beta},{args.gamma})"]
    )
    xlabel = f"{args.error_type}"
    if args.error_type == "geodesic_length":
        xlabel = "geodesic distance"
    ylabel = "$|\Delta E|$ (kcal/mol)"
else:
    x = np.array(df[args.error_type.lower()])
    xlabel = f"{args.error_type} ($\AA$)"
    if args.error_type == "DMAE":
        xlabel = f"D-MAE ($\AA$)"
    ylabel = "$|\Delta E|$ (kcal/mol)"


if args.loglog:
    corr = scipy.stats.pearsonr(np.log(x), np.log(y))
else:
    corr = scipy.stats.pearsonr(x, y)
print(f"Pearson's r: {corr}")
if not args.visualize:
    exit("Debug: Check only corr")


fontsize = 24
fig = plt.figure(figsize=(6.5, 5.5))
plt.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
plt.scatter(x, y, marker=".")
plt.xlabel(xlabel, fontsize=fontsize)
plt.ylabel(ylabel, fontsize=fontsize)

if args.loglog:
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(2 * 1e-1, None)
    if args.error_type == "RMSD":
        plt.xlim(6 * 1e-3, 2.0)
    elif args.error_type == "DMAE":
        plt.xlim(6 * 1e-3, 6 * 1e-1)
    elif args.error_type in ["q_norm", "geodesic_length"]:
        plt.xlim(2 * 1e-2, 1.3)
    else:
        raise ValueError()
else:
    # set x & y range
    if args.error_type == "RMSD":
        plt.xlim(0, 0.5)
        plt.ylim(0, 20.0)
    elif args.error_type == "DMAE":
        plt.xlim(0, 0.2)
        plt.ylim(0, 20.0)
    elif args.error_type in ["q_norm", "geodesic_length"]:
        plt.xlim(0, 0.3)
        plt.ylim(0, 20.0)
    else:
        raise ValueError()

plt.tick_params(axis="both", which="major", labelsize=fontsize)
plt.tight_layout()
save_filename = f"./{args.error_type}.pdf"
plt.savefig(save_filename)
print(f"save {save_filename}")
plt.show()


# if args.error_type in ["q_norm", "geodesic_length"]:
#     re = 1.5
#
#     fig = plt.figure(figsize=(8, 8))
#     fig.suptitle(f"q(r) with re={re}", fontsize=16)
#
#     r = np.linspace(0, 10, 500)
#
#     ratio = r / re
#     val1 = np.exp(args.alpha * (1 - ratio))
#     val2 = args.beta / ratio
#
#     val_ref = np.exp(1.7 * (1 - ratio))
#
#     plt.plot(r, val1, label=f"exp({args.alpha} * (1 - r/re))")
#     plt.plot(r, val2, label=f"{args.beta} * (re / r)")
#     plt.plot(r, val_ref, label="exp(1.7 * (1 - r/re))")
#     plt.plot(r, val1 + val2, label="q(r)")
#     plt.axvline(re)
#     plt.axvline(0, color="k")
#     plt.axhline(0, color="k")
#     plt.xlabel("r")
#     plt.legend()
#     plt.show()
