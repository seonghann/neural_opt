import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--visualize",
    action="store_true",
    help="visualize scatter plot",
)
parser.add_argument("--sampling_csv", type=str, help="sampling csv filepath")
parser.add_argument("--analyzing_csv", type=str, help="analyzing csv filepath")
parser.add_argument("--fontsize", type=int, default=15, help="base font size for plots (default: 15)")
# parser.add_argument("--save_figures", action="store_true", help="save figures to files")
parser.add_argument(
    "--save",
    type=str,
    default=None,
    help="Save plot to file (e.g., plot.png, plot.svg)",
)
args = parser.parse_args()
print(args)
###############################################

# import sys
import numpy as np
import pandas as pd

pd.options.display.max_rows = None


df = pd.read_csv(args.sampling_csv)
print(df)
print(f"len(df)={len(df)}")
print("         perr_straight       perr_projected")
print(f"Mean  : {df['perr_straight'].mean()}, {df['perr_projected'].mean()}")
print(f"Median: {df['perr_straight'].median()}, {df['perr_projected'].median()}")



df = pd.read_csv(args.analyzing_csv)

required_columns = ["delta_E_xT", "delta_E_xt2", "delta_E_xt"]
plot_energy = all(col in df.columns for col in required_columns)

print(df)
print("Mean error: Cartesian sampling, Riemannian sampling, Non-equilibrium")
print("RMSD", df["rmsd"].mean(), df["_rmsd"].mean(), df["__rmsd"].mean())
print("DMAE", df["dmae"].mean(), df["_dmae"].mean(), df["__dmae"].mean())
print("q_norm", df["q_norm"].mean(), df["_q_norm"].mean(), df["__q_norm"].mean())
if plot_energy:
    print("dE", df["delta_E_xt"].mean(), df["delta_E_xt2"].mean(), df["delta_E_xT"].mean())

print("Median error: Cartesian sampling, Riemannian sampling, Non-equilibrium")
print("RMSD", df["rmsd"].median(), df["_rmsd"].median(), df["__rmsd"].median())
print("DMAE", df["dmae"].median(), df["_dmae"].median(), df["__dmae"].median())
print("q_norm", df["q_norm"].median(), df["_q_norm"].median(), df["__q_norm"].median())
if plot_energy:
    print("dE", df["delta_E_xt"].median(), df["delta_E_xt2"].median(), df["delta_E_xT"].median())

if not args.visualize:
    exit(1)

## Plot distribution of RMSD, DMAE, and q-norm
import matplotlib.pyplot as plt

# Set font sizes based on base fontsize
base_fontsize = args.fontsize
title_fontsize = base_fontsize + 2
label_fontsize = base_fontsize
tick_fontsize = base_fontsize - 2
legend_fontsize = base_fontsize - 1

# Set default font sizes for matplotlib
plt.rcParams.update({
    'font.size': base_fontsize,
    'axes.titlesize': title_fontsize,
    'axes.labelsize': label_fontsize,
    'xtick.labelsize': tick_fontsize,
    'ytick.labelsize': tick_fontsize,
    'legend.fontsize': legend_fontsize,
})


def plot_hist_with_gaussian(data, label, color, range, bin_edges=20, gaussian=False):
    mu = np.mean(data)
    sigma = np.std(data)
    n, bins, patches = plt.hist(
        data,
        bins=bin_edges,
        alpha=0.3,
        label=label,
        color=color,
        range=range,
        # edgecolor=color,
        edgecolor="dark"+color,
        # edgecolor="black",
        # histtype="step",
        histtype="stepfilled",
        linewidth=2,
    )
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((bins - mu) / sigma) ** 2)
    if gaussian:
        plt.plot(bins, y, "--", color=color, label=label)


# Load data
# Plot histogram
bins = 50
figsize = (6, 4)

figure1 = plt.figure(figsize=figsize)
range_max = max(df[["rmsd", "_rmsd", "__rmsd"]].max())
plot_hist_with_gaussian(df["__rmsd"], "Non-equilibrium", "orange", (0, range_max), bins)
plot_hist_with_gaussian(df["_rmsd"], "Riemannian", "blue", (0, range_max), bins)
plot_hist_with_gaussian(df["rmsd"], "Euclidean", "green", (0, range_max), bins)
plt.xlabel("RMSD ($\AA$)", fontsize=label_fontsize)
plt.ylabel("Number of samples", fontsize=label_fontsize)
plt.xlim(0, range_max)
plt.legend(loc="upper right", fontsize=legend_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.tight_layout()


figure2 = plt.figure(figsize=figsize)
range_max = max(df[["dmae", "_dmae", "__dmae"]].max())
plot_hist_with_gaussian(df["__dmae"], "Non-equilibrium", "orange", (0, range_max), bins)
plot_hist_with_gaussian(df["_dmae"], "Riemannian", "blue", (0, range_max), bins)
plot_hist_with_gaussian(df["dmae"], "Euclidean", "green", (0, range_max), bins)
plt.xlabel("D-MAE ($\AA$)", fontsize=label_fontsize)
plt.ylabel("Number of samples", fontsize=label_fontsize)
plt.xlim(0, range_max)
plt.legend(loc="upper right", fontsize=legend_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.tight_layout()

figure3 = plt.figure(figsize=figsize)
# range_max = max(df[["q_norm", "_q_norm", "__q_norm"]].max())  # too log
range_max = 3.0
plot_hist_with_gaussian(df["__q_norm"], "Non-equilibrium", "orange", (0, range_max), bins)
plot_hist_with_gaussian(df["_q_norm"], "Riemannian", "blue", (0, range_max), bins)
plot_hist_with_gaussian(df["q_norm"], "Euclidean", "green", (0, range_max), bins)
plt.xlabel("$\|q_0 - q_t\|_2$", fontsize=label_fontsize)
plt.ylabel("Number of samples", fontsize=label_fontsize)
plt.xlim(0, range_max)
plt.legend(loc="upper right", fontsize=legend_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.tight_layout()


figure4 = None
if plot_energy:
    figure4 = plt.figure(figsize=figsize)
    # range_max = max(df[["delta_E_xt", "delta_E_xt2", "delta_E_xT"]].max())  # too log
    range_max = 300
    plot_hist_with_gaussian(df["delta_E_xT"], "Non-equilibrium", "orange", (0, range_max), bins)
    plot_hist_with_gaussian(df["delta_E_xt2"], "Riemannian", "blue", (0, range_max), bins)
    plot_hist_with_gaussian(df["delta_E_xt"], "Euclidean", "green", (0, range_max), bins)
    plt.xlabel("$|\Delta E|$ (kcal/mol)", fontsize=label_fontsize)
    plt.ylabel("Number of samples", fontsize=label_fontsize)
    plt.xlim(0, range_max)
    plt.legend(loc="upper right", fontsize=legend_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.tight_layout()

plt.show()


if args.save is not None:
    if not args.visualize:
        print("To save the distribution plots, you can use --visualize")
    figure1.savefig(f"{args.save}.rmsd.svg", dpi=300, bbox_inches='tight')
    figure2.savefig(f"{args.save}.dmae.svg", dpi=300, bbox_inches='tight')
    figure3.savefig(f"{args.save}.q_norm.svg", dpi=300, bbox_inches='tight')
    if figure4 is not None:
        figure4.savefig(f"{args.save}.delta_E.svg", dpi=300, bbox_inches='tight')
    print(f"Save figures to files: {args.save}.rmsd.svg, {args.save}.dmae.svg, {args.save}.q_norm.svg")
