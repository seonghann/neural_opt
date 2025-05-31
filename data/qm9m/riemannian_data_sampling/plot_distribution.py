import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--visualize",
    action="store_true",
    help="visualize scatter plot",
)
parser.add_argument("--sampling_csv", type=str, help="sampling csv filepath")
parser.add_argument("--analyzing_csv", type=str, help="analyzing csv filepath")
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
print(df)
print("Mean error: Cartesian sampling, Riemannian sampling, MMFF")
print("RMSD", df["rmsd"].mean(), df["_rmsd"].mean(), df["__rmsd"].mean())
print("DMAE", df["dmae"].mean(), df["_dmae"].mean(), df["__dmae"].mean())
print("q_norm", df["q_norm"].mean(), df["_q_norm"].mean(), df["__q_norm"].mean())
print("Median error: Cartesian sampling, Riemannian sampling, MMFF")
print("RMSD", df["rmsd"].median(), df["_rmsd"].median(), df["__rmsd"].median())
print("DMAE", df["dmae"].median(), df["_dmae"].median(), df["__dmae"].median())
print("q_norm", df["q_norm"].median(), df["_q_norm"].median(), df["__q_norm"].median())

if not args.visualize:
    exit(1)

## Plot distribution of RMSD, DMAE, and q-norm
import matplotlib.pyplot as plt


def plot_hist_with_gaussian(data, label, color, range, bin_edges=20):
    mu = np.mean(data)
    sigma = np.std(data)
    n, bins, patches = plt.hist(
        data, bins=bin_edges, alpha=0.5, label=label, color=color, range=range
    )
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((bins - mu) / sigma) ** 2)
    plt.plot(bins, y, "--", color=color, label=label)


# Load data
# Plot histogram
bins = 150
figure1 = plt.figure(figsize=(12, 6))
range_max = max(df[["rmsd", "_rmsd", "__rmsd"]].max())
plot_hist_with_gaussian(df["rmsd"], "Cartesian", "blue", (0, range_max), bins)
plot_hist_with_gaussian(df["_rmsd"], "Riemannian", "orange", (0, range_max), bins)
plot_hist_with_gaussian(df["__rmsd"], "MMFF", "green", (0, range_max), bins)
plt.title("RMSD Distribution Comparison")
plt.xlabel("RMSD Value")
plt.ylabel("Number of Samples")
plt.legend(loc="upper right")


figure2 = plt.figure(figsize=(12, 6))
range_max = max(df[["dmae", "_dmae", "__dmae"]].max())
plot_hist_with_gaussian(df["dmae"], "Cartesian", "blue", (0, range_max), bins)
plot_hist_with_gaussian(df["_dmae"], "Riemannian", "orange", (0, range_max), bins)
plot_hist_with_gaussian(df["__dmae"], "MMFF", "green", (0, range_max), bins)
plt.title("DMAE Distribution Comparison")
plt.xlabel("DMAE Value")
plt.ylabel("Number of Samples")
plt.legend(loc="upper right")

figure3 = plt.figure(figsize=(12, 6))
range_max = max(df[["q_norm", "_q_norm", "__q_norm"]].max())
plot_hist_with_gaussian(df["q_norm"], "Cartesian", "blue", (0, range_max), bins)
plot_hist_with_gaussian(df["_q_norm"], "Riemannian", "orange", (0, range_max), bins)
plot_hist_with_gaussian(df["__q_norm"], "MMFF", "green", (0, range_max), bins)
plt.title("q_norm Distribution Comparison")
plt.xlabel("q_norm Value")
plt.ylabel("Number of Samples")
plt.legend(loc="upper right")
plt.show()
