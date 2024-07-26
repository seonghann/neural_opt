"""
correlation을 계산 및 scatter plot visualization
"""

## Read csv file
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha",
    type=float,
    help="coefficient alpha for the Morse scaler",
    # required=True,
    default=None,
)
parser.add_argument(
    "--beta",
    type=float,
    help="coefficient beta for the Morse scaler",
    # required=True,
    default=None,
)
parser.add_argument(
    "--gamma",
    type=float,
    help="coefficient gamma for the Morse scaler",
    # required=True,
    default=0.0,
)
parser.add_argument(
    "--error_type",
    type=str,
    help="structural error type",
    choices=["DMAE", "RMSD", "q_norm"],
    required=True,
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="visualize scatter plot",
)
parser.add_argument("--input_csv", type=str, help="input csv filepath")
args = parser.parse_args()
print(args)



# df = pd.read_csv("./qm9m.csv")
df = pd.read_csv(args.input_csv)
print(df)
y = np.array(df["dE"])

if args.error_type == "q_norm":
    x = np.array(
        df[args.error_type.lower()+f"({args.alpha},{args.beta},{args.gamma})"]
    )
    xlabel = f"{args.error_type}"
    ylabel = "$|\Delta E|$ (kcal/mol)"
else:
    x = np.array(
        df[args.error_type.lower()]
    )
    xlabel = f"{args.error_type} ($\AA$)"
    ylabel = "$|\Delta E|$ (kcal/mol)"



## plot scatter
## refer https://stackoverflow.com/questions/37008112/matplotlib-plotting-histogram-plot-just-above-scatter-plot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats

corr = scipy.stats.pearsonr(x, y)
print(f"Pearson's r: {corr}")
# print(f"alpha={alpha}, beta={beta}: corr={corr[0]}")
# print(f"alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}: corr={corr[0]}")
if not args.visualize:
    exit("Debug: Check only corr")

fig = plt.figure(figsize=(8, 8))
# fig.suptitle('GFN2-xTB vs wB97xd3', fontsize=16)
fig.suptitle('MMFF vs DFT', fontsize=16)
gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)

ax_main.scatter(x,y,marker='.')
ax_main.set(xlabel=xlabel, ylabel=ylabel)

ax_xDist.hist(x,bins=100,align='mid')
ax_xDist.set(ylabel='count')

ax_yDist.hist(y,bins=100,orientation='horizontal',align='mid')
ax_yDist.set(xlabel='count')
plt.ylim(0,)
# plt.xlim(0,)
ax_main.set_xlim(0, )
# plt.show()


if args.error_type == "q_norm":
    re = 1.5

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f'q(r) with re={re}', fontsize=16)

    r = np.linspace(0, 10, 500)

    ratio = r / re
    val1 = np.exp(args.alpha * (1 - ratio))
    val2 = args.beta / ratio

    val_ref = np.exp(1.7 * (1 - ratio))

    plt.plot(r, val1, label=f"exp({args.alpha} * (1 - r/re))")
    plt.plot(r, val2, label=f"{args.beta} * (re / r)")
    plt.plot(r, val_ref, label="exp(1.7 * (1 - r/re))")
    plt.plot(r, val1+val2, label="q(r)")
    plt.axvline(re)
    plt.axvline(0, color="k")
    plt.axhline(0, color="k")
    plt.xlabel("r")
    plt.legend()
    plt.show()
