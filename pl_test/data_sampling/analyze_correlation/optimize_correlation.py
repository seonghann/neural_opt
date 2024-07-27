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
    default=None,
)
args = parser.parse_args()
print(args)
#####################################################################################



def com_to_xyz():
    pass






import sys
sys.path.append("/home/share/DATA/NeuralOpt/utils")
from calcDMAEfromXYZ import calcDMAEfromXYZ
from calcRMSDfromXYZ import calcRMSDfromXYZ
from calcDefinedErrorfromXYZ import calcMorseErrorfromXYZ
from calcGeodesicLength import calcGeodesicLength, calcCoulombNormfromXYZ


alpha, beta = 1.7, 0.01 # 0.6377882783025347
alpha, beta = 1.4, 0.6  # best: 0.6484680544492742
# alpha, beta = 1.5, 0.6
alpha, beta = 1.7, 0.01 # 0.6377882783025347
# alpha, beta = args.alpha, args.beta
alpha, beta, gamma = args.alpha, args.beta, args.gamma


print(f"Debug: alpha={alpha}, beta={beta}")


data_dir = "/home/share/DATA/jhwoo_TS2/qm_results"
# prb_dir = "./orca_sp_log_gfn2"
# prb_dir = "./orca_sp_log_pm7"
# ref_dir = "./orca_sp_log_wb97xd3"
prb_dir = f"{data_dir}/orca_sp_log_pm7"
ref_dir = f"{data_dir}/orca_sp_log_wb97xd3"

# prb_key = "gfn2"
prb_key = "pm7"
ref_key = "wb97xd3"

# prb_dir = "./orca_sp_log_lst/"
# prb_key = "lst"


## set structural error type
# error_type = ["RMSD", "D-MAE", "D-RMSD", "D-MAPE", "D-RMSPD", "MorseError-MAE", "MorseError-RMSD"][-2]
error_type = ["RMSD", "D-MAE", "D-RMSD", "D-MAPE", "D-RMSPD", "MorseError-MAE", "MorseError-norm"][-1]
# error_type = ["RMSD", "D-MAE", "D-RMSD", "D-MAPE", "D-RMSPD", "MorseError-MAE", "MorseError-norm"][1]
# error_type = ["RMSD", "D-MAE", "D-RMSD", "D-MAPE", "D-RMSPD", "MorseError-MAE", "MorseError-norm"][0]
# error_type = "MorseError-geodesic"
# error_type = "Coulomb-norm"


error_type = "RMSD"
error_type = "D-MAE"
# error_type = "MorseError-geodesic"
error_type = "MorseError-norm"

weighted_type = [None, "mass", "charge", "valence charge"][0]

# error_type = "D-MAE"; weighted_type = "charge"  # 

# weighted_type = "valence charge2"

print(f"* error_type: {error_type}")
print(f"* weighted_type: {weighted_type}")


# load test indices from 'data_split.pkl'
import pickle
with open("/home/share/DATA/NeuralOpt/data/data_split.pkl", "rb") as f:
    data_split = pickle.load(f)
# indices = data_split["test"]
indices = data_split["valid_index"]

if prb_key == "pm7":
    import pandas as pd
    # filename = "./final.json"
    # filename = "./final.20240103.json"
    filename = f"{data_dir}/final.20240103.json"
    print(f"load {filename}")
    df = pd.read_json(path_or_buf=filename, orient="records")
    pm7_indices = df["rxn index"].values.tolist()
    _indices = [i for i in indices if i in pm7_indices]
    indices = _indices



## Get structural errors
geom_err_list = []
for i in indices:
    ii = str(i).zfill(6)
    prb_xyz = f"{prb_dir}/rxn{ii}/ts{ii}.{prb_key}.xyz"
    ref_xyz = f"{ref_dir}/rxn{ii}/ts{ii}.{ref_key}.xyz"

    try:
        if error_type == "RMSD":
            if weighted_type is None:
                mass_weighted = False
            elif weighted_type == "mass":
                mass_weighted = True
            else:
                raise NotImplementedError
            _, err = calcRMSDfromXYZ(ref_xyz, prb_xyz, mass_weighted=mass_weighted)
            err = float(err)
        elif error_type == "D-MAE":
            err, _ = calcDMAEfromXYZ(ref_xyz, prb_xyz, weighted_type=weighted_type)
        elif error_type == "D-RMSD":
            err, _ = calcDMAEfromXYZ(ref_xyz, prb_xyz, weighted_type=weighted_type, rmsd=True)
        elif error_type == "D-MAPE":
            _, err = calcDMAEfromXYZ(ref_xyz, prb_xyz, weighted_type=weighted_type)
        elif error_type == "D-RMSPD":
            _, err = calcDMAEfromXYZ(ref_xyz, prb_xyz, weighted_type=weighted_type, rmsd=True)
        elif error_type == "Coulomb-norm":
            err = calcCoulombNormfromXYZ(ref_xyz, prb_xyz)
        elif error_type == "MorseError-MAE":
            err = calcMorseErrorfromXYZ(ref_xyz, prb_xyz, norm=False)
            err = float(err)
        # elif error_type == "MorseError-RMSD":
        elif error_type == "MorseError-norm":
            # err = calcMorseErrorfromXYZ(ref_xyz, prb_xyz, norm=True)
            err = calcMorseErrorfromXYZ(ref_xyz, prb_xyz, norm=True, alpha=alpha, beta=beta, gamma=gamma)
            err = float(err)
        elif error_type == "MorseError-geodesic":
            geodesic_xyz = f"./alpha_{alpha}_beta_{beta}/rxn{ii}-wb97xd3-geodesic.xyz"
            err = calcGeodesicLength(geodesic_xyz, alpha=alpha, beta=beta)
            # print(f"Debug: geodesic_xyz={geodesic_xyz}")
            # print(f"Debug: err={err}")
        else:
            raise NotImplementedError
        geom_err_list.append(err)
    # except:
    except Exception as e:
        print(f"Error in {ii}")
        print("Exception:\n", e)
        raise Exception


## Get energy errors
def read_energy_from_orca_log(filename):
    key = "FINAL SINGLE POINT ENERGY"

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if key in line:
                energy = float(line.split()[-1])
    return energy

dE_list = []
for i in indices:
    ii = str(i).zfill(6)
    prb_log = f"{prb_dir}/rxn{ii}/ts{ii}.{prb_key}.out"
    ref_log = f"{ref_dir}/rxn{ii}/ts{ii}.{ref_key}.out"
    prb_e = read_energy_from_orca_log(prb_log)
    ref_e = read_energy_from_orca_log(ref_log)
    hartree2kcal = 627.509
    de = abs(prb_e - ref_e) * hartree2kcal  # absolute energy error
    # de = (prb_e - ref_e) * hartree2kcal  # signed error
    dE_list.append(de)

# print(f"dE_list=\n{dE_list}")
print(f"Debug: geom_err_list = {geom_err_list}")
print(f"{error_type} (ang): {sum(geom_err_list)/len(geom_err_list):0.3f} ({min(geom_err_list):0.3f} ~ {max(geom_err_list):0.3f})")
print(f"dE (kcal/mol): {sum(dE_list)/len(dE_list):0.3f} ({min(dE_list):0.3f} ~ {max(dE_list):0.3f})")
print(f"dE median (kcal/mol): {sorted(dE_list)[len(dE_list)//2]:0.3f}")



## plot scatter
## refer https://stackoverflow.com/questions/37008112/matplotlib-plotting-histogram-plot-just-above-scatter-plot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

x = np.array(geom_err_list)
if weighted_type:
    xlabel = f"{weighted_type}-weighted {error_type} ($\AA$)"
else:
    xlabel = f"{error_type} ($\AA$)"

y = np.array(dE_list)
ylabel = "$|\Delta E|$ (kcal/mol)"


#############################################################################################################
## Screening outlier samples
# screening_type = "RMSD"; screening_thr = 0.5  # reference: GeoDiff, COV threshold for QM9
screening_type = "DMAE"; screening_thr = 0.2  # reference: TSDiff, COV threshold for Grambow
# screening_type = "DMAE"; screening_thr = 0.1  # reference: TSDiff, COV threshold for Grambow

dmae_err_list = []
for i in indices:
    ii = str(i).zfill(6)
    prb_xyz = f"{prb_dir}/rxn{ii}/ts{ii}.{prb_key}.xyz"
    ref_xyz = f"{ref_dir}/rxn{ii}/ts{ii}.{ref_key}.xyz"

#     if weighted_type is None:
#         mass_weighted = False
#     elif weighted_type == "mass":
#         mass_weighted = True
#     else:
#         raise NotImplementedError

    if screening_type == "DMAE":
        err, _ = calcDMAEfromXYZ(ref_xyz, prb_xyz) #weighted_type=weighted_type)
    elif screening_type == "RMSD":
        _, err = calcRMSDfromXYZ(ref_xyz, prb_xyz) #, mass_weighted=mass_weighted)
        err = float(err)
    else:
        raise NotImplementedError
    dmae_err_list.append(err)
dmae_err_list = np.array(dmae_err_list)

## remove y values of larger than 0.5
indices, = np.where(dmae_err_list < screening_thr)
x = x[indices]
y = y[indices]

print(f"screening samples with {screening_type} > {screening_thr}")
print(f"# of samples: {len(dmae_err_list)} -> {len(x)}")
#############################################################################################################



import scipy.stats
corr = scipy.stats.pearsonr(x, y)
print(f"Pearson's r: {corr}")
# print(f"alpha={alpha}, beta={beta}: corr={corr[0]}")
print(f"alpha={alpha}, beta={beta}, gamma={gamma}: corr={corr[0]}")
# exit("Debug: Check only corr")

fig = plt.figure(figsize=(8, 8))
# fig.suptitle('GFN2-xTB vs wB97xd3', fontsize=16)
fig.suptitle('PM7 vs wB97xd3', fontsize=16)
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

plt.show()
