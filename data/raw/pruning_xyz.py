import glob
import os.path as osp
# read xyz file

xyz_files = glob.glob("source/*.xyz")

for xyz_file in xyz_files:
    with open(xyz_file, "r") as f:
        lines = f.readlines()

    save_file = osp.join("GeodesictoDFT", osp.basename(xyz_file))
    with open(save_file, "w") as f:
        for line in lines:
            # if line has "rxn_smarts=~~" then capsulate the ~~ with ""
            if "rxn_smarts=" in line:
                line = line.replace("rxn_smarts=", "rxn_smarts=\"")
                line = line.replace("\n", "\"\n")
            f.write(line)
