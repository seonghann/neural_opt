"""
Processing incorrect data from QM9M.
For the test set, identify and save the indices of samples with the following two issues into a file (./wrong_samples.pkl):

1. MMFF structure is identical to the DFT structure.
2. Molecule consists of multiple disconnected components (can be detected via SMILES or SMARTS).
"""

import pickle
from ase.io import iread
import numpy as np


with open("data_split.pkl", "rb") as f:
    data_split = pickle.load(f)


rmsd_err_idx = []
smarts_err_idx = []
smarts_warning_idx = []

for i, idx in enumerate(data_split["test_index"]):
    filename = f"./MMFFtoDFT_input/idx{idx}.xyz"
    atoms_DFT, atoms_MMFF = list(iread(filename))

    # assert atoms_DFT.info["smarts"] == atoms_MMFF.info["smarts"]
    if not atoms_DFT.info["smarts"] == atoms_MMFF.info["smarts"]:
        print(f"Warning: SMARTS of DFT != SMARTS of MMFF")
        smarts_warning_idx.append(idx)
    smarts = atoms_DFT.info["smarts"]
    rmsd = np.sqrt(
        np.mean(np.square(atoms_DFT.positions - atoms_MMFF.positions).sum(axis=-1))
    )

    print(f"i={i}, idx={idx}, rmsd={round(rmsd, 3)}")

    # 1. check RMSD
    if rmsd < 1e-9:
        print(f"Error: very small RMSD")
        rmsd_err_idx.append(idx)

    # 2. check SMARTS
    if "." in smarts:
        print(f"Error: '.' in SMARTS!!!")
        smarts_err_idx.append(idx)

print("rmsd_err_idx: \n", rmsd_err_idx)
print("smarts_err_idx: \n", smarts_err_idx)
print("smarts_warning_idx: \n", smarts_warning_idx)

# save indices
err_idx = rmsd_err_idx + smarts_err_idx
err_idx = sorted(err_idx)

save_filename = "wrong_samples.pkl"
with open(save_filename, "wb") as f:
    pickle.dump(err_idx, f)
    print("Save ", save_filename)
