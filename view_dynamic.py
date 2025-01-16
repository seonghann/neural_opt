"""
Evaluate accuracy of generated trajectories

(Usage)
    >>> python view_dynamic.py \
    --config_yaml configs/sampling.h.condensed2.yaml \
    --prb_pt save_dynamic.pt
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_yaml", type=str)
parser.add_argument("--prb_pt", type=str)
# parser.add_argument("--align_target", type=str, default="DMAE", choices=["DMAE", "RMSD"])
args = parser.parse_args()
print(args)
###########################################################################

import torch
import numpy as np
from omegaconf import OmegaConf
from dataset.data_module import load_datamodule

import sys
sys.path.append("./unit_test")
from geodesic_h_transform_test import RMSD, DMAE


def remap2symbol(atomic_numbers):
    MAPPING = {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"}
    symbols = [MAPPING[num.item()] for num in atomic_numbers]
    return symbols


def write_xyz(filename, atoms, coords, mode="w", comment="\n"):
    """Write atom names and coordinate data to XYZ file

    Args:
        filename:   Name of xyz data file
        atoms:      Iterable of atom names
        coords:     Coordinates, must be of shape nimages*natoms*3
    """
    natoms = len(atoms)
    with open(filename, mode) as f:
        for i, X in enumerate(np.atleast_3d(coords)):
            f.write("%d\n" % natoms)
            # f.write("Frame %d\n" % i)
            f.write(comment)
            for a, Xa in zip(atoms, X):
                f.write(" {:3} {:21.12f} {:21.12f} {:21.12f}\n".format(a, *Xa))


if __name__ == "__main__":
    ###########################################################################
    ## Load reference pos

    datamodule = load_datamodule(OmegaConf.load(args.config_yaml))
    for batch in datamodule.test_dataloader():
        print(f"Debug: batch.idx={batch.idx}")

    pos_ref_list = []
    for batch in datamodule.test_dataloader():
        natoms = batch.batch.bincount()
        pos_ref = batch.pos[:, 0, :].split(natoms.tolist())
        pos_ref_list.extend(pos_ref)
    ###########################################################################


    ###########################################################################
    ## Load prb pos
    data = torch.load(args.prb_pt, map_location="cpu")
    print(f"Load {args.prb_pt}")

    pos_traj_list = []
    atom_type_list = []
    smarts_list = []
    for batch in data:
        natoms = batch.batch.bincount()
        pos_traj = batch.pos_traj  # shape: (ntraj,)---(natoms, 3)
        pos_traj = torch.stack(pos_traj)  # shape: (ntraj, natoms, 3)
        pos_traj = pos_traj.transpose(0, 1)  # shape: (natoms, ntraj, 3)
        pos_traj = pos_traj.split(natoms.tolist())

        pos_traj_list.extend(pos_traj)

        atom_type = batch.atom_type.split(natoms.tolist())
        atom_type_list.extend(atom_type)

        smarts = batch.smarts
        smarts_list.extend(smarts)
    ###########################################################################


    ###########################################################################
    ## Calculate DMAE, RMSD

    for i, (pos_ref, pos_gen) in enumerate(zip(pos_ref_list, pos_traj_list)):
        # atom_type = atom_type_list[i]
        # smarts = smarts_list[i]

        pos = pos_gen.transpose(0, 1)  # shape: (ntraj, natoms, 3)

        ## Calculate metrics
        rmsd = [round(RMSD(pos_ref, x).item(), 3) for x in pos]
        dmae = [round(DMAE(pos_ref, x).item(), 3) for x in pos]
        # q_norm = [_q.norm(dim=-1).item() for _q in target_q.split(nedges.tolist())]
        # print(f"rxn_idx={rxn_idx[i]}")
        print(f"i={i}")
        print(f"rmsd (traj): {rmsd}")
        print(f"dmae (traj): {dmae}")
        print(f"rmsd (start -> final): {rmsd[0]} -> {rmsd[-1]}")
        print(f"dmae (start -> final): {dmae[0]} -> {dmae[-1]}")

        # comment = f"rxn_idx={rxn_idx[i]}\n"
        # # write_xyz(filename, symbols[i], pos.cpu().numpy(), comment=comment)
        # write_xyz(filename, symbols[i], pos.cpu().numpy(), comment=comment, mode="a")
        # print(f"Write {filename}")

    ###########################################################################
#     exit("DEBUG")
# 
# rxn_idx = torch.load("data/processed/test_proc.pt")[0]["rxn_idx"]
# data = data[0]
# 
# 
# natoms = data.batch.bincount()
# full_edge, _, _ = data.full_edge(upper_triangle=True)
# nedges = data.batch.index_select(0, full_edge[0]).bincount()
# 
# # pos_traj = data.pos_traj  # shape: (ntraj,)---(natoms, 3)
# # pos_traj = torch.stack(pos_traj)  # shape: (ntraj, natoms, 3)
# # pos_traj = pos_traj.transpose(0, 1)  # shape: (natoms, ntraj, 3)
# # 
# # atom_type = data.atom_type.split(natoms.tolist())
# # symbols = [remap2symbol(sym) for sym in atom_type]
# # pos_traj = pos_traj.split(natoms.tolist())
# 
# 
# 
# filename = "traj.xyz"
# with open(filename, "w") as f: f.write("")
# for i in range(len(pos_traj)):
# 
#     pos = pos_traj[i].transpose(0, 1) 
# 
#     ## Calculate metrics
#     rmsd = [round(RMSD(pos_ref[i], x).item(), 3) for x in pos]
#     dmae = [round(DMAE(pos_ref[i], x).item(), 3) for x in pos]
#     # q_norm = [_q.norm(dim=-1).item() for _q in target_q.split(nedges.tolist())]
#     print(f"rxn_idx={rxn_idx[i]}")
#     print(f"rmsd (traj): {rmsd}")
#     print(f"dmae (traj): {dmae}")
#     print(f"rmsd (start -> final): {rmsd[0]} -> {rmsd[-1]}")
#     print(f"dmae (start -> final): {dmae[0]} -> {dmae[-1]}")
# 
#     comment = f"rxn_idx={rxn_idx[i]}\n"
#     # write_xyz(filename, symbols[i], pos.cpu().numpy(), comment=comment)
#     write_xyz(filename, symbols[i], pos.cpu().numpy(), comment=comment, mode="a")
#     print(f"Write {filename}")
