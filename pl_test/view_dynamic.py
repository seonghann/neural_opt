import torch
import numpy as np
from omegaconf import OmegaConf
from dataset.data_module import GrambowDataModule

import sys
sys.path.append("./unit_test")
from geodesic_h_transform_test import RMSD, DMAE


def remap2symbol(atomic_numbers):
    MAPPING = {0: "H", 1: "C", 2: "N", 3: "O"}
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


# datamodule = GrambowDataModule(OmegaConf.load("configs/sampling.schnet.yaml"))
# datamodule = GrambowDataModule(OmegaConf.load("configs/sampling.schnet.fast.yaml"))
datamodule = GrambowDataModule(OmegaConf.load("configs/sampling.cfm.condensed2.yaml"))
for batch in datamodule.test_dataloader():
    print(batch.rxn_idx)


save_dynamic = "./save_dynamic.pt"
data = torch.load(save_dynamic)
# data = torch.load(save_dynamic, map_location=torch.device('cpu'))
print(f"Load {save_dynamic}")


rxn_idx = torch.load("data/processed/test_proc.pt")[0]["rxn_idx"]
data = data[0]


natoms = data.batch.bincount()
full_edge, _, _ = data.full_edge(upper_triangle=True)
nedges = data.batch.index_select(0, full_edge[0]).bincount()

pos_traj = data.pos_traj  # shape: (ntraj,)---(natoms, 3)
pos_traj = torch.stack(pos_traj)  # shape: (ntraj, natoms, 3)
pos_traj = pos_traj.transpose(0, 1)  # shape: (natoms, ntraj, 3)

atom_type = data.atom_type.split(natoms.tolist())
symbols = [remap2symbol(sym) for sym in atom_type]
pos_traj = pos_traj.split(natoms.tolist())


pos_ref = None
for batch in datamodule.test_dataloader():
    pos_ref = batch.pos[:, 0, :]
pos_ref = pos_ref.split(natoms.tolist())


filename = "traj.xyz"
with open(filename, "w") as f: f.write("")
for i in range(len(pos_traj)):

    pos = pos_traj[i].transpose(0, 1) 

    ## Calculate metrics
    rmsd = [round(RMSD(pos_ref[i], x).item(), 3) for x in pos]
    dmae = [round(DMAE(pos_ref[i], x).item(), 3) for x in pos]
    # q_norm = [_q.norm(dim=-1).item() for _q in target_q.split(nedges.tolist())]
    print(f"rxn_idx={rxn_idx[i]}")
    print(f"rmsd (traj): {rmsd}")
    print(f"dmae (traj): {dmae}")
    print(f"rmsd (start -> final): {rmsd[0]} -> {rmsd[-1]}")
    print(f"dmae (start -> final): {dmae[0]} -> {dmae[-1]}")

    comment = f"rxn_idx={rxn_idx[i]}\n"
    # write_xyz(filename, symbols[i], pos.cpu().numpy(), comment=comment)
    write_xyz(filename, symbols[i], pos.cpu().numpy(), comment=comment, mode="a")
    print(f"Write {filename}")
