"""
Evaluate accuracy of generated geometries

(Usage)
    >>> python eval_accuracy.py reproduce/wb97xd3/samples_all.pkl
    >>> python eval_accuracy.py \
    --config_yaml configs/sampling.h.condensed2.yaml \
    --prb_pt save_dynamic.pt
    --align_target DMAE
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_yaml", type=str)
parser.add_argument("--prb_pt", type=str)
parser.add_argument("--align_target", type=str, default="DMAE", choices=["DMAE", "RMSD"])
args = parser.parse_args()
print(args)
###########################################################################

import torch
import numpy as np
import pickle
from ase import Atoms
from ase.build.rotate import minimize_rotation_and_translation
from math import sqrt


def remap2atomic_numbers(atomic_numbers):
    MAPPING = {0: 1, 1: 6, 2: 7, 3: 8}
    symbols = [MAPPING[num.item()] for num in atomic_numbers]
    return symbols


def calc_RMSD(pos1, pos2):
    return sqrt(((pos1 - pos2) ** 2).sum() / len(pos1))


def get_min_rmsd_match(matches, ref_atoms, prb_atoms):
    rmsds = []
    ref_atoms = ref_atoms.copy()
    prb_atoms = prb_atoms.copy()

    ref_pos = ref_atoms.positions.copy()
    prb_pos = prb_atoms.positions.copy()

    for match in matches:
        prb_atoms.positions = prb_pos[list(match)].copy()
        minimize_rotation_and_translation(ref_atoms, prb_atoms)
        match_pos = prb_atoms.positions
        rmsd = calc_RMSD(ref_pos, match_pos)
        rmsds.append(rmsd)
    return list(matches[rmsds.index(min(rmsds))]), rmsds[0], min(rmsds)


# # def calc_RMSD2(mol1: ase.Atoms, mol2: ase.Atoms, mass_weighted: bool = False):
def calc_RMSD2(mol1: Atoms, mol2: Atoms, mass_weighted: bool = False):
    """minimize_rotation_and_translation() + calc_RMSD()"""
    minimize_rotation_and_translation(mol1, mol2)
    p1, p2 = mol1.positions, mol2.positions
    if mass_weighted:
        mass = mol1.get_masses()
        rmsd = np.sqrt(np.mean(np.sum(mass.reshape(-1, 1) * (p1 - p2) ** 2, axis=1)))
    else:
        rmsd = np.sqrt(np.mean(np.sum((p1 - p2) ** 2, axis=1)))
    return rmsd

def get_substruct_matches(smarts):
    from rdkit import Chem
    smarts_r, smarts_p = smarts.split(">>")
    mol_r = Chem.MolFromSmarts(smarts_r)
    mol_p = Chem.MolFromSmarts(smarts_p)

    matches_r = list(mol_r.GetSubstructMatches(mol_r, uniquify=False))
    map_r = np.array([atom.GetAtomMapNum() for atom in mol_r.GetAtoms()]) - 1
    map_r_inv = np.argsort(map_r)
    for i in range(len(matches_r)):
        matches_r[i] = tuple(map_r[np.array(matches_r[i])[map_r_inv]])

    matches_p = list(mol_p.GetSubstructMatches(mol_p, uniquify=False))
    map_p = np.array([atom.GetAtomMapNum() for atom in mol_p.GetAtoms()]) - 1
    map_p_inv = np.argsort(map_p)
    for i in range(len(matches_p)):
        matches_p[i] = tuple(map_p[np.array(matches_p[i])[map_p_inv]])

    matches = set(matches_r) & set(matches_p)
    # matches = set(matches_r) | set(matches_p); print(f"Debug: change & -> |")
    matches = list(matches)
    matches.sort()
    return matches


def align_jhwoo(
    ref_atoms,
    prb_atoms,
    # ref_smarts,
    # prb_smarts,
    smiles,
    silent = False,
    target = "RMSD",
):
    # Refer to: /home/share/DATA/NeuralOpt/utils/alignXYZ.py
    import sys
    sys.path.append("/home/share/DATA/NeuralOpt/utils")
    # from alignXYZ import get_substruct_matches, get_min_dmae_match
    from alignXYZ import get_min_dmae_match

    # print(f"Debug: smiles={smiles}")
    matches = get_substruct_matches(smiles)
    # print(f"Debug: matches={matches}")

    ref_pos = ref_atoms.positions
    prb_pos = prb_atoms.positions

    if target == "DMAE":
        match, init_target, final_target = get_min_dmae_match(matches, ref_pos, prb_pos)
    elif target == "RMSD":
        match, init_target, final_target = get_min_rmsd_match(matches, ref_atoms, prb_atoms)
    else:
        raise NotImplementedError(f"target={target}, it should be one of ['DMAE', 'RMSD']")

    if not silent:
        if abs(init_target - final_target) > 1e-5:
            print(f"[{i}]: {init_target} -> {final_target}")
        else:
            print(f"[{i}]: {final_target}")

    ret_pos = prb_pos[match]
    ret_atoms = prb_atoms[match]
    return torch.from_numpy(ret_pos)



if __name__ == "__main__":
    ###########################################################################
    ## Load reference pos
    from dataset.data_module import GrambowDataModule
    from omegaconf import OmegaConf

    datamodule = GrambowDataModule(OmegaConf.load(args.config_yaml))
    for batch in datamodule.test_dataloader():
        print(f"Debug: batch.rxn_idx={batch.rxn_idx}")

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

    pos_list = []
    atom_type_list = []
    smarts_list = []
    for batch in data:
        natoms = batch.batch.bincount()
        pos = batch.pos.split(natoms.tolist())
        pos_list.extend(pos)

        atom_type = batch.atom_type.split(natoms.tolist())
        atom_type_list.extend(atom_type)

        smarts = batch.smarts
        smarts_list.extend(smarts)
    ###########################################################################


    ## Calculate DMAE, RMSD
    rmsd_list = []
    dmae_list = []
    # q_norm_list = []  # TODO:
    for i, (pos_ref, pos_gen) in enumerate(zip(pos_ref_list, pos_list)): 
        atom_type = atom_type_list[i]

        smiles = smarts_list[i]

        atom_type = remap2atomic_numbers(atom_type)
        atoms_ref = Atoms(symbols=atom_type, positions=pos_ref)
        atoms_gen = Atoms(symbols=atom_type, positions=pos_gen)
        pos_gen = align_jhwoo(atoms_ref, atoms_gen, smiles, target=args.align_target)

        atoms_ref = Atoms(symbols=atom_type, positions=pos_ref)
        atoms_gen = Atoms(symbols=atom_type, positions=pos_gen)
        rmsd = calc_RMSD2(atoms_ref, atoms_gen)

        dm_ref = torch.cdist(pos_ref, pos_ref)
        dm_gen = torch.cdist(pos_gen, pos_gen)
        dmae = torch.triu(abs(dm_ref - dm_gen), diagonal=1).sum()
        dmae /= len(pos_ref) * (len(pos_ref) - 1) / 2

        rmsd_list.append(rmsd)
        dmae_list.append(dmae)

        print(f"it={i}: rmsd={rmsd}, dmae={dmae}")
    rmsd_list = torch.tensor(rmsd_list)
    dmae_list = torch.tensor(dmae_list)

    print(f"dmae_list.sort()[0]=\n{dmae_list.sort()[0]}")
    print(f"rmsd_list.sort()[0]=\n{rmsd_list.sort()[0]}")
    print(f"RMSD (mean  ): {rmsd_list.mean()}")
    print(f"DMAE (mean  ): {dmae_list.mean()}")
    print(f"RMSD (median): {rmsd_list.median()}")
    print(f"DMAE (median): {dmae_list.median()}")
