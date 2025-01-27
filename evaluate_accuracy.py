"""
Evaluate accuracy of generated geometries

(Usage)
    >>> python eval_accuracy2.py \
    --config_yaml configs/sampling.h.condensed2.yaml \
    --prb_pt save_dynamic.pt \
    --align_target DMAE
"""
import torch
import numpy as np
from ase import Atoms
from ase.build.rotate import minimize_rotation_and_translation
from math import sqrt


def remap2atomic_numbers(atomic_numbers):
    MAPPING = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
    symbols = [MAPPING[num.item()] for num in atomic_numbers]
    return symbols


def calc_RMSD(pos1, pos2):
    # return sqrt(((pos1 - pos2) ** 2).sum() / len(pos1))
    rmsd = np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1)))
    return rmsd


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

    def _get_substruct_matches(smarts):
        mol = Chem.MolFromSmarts(smarts)
        matches = list(mol.GetSubstructMatches(mol, uniquify=False))
        map = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()]) - 1
        map_inv = np.argsort(map)
        for i in range(len(matches)):
            matches[i] = tuple(map[np.array(matches[i])[map_inv]])
        return matches

    smarts_list = smarts.split(">>")
    if len(smarts_list) == 2:
        smarts_r, smarts_p = smarts_list
        matches_r = _get_substruct_matches(smarts_r)
        matches_p = _get_substruct_matches(smarts_p)
        matches = set(matches_r) & set(matches_p)
        # matches = set(matches_r) | set(matches_p); print(f"Debug: change & -> |")
    elif len(smarts_list) == 1:
        smarts = smarts_list[0]
        matches = _get_substruct_matches(smarts)
        matches = set(matches)
    else:
        raise ValueError()

    matches = list(matches)
    matches.sort()
    return matches


def align_jhwoo(
    ref_atoms,
    prb_atoms,
    # ref_smarts,
    # prb_smarts,
    smarts,
    silent = False,
    target = "RMSD",
):
    # Refer to: /home/share/DATA/NeuralOpt/utils/alignXYZ.py
    import sys
    sys.path.append("/home/share/DATA/NeuralOpt/utils")
    # from alignXYZ import get_substruct_matches, get_min_dmae_match
    from alignXYZ import get_min_dmae_match

    matches = get_substruct_matches(smarts)

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
            print(f"[{i}]: {init_target} > {final_target}")
        else:
            print(f"[{i}]: {final_target}")

    ret_pos = prb_pos[match]
    ret_atoms = prb_atoms[match]
    return torch.from_numpy(ret_pos)


def calc_DMAE(pos_ref, pos_prb):
    dm_ref = torch.cdist(pos_ref, pos_ref)
    dm_prb = torch.cdist(pos_prb, pos_prb)
    dmae = torch.triu(abs(dm_ref - dm_prb), diagonal=1).sum()
    dmae /= len(pos_ref) * (len(pos_ref) - 1) / 2
    return dmae


def calc_q_norm(solver, pos_ref, pos_prb, atom_type):
    natoms = len(pos_ref)
    edge_index = torch.triu_indices(natoms, natoms, offset=1)

    q_prb = solver.compute_q(edge_index, atom_type, pos_prb)
    q_ref = solver.compute_q(edge_index, atom_type, pos_ref)

    norm_err = (q_prb - q_ref).square().sum().sqrt()
    return norm_err


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str)
    parser.add_argument("--prb_pt", type=str)
    parser.add_argument("--align_target", type=str, default="none", choices=["DMAE", "RMSD", "none"])
    parser.add_argument("--save_dir", type=str, required=False, default="", help="save xyz files")
    parser.add_argument("--ban_index", type=str, required=False, default=None, help="file path containing ban indices (.pkl)")
    parser.add_argument("--save_csv", type=str, required=False, default=None, help="save accuracies (.csv)")
    args = parser.parse_args()
    print(args)

    import pickle

    if args.ban_index is None:
        ban_index = []
    else:
        ban_index = pickle.load(open(args.ban_index, "rb"))


    # NOTE: Load reference pos
    from dataset.data_module import load_datamodule
    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config_yaml)
    datamodule = load_datamodule(config)

    pos_ref_list = []
    data_idx_list = []
    for batch in datamodule.test_dataloader():
        natoms = batch.batch.bincount()
        pos_ref = batch.pos[:, 0, :].split(natoms.tolist())
        pos_ref_list.extend(pos_ref)

        data_idx_list.extend(batch.idx)


    # NOTE: Load prb pos
    data = torch.load(args.prb_pt, map_location="cpu")
    print(f"Load {args.prb_pt}")

    pos_list = []
    xT_list = []  # starting geometries (e.g., MMFF of MMFFtoDFT)
    atom_type_list = []
    smarts_list = []
    for batch in data:
        natoms = batch.batch.bincount()
        pos = batch.pos.split(natoms.tolist())
        pos_list.extend(pos)

        xT = batch.pos_traj[0]
        xT = xT.split(natoms.tolist())
        xT_list.extend(xT)

        atom_type = batch.atom_type.split(natoms.tolist())
        atom_type_list.extend(atom_type)

        smarts = batch.smarts
        smarts_list.extend(smarts)


    # NOTE: Calculate RMSD, DMAE, q_norm
    from utils.geodesic_solver import GeodesicSolver

    geodesic_solver = GeodesicSolver(config.manifold)
    # q_type = "morse"

    ## Calculate DMAE, RMSD
    rmsd_list = []
    dmae_list = []
    rmsd_list_xT = []
    dmae_list_xT = []
    q_norm_list = []
    q_norm_list_xT = []
    data_idx_list2 = []
 
    for i, (pos_ref, pos_gen, xT) in enumerate(zip(pos_ref_list, pos_list, xT_list)): 
        data_idx = data_idx_list[i]
        if data_idx in ban_index:
            print(f"Pass data_idx={data_idx} because it's in ban_index.")
            continue
        data_idx_list2.append(data_idx)

        _atom_type = atom_type_list[i]

        smarts = smarts_list[i]

        atom_type = remap2atomic_numbers(_atom_type)
        atoms_ref = Atoms(symbols=atom_type, positions=pos_ref)
        atoms_gen = Atoms(symbols=atom_type, positions=pos_gen)
        atoms_xT = Atoms(symbols=atom_type, positions=xT)

        if args.align_target.lower() != "none":
            pos_gen = align_jhwoo(atoms_ref, atoms_gen, smarts, target=args.align_target)
            xT = align_jhwoo(atoms_ref, atoms_xT, smarts, target=args.align_target)
            atoms_gen = Atoms(symbols=atom_type, positions=pos_gen)
            atoms_xT = Atoms(symbols=atom_type, positions=xT)

        rmsd = calc_RMSD2(atoms_ref, atoms_gen)
        rmsd_xT = calc_RMSD2(atoms_ref, atoms_xT)

        dmae = calc_DMAE(pos_ref, pos_gen)
        dmae_xT = calc_DMAE(pos_ref, xT)

        q_norm = calc_q_norm(geodesic_solver, pos_ref, pos_gen, _atom_type)
        q_norm_xT = calc_q_norm(geodesic_solver, pos_ref, xT, _atom_type)

        rmsd_list.append(rmsd)
        dmae_list.append(dmae)
        rmsd_list_xT.append(rmsd_xT)
        dmae_list_xT.append(dmae_xT)
        q_norm_list.append(q_norm)
        q_norm_list_xT.append(q_norm_xT)

        # print(f"it: {i}, data_idx: {data_idx}, rmsd: {rmsd_xT} > {rmsd}, dmae: {dmae_xT} > {dmae}, q_norm: {q_norm_xT} > {q_norm}")

    rmsd_list = torch.tensor(rmsd_list)
    dmae_list = torch.tensor(dmae_list)
    rmsd_list_xT = torch.tensor(rmsd_list_xT)
    dmae_list_xT = torch.tensor(dmae_list_xT)
    q_norm_list = torch.tensor(q_norm_list)
    q_norm_list_xT = torch.tensor(q_norm_list_xT)

    import pandas as pd
    pd.set_option('display.float_format', '{:.3f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_seq_items', None)
    pd.set_option('display.max_colwidth', 500)
    pd.set_option('expand_frame_repr', True)

    save_data = {
        "data_idx": data_idx_list2,
        "rmsd_xT": rmsd_list_xT,
        "rmsd": rmsd_list,
        "dmae_xT": dmae_list_xT,
        "dmae": dmae_list,
        "q_norm_xT": q_norm_list_xT,
        "q_norm": q_norm_list,
    }
    df = pd.DataFrame(save_data)
    df = df.sort_values(by="rmsd"); print("Sorted with rmsd")
    print(df)

    if args.save_csv is not None:
        df.to_csv(args.save_csv)
        print(f"Save {args.save_csv}")

    print("=" * 100)
    print("Statistics")
    print("# of data: ", len(rmsd_list))
    print(f"[x0 vs predicted]")
    print(f"RMSD   (mean, median): {rmsd_list.mean():.3g}, {rmsd_list.median():.3g}")
    print(f"DMAE   (mean, median): {dmae_list.mean():.3g}, {dmae_list.median():.3g}")
    print(f"q_norm (mean, median): {q_norm_list.mean():.3g}, {q_norm_list.median():.3g}")

    print(f"[x0 vs xT]")
    print(f"RMSD   (mean, median): {rmsd_list_xT.mean():.3g}, {rmsd_list_xT.median():.3g}")
    print(f"DMAE   (mean, median): {dmae_list_xT.mean():.3g}, {dmae_list_xT.median():.3g}")
    print(f"q_norm (mean, median): {q_norm_list_xT.mean():.3g}, {q_norm_list_xT.median():.3g}")

    print(f"x0 vs xT")
    print(f"RMSD   (Q1, Q3): {np.percentile(rmsd_list_xT, 25):.3g}, {np.percentile(rmsd_list_xT, 75):.3g}")
    print(f"DMAE   (Q1, Q3): {np.percentile(dmae_list_xT, 25):.3g}, {np.percentile(dmae_list_xT, 75):.3g}")
    print(f"q_norm (Q1, Q3): {np.percentile(q_norm_list_xT, 25):.3g}, {np.percentile(q_norm_list_xT, 75):.3g}")
    print("=" * 100)


    # NOTE: Save xyz files
    if args.save_dir:
        import os

        os.mkdir(args.save_dir)
        print(f"mkdir {args.save_dir}")
        for i, (pos_ref, pos_gen, xT) in enumerate(zip(pos_ref_list, pos_list, xT_list)):
            idx = data_idx_list[i]

            _atom_type = atom_type_list[i]

            smarts = smarts_list[i]
            atom_type = remap2atomic_numbers(_atom_type)
            atoms_gen = Atoms(symbols=atom_type, positions=pos_gen)
            comment = f"idx={idx} smarts=\"{smarts}\" rmsd={rmsd_list[i]} dmae={dmae_list[i]} q_norm={q_norm_list[i]}"
            filename = f"./{args.save_dir}/idx{idx}.xyz"
            atoms_gen.write(filename, comment=comment)
            print(f"Save {filename}")
