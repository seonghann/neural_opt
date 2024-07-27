import re
from math import sqrt

import ase
from ase import Atoms
from ase.io import read
from ase.build.rotate import minimize_rotation_and_translation
import numpy as np
import torch

# fmt: off
ATOMIC_RADIUS = dict(H=0.31, He=0.28,
                     Li=1.28, Be=0.96, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Ne=0.58,
                     Na=1.66, Mg=1.41, Al=1.21, Si=1.11, P=1.07, S=1.05, Cl=1.02, Ar=1.06)
# fmt: on


def get_rijlist_and_re(mol, threshold=np.inf):
    from scipy.spatial import KDTree

    geom = mol.get_positions()
    assert len(geom.shape) == 2

    rijset = set()
    tree = KDTree(geom)
    pairs = tree.query_pairs(threshold)
    rijset.update(pairs)

    rijlist = sorted(rijset)

    atoms = mol.get_chemical_symbols()
    radius = np.array([ATOMIC_RADIUS.get(atom.capitalize(), 1.5) for atom in atoms])
    re = np.array([radius[i] + radius[j] for i, j in rijlist])
    return rijlist, re


def compute_rij(geom, rij_list):
    nrij = len(rij_list)
    rij = np.zeros(nrij)
    bmat = np.zeros((nrij, len(geom), 3))
    for idx, (i, j) in enumerate(rij_list):
        dvec = geom[i] - geom[j]
        rij[idx] = r = np.sqrt(
            dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]
        )
        grad = dvec / r
        bmat[idx, i] = grad
        bmat[idx, j] = -grad
    return rij, bmat


def compute_wij(geom, rij_list, func):
    geom = np.asarray(geom).reshape(-1, 3)
    nrij = len(rij_list)
    rij, bmat = compute_rij(geom, rij_list)
    wij, dwdr = func(rij)
    for idx, grad in enumerate(dwdr):
        bmat[idx] *= grad
    return wij, bmat.reshape(nrij, -1)


def morse_scaler(re=1.5, alpha=1.7, beta=0.01, gamma=0.0):
    def scaler(x):
        ratio = x / re
        val1 = np.exp(alpha * (1 - ratio))
        val2 = beta / ratio
        val3 = gamma * ratio
        dval = -alpha / re * val1 - val2 / x + gamma / re
        return val1 + val2 + val3, dval

    return scaler


def get_morse_qij(mol, alpha=1.7, beta=0.01, gamma=0.0):
    rijlist, re = get_rijlist_and_re(mol)
    wij, _ = compute_wij(mol.positions, rijlist, morse_scaler(re, alpha, beta, gamma))
    return wij


def calc_morse_error(
    mol1: ase.Atoms, mol2: ase.Atoms, norm: bool = True, alpha=1.7, beta=0.01, gamma=0.0
):
    assert mol1.get_chemical_symbols() == mol2.get_chemical_symbols()
    qij1 = get_morse_qij(mol1, alpha, beta, gamma)
    qij2 = get_morse_qij(mol2, alpha, beta, gamma)

    natoms = len(mol1)
    assert len(qij1) == natoms * (natoms - 1) / 2

    if norm:
        retval = np.linalg.norm(qij1 - qij2)
        # retval = np.sqrt(((qij1 - qij2)**2).mean())
    else:
        retval = abs(qij1 - qij2).mean()
    return retval


def read_gaussian_com(file_path):
    """Read .com file and return ase.Atoms object"""
    elements = []
    positions = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Skip Link 0 commands and route section
    idx = 0
    while lines[idx].strip() and lines[idx].startswith("%"):
        idx += 1
    while lines[idx].strip() and lines[idx].startswith("#"):
        idx += 1

    # Skip title section
    while not lines[idx].strip():
        idx += 1
    idx += 1
    while lines[idx].strip():
        idx += 1
    idx += 1

    # Skip charge and multiplicity line
    charge_mult = lines[idx].strip().split()
    charge = int(charge_mult[0])
    multiplicity = int(charge_mult[1])
    idx += 1

    # Read atoms and positions
    while idx < len(lines) and lines[idx].strip():
        parts = lines[idx].strip().split()
        elements.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        idx += 1

    atoms = Atoms(symbols=elements, positions=positions)
    return atoms


def calc_RMSD(mol1: Atoms, mol2: Atoms, mass_weighted: bool = False):
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


def get_min_q_norm_match(matches, ref_atoms, prb_atoms):
    q_norms = []
    ref_atoms = ref_atoms.copy()
    prb_atoms = prb_atoms.copy()

    prb_pos = prb_atoms.positions.copy()
    for match in matches:
        prb_atoms.positions = prb_pos[list(match)].copy()
        q_norm = calc_morse_error(
            ref_atoms,
            prb_atoms,
            norm=True,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
        q_norms.append(q_norm)
    return list(matches[q_norms.index(min(q_norms))]), q_norms[0], min(q_norms)


def calc_DMAE(pos_ref, pos_prb):
    from scipy.spatial.distance import cdist

    dm_ref = cdist(pos_ref, pos_ref)
    dm_prb = cdist(pos_prb, pos_prb)
    dmae = np.triu(abs(dm_ref - dm_prb), k=1).sum()
    dmae /= len(pos_ref) * (len(pos_ref) - 1) / 2
    return dmae


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        type=float,
        help="coefficient alpha for the Morse scaler",
        default=None,
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="coefficient beta for the Morse scaler",
        default=None,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="coefficient gamma for the Morse scaler",
        default=0.0,
    )
    # parser.add_argument(
    #     "--align_target",
    #     type=str,
    #     help="align target",
    #     choices=["DMAE", "RMSD", "q_norm", "None"],
    #     default="None",
    # )
    parser.add_argument(
        "--error_type",
        type=str,
        help="structural error type",
        choices=["DMAE", "RMSD", "q_norm"],
        required=True,
    )
    parser.add_argument("--input_csv", type=str, help="input csv filepath")
    parser.add_argument("--output_csv", type=str, help="output csv filepath")
    parser.add_argument(
        "--dft_results_path1", type=str, help="dft results path (xyz of \tilde{x})"
    )
    parser.add_argument(
        "--dft_results_path2", type=str, help="dft results path (xyz of x)"
    )
    args = parser.parse_args()
    print(args)

    import pandas as pd

    # df = pd.read_csv("./QM9M_SP_CALC/QM9M_SP.csv")
    # df = pd.read_csv("./qm9m.csv")
    df = pd.read_csv(args.input_csv)
    print(df)

    if args.error_type == "q_norm":
        key = args.error_type.lower() + f"({args.alpha},{args.beta},{args.gamma})"
    else:
        key = args.error_type.lower()
    if key in df.keys():
        # exit(f"{key} is already in ./qm9m.csv")
        exit(f"{key} is already in {args.input_csv}")

    ## Calculate structural erros: DMAE, RMSD, q-norm
    error_list = []
    for i, idx in enumerate(df["index"]):
        smarts = df[df["index"] == idx]["smarts"].values.item()

        atoms_mmff = read_gaussian_com(f"{args.dft_results_path1}/idx{idx}/input.com")
        atoms_dft = read_gaussian_com(f"{args.dft_results_path2}/idx{idx}/input.com")
        atom_type = atoms_dft.get_atomic_numbers()

        if args.error_type == "RMSD":
            err = calc_RMSD(atoms_mmff, atoms_dft)
        elif args.error_type == "DMAE":
            err = calc_DMAE(atoms_mmff.positions, atoms_dft.positions)
        elif args.error_type == "q_norm":
            err = calc_morse_error(
                atoms_mmff,
                atoms_dft,
                norm=True,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
        else:
            raise NotImplementedError()
        print(i, idx, err)

        error_list.append(err)

    assert len(error_list) == len(df)
    if args.error_type == "q_norm":
        df[args.error_type.lower() + f"({args.alpha},{args.beta},{args.gamma})"] = (
            error_list
        )
    else:
        df[args.error_type.lower()] = error_list

    print(df)
    # save_file_path = "./qm9m.csv"
    save_file_path = args.output_csv
    df.to_csv(save_file_path, index=False)
    exit(f"DEBUG: save {save_file_path}")
