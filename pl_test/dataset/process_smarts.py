
from utils.chem import ATOM_ENCODER, BOND_TYPES_ENCODER, BOND_TYPES_DECODER, ATOMIC_NUMBERS

import rdkit.Chem as Chem
import rdkit

import numpy as np
import torch



def process_smarts(rxn_smarts):
    r_smarts, p_smarts = rxn_smarts.split(">>")

    # sanitize
    if isinstance(r_smarts, str) and isinstance(p_smarts, str):
        try:
            r = Chem.MolFromSmarts(r_smarts)
            Chem.SanitizeMol(r)
        except:
            print("Sanitization failed for", r_smarts)
            raise ValueError

        try:
            p = Chem.MolFromSmarts(p_smarts)
            Chem.SanitizeMol(p)
        except:
            print("Sanitization failed for", p_smarts)
            raise ValueError
    else:
        r, p = r_smarts, p_smarts

    N = r.GetNumAtoms()

    # Consider perumation based on atom map number
    r_perm = np.array([a.GetAtomMapNum() for a in r.GetAtoms()]) - 1
    p_perm = np.array([a.GetAtomMapNum() for a in p.GetAtoms()]) - 1
    r_perm_inv = np.argsort(r_perm)
    p_perm_inv = np.argsort(p_perm)

    # extract atomic features
    r_feat = []
    p_feat = []

    atom_numbers = []

    for atom in np.array(r.GetAtoms())[r_perm_inv]:
        atom_numbers.append(atom.GetAtomicNum())
        atomic_feat = []
        for k, v in ATOM_ENCODER.items():
            feat = getattr(atom, k)()
            if feat not in v:
                raise ValueError(f"While processing {r_smarts}, ({k}) atom feature {feat} is not in {v}")
                v.update({feat: len(v)})
            atomic_feat.append(v[feat])
        r_feat.append(atomic_feat)

    for atom in np.array(p.GetAtoms())[p_perm_inv]:
        atomic_feat = []
        for k, v in ATOM_ENCODER.items():
            feat = getattr(atom, k)()
            if feat not in v:
                raise ValueError(f"While processing {p_smarts}, ({k}) atom feature {feat} is not in {v}")
                v.update({feat: len(v)})
            atomic_feat.append(v[feat])
        p_feat.append(atomic_feat)

    r_feat = torch.tensor(r_feat, dtype=torch.long)
    p_feat = torch.tensor(p_feat, dtype=torch.long)

    # extract bond feature
    r_adj = Chem.rdmolops.GetAdjacencyMatrix(r)
    p_adj = Chem.rdmolops.GetAdjacencyMatrix(p)
    r_adj_perm = r_adj[r_perm_inv, :].T[r_perm_inv, :].T
    p_adj_perm = p_adj[p_perm_inv, :].T[p_perm_inv, :].T
    adj = r_adj_perm + p_adj_perm
    row, col = adj.nonzero()

    _nonbond = 0
    p_edge_type = []
    for i, j in zip(p_perm_inv[row], p_perm_inv[col]):
        b = p.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            p_edge_type.append(BOND_TYPES_ENCODER[b.GetBondType()])
        elif b is None:
            p_edge_type.append(_nonbond)

    r_edge_type = []
    for i, j in zip(r_perm_inv[row], r_perm_inv[col]):
        b = r.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            r_edge_type.append(BOND_TYPES_ENCODER[b.GetBondType()])
        elif b is None:
            r_edge_type.append(_nonbond)

    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    r_edge_type = torch.tensor(r_edge_type)
    p_edge_type = torch.tensor(p_edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    r_edge_type = r_edge_type[perm]
    p_edge_type = p_edge_type[perm]

    atom_numbers = [ATOMIC_NUMBERS[atom_number] for atom_number in atom_numbers]
    atom_type = torch.tensor(atom_numbers, dtype=torch.long)

    row, col = edge_index

    return atom_type, edge_index, r_edge_type, p_edge_type, r_feat, p_feat


if __name__ == "__main__":
    # for debug
    import pandas as pd
    df = pd.read_csv("smarts.csv")
    smarts_list = df.AAM.tolist()
    print(BOND_TYPES_DECODER)
    for smarts in smarts_list[:5]:
        process_smarts(smarts)
