from utils.chem import ATOM_ENCODER, BOND_TYPES_ENCODER, BOND_TYPES_DECODER, ATOMIC_NUMBERS

import rdkit.Chem as Chem
import rdkit

import numpy as np
import torch


# TODO: process_smarts 와 process_smarts_single을 하나의 함수로 합치기.
def process_smarts_single(
    smarts: str,
):
    """Processing SMARTS"""

    mol = Chem.MolFromSmarts(smarts)
    try:
        Chem.SanitizeMol(mol)
    except:
        print("Sanitization failed for", smarts)
        raise ValueError

    N = mol.GetNumAtoms()

    # Consider perumation based on atom map number
    perm = np.array([a.GetAtomMapNum() for a in mol.GetAtoms()]) - 1
    perm_inv = np.argsort(perm)

    # extract atomic features
    node_feat = []
    atom_numbers = []

    for atom in np.array(mol.GetAtoms())[perm_inv]:
        atom_numbers.append(atom.GetAtomicNum())
        atomic_feat = []
        for k, v in ATOM_ENCODER.items():
            feat = getattr(atom, k)()
            if feat not in v:
                raise ValueError(f"While processing {smarts}, ({k}) atom feature {feat} is not in {v}")
                v.update({feat: len(v)})
            atomic_feat.append(v[feat])
        node_feat.append(atomic_feat)

    node_feat = torch.tensor(node_feat, dtype=torch.long)

    # extract bond feature
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    adj_perm = adj[perm_inv, :].T[perm_inv, :].T
    adj = adj_perm
    row, col = adj.nonzero()

    _nonbond = 0
    edge_type = []
    for i, j in zip(perm_inv[row], perm_inv[col]):
        b = mol.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            edge_type.append(BOND_TYPES_ENCODER[b.GetBondType()])
        elif b is None:
            edge_type.append(_nonbond)

    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    atom_numbers = [ATOMIC_NUMBERS[atom_number] for atom_number in atom_numbers]
    atom_type = torch.tensor(atom_numbers, dtype=torch.long)

    row, col = edge_index

    return atom_type, edge_index, edge_type, node_feat


def process_smarts(rxn_smarts):
    """Processing reaction SMARTS"""
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
    print(f"BOND_TYPES_ENCODER: {BOND_TYPES_ENCODER}")
    print("==" * 50)
    rxn_smarts = "[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]>>[C:1]([C:2]([N:3]=[O:4])=[N+:6]=[N-:5])([H:7])([H:8])[H:9]"
    # rxn_smarts = "[#6:1](-[#6:2]1:[#6:3](:[#6:4](:[#6:5](:[#7:6]:1-[#6:7](-[#7:8](-[H:16])-[H:17])=[#8:9])-[H:15])-[H:14])-[H:13])(-[H:10])(-[H:11])-[H:12]>>[#6:1](-[#6:2]1:[#6:3](:[#6:4](:[#6:5](:[#7:6]:1-[#6:7](-[#7:8](-[H:16])-[H:17])=[#8:9])-[H:15])-[H:14])-[H:13])(-[H:10])(-[H:11])-[H:12]"
    atom_type, edge_index, r_edge_type, p_edge_type, r_feat, p_feat = process_smarts(rxn_smarts)
    print(f"* rxn_smarts: \n{rxn_smarts}")
    print(f"* atom_type: \n{atom_type}")
    print(f"* edge_index: \n{edge_index}")
    print(f"* r_edge_type: \n{r_edge_type}")
    print(f"* p_edge_type: \n{p_edge_type}")
    print(f"* r_feat: \n{r_feat}")
    print(f"* p_feat: \n{p_feat}")


    print("==" * 50)
    smarts = "[#6:1](-[#6:2]1:[#6:3](:[#6:4](:[#6:5](:[#7:6]:1-[#6:7](-[#7:8](-[H:16])-[H:17])=[#8:9])-[H:15])-[H:14])-[H:13])(-[H:10])(-[H:11])-[H:12]"
    smarts = "[#7:1](~[#6:2]1~[#7:3]~[#6:4](~[#7:5](~[#6:6](~[#8:7])~[H:13])~[H:12])~[#7:8]~[#8:9]~1)(~[H:10])~[H:11]"
    # smarts = "[#6:1](-[#6:2](-[#6:3]1(-[#6:4](-[#6:5](-[H:16])(-[H:17])-[H:18])(-[H:14])-[H:15])-[#6:6](-[#6@:7]-1(-[#6:8](-[H:22])(-[H:23])-[H:24])-[H:21])(-[H:19])-[H:20])(-[H:12])-[H:13])(-[H:9])(-[H:10])-[H:11]"
    atom_type, edge_index, edge_type, node_feat = process_smarts_single(smarts)
    print(f"* smarts: \n{smarts}")
    print(f"* atom_type: \n{atom_type}")
    print(f"* edge_index: \n{edge_index}")
    print(f"* edge_type: \n{edge_type}")
    print(f"* node_feat: \n{node_feat}")
