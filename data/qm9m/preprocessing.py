import os
from rdkit import Chem
from ase import Atoms
from ase.io import write


# Function to read an SDF file
def read_sdf(filename):
    suppl = Chem.SDMolSupplier(filename, removeHs=False)
    return [mol for mol in suppl if mol is not None]


# Function to convert a molecule to a SMILES string with atom mapping
# def mol_to_smiles_with_mapping(mol):
#     for atom in mol.GetAtoms():
#         atom.SetAtomMapNum(atom.GetIdx() + 1)  # Atom-mapping 번호 설정
#     return Chem.MolToSmiles(mol)


def mol_to_smarts(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)  # Set atom-mapping numbers
    return Chem.MolToSmarts(mol)


# Function to extract xyz coordinates from a molecule
def get_xyz_coordinates(mol):
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coordinates = []
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coordinates.append((pos.x, pos.y, pos.z))
    return coordinates


# Function to create an ase.Atoms object from an RDKit molecule
def mol_to_ase_atoms(mol):
    coordinates = get_xyz_coordinates(mol)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(positions=coordinates, symbols=symbols)


if __name__ == "__main__":
    import argparse

    ps = argparse.ArgumentParser(
        description="Data processing: sdf to xyz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ps.add_argument("--sdf_path", type=str, help="sdf files directory path")
    ps.add_argument("--save_dir", type=str, help="save directory path")
    args = ps.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        print(f"make directory {args.save_dir}")

    for idx in range(1, 133885 + 1):
        # sdf to smiles and atoms object
        sdf_file = f"{args.sdf_path}/{idx}.sdf"
        print(f"Read {sdf_file}")
        mol = read_sdf(sdf_file)[0]
        # smiles = mol_to_smiles_with_mapping(mol)
        smarts = mol_to_smarts(mol)
        atoms = mol_to_ase_atoms(mol)
        print("idx=", idx, smarts, atoms)

        # Write in xyz
        # comment = f"DFT(B3LYP/6-31G(2df,p)) idx={idx} smarts=\"{smarts}\""
        comment = (
            f'DFT(B3LYP/6-31G(2df,p)) idx={idx} GeodesicLength=0.0 smarts="{smarts}"'
        )
        filename = f"{args.save_dir}/idx{idx}.xyz"
        write(filename, atoms, format="xyz", append=False, comment=comment)

        # sdf to smiles and atoms object
        sdf_file = f"{args.sdf_path}/{idx}.mmff.sdf"
        print(f"Read {sdf_file}")
        mol = read_sdf(sdf_file)[0]
        # smiles_mmff = mol_to_smiles_with_mapping(mol)
        smarts_mmff = mol_to_smarts(mol)
        atoms = mol_to_ase_atoms(mol)
        print("idx=", idx, smarts_mmff, atoms)

        # Write in xyz
        comment = f'MMFF94 idx={idx} GeodesicLength=0.0 smarts="{smarts_mmff}"'
        write(filename, atoms, format="xyz", append=True, comment=comment)
        print(f"Write {filename}")

        # assert smiles == smiles_mmff, print(f"{smiles}\n{smiles_mmff}")
        if smarts != smarts_mmff:
            print(f"Warning: smarts & smarts_mmff=\n{smarts}\n{smarts_mmff}")
