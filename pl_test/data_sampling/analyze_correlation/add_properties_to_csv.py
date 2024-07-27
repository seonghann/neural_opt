"""
Add dE and SMARTS information to csv file
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, help="input csv filepath")
parser.add_argument("--output_csv", type=str, help="output csv filepath")
parser.add_argument("--sdf_path", type=str, help="data sdf file path (needed to read smarts information)")
args = parser.parse_args()
print(args)


import pandas as pd
from rdkit import Chem
from ase.units import Hartree, kcal, mol

# df = pd.read_csv("./QM9M_SP_CALC/QM9M_SP.csv")
df = pd.read_csv(args.input_csv)
print(df)


## Add energy difference (dE)
df["dE"] = abs(df['DFT_energy'] - df['MMFF_energy']) * Hartree * mol / kcal
print(df)


## Add SMARTS
# SDF 파일을 읽어오는 함수
def read_sdf(filename):
    suppl = Chem.SDMolSupplier(filename, removeHs=False)
    return [mol for mol in suppl if mol is not None]

def mol_to_smarts(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)  # Atom-mapping 번호 설정
    return Chem.MolToSmarts(mol)


smarts_list = []
for idx in df["index"]:
    # sdf_file = f"../../sdf_files/{idx}.sdf"  # DFT results
    sdf_file = f"{args.sdf_path}/{idx}.sdf"  # DFT results
    print(f"Read {sdf_file}")
    mol = read_sdf(sdf_file)[0]
    smarts = mol_to_smarts(mol)
    smarts_list.append(smarts)

assert len(smarts_list) == len(df)


df["smarts"] = smarts_list
print(df)

# save_file_path = "./qm9m.csv"
save_file_path = args.output_csv
df.to_csv(save_file_path, index=False)
exit(f"DEBUG: save {save_file_path}")
