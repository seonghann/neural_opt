import argparse

ps = argparse.ArgumentParser(
    description="Visualize xyz file using ase.visualize",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
ps.add_argument("--xyz", type=str, help="xyz file path")
args = ps.parse_args()
print(args)


from ase.io import iread
from ase.visualize import view


atoms = iread(args.xyz)
view(atoms)
