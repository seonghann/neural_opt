import rdkit.Chem as Chem

_hybrid = Chem.rdchem.HybridizationType.values
_chiral = Chem.rdchem.ChiralType.values
HYBRID_ENCODER = {v: k for k, v in Chem.rdchem.HybridizationType.values.items()}
HYBRID_DECODER = {k: v for k, v in Chem.rdchem.HybridizationType.values.items()}
CHIRAL_ENCODER = {v: k for k, v in Chem.rdchem.ChiralType.values.items()}
CHIRAL_DECODER = {k: v for k, v in Chem.rdchem.ChiralType.values.items()}

# only use non-bond, single, double, triple, aromatic
BOND_TYPES_DECODER = {}
BOND_TYPES_ENCODER = {}
for i, k in enumerate(["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]):
    bond = Chem.rdchem.BondType.names[k]
    BOND_TYPES_DECODER[i + 1] = bond
    BOND_TYPES_ENCODER[bond] = i + 1
BOND_TYPES_ENCODER[None] = 0
BOND_TYPES_DECODER[0] = None


ATOM_ENCODER = {
    "GetIsAromatic": {
        False: 0,
        True: 1
    },
    "GetFormalCharge": {
        -2: 0,
        -1: 1,
        0: 2,
        1: 3,
        2: 4
    },
    "GetTotalNumHs": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
    },
    "GetTotalValence": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
    },
    "GetTotalDegree": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
    },
    "GetHybridization": HYBRID_ENCODER,
    "GetChiralTag": CHIRAL_ENCODER,
    "IsInRing": {
        False: 0,
        True: 1
    },
}

ATOMIC_NUMBERS = {
    1: 0,  # H
    6: 1,  # C
    7: 2,  # N
    8: 3,  # O
}

ATOMIC_RADII = dict(H=0.31, He=0.28,
                    Li=1.28, Be=0.96, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Ne=0.58,
                    Na=1.66, Mg=1.41, Al=1.21, Si=1.11, P=1.07, S=1.05, Cl=1.02, Ar=1.06)

PERIODIC_TABLE = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
                  11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar'}

ATOMIC_RADII_LIST = [ATOMIC_RADII[PERIODIC_TABLE[i]] for i in list(ATOMIC_NUMBERS.keys())]
