from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import warnings
import tempfile
import os


def atoms_to_smarts(atoms, sanitize=True, remove_hs=False, atom_mapping=False):
    """
    Convert ASE Atoms object to SMARTS pattern (with bond information)

    Parameters:
    -----------
    atoms : ase.Atoms
        ASE Atoms object
    sanitize : bool
        Whether to sanitize molecular structure (default: True)
    remove_hs : bool
        Whether to remove hydrogen atoms (default: False)
    atom_mapping : bool
        Whether to include atom mapping numbers (default: False)

    Returns:
    --------
    str : SMARTS pattern string
    """

    # Create temporary XYZ file from ASE Atoms
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        temp_xyz_path = f.name

        # Write XYZ format
        n_atoms = len(atoms)
        f.write(f"{n_atoms}\n")
        f.write("Generated from ASE Atoms\n")

        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        for symbol, pos in zip(symbols, positions):
            f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    try:
        # Read molecule from temporary XYZ file
        mol = Chem.MolFromXYZFile(temp_xyz_path)

        if mol is None:
            raise ValueError("Cannot convert ASE Atoms to RDKit molecule")

        # Automatically generate bond information (important!)
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except ValueError as e:
            warnings.warn(
                f"Bond determination failed: {e}. Ensure the molecule has valid connectivity."
            )
            return None

        # Add atom mapping numbers (optional)
        if atom_mapping:
            for i, atom in enumerate(mol.GetAtoms()):
                atom.SetAtomMapNum(i + 1)

        # Sanitize molecular structure
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                warnings.warn(
                    f"Molecular sanitization failed: {e}. Using original structure."
                )

        # Remove hydrogens (optional)
        if remove_hs:
            mol = Chem.RemoveHs(mol)

        # Generate SMARTS
        smarts = Chem.MolToSmarts(mol)

        return smarts

    finally:
        # Clean up temporary file
        if os.path.exists(temp_xyz_path):
            os.unlink(temp_xyz_path)


def xyz_to_smarts(xyz_filename, sanitize=True, remove_hs=False, atom_mapping=False):
    """
    Convert XYZ file to SMARTS pattern (with bond information)

    Parameters:
    -----------
    xyz_filename : str
        Path to XYZ file
    sanitize : bool
        Whether to sanitize molecular structure (default: True)
    remove_hs : bool
        Whether to remove hydrogen atoms (default: False)
    atom_mapping : bool
        Whether to include atom mapping numbers (default: False)

    Returns:
    --------
    str : SMARTS pattern string
    """

    # Read molecule from XYZ file
    mol = Chem.MolFromXYZFile(xyz_filename)

    if mol is None:
        raise ValueError(f"Cannot read XYZ file: {xyz_filename}")

    # Automatically generate bond information (important!)
    rdDetermineBonds.DetermineBonds(mol)

    # Add atom mapping numbers (optional)
    if atom_mapping:
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)

    # Sanitize molecular structure
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            warnings.warn(
                f"Molecular sanitization failed: {e}. Using original structure."
            )

    # Remove hydrogens (optional)
    if remove_hs:
        mol = Chem.RemoveHs(mol)

    # Generate SMARTS
    smarts = Chem.MolToSmarts(mol)

    return smarts


def debug_mol_info(xyz_filename):
    """
    Debugging function for molecular information
    """
    mol = Chem.MolFromXYZFile(xyz_filename)

    if mol is None:
        print("Cannot read molecule.")
        return

    print(f"Number of atoms: {mol.GetNumAtoms()}")
    print(f"Number of bonds (before): {mol.GetNumBonds()}")

    # Generate bond information
    rdDetermineBonds.DetermineBonds(mol)

    print(f"Number of bonds (after): {mol.GetNumBonds()}")

    # Atom information
    print("\nAtom information:")
    for atom in mol.GetAtoms():
        print(
            f"  {atom.GetIdx()}: {atom.GetSymbol()} (connected atoms: {atom.GetDegree()})"
        )

    # Bond information
    print("\nBond information:")
    for bond in mol.GetBonds():
        print(
            f"  {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}: {bond.GetBondType()}"
        )


# Usage example
if __name__ == "__main__":
    xyz_file = "molecule.xyz"  # Change to actual XYZ file path
    xyz_file = "idx64.xyz"  # Change to actual XYZ file path

    try:
        # Print debugging information
        print("=== Molecular Information ===")
        debug_mol_info(xyz_file)

        print("\n=== SMARTS Results ===")
        # Basic SMARTS
        smarts = xyz_to_smarts(xyz_file)
        print(f"SMARTS: {smarts}")

        # SMARTS with atom mapping
        smarts_mapped = xyz_to_smarts(xyz_file, atom_mapping=True)
        print(f"SMARTS (with mapping): {smarts_mapped}")

    except Exception as e:
        print(f"Error: {e}")
