import argparse
from tqdm import tqdm
import torch
from typing import List, Dict
from rdkit_utils import atoms_to_smarts
from ase import Atoms
from ase.io import write


def split_batch_by_idx_m(batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    idx_m = batch['_idx_m']
    n_atoms = batch['_n_atoms']
    unique_samples = idx_m.unique(sorted=True)
    
    # atomic-level fields (with shape [sum(n_atoms), ...])
    atom_level_fields = ['_atomic_numbers', '_positions', '_positions_eq',
                         '_positions_noneq', '_atomic_numbers_noneq']
    
    # sample-level fields (with shape [num_samples, ...])
    sample_level_fields = ['_idx', 'energy', 'rmsd', '_n_atoms',
                           '_cell', '_pbc', 'energy_noneq', 'rmsd_noneq']
    
    # prepare for atom-level split
    atom_splits = torch.cumsum(n_atoms, dim=0).tolist()
    atom_starts = [0] + atom_splits[:-1]
    
    per_sample_data = []
    for i, sample_idx in enumerate(unique_samples.tolist()):
        sample = {}
        
        # sample-level fields
        for key in sample_level_fields:
            if key in batch:
                sample[key] = batch[key][i]
        
        # atom-level fields
        start = atom_starts[i]
        end = atom_splits[i]
        for key in atom_level_fields:
            if key in batch:
                sample[key] = batch[key][start:end]
        
        per_sample_data.append(sample)

    return per_sample_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate xyz files from a batch of data",
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to the input file containing the batch data (.pt)"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Directory to save the output xyz files"
    )
    parser.add_argument(
        "--start_idx", type=int, required=True,
        help="Starting index for naming output files"
    )
    parser.add_argument(
        "--end_idx", type=int, required=True,
        help="Ending index for naming output files (inclusive)"
    )
    args = parser.parse_args()

    dataloader = torch.load(args.input_file)
    print("Loaded data from", args.input_file)

    # total number of samples
    n_samples = sum(len(batch["_idx"]) for batch in dataloader)
    assert n_samples == args.end_idx - args.start_idx + 1, \
        f"Expected {args.end_idx - args.start_idx + 1} samples, but got {n_samples}."
    print(f"Total number of samples: {n_samples}")

    idx = args.start_idx
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        split_batch = split_batch_by_idx_m(batch) 

        for sample in split_batch:
            # Extract the necessary fields
            positions = sample['_positions'].numpy()
            numbers = sample['_atomic_numbers'].numpy()

            # Create Atoms object
            atoms = Atoms(positions=sample['_positions_eq'].numpy(),
                             numbers=sample['_atomic_numbers'].numpy())
            smarts = atoms_to_smarts(atoms, sanitize=True, remove_hs=False, atom_mapping=True)
            _idx = sample['_idx'].item()
            comment = (
                f'equilibrium idx={idx} GeodesicLength=0.0 smarts="{smarts}" _idx={_idx}'
            )
            print(f"Debug: comment=\n{comment}")

            filename = f"{args.save_dir}/idx{idx}.xyz"
            write(filename, atoms, format="xyz", append=False, comment=comment)


            # Write the noneq structure
            atoms = Atoms(positions=sample['_positions_noneq'].numpy(),
                          numbers=sample['_atomic_numbers_noneq'].numpy())
            comment = (
                f'non-equilibrium idx={idx} GeodesicLength=0.0 smarts="{smarts}" _idx={_idx}'
            )
            write(filename, atoms, format="xyz", append=True, comment=comment)
            print(f"Write {filename}")

            idx += 1