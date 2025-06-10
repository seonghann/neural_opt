"""
dataloader로부터 xyz 파일로 저장하는 스크립트 (for R-DSM training)

각 xyz 파일은 2개의 구조 정보로 구성. (eq and non-eq structures)
"""
import argparse
import os
from random import choice
import time

import torch
from torch.utils.data import DataLoader
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from tqdm import tqdm
from pytorch_lightning import seed_everything

from morered.datasets import QM7X
from morered.utils import batch_center_systems, batch_rmsd
from morered.noise_schedules import PolynomialSchedule
from morered.processes import VPGaussianDDPM
from morered.sampling import MoreRedJT

from parse_noneq import load_system_by_key


def sample_noneq_sample(
    smiles_id: int,
    stereo_iso_id: int,
    conform_id: int,
    mode: str = "random",
    base_dir: str = "../data/",
):
    if mode not in ["random", "smallest", "largest", "median"]:
        raise ValueError(
            f"Unsupported mode: {mode}. Choose from ['random','smallest','largest','median']."
        )

    key = (smiles_id, stereo_iso_id, conform_id)

    # 100개의 non-equilibrium 구조 중 선택.
    n_samples = 100
    if mode  == "random":
        noneq_key = choice(range(1, n_samples + 1))
        data = load_system_by_key(key, noneq_key, base_dir=base_dir)
        return data
    else:
        data_list = [
            load_system_by_key(key, noneq_key, base_dir=base_dir) for noneq_key in range(1, n_samples + 1)
        ]
        rmsd_list = torch.tensor([data["rmsd"] for data in data_list]).reshape(-1)
        sorted_indices = sorted(range(len(rmsd_list)), key=lambda i: rmsd_list[i])

        if mode == "smallest":
            chosen_idx = sorted_indices[0]
        elif mode == "largest":
            chosen_idx = sorted_indices[-1]
        else:
            # mode == "median"
            mid = len(sorted_indices) // 2
            chosen_idx = sorted_indices[mid]
        return data_list[chosen_idx]


def batch_process_noneq(batch, metadata, suffix="_noneq", sample_mode="random", datapath="../data/"):
    """
    Augments a batch of equilibrium molecular structures with corresponding non-equilibrium data.

    For each structure in the input `batch`, this function uses metadata to find a corresponding
    non-equilibrium conformation (e.g., sampled from distortions) and appends its features to the
    batch dictionary under new key names with a suffix (e.g., "_noneq").

    The function also stores the original (equilibrium) positions under the key "_positions_eq",
    replaces the active input coordinates "R" with the perturbed ones from the non-equilibrium
    data, and merges all available atomic/molecular-level features.
    """
    tmp_batch = {key: val.clone() for key, val in batch.items()}
    reference = batch[properties.R].clone()
    tmp_batch[properties.R + "_eq"] = reference

    # Extract mapping keys from metadata using global indices
    print("Mapping keys from metadata...")
    smiles_ids = [metadata["groups_ids"]["smiles_id"][i] for i in batch["_idx"]]
    stereo_iso_ids = [
        metadata["groups_ids"]["stereo_iso_id"][i] for i in batch["_idx"]
    ]
    conform_ids = [
        metadata["groups_ids"]["conform_id"][i] for i in batch["_idx"]
    ]

    # Get non-equilibrium samples
    sampled = [
        sample_noneq_sample(sid, iid, cid, mode=sample_mode, base_dir=datapath)
        for sid, iid, cid in zip(smiles_ids, stereo_iso_ids, conform_ids)
    ]
    batch_sampled = _atoms_collate_fn(sampled)

    # Add all non-eq features with modified key names
    for k, v in batch_sampled.items():
        new_key = k + suffix
        tmp_batch[new_key] = v

    assert torch.equal(
        tmp_batch[properties.n_atoms], tmp_batch[properties.n_atoms + suffix]
    ), "Number of atoms mismatch between equilibrium and non-equilibrium structures."
    assert torch.equal(
        tmp_batch["_atomic_numbers"], tmp_batch["_atomic_numbers" + suffix]
    ), "Atomic numbers mismatch between equilibrium and non-equilibrium structures."

    # Update perturbed positions as input
    tmp_batch[properties.R] = tmp_batch[properties.R + suffix]
    return tmp_batch


def _batch_center_systems_inplace(batch):
    batch[properties.R] = batch_center_systems(
        batch[properties.R],
        batch[properties.idx_m],
        batch[properties.n_atoms],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Denoise QM7-X non-equilibrium structures using the MoreRed-JT sampler"
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="../data",
        help="Path to the QM7-X database file (e.g., qm7x.db)",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        required=True,
        help="Path to the split.npz file defining train/test splits",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for DataLoader (default: 10)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained MoreRed-JT model file",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Number of batches to process (default: all batches)",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Output path to save the results (.pt) (default: None)",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="random",
        choices=["random", "smallest", "largest", "median"],
        help="Sampling mode for non-equilibrium structures (default: random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for DataLoader (default: 8)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the test DataLoader (default: False)",
    )
    parser.add_argument(
        "--denoising",
        action="store_true",
        help="Enable denoising of non-equilibrium structures using the MoreRed-JT sampler (default: False)",
    )
    parser.add_argument(
        "--preprocessed_data",
        type=str,
        default=None,
        help="Path to preprocessed data (.pt) or None to load from DataLoader",
    )
    parser.add_argument(
        "--dataloader_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split of the DataLoader to use (default: test)",
    )
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed, workers=True)

    if args.denoising:
        assert os.path.exists(
                args.model_path
            ), f"model_path {args.model_path} does not exist."

        # Define sampler (MoreRed-JT)
        noise_schedule = PolynomialSchedule(
            T=1000, s=1e-5, dtype=torch.float64, variance_type="lower_bound"
        )
        diff_proc = VPGaussianDDPM(
            noise_schedule, noise_key="eps", invariant=True, dtype=torch.float64
        )
        morered_jt = torch.load(args.model_path, map_location="cpu")
        print(f"Loaded model from {args.model_path}")
        mrd_jt_sampler = MoreRedJT(
            diff_proc,
            morered_jt,
            noise_pred_key="eps_pred",
            time_key="t",
            time_pred_key="t_pred",
            convergence_step=0,
            cutoff=5.0,
            recompute_neighbors=False,
            save_progress=True,
            progress_stride=1,
            results_on_cpu=True,
        )

    if args.preprocessed_data:
        test_dataloader = torch.load(args.preprocessed_data)
        print(f"Loaded preprocessed data from {args.preprocessed_data}")
    else:
        # Load equilibrium data module
        assert os.path.exists(args.datapath), f"datapath {args.datapath} does not exist."
        assert os.path.exists(
            args.split_file
        ), f"split_file {args.split_file} does not exist."
        datapath_db = os.path.join(args.datapath, "qm7x.db")
        data_eq = QM7X(
            datapath=datapath_db,
            raw_data_path=None,
            remove_duplicates=True,
            only_equilibrium=True,
            batch_size=args.batch_size,
            split_file=args.split_file,
            num_workers=args.num_workers,
            distance_unit="Ang",
            property_units={"energy": "eV"},
            load_properties=["energy", "rmsd"],
        )
        data_eq.setup()

        # Load full data module
        data_all = QM7X(
            datapath=datapath_db,
            raw_data_path=None,
            remove_duplicates=True,
            batch_size=args.batch_size,
            split_file=args.split_file,
            num_workers=args.num_workers,
            distance_unit="Ang",
            property_units={"energy": "eV"},
            load_properties=["energy", "rmsd"],
        )
        data_all.setup()
        print(f"Loaded data from {datapath_db} with split file {args.split_file}")

        if args.dataloader_split == "train":
            _test_dataloader = data_eq.train_dataloader()
        elif args.dataloader_split == "val":
            _test_dataloader = data_eq.val_dataloader()
        elif args.dataloader_split == "test":
            _test_dataloader = data_eq.test_dataloader()
        else:
            raise ValueError(
                f"Unsupported dataloader split: {args.dataloader_split}. Choose from ['train', 'val', 'test']."
            )

        test_dataloader = DataLoader(
            _test_dataloader.dataset,
            batch_size=_test_dataloader.batch_size,
            shuffle=args.shuffle,
            collate_fn=getattr(_test_dataloader, "collate_fn", None),
            num_workers=getattr(_test_dataloader, "num_workers", 0),
        )

    total_batches = len(test_dataloader)
    num_batches = args.num_batches if args.num_batches is not None else total_batches

    results = []
    for i, batch in tqdm(enumerate(test_dataloader), total=num_batches):
        if i >= num_batches:
            break

        if args.preprocessed_data is None:
            print("Processing batch...")
            start = time.perf_counter()
            batch = batch_process_noneq(batch, data_all.dataset.metadata, sample_mode=args.sample_mode, datapath=args.datapath)
            _batch_center_systems_inplace(batch)
            print(f"Batch {i+1}/{num_batches} processed in {time.perf_counter() - start:.4f} seconds")

        if args.denoising:
            # Denoise non-equilibrium structures using the MoreRed-JT sampler
            print("Start denoising...")
            start = time.perf_counter()
            relaxed_jt, num_steps_jt, hist_jt = mrd_jt_sampler.denoise(
                batch, max_steps=1000
            )
            print(f"Batch {i+1}/{num_batches} denoised in {time.perf_counter() - start:.4f} seconds")

            tmp_batch = {key: val.clone() for key, val in batch.items()}
            tmp_batch.update({properties.R: relaxed_jt[properties.R]})
            tmp_batch.update({"num_steps": num_steps_jt})

            results.append(tmp_batch)
            rmsd_noneq = batch["rmsd_noneq"]
            tmp_batch["rmsd_denoised"] = batch_rmsd(
                tmp_batch[properties.R + "_eq"], tmp_batch
            )
            print(f"RMSD(non-eq): {rmsd_noneq}")
            print(f"RMSD(denoised): {tmp_batch['rmsd_denoised']}")
        else:
            results.append(batch)

    # print(f"Debug: results=\n{results}")
    # Save results (.pt)
    if args.results_path is not None:
        torch.save(results, args.results_path)
        print(f"Results saved to {args.results_path}")


if __name__ == "__main__":
    main()
