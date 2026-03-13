"""
Analysis: Riemannian Noise Sampling — DFT vs MMFF vs Riemannian Noisy
Date: 2026-03-12
Related progress log: docs/progress/260311_reproduction_pipeline.md

Problem:
    Reproduce Figure 2 of the paper: compare RMSD distributions of
    DFT vs MMFF and DFT vs Riemannian noisy structures.
    Validate that our ODE-generated Riemannian noise produces
    geometrically reasonable perturbations.

Judgment Criteria:
    1. Riemannian noisy RMSD distribution should overlap with MMFF RMSD range
    2. Per-molecule RMSD(DFT, Riemannian) should be finite and non-zero
    3. Statistics should be qualitatively consistent with paper Table 1

Conclusion:
    (filled after running)

Usage: python analyze/260312_riemannian_noise_comparison.py
"""

import glob
import re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def compute_rmsd(pos1, pos2):
    """RMSD between two position arrays (N, 3)."""
    diff = pos1 - pos2
    return np.sqrt((diff ** 2).sum(axis=1).mean())


def compute_dmae(pos1, pos2, edge_index):
    """D-MAE: mean absolute error of pairwise distances."""
    d1 = np.linalg.norm(pos1[edge_index[0]] - pos1[edge_index[1]], axis=1)
    d2 = np.linalg.norm(pos2[edge_index[0]] - pos2[edge_index[1]], axis=1)
    return np.abs(d1 - d2).mean()


def parse_xyz_file(path):
    """Parse our 3-frame xyz file: pos_0, pos_t, pos_target.
    Returns pos_0, pos_t, time_step, idx."""
    with open(path) as f:
        lines = f.readlines()

    # First frame: pos_0 with metadata
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()

    # Parse metadata from comment
    idx_match = re.search(r'idx=(\d+)', comment)
    ts_match = re.search(r'time_step=(\d+)', comment)
    idx = int(idx_match.group(1)) if idx_match else None
    time_step = int(ts_match.group(1)) if ts_match else None

    # Parse pos_0
    pos_0 = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        pos_0.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pos_0 = np.array(pos_0)

    # Second frame: pos_t
    offset = 2 + n_atoms
    n_atoms2 = int(lines[offset].strip())
    assert n_atoms2 == n_atoms
    pos_t = []
    for i in range(offset + 2, offset + 2 + n_atoms):
        parts = lines[i].split()
        pos_t.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pos_t = np.array(pos_t)

    return pos_0, pos_t, time_step, idx


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from omegaconf import OmegaConf
    from src.dataset.data_module import load_datamodule

    # --- Load dataset to get MMFF positions ---
    config = OmegaConf.load('data/qm9m/riemannian_data_sampling/riemannian_data_sampling.yaml')
    config.train.batch_size = 300
    dm = load_datamodule(config)
    test_dl = dm.test_dataloader()

    # Collect DFT and MMFF positions indexed by data.idx
    print("Loading DFT and MMFF positions from dataset...")
    dft_positions = {}  # idx -> (n_atoms, 3)
    mmff_positions = {}
    for batch in test_dl:
        for i in range(len(batch)):
            data_i = batch[i]
            idx = data_i.idx
            if hasattr(idx, 'item'):
                idx = idx.item()
            pos = data_i.pos.numpy()  # (n_atoms, n_frames, 3)
            dft_positions[idx] = pos[:, 0, :]   # DFT (first frame)
            mmff_positions[idx] = pos[:, -1, :]  # MMFF (last frame)
    print(f"  Loaded {len(dft_positions)} molecules from test set")

    # --- Parse Riemannian noisy xyz files ---
    xyz_dir = '/tmp/verify_riemannian_xyz3'
    xyz_files = sorted(glob.glob(f'{xyz_dir}/*.xyz'))
    print(f"  Found {len(xyz_files)} Riemannian noisy xyz files")

    rmsd_dft_mmff = []
    rmsd_dft_riemannian = []
    dmae_dft_mmff = []
    dmae_dft_riemannian = []
    time_steps = []
    matched_count = 0

    for xyz_file in xyz_files:
        pos_0_xyz, pos_t_xyz, ts, idx = parse_xyz_file(xyz_file)

        if idx not in dft_positions:
            continue

        pos_dft = dft_positions[idx]
        pos_mmff = mmff_positions[idx]
        n_atoms = pos_dft.shape[0]

        # Verify pos_0 from xyz matches DFT from dataset
        # (they should be very close, possibly recentered)
        pos_0_centered = pos_0_xyz - pos_0_xyz.mean(axis=0)
        pos_dft_centered = pos_dft - pos_dft.mean(axis=0)

        # RMSD
        r_mmff = compute_rmsd(pos_dft_centered, pos_mmff - pos_mmff.mean(axis=0))
        r_riem = compute_rmsd(pos_dft_centered, pos_t_xyz - pos_t_xyz.mean(axis=0))

        # D-MAE (upper triangle pairwise)
        edge_idx = np.array(np.triu_indices(n_atoms, k=1))

        d_mmff = compute_dmae(pos_dft, pos_mmff, edge_idx)
        d_riem = compute_dmae(pos_0_xyz, pos_t_xyz, edge_idx)

        rmsd_dft_mmff.append(r_mmff)
        rmsd_dft_riemannian.append(r_riem)
        dmae_dft_mmff.append(d_mmff)
        dmae_dft_riemannian.append(d_riem)
        time_steps.append(ts)
        matched_count += 1

    print(f"  Matched {matched_count} molecules\n")

    rmsd_dft_mmff = np.array(rmsd_dft_mmff)
    rmsd_dft_riemannian = np.array(rmsd_dft_riemannian)
    dmae_dft_mmff = np.array(dmae_dft_mmff)
    dmae_dft_riemannian = np.array(dmae_dft_riemannian)
    time_steps = np.array(time_steps)

    # --- Summary Statistics ---
    print("=" * 60)
    print("RMSD (Å): DFT vs X")
    print(f"  {'':20s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Max':>8s}")
    print(f"  {'MMFF':20s} {rmsd_dft_mmff.mean():8.4f} {np.median(rmsd_dft_mmff):8.4f} {rmsd_dft_mmff.std():8.4f} {rmsd_dft_mmff.max():8.4f}")
    print(f"  {'Riemannian noisy':20s} {rmsd_dft_riemannian.mean():8.4f} {np.median(rmsd_dft_riemannian):8.4f} {rmsd_dft_riemannian.std():8.4f} {rmsd_dft_riemannian.max():8.4f}")
    print()
    print("D-MAE (Å): DFT vs X")
    print(f"  {'':20s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Max':>8s}")
    print(f"  {'MMFF':20s} {dmae_dft_mmff.mean():8.4f} {np.median(dmae_dft_mmff):8.4f} {dmae_dft_mmff.std():8.4f} {dmae_dft_mmff.max():8.4f}")
    print(f"  {'Riemannian noisy':20s} {dmae_dft_riemannian.mean():8.4f} {np.median(dmae_dft_riemannian):8.4f} {dmae_dft_riemannian.std():8.4f} {dmae_dft_riemannian.max():8.4f}")
    print()
    print(f"Paper reference (Table 1, full test set):")
    print(f"  MMFF     RMSD: mean=0.200, median=0.137")
    print(f"  R-DM     RMSD: mean=0.104, median=0.031  (after training)")
    print(f"  MMFF     D-MAE: mean=0.0717, median=0.0571")
    print(f"  NOTE: Our Riemannian noisy is BEFORE training — it's the noise")
    print(f"        injection itself, not model output. Compare to MMFF baseline.")
    print("=" * 60)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # RMSD histograms
    ax = axes[0, 0]
    bins = np.linspace(0, max(rmsd_dft_mmff.max(), rmsd_dft_riemannian.max()) * 1.1, 30)
    ax.hist(rmsd_dft_mmff, bins=bins, alpha=0.6, label=f'MMFF (n={len(rmsd_dft_mmff)})', color='blue')
    ax.hist(rmsd_dft_riemannian, bins=bins, alpha=0.6, label=f'Riemannian (n={len(rmsd_dft_riemannian)})', color='red')
    ax.set_xlabel('RMSD (Å)')
    ax.set_ylabel('Count')
    ax.set_title('RMSD: DFT vs X')
    ax.legend()
    ax.axvline(rmsd_dft_mmff.mean(), color='blue', linestyle='--', alpha=0.5)
    ax.axvline(rmsd_dft_riemannian.mean(), color='red', linestyle='--', alpha=0.5)

    # D-MAE histograms
    ax = axes[0, 1]
    bins = np.linspace(0, max(dmae_dft_mmff.max(), dmae_dft_riemannian.max()) * 1.1, 30)
    ax.hist(dmae_dft_mmff, bins=bins, alpha=0.6, label='MMFF', color='blue')
    ax.hist(dmae_dft_riemannian, bins=bins, alpha=0.6, label='Riemannian', color='red')
    ax.set_xlabel('D-MAE (Å)')
    ax.set_ylabel('Count')
    ax.set_title('D-MAE: DFT vs X')
    ax.legend()

    # RMSD vs time_step scatter
    ax = axes[1, 0]
    ax.scatter(time_steps, rmsd_dft_riemannian, alpha=0.5, s=20, label='Riemannian', color='red')
    ax.axhline(rmsd_dft_mmff.mean(), color='blue', linestyle='--', label=f'MMFF mean={rmsd_dft_mmff.mean():.3f}')
    ax.set_xlabel('time_step t')
    ax.set_ylabel('RMSD (Å)')
    ax.set_title('Riemannian RMSD vs noise level (time_step)')
    ax.legend()

    # D-MAE vs time_step scatter
    ax = axes[1, 1]
    ax.scatter(time_steps, dmae_dft_riemannian, alpha=0.5, s=20, label='Riemannian', color='red')
    ax.axhline(dmae_dft_mmff.mean(), color='blue', linestyle='--', label=f'MMFF mean={dmae_dft_mmff.mean():.4f}')
    ax.set_xlabel('time_step t')
    ax.set_ylabel('D-MAE (Å)')
    ax.set_title('Riemannian D-MAE vs noise level (time_step)')
    ax.legend()

    plt.tight_layout()
    out_path = 'analyze/260312_riemannian_noise_comparison.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
