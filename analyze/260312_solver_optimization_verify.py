"""
Analysis: ODE Solver Optimization Verification
Date: 2026-03-12
Related progress log: docs/progress/260312_riemannian_noise_optimization.md

Problem:
    Verify that CPU-offloaded pinv in solver.py produces numerically
    consistent results vs the original GPU vmap(pinv) implementation.
    Also measure end-to-end speed improvement on batch_geodesic_ode_solve.

Judgment Criteria:
    1. Full ODE solve: output pos RMSD < 1e-4 between old and new
    2. Speed improvement > 3x on same batch
    3. ban_index set identical

Conclusion:
    (filled after running)

Usage: /venv/neural_opt/bin/python analyze/260312_solver_optimization_verify.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf
from dataset.data_module import load_datamodule
from manifold.solver import GeodesicSolver, redefine_edge_index, redefine_with_pad
from manifold.graph import MolGraph
import manifold.solver as solver_module
from torch import vmap


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = OmegaConf.load('data/qm9m/riemannian_data_sampling/riemannian_data_sampling.yaml')
    config.train.batch_size = 100
    dm = load_datamodule(config)
    test_dl = dm.test_dataloader()

    ode_config = OmegaConf.create({
        'ode_solver': {
            'alpha': 1.7, 'beta': 0.01, 'gamma': 0.0, 'svd_tol': 1e-2,
        }
    })
    solver = GeodesicSolver(ode_config)

    batch_data = next(iter(test_dl))
    batch_data = batch_data.to(device)

    graph = MolGraph.from_batch(batch_data)
    edge_index = graph.full_edge(upper_triangle=True)[0]
    pos = batch_data.pos[:, 0, :]
    batch_idx = graph.batch
    num_nodes = batch_data.ptr[1:] - batch_data.ptr[:-1]
    B = num_nodes.size(0)

    # Generate q_dot (noise)
    torch.manual_seed(42)
    n = num_nodes.max().item()
    e = n * (n - 1) // 2
    print(f"B={B}, n={n}, e={e}")

    # Compute initial q_dot
    q = solver.compute_q(edge_index, graph.atom_type, pos)
    q_dot = torch.randn_like(q) * 0.1

    # ===== Test with NEW (CPU-offloaded pinv) =====
    print("\n--- NEW (CPU-offloaded pinv) ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    init_new, last_new, iter_new, idx_tensor_new, stats_new = solver.batch_geodesic_ode_solve(
        pos.clone(), q_dot.clone(), edge_index, graph.atom_type, batch_idx, num_nodes,
        q_type="morse", num_iter=10, ref_dt=0.05, max_dt=0.1, min_dt=1e-3,
        max_iter=1000, err_thresh=0.05, method="Heun"
    )
    torch.cuda.synchronize()
    new_time = time.perf_counter() - t0
    print(f"  Time: {new_time:.2f}s")
    print(f"  Iterations: {iter_new}")
    print(f"  Ban index: {stats_new['ban_index']}")

    # ===== Test with OLD (GPU vmap pinv) =====
    print("\n--- OLD (GPU vmap pinv) ---")
    # Monkey-patch back to old
    old_pinv = lambda J, rtol=1e-4, atol=None: vmap(torch.linalg.pinv)(
        J.to_dense() if J.is_sparse else J, rtol=rtol, atol=atol
    )
    old_svd = lambda J: vmap(torch.linalg.svd)(J.to_dense() if J.is_sparse else J)

    saved_pinv = solver_module.batch_pinv1
    saved_svd = solver_module.batch_svd1
    solver_module.batch_pinv1 = old_pinv
    solver_module.batch_svd1 = old_svd

    torch.manual_seed(42)
    q_dot_old = torch.randn_like(q) * 0.1

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    init_old, last_old, iter_old, idx_tensor_old, stats_old = solver.batch_geodesic_ode_solve(
        pos.clone(), q_dot_old.clone(), edge_index, graph.atom_type, batch_idx, num_nodes,
        q_type="morse", num_iter=10, ref_dt=0.05, max_dt=0.1, min_dt=1e-3,
        max_iter=1000, err_thresh=0.05, method="Heun"
    )
    torch.cuda.synchronize()
    old_time = time.perf_counter() - t0
    print(f"  Time: {old_time:.2f}s")
    print(f"  Iterations: {iter_old}")
    print(f"  Ban index: {stats_old['ban_index']}")

    # Restore
    solver_module.batch_pinv1 = saved_pinv
    solver_module.batch_svd1 = saved_svd

    # ===== Compare =====
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Position comparison
    x_new = last_new['x']
    x_old = last_old['x']
    rmsd_per_mol = ((x_new - x_old) ** 2).sum(dim=-1).mean(dim=-1).sqrt()
    print(f"\nPosition RMSD (new vs old):")
    print(f"  Mean:   {rmsd_per_mol.mean():.2e} A")
    print(f"  Max:    {rmsd_per_mol.max():.2e} A")
    print(f"  Median: {rmsd_per_mol.median():.2e} A")

    # Velocity comparison
    xdot_new = last_new['x_dot']
    xdot_old = last_old['x_dot']
    xdot_err = (xdot_new - xdot_old).norm() / xdot_old.norm()
    print(f"\nx_dot relative error: {xdot_err:.2e}")

    # q comparison
    q_new = last_new['q']
    q_old = last_old['q']
    q_err = (q_new - q_old).norm() / q_old.norm()
    print(f"q relative error: {q_err:.2e}")

    # Exclude banned molecules from RMSD
    all_ban = set()
    if stats_new['ban_index'].numel() > 0:
        all_ban.update(stats_new['ban_index'].cpu().tolist())
    if stats_old['ban_index'].numel() > 0:
        all_ban.update(stats_old['ban_index'].cpu().tolist())
    if all_ban:
        valid_mask = torch.ones(B, dtype=torch.bool)
        for idx in all_ban:
            valid_mask[idx] = False
        rmsd_valid = rmsd_per_mol[valid_mask]
        print(f"\nExcluding banned molecules {all_ban}:")
        print(f"  Mean RMSD:   {rmsd_valid.mean():.2e} A")
        print(f"  Max RMSD:    {rmsd_valid.max():.2e} A")
        print(f"  Median RMSD: {rmsd_valid.median():.2e} A")

    # Ban index
    ban_new = set(stats_new['ban_index'].cpu().tolist()) if stats_new['ban_index'].numel() > 0 else set()
    ban_old = set(stats_old['ban_index'].cpu().tolist()) if stats_old['ban_index'].numel() > 0 else set()
    print(f"\nBan index match: {ban_new == ban_old}")
    if ban_new != ban_old:
        print(f"  New: {ban_new}")
        print(f"  Old: {ban_old}")

    # Speed
    speedup = old_time / new_time
    print(f"\nSpeed:")
    print(f"  Old (GPU SVD): {old_time:.2f}s")
    print(f"  New (CPU pinv): {new_time:.2f}s")
    print(f"  Speedup: {speedup:.1f}x")

    # Verdict
    print("\n" + "=" * 60)
    rmsd_ok = rmsd_per_mol.max() < 1e-3
    speed_ok = speedup > 3
    ban_ok = ban_new == ban_old
    print(f"RMSD < 1e-3:    {'PASS' if rmsd_ok else 'FAIL'} (max={rmsd_per_mol.max():.2e})")
    print(f"Speedup > 3x:   {'PASS' if speed_ok else 'FAIL'} ({speedup:.1f}x)")
    print(f"Ban index match: {'PASS' if ban_ok else 'FAIL'}")
    print(f"Overall:         {'PASS' if (rmsd_ok and speed_ok and ban_ok) else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
