"""
Analysis: ODE Solver Component-Level Profiling
Date: 2026-03-12
Related progress log: docs/progress/260312_riemannian_noise_optimization.md

Problem:
    Measure exact wall-clock time breakdown within batch_geodesic_ode_solve()
    to identify which sub-operations are the true bottleneck:
    - sparse_batch_jacobian_q (computation + index_select)
    - sparse_batch_hessian_q (computation + index_select)
    - to_dense() conversions (Jacobian, Hessian)
    - batch_pinv1 (SVD via vmap)
    - bmm (Hessian @ J_inv)
    - einsum (Christoffel contraction)

Judgment Criteria:
    1. Identify which component(s) take >30% of per-step time
    2. Confirm whether sparse->dense conversion is significant
    3. Determine if pinv (SVD) dominates as hypothesized

Conclusion:
    (filled after running)

Usage: /venv/ASBS_TB/bin/python analyze/260312_solver_profiling.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import vmap
from omegaconf import OmegaConf
from src.dataset.data_module import load_datamodule
from src.manifold.solver import (
    GeodesicSolver,
    redefine_edge_index,
    redefine_with_pad,
    batch_pinv1,
)
from src.manifold.graph import MolGraph


def profile_single_advance(solver, x, x_dot, index_tensor, atom_type, done, dt, device):
    """Profile one call to _advance() with component-level timing."""
    B = (~done).sum()
    not_done_index = torch.where(~done)[0]
    n = x.size(1)
    n3 = n * 3

    timings = {}

    # 1. Jacobian computation (sparse)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    J_sparse = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=atom_type)
    J_sparse = J_sparse.index_select(0, not_done_index)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['jacobian_sparse'] = time.perf_counter() - t0

    # 2. J.to_dense() for q_dot computation (done in batch_advance_heun)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    J_dense = J_sparse.to_dense()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['J_to_dense'] = time.perf_counter() - t0

    # 3. Hessian computation (sparse)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    hess_sparse = solver.sparse_batch_hessian_q(index_tensor, x, atom_type=atom_type)
    hess_sparse = hess_sparse.index_select(0, not_done_index)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['hessian_sparse'] = time.perf_counter() - t0

    # 4. Hessian to_dense() (done inside batch_christoffel via bmm)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    hess_dense = hess_sparse.to_dense()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['H_to_dense'] = time.perf_counter() - t0

    # 5. batch_pinv1 (SVD via vmap) — note: also calls J.to_dense() internally
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    J_inv = batch_pinv1(J_sparse, rtol=1e-4, atol=solver.svd_tol).transpose(-1, -2)  # (B, e, 3n)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['pinv_svd'] = time.perf_counter() - t0

    # 5b. pinv with pre-densified J (isolate SVD from to_dense)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    J_inv2 = vmap(torch.linalg.pinv)(J_dense, rtol=1e-4, atol=solver.svd_tol).transpose(-1, -2)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['pinv_svd_dense_input'] = time.perf_counter() - t0

    # 6. bmm: hess @ J_inv
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    christoffel = torch.bmm(hess_dense, J_inv).transpose(-1, -2).reshape(B, n3, n3, n3)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['bmm_christoffel'] = time.perf_counter() - t0

    # 7. einsum contraction
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_ddot = -torch.einsum("bj,bkij,bi->bk",
                            x_dot[~done].reshape(B, -1),
                            christoffel,
                            x_dot[~done].reshape(B, -1))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['einsum'] = time.perf_counter() - t0

    # 8. Full _advance() call for reference total
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    J2 = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=atom_type)
    J2 = J2.index_select(0, not_done_index)
    solver._advance(done, x.clone(), x_dot.clone(), index_tensor, atom_type, dt, q_type="morse", J=J2)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['full_advance'] = time.perf_counter() - t0

    return timings


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config and data
    config = OmegaConf.load('data/qm9m/riemannian_data_sampling/riemannian_data_sampling.yaml')
    config.train.batch_size = 100
    dm = load_datamodule(config)
    test_dl = dm.test_dataloader()

    solver_config = OmegaConf.create({
        'ode_solver': {
            'alpha': 1.7,
            'beta': 0.01,
            'gamma': 0.0,
            'svd_tol': 1e-2,
        }
    })
    solver = GeodesicSolver(solver_config)

    # Get one batch
    batch_data = next(iter(test_dl))
    batch_data = batch_data.to(device)

    graph = MolGraph.from_batch(batch_data)
    edge_index = graph.full_edge(upper_triangle=True)[0]
    pos = batch_data.pos[:, 0, :]  # DFT positions
    atom_type = batch_data.x  # atom type stored as .x in PyG
    batch_idx = graph.batch
    num_nodes = batch_data.ptr[1:] - batch_data.ptr[:-1]

    B = num_nodes.size(0)
    n = num_nodes.max().item()
    e = n * (n - 1) // 2
    print(f"Batch: B={B}, n={n}, e={e}")

    # Prepare batched tensors
    index_tensor = redefine_edge_index(edge_index, batch_idx, num_nodes)
    x = redefine_with_pad(pos, batch_idx)  # (B, n, 3)
    at = redefine_with_pad(atom_type, batch_idx, padding_value=-1)  # (B, n)

    # Create dummy x_dot and done
    x_dot = torch.randn_like(x) * 0.01
    done = torch.zeros(B, dtype=torch.bool, device=device)
    dt = torch.full((B,), 0.01, device=device)

    # Warm up
    print("\nWarming up...")
    for _ in range(2):
        J = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=at)
        not_done_index = torch.where(~done)[0]
        J = J.index_select(0, not_done_index)
        solver._advance(done, x.clone(), x_dot.clone(), index_tensor, at, dt, q_type="morse", J=J)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Profile multiple runs
    n_runs = 5
    print(f"\nProfiling {n_runs} runs...")
    all_timings = []
    for run in range(n_runs):
        timings = profile_single_advance(solver, x, x_dot, index_tensor, at, done, dt, device)
        all_timings.append(timings)
        print(f"  Run {run}: full_advance={timings['full_advance']:.4f}s")

    # Aggregate
    keys = list(all_timings[0].keys())
    avg = {k: sum(t[k] for t in all_timings) / n_runs for k in keys}

    # Report
    total = avg['full_advance']
    print("\n" + "=" * 70)
    print(f"PROFILING RESULTS (B={B}, n={n}, e={e}, device={device})")
    print(f"{'Component':<30s} {'Time (ms)':>10s} {'% of total':>10s}")
    print("-" * 70)

    component_order = [
        ('jacobian_sparse', 'Jacobian (sparse compute)'),
        ('J_to_dense', 'J.to_dense()'),
        ('hessian_sparse', 'Hessian (sparse compute)'),
        ('H_to_dense', 'H.to_dense()'),
        ('pinv_svd', 'pinv (SVD, sparse input)'),
        ('pinv_svd_dense_input', 'pinv (SVD, dense input)'),
        ('bmm_christoffel', 'bmm (H @ J_inv)'),
        ('einsum', 'einsum contraction'),
        ('full_advance', 'TOTAL (_advance)'),
    ]

    for key, label in component_order:
        ms = avg[key] * 1000
        pct = avg[key] / total * 100 if total > 0 else 0
        marker = " <<<" if pct > 25 and key != 'full_advance' else ""
        print(f"  {label:<28s} {ms:>10.2f} {pct:>9.1f}%{marker}")

    print("=" * 70)

    # Derived insights
    sparse_overhead = avg['J_to_dense'] + avg['H_to_dense']
    svd_only = avg['pinv_svd_dense_input']
    print(f"\nDerived insights:")
    print(f"  Sparse->dense total:  {sparse_overhead*1000:.2f}ms ({sparse_overhead/total*100:.1f}%)")
    print(f"  SVD-only (no to_dense): {svd_only*1000:.2f}ms ({svd_only/total*100:.1f}%)")
    print(f"  pinv overhead from to_dense: {(avg['pinv_svd']-svd_only)*1000:.2f}ms")

    # Also test different batch sizes using pre-computed padded tensors
    print("\n\nBATCH SIZE SCALING:")
    print(f"{'Batch':>6s} {'advance_ms':>12s} {'per_mol_ms':>12s}")
    print("-" * 40)
    for bs in [10, 25, 50, 100]:
        if bs > B:
            continue
        done_sub = torch.zeros(bs, dtype=torch.bool, device=device)
        dt_sub = dt[:bs]

        # Subset padded tensors and rebuild index_tensor
        mask = batch_idx < bs
        pos_sub = pos[mask]
        ei_mask = batch_idx[edge_index[0]] < bs
        ei_sub = edge_index[:, ei_mask]
        at_raw_sub = atom_type[mask]
        batch_sub = batch_idx[mask]
        nn_sub = num_nodes[:bs]
        it_sub = redefine_edge_index(ei_sub, batch_sub, nn_sub)
        x_sub = redefine_with_pad(pos_sub, batch_sub)
        at_sub = redefine_with_pad(at_raw_sub, batch_sub, padding_value=-1)
        x_dot_sub = torch.randn_like(x_sub) * 0.01

        # Warm up
        J_ = solver.sparse_batch_jacobian_q(it_sub, x_sub, atom_type=at_sub)
        ndi_ = torch.where(~done_sub)[0]
        J_ = J_.index_select(0, ndi_)
        solver._advance(done_sub, x_sub.clone(), x_dot_sub.clone(), it_sub, at_sub, dt_sub, q_type="morse", J=J_)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Time
        times = []
        for _ in range(3):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            J_ = solver.sparse_batch_jacobian_q(it_sub, x_sub, atom_type=at_sub)
            J_ = J_.index_select(0, ndi_)
            solver._advance(done_sub, x_sub.clone(), x_dot_sub.clone(), it_sub, at_sub, dt_sub, q_type="morse", J=J_)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_t = sum(times) / len(times) * 1000
        print(f"  {bs:>4d}   {avg_t:>10.1f}ms  {avg_t/bs:>10.02f}ms")

    print("\n[Done]")


if __name__ == "__main__":
    main()
