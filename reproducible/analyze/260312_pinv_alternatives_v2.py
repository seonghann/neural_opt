"""
Analysis: pinv Alternatives v2 — Eigendecomposition Approach
Date: 2026-03-12
Related progress log: docs/progress/260312_riemannian_noise_optimization.md

Problem:
    SVD-based pinv takes 98% of ODE solver time.
    v1 showed Cholesky is fast but doesn't match SVD truncation semantics.
    Test eigendecomposition-based approaches that correctly truncate
    small singular values while being faster than full SVD.

    Key insight: J is (B, e, 3n) with e > 3n.
    JtJ = J^T @ J is (B, 3n, 3n) — symmetric positive semi-definite.
    eigh(JtJ) gives eigenvalues = sigma^2, eigenvectors = V.
    Truncated pinv = V @ diag(1/sigma, truncated) @ V^T @ J^T / sigma

Judgment Criteria:
    1. Speed: > 10x faster than vmap(pinv)
    2. Accuracy: rel error < 1e-3 vs SVD-based pinv
    3. Christoffel x_ddot rel error < 1e-3

Conclusion:
    (filled after running)

Usage: /venv/neural_opt/bin/python analyze/260312_pinv_alternatives_v2.py
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
)
from src.manifold.graph import MolGraph


def pinv_vmap_svd(J, rtol=1e-4, atol=1e-2):
    """Original: vmap(torch.linalg.pinv)"""
    return vmap(torch.linalg.pinv)(J, rtol=rtol, atol=atol)


def pinv_eigh(J, rtol=1e-4, atol=1e-2):
    """Eigendecomposition-based pinv.
    J: (B, e, 3n) with e > 3n
    JtJ = J^T J: (B, 3n, 3n), symmetric PSD
    eigenvalues of JtJ = sigma^2
    pinv(J) = V @ diag(1/sigma_trunc) @ V^T @ J^T / sigma_trunc
    """
    B, e, n3 = J.shape
    Jt = J.transpose(-1, -2)  # (B, 3n, e)
    JtJ = torch.bmm(Jt, J)  # (B, 3n, 3n)

    # eigh returns eigenvalues in ascending order
    eigvals, V = torch.linalg.eigh(JtJ)  # eigvals: (B, 3n), V: (B, 3n, 3n)

    # sigma = sqrt(max(eigvals, 0))
    sigma_sq = eigvals.clamp(min=0)  # (B, 3n)
    sigma = sigma_sq.sqrt()

    # Truncation mask: same semantics as torch.linalg.pinv
    # threshold = max(atol, rtol * sigma_max)
    sigma_max = sigma.max(dim=-1, keepdim=True).values  # (B, 1)
    threshold = torch.clamp(rtol * sigma_max, min=atol)  # (B, 1)
    mask = sigma > threshold  # (B, 3n)

    # Compute 1/sigma where mask is True, else 0
    inv_sigma = torch.where(mask, 1.0 / sigma.clamp(min=1e-30), torch.zeros_like(sigma))  # (B, 3n)

    # pinv(J) = V @ diag(1/sigma^2) @ V^T @ J^T
    # = V @ diag(inv_sigma^2) @ (V^T @ J^T)
    VtJt = torch.bmm(V.transpose(-1, -2), Jt)  # (B, 3n, e)
    scaled = inv_sigma.unsqueeze(-1) ** 2 * VtJt  # (B, 3n, e)
    result = torch.bmm(V, scaled)  # (B, 3n, e)
    return result


def pinv_eigh_v2(J, rtol=1e-4, atol=1e-2):
    """Eigendecomposition-based pinv, alternative formulation.
    pinv(J) = V @ diag(1/sigma_sq_trunc) @ V^T @ J^T
    Directly uses eigenvalues (no sqrt).
    """
    B, e, n3 = J.shape
    Jt = J.transpose(-1, -2)  # (B, 3n, e)
    JtJ = torch.bmm(Jt, J)  # (B, 3n, 3n)

    eigvals, V = torch.linalg.eigh(JtJ)  # eigvals: (B, 3n), V: (B, 3n, 3n)

    sigma_sq = eigvals.clamp(min=0)
    sigma = sigma_sq.sqrt()

    sigma_max = sigma.max(dim=-1, keepdim=True).values
    threshold = torch.clamp(rtol * sigma_max, min=atol)
    mask = sigma > threshold

    # 1/sigma_sq where valid
    inv_sigma_sq = torch.where(mask, 1.0 / sigma_sq.clamp(min=1e-30), torch.zeros_like(sigma_sq))

    VtJt = torch.bmm(V.transpose(-1, -2), Jt)  # (B, 3n, e)
    scaled = inv_sigma_sq.unsqueeze(-1) * VtJt  # (B, 3n, e)
    result = torch.bmm(V, scaled)  # (B, 3n, e)
    return result


def pinv_svd_manual(J, rtol=1e-4, atol=1e-2):
    """Manual truncated SVD (batched, no vmap).
    Compute SVD, apply truncation, reconstruct pinv.
    """
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    # U: (B, e, k), S: (B, k), Vh: (B, k, 3n) where k = min(e, 3n)

    S_max = S.max(dim=-1, keepdim=True).values
    threshold = torch.clamp(rtol * S_max, min=atol)
    mask = S > threshold

    inv_S = torch.where(mask, 1.0 / S.clamp(min=1e-30), torch.zeros_like(S))

    # pinv = Vh^T @ diag(1/S) @ U^T
    return torch.bmm(Vh.transpose(-1, -2) * inv_S.unsqueeze(-2), U.transpose(-1, -2))


def pinv_svd_manual_reduced(J, rtol=1e-4, atol=1e-2):
    """Manual truncated SVD with reduced computation.
    Since we only need pinv(J) = V @ diag(1/S) @ U^T,
    and J is (B, e, 3n) with e >> 3n, use full_matrices=False.
    """
    # economy SVD: U(B,e,3n), S(B,3n), Vh(B,3n,3n)
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)

    S_max = S.max(dim=-1, keepdim=True).values
    threshold = torch.clamp(rtol * S_max, min=atol)
    mask = S > threshold

    inv_S = torch.where(mask, 1.0 / S.clamp(min=1e-30), torch.zeros_like(S))

    # pinv = Vh^T @ diag(1/S) @ U^T = (B, 3n, 3n) @ (B, 3n, e) = (B, 3n, e)
    return torch.bmm(Vh.transpose(-1, -2) * inv_S.unsqueeze(-2), U.transpose(-1, -2))


def pinv_cholesky_eig_truncated(J, rtol=1e-4, atol=1e-2):
    """Cholesky-like but with eigenvalue truncation.
    Uses eigh to find eigenvalues, truncates, then reconstructs.
    Same as pinv_eigh_v2 but might use different numerical path.
    """
    B, e, n3 = J.shape
    Jt = J.transpose(-1, -2)
    JtJ = torch.bmm(Jt, J)

    # Force symmetry
    JtJ = (JtJ + JtJ.transpose(-1, -2)) / 2

    eigvals, V = torch.linalg.eigh(JtJ)
    sigma = eigvals.clamp(min=0).sqrt()
    sigma_max = sigma.max(dim=-1, keepdim=True).values
    threshold = torch.clamp(rtol * sigma_max, min=atol)
    mask = sigma > threshold

    inv_eigvals = torch.where(mask, 1.0 / eigvals.clamp(min=1e-30), torch.zeros_like(eigvals))

    # pinv = V diag(1/lambda) V^T J^T
    VtJt = torch.bmm(V.transpose(-1, -2), Jt)
    scaled = inv_eigvals.unsqueeze(-1) * VtJt
    return torch.bmm(V, scaled)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = OmegaConf.load('data/qm9m/riemannian_data_sampling/riemannian_data_sampling.yaml')
    config.train.batch_size = 100
    dm = load_datamodule(config)
    test_dl = dm.test_dataloader()

    solver_config = OmegaConf.create({
        'ode_solver': {'alpha': 1.7, 'beta': 0.01, 'gamma': 0.0, 'svd_tol': 1e-2}
    })
    solver = GeodesicSolver(solver_config)

    batch_data = next(iter(test_dl))
    batch_data = batch_data.to(device)

    graph = MolGraph.from_batch(batch_data)
    edge_index = graph.full_edge(upper_triangle=True)[0]
    pos = batch_data.pos[:, 0, :]
    atom_type = batch_data.x
    batch_idx = graph.batch
    num_nodes = batch_data.ptr[1:] - batch_data.ptr[:-1]

    B = num_nodes.size(0)
    n = num_nodes.max().item()
    e = n * (n - 1) // 2

    index_tensor = redefine_edge_index(edge_index, batch_idx, num_nodes)
    x = redefine_with_pad(pos, batch_idx)
    at = redefine_with_pad(atom_type, batch_idx, padding_value=-1)

    J_sparse = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=at)
    J_dense = J_sparse.to_dense()
    print(f"B={B}, n={n}, e={e}, J shape={J_dense.shape}")

    # Reference
    print("\nComputing reference (vmap SVD)...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ref = pinv_vmap_svd(J_dense.clone())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    ref_time = time.perf_counter() - t0
    print(f"  vmap SVD: {ref_time*1000:.1f}ms")

    # Test methods
    methods = [
        ("vmap(pinv)", lambda: pinv_vmap_svd(J_dense, rtol=1e-4, atol=1e-2)),
        ("eigh (sqrt)", lambda: pinv_eigh(J_dense, rtol=1e-4, atol=1e-2)),
        ("eigh (direct)", lambda: pinv_eigh_v2(J_dense, rtol=1e-4, atol=1e-2)),
        ("eigh (trunc)", lambda: pinv_cholesky_eig_truncated(J_dense, rtol=1e-4, atol=1e-2)),
        ("SVD manual", lambda: pinv_svd_manual(J_dense, rtol=1e-4, atol=1e-2)),
        ("SVD reduced", lambda: pinv_svd_manual_reduced(J_dense, rtol=1e-4, atol=1e-2)),
    ]

    print(f"\n{'Method':<20s} {'Time (ms)':>10s} {'Speedup':>8s} {'Rel Error':>12s} {'Max Error':>12s}")
    print("-" * 70)

    best_name, best_time, best_fn = None, float('inf'), None

    for name, fn in methods:
        try:
            result = fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except Exception as ex:
            print(f"  {name:<18s}  FAILED: {ex}")
            continue

        times = []
        for _ in range(5):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        speedup = ref_time * 1000 / avg_ms

        rel_err = (result - ref).norm() / ref.norm()
        max_err = (result - ref).abs().max()

        marker = ""
        if rel_err < 1e-3 and avg_ms < best_time:
            best_name, best_time, best_fn = name, avg_ms, fn
            marker = " <-- BEST"
        elif rel_err < 1e-3:
            marker = " (accurate)"

        print(f"  {name:<18s} {avg_ms:>10.1f} {speedup:>7.1f}x {rel_err:>11.2e} {max_err:>11.2e}{marker}")

    # End-to-end validation with best method
    if best_fn is not None:
        print(f"\n\nEnd-to-end validation with '{best_name}':")
        done = torch.zeros(B, dtype=torch.bool, device=device)
        x_dot = torch.randn_like(x) * 0.01
        dt = torch.full((B,), 0.01, device=device)
        not_done_index = torch.where(~done)[0]
        J_for_advance = J_sparse.index_select(0, not_done_index)

        # Original _advance
        dx_ref, dx_dot_ref = solver._advance(done, x.clone(), x_dot.clone(), index_tensor, at, dt, q_type="morse", J=J_for_advance)

        # _advance with modified pinv (monkey-patch temporarily)
        import src.manifold.solver as solver_module
        original_pinv = solver_module.batch_pinv1

        if 'eigh' in best_name:
            if 'direct' in best_name:
                solver_module.batch_pinv1 = lambda J, rtol=1e-4, atol=None: pinv_eigh_v2(J.to_dense() if J.is_sparse else J, rtol=rtol, atol=atol if atol else 1e-2)
            else:
                solver_module.batch_pinv1 = lambda J, rtol=1e-4, atol=None: pinv_eigh(J.to_dense() if J.is_sparse else J, rtol=rtol, atol=atol if atol else 1e-2)
        elif 'SVD' in best_name:
            solver_module.batch_pinv1 = lambda J, rtol=1e-4, atol=None: pinv_svd_manual_reduced(J.to_dense() if J.is_sparse else J, rtol=rtol, atol=atol if atol else 1e-2)

        J_for_advance2 = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=at)
        J_for_advance2 = J_for_advance2.index_select(0, not_done_index)
        dx_new, dx_dot_new = solver._advance(done, x.clone(), x_dot.clone(), index_tensor, at, dt, q_type="morse", J=J_for_advance2)

        # Restore
        solver_module.batch_pinv1 = original_pinv

        dx_err = (dx_new - dx_ref).norm() / dx_ref.norm()
        dx_dot_err = (dx_dot_new - dx_dot_ref).norm() / dx_dot_ref.norm()
        print(f"  dx relative error:     {dx_err:.2e}")
        print(f"  dx_dot relative error: {dx_dot_err:.2e}")

        # Full _advance timing comparison
        print(f"\n  Full _advance timing comparison:")

        # Original
        times = []
        for _ in range(3):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            J_ = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=at)
            J_ = J_.index_select(0, not_done_index)
            solver._advance(done, x.clone(), x_dot.clone(), index_tensor, at, dt, q_type="morse", J=J_)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        orig_ms = sum(times) / len(times) * 1000

        # Modified
        if 'eigh' in best_name:
            if 'direct' in best_name:
                solver_module.batch_pinv1 = lambda J, rtol=1e-4, atol=None: pinv_eigh_v2(J.to_dense() if J.is_sparse else J, rtol=rtol, atol=atol if atol else 1e-2)
            else:
                solver_module.batch_pinv1 = lambda J, rtol=1e-4, atol=None: pinv_eigh(J.to_dense() if J.is_sparse else J, rtol=rtol, atol=atol if atol else 1e-2)
        elif 'SVD' in best_name:
            solver_module.batch_pinv1 = lambda J, rtol=1e-4, atol=None: pinv_svd_manual_reduced(J.to_dense() if J.is_sparse else J, rtol=rtol, atol=atol if atol else 1e-2)

        times = []
        for _ in range(3):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            J_ = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=at)
            J_ = J_.index_select(0, not_done_index)
            solver._advance(done, x.clone(), x_dot.clone(), index_tensor, at, dt, q_type="morse", J=J_)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        new_ms = sum(times) / len(times) * 1000

        solver_module.batch_pinv1 = original_pinv

        print(f"  Original _advance: {orig_ms:.1f}ms")
        print(f"  Modified _advance: {new_ms:.1f}ms")
        print(f"  Speedup:           {orig_ms/new_ms:.1f}x")

    print("\n[Done]")


if __name__ == "__main__":
    main()
