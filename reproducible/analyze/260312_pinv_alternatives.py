"""
Analysis: pinv Alternative Methods Benchmark
Date: 2026-03-12
Related progress log: docs/progress/260312_riemannian_noise_optimization.md

Problem:
    vmap(torch.linalg.pinv) consumes 98% of ODE solver time.
    Test alternative pseudo-inverse implementations:
    1. Native batched torch.linalg.pinv (no vmap)
    2. Normal equations: (J^T J)^{-1} J^T via Cholesky
    3. Normal equations: (J^T J + eps*I)^{-1} J^T via Cholesky (Tikhonov)
    4. QR decomposition based
    5. torch.linalg.lstsq

Judgment Criteria:
    1. Speed improvement relative to vmap(pinv)
    2. Numerical accuracy: ||pinv_new - pinv_svd||_F / ||pinv_svd||_F < 1e-3
    3. Must handle rank-deficient cases (svd_tol=1e-2)

Conclusion:
    (filled after running)

Usage: /venv/neural_opt/bin/python analyze/260312_pinv_alternatives.py
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


def pinv_native_batched(J, rtol=1e-4, atol=1e-2):
    """torch.linalg.pinv supports batch dimensions natively."""
    return torch.linalg.pinv(J, rtol=rtol, atol=atol)


def pinv_cholesky(J, atol=1e-2):
    """Normal equations with Cholesky: pinv(J) = (J^T J + eps I)^{-1} J^T
    J: (B, e, 3n) where e > 3n (over-determined)
    """
    B, e, n3 = J.shape
    JtJ = torch.bmm(J.transpose(-1, -2), J)  # (B, 3n, 3n)
    reg = atol * torch.eye(n3, device=J.device, dtype=J.dtype).unsqueeze(0)
    L = torch.linalg.cholesky(JtJ + reg)  # (B, 3n, 3n)
    Jt = J.transpose(-1, -2)  # (B, 3n, e)
    return torch.cholesky_solve(Jt, L)  # (B, 3n, e)


def pinv_cholesky_ex(J, atol=1e-2):
    """Cholesky with cholesky_ex (returns info for error checking)."""
    B, e, n3 = J.shape
    JtJ = torch.bmm(J.transpose(-1, -2), J)  # (B, 3n, 3n)
    reg = atol * torch.eye(n3, device=J.device, dtype=J.dtype).unsqueeze(0)
    L, info = torch.linalg.cholesky_ex(JtJ + reg)  # (B, 3n, 3n)
    Jt = J.transpose(-1, -2)  # (B, 3n, e)
    return torch.cholesky_solve(Jt, L)  # (B, 3n, e)


def pinv_lu_solve(J, atol=1e-2):
    """Normal equations with LU solve: (J^T J + eps I) X = J^T"""
    B, e, n3 = J.shape
    JtJ = torch.bmm(J.transpose(-1, -2), J)  # (B, 3n, 3n)
    reg = atol * torch.eye(n3, device=J.device, dtype=J.dtype).unsqueeze(0)
    Jt = J.transpose(-1, -2)  # (B, 3n, e)
    return torch.linalg.solve(JtJ + reg, Jt)  # (B, 3n, e)


def pinv_qr(J, atol=1e-2):
    """QR-based pseudo-inverse.
    J^T = QR => pinv(J) = Q R^{-T}
    J: (B, e, 3n), J^T: (B, 3n, e)
    """
    Jt = J.transpose(-1, -2).contiguous()  # (B, 3n, e)
    Q, R = torch.linalg.qr(Jt, mode='reduced')  # Q: (B, 3n, 3n), R: (B, 3n, e)
    # pinv(J) = Q @ R^{-T} ... but R is (3n, e), R^T is (e, 3n)
    # Actually: J = R^T Q^T, so pinv(J) = Q (R^T)^{-1}
    # R^T: (B, e, 3n) — not square, can't invert directly
    # Use: pinv(J) = (J^T J)^{-1} J^T = (R^T Q^T Q R)^{-1} R^T Q^T = R^{-1} (R^T)^{-1} R^T Q^T = R^{-1} Q^T
    # Wait: J^T = QR, J = R^T Q^T
    # pinv(J) = (J^T J)^{-1} J^T = (QR R^T Q^T)^{-1} QR
    # This doesn't simplify nicely for non-square R.
    # Instead use lstsq approach: min ||J x - b|| for each column of I
    # Or: J^T = QR where Q:(B,3n,k), R:(B,k,e), k=min(3n,e)=3n
    # pinv(J) = Q @ pinv(R^T) ... still need pinv of R^T

    # Simpler: use QR of J directly
    # J: (B, e, 3n), QR: Q(B,e,3n), R(B,3n,3n)
    Q2, R2 = torch.linalg.qr(J, mode='reduced')  # Q2: (B, e, 3n), R2: (B, 3n, 3n)
    # pinv(J) = pinv(Q2 R2) = R2^{-1} Q2^T (since Q2^T Q2 = I)
    R2_inv = torch.linalg.solve_triangular(R2, torch.eye(R2.shape[-1], device=J.device, dtype=J.dtype).unsqueeze(0).expand_as(R2), upper=True)
    return torch.bmm(R2_inv, Q2.transpose(-1, -2))  # (B, 3n, e)


def pinv_lstsq(J, atol=1e-2):
    """Use lstsq to solve J @ X = I, giving pinv(J) = X."""
    B, e, n3 = J.shape
    I = torch.eye(e, device=J.device, dtype=J.dtype).unsqueeze(0).expand(B, -1, -1)
    result = torch.linalg.lstsq(J, I, rcond=atol)
    return result.solution  # (B, 3n, e)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
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
    print(f"B={B}, n={n}, e={e}, J shape=({B}, {e}, {3*n})")

    # Build J
    index_tensor = redefine_edge_index(edge_index, batch_idx, num_nodes)
    x = redefine_with_pad(pos, batch_idx)
    at = redefine_with_pad(atom_type, batch_idx, padding_value=-1)

    J_sparse = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=at)
    J_dense = J_sparse.to_dense()
    print(f"J_dense shape: {J_dense.shape}, dtype: {J_dense.dtype}")

    # Reference: vmap SVD
    print("\nComputing reference (vmap SVD)...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ref = pinv_vmap_svd(J_dense, rtol=1e-4, atol=1e-2)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    ref_time = time.perf_counter() - t0
    print(f"  vmap SVD: {ref_time*1000:.1f}ms")

    # Test alternatives
    methods = [
        ("vmap(pinv)", lambda: pinv_vmap_svd(J_dense, rtol=1e-4, atol=1e-2)),
        ("native pinv", lambda: pinv_native_batched(J_dense, rtol=1e-4, atol=1e-2)),
        ("cholesky", lambda: pinv_cholesky(J_dense, atol=1e-2)),
        ("cholesky_ex", lambda: pinv_cholesky_ex(J_dense, atol=1e-2)),
        ("LU solve", lambda: pinv_lu_solve(J_dense, atol=1e-2)),
        ("QR", lambda: pinv_qr(J_dense, atol=1e-2)),
        ("lstsq", lambda: pinv_lstsq(J_dense, atol=1e-2)),
    ]

    print(f"\n{'Method':<20s} {'Time (ms)':>10s} {'Speedup':>8s} {'Rel Error':>12s} {'Max Error':>12s}")
    print("-" * 70)

    for name, fn in methods:
        # Warm up
        try:
            result = fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except Exception as ex:
            print(f"  {name:<18s}  FAILED: {ex}")
            continue

        # Time (3 runs)
        times = []
        for _ in range(3):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        speedup = ref_time * 1000 / avg_ms

        # Accuracy
        rel_err = (result - ref).norm() / ref.norm()
        max_err = (result - ref).abs().max()

        print(f"  {name:<18s} {avg_ms:>10.1f} {speedup:>7.1f}x {rel_err:>11.2e} {max_err:>11.2e}")

    # Also test: does the Christoffel result change?
    print("\n\nEnd-to-end Christoffel validation:")
    print("Computing Christoffel with vmap SVD...")
    done = torch.zeros(B, dtype=torch.bool, device=device)
    x_dot = torch.randn_like(x) * 0.01
    not_done_index = torch.where(~done)[0]
    J_for_christoffel = J_sparse.index_select(0, not_done_index)

    # Original Christoffel
    christoffel_ref = solver.batch_christoffel(index_tensor, at, x, J_for_christoffel, not_done_index, q_type="morse")
    x_ddot_ref = -torch.einsum("bj,bkij,bi->bk", x_dot.reshape(B, -1), christoffel_ref, x_dot.reshape(B, -1))

    # Christoffel with Cholesky pinv
    print("Computing Christoffel with Cholesky pinv...")
    n3 = n * 3
    hess = solver.sparse_batch_hessian_q(index_tensor, x, atom_type=at)
    hess = hess.index_select(0, not_done_index)

    J_dense_sub = J_for_christoffel.to_dense()
    J_inv_chol = pinv_cholesky(J_dense_sub, atol=1e-2).transpose(-1, -2)  # (B, e, 3n)
    christoffel_chol = torch.bmm(hess.to_dense(), J_inv_chol).transpose(-1, -2).reshape(B, n3, n3, n3)
    x_ddot_chol = -torch.einsum("bj,bkij,bi->bk", x_dot.reshape(B, -1), christoffel_chol, x_dot.reshape(B, -1))

    # Compare
    christoffel_err = (christoffel_chol - christoffel_ref).norm() / christoffel_ref.norm()
    xddot_err = (x_ddot_chol - x_ddot_ref).norm() / x_ddot_ref.norm()
    print(f"  Christoffel relative error: {christoffel_err:.2e}")
    print(f"  x_ddot relative error:      {xddot_err:.2e}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
