"""
Analysis: CPU vs GPU pinv — SVD on CPU with multiprocessing
Date: 2026-03-12
Related progress log: docs/progress/260312_riemannian_noise_optimization.md

Problem:
    GPU SVD on small matrices (300x75) is extremely slow (5.5s for B=100).
    Test whether CPU pinv with multi-core parallelism is faster.
    Strategy: GPU→CPU transfer, CPU pinv, CPU→GPU transfer.

Judgment Criteria:
    1. CPU pinv + transfer overhead < GPU pinv (5.5s)
    2. Numerical match: rel error < 1e-5 vs GPU SVD
    3. Scaling: test B=10,50,100,200

Conclusion:
    (filled after running)

Usage: /venv/neural_opt/bin/python analyze/260312_pinv_cpu_benchmark.py
"""

import sys
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import vmap
from omegaconf import OmegaConf
from dataset.data_module import load_datamodule
from manifold.solver import GeodesicSolver, redefine_edge_index, redefine_with_pad
from manifold.graph import MolGraph


def main():
    device = torch.device('cuda')
    print(f"CPU cores: {os.cpu_count()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")

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

    index_tensor = redefine_edge_index(edge_index, batch_idx, num_nodes)
    x = redefine_with_pad(pos, batch_idx)
    at = redefine_with_pad(atom_type, batch_idx, padding_value=-1)

    J_sparse = solver.sparse_batch_jacobian_q(index_tensor, x, atom_type=at)
    J_gpu = J_sparse.to_dense()  # (B, e, 3n) on GPU
    print(f"J shape: {J_gpu.shape}, dtype: {J_gpu.dtype}")

    # --- Benchmark GPU pinv (reference) ---
    print("\n=== GPU pinv (vmap SVD) ===")
    # warmup
    _ = vmap(torch.linalg.pinv)(J_gpu, rtol=1e-4, atol=1e-2)
    torch.cuda.synchronize()

    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref_gpu = vmap(torch.linalg.pinv)(J_gpu, rtol=1e-4, atol=1e-2)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    gpu_ms = sum(times) / len(times) * 1000
    print(f"  Time: {gpu_ms:.1f}ms")

    # --- Benchmark CPU pinv (native batched) ---
    print("\n=== CPU pinv (batched, native threads) ===")
    for num_threads in [1, 16, 64, 128, 256, 512]:
        torch.set_num_threads(num_threads)

        J_cpu = J_gpu.cpu()  # warmup transfer
        _ = torch.linalg.pinv(J_cpu, rtol=1e-4, atol=1e-2)

        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            J_cpu = J_gpu.cpu()
            result_cpu = torch.linalg.pinv(J_cpu, rtol=1e-4, atol=1e-2)
            result_back = result_cpu.to(device)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        total_ms = sum(times) / len(times) * 1000

        # Breakdown: transfer + compute
        times_transfer = []
        times_compute = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            J_cpu = J_gpu.cpu()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            result_cpu = torch.linalg.pinv(J_cpu, rtol=1e-4, atol=1e-2)
            t2 = time.perf_counter()
            result_back = result_cpu.to(device)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            times_transfer.append((t1 - t0 + t3 - t2) * 1000)
            times_compute.append((t2 - t1) * 1000)

        xfer_ms = sum(times_transfer) / len(times_transfer)
        comp_ms = sum(times_compute) / len(times_compute)

        # Accuracy
        err = (result_back - ref_gpu).norm() / ref_gpu.norm()
        speedup = gpu_ms / total_ms

        print(f"  threads={num_threads:>3d}: total={total_ms:>8.1f}ms  (xfer={xfer_ms:.1f}ms + compute={comp_ms:.1f}ms)  {speedup:>6.1f}x  err={err:.2e}")

    # --- Benchmark CPU vmap pinv ---
    print("\n=== CPU vmap(pinv) ===")
    for num_threads in [64, 256]:
        torch.set_num_threads(num_threads)
        J_cpu = J_gpu.cpu()
        _ = vmap(torch.linalg.pinv)(J_cpu, rtol=1e-4, atol=1e-2)

        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            J_cpu = J_gpu.cpu()
            result_cpu = vmap(torch.linalg.pinv)(J_cpu, rtol=1e-4, atol=1e-2)
            result_back = result_cpu.to(device)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        total_ms = sum(times) / len(times) * 1000
        err = (result_back - ref_gpu).norm() / ref_gpu.norm()
        speedup = gpu_ms / total_ms
        print(f"  threads={num_threads:>3d}: total={total_ms:>8.1f}ms  {speedup:>6.1f}x  err={err:.2e}")

    # --- Also test CPU eigh-based approach ---
    print("\n=== CPU eigh-based pinv ===")
    def pinv_eigh_cpu(J, rtol=1e-4, atol=1e-2):
        Jt = J.transpose(-1, -2)
        JtJ = torch.bmm(Jt, J)
        eigvals, V = torch.linalg.eigh(JtJ)
        sigma = eigvals.clamp(min=0).sqrt()
        sigma_max = sigma.max(dim=-1, keepdim=True).values
        threshold = torch.clamp(rtol * sigma_max, min=atol)
        mask = sigma > threshold
        inv_eigvals = torch.where(mask, 1.0 / eigvals.clamp(min=1e-30), torch.zeros_like(eigvals))
        VtJt = torch.bmm(V.transpose(-1, -2), Jt)
        return torch.bmm(V, inv_eigvals.unsqueeze(-1) * VtJt)

    for num_threads in [64, 256]:
        torch.set_num_threads(num_threads)
        J_cpu = J_gpu.cpu()
        _ = pinv_eigh_cpu(J_cpu)

        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            J_cpu = J_gpu.cpu()
            result_cpu = pinv_eigh_cpu(J_cpu)
            result_back = result_cpu.to(device)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        total_ms = sum(times) / len(times) * 1000
        err = (result_back - ref_gpu).norm() / ref_gpu.norm()
        speedup = gpu_ms / total_ms
        print(f"  threads={num_threads:>3d}: total={total_ms:>8.1f}ms  {speedup:>6.1f}x  err={err:.2e}")

    # --- Also run the eigh on GPU for comparison ---
    print("\n=== GPU eigh-based pinv ===")
    def pinv_eigh_gpu(J, rtol=1e-4, atol=1e-2):
        Jt = J.transpose(-1, -2)
        JtJ = torch.bmm(Jt, J)
        eigvals, V = torch.linalg.eigh(JtJ)
        sigma = eigvals.clamp(min=0).sqrt()
        sigma_max = sigma.max(dim=-1, keepdim=True).values
        threshold = torch.clamp(rtol * sigma_max, min=atol)
        mask = sigma > threshold
        inv_eigvals = torch.where(mask, 1.0 / eigvals.clamp(min=1e-30), torch.zeros_like(eigvals))
        VtJt = torch.bmm(V.transpose(-1, -2), Jt)
        return torch.bmm(V, inv_eigvals.unsqueeze(-1) * VtJt)

    _ = pinv_eigh_gpu(J_gpu)
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pinv_eigh_gpu(J_gpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    eigh_gpu_ms = sum(times) / len(times) * 1000
    err = (result - ref_gpu).norm() / ref_gpu.norm()
    print(f"  Time: {eigh_gpu_ms:.1f}ms  {gpu_ms/eigh_gpu_ms:.1f}x  err={err:.2e}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
