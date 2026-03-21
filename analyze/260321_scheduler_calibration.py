"""
Analysis: Discrete vs Continuous Scheduler Calibration Verification
Date: 2026-03-21
Related progress log: docs/progress/260321_port_experiment_branches.md

Problem:
    Verify that SigmoidDiffusionScheduler (c=12, beta_start=1e-7, beta_end=10)
    was correctly calibrated to match TSDiffNoiseScheduler (beta_start=1e-7,
    beta_end=2e-3, num_steps=5000).

    The user's notebook (continuous_diffusion_scheduler.ipynb on
    origin/time_embedding_ablation) shows this calibration was already done.
    We need to verify:
      1. alpha_bar (discrete) vs alpha(t) (continuous) — are they equivalent?
      2. alpha_bar_sq vs alpha(t)^2 — which mapping is correct?
      3. sigma relationship: sqrt(1-alpha_bar^2) vs sigma(t)?

    Key: the discrete and continuous schedulers use DIFFERENT beta_end values.
    The continuous scheduler's beta_end=10 was chosen to make the alpha curves match
    after the integral/cumprod difference.

Judgment Criteria:
    - Max relative error < 1% at corresponding time points
    - t_disc/N ↔ t_cont mapping should be linear (simple conversion)

Conclusion:
    <filled after running>

Usage: python analyze/260321_scheduler_calibration.py
"""
import sys
sys.path.insert(0, "/home/seonghwan/RxnExpPipe/neural_opt")

import torch
import numpy as np

from src.diffusion.noise_scheduler import TSDiffNoiseScheduler
from src.diffusion.continuous_scheduler import SigmoidDiffusionScheduler


def print_table(title, headers, rows):
    """Print a formatted comparison table."""
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    header_str = "".join(f"{h:>16}" for h in headers)
    print(header_str)
    print("-" * 120)
    for row in rows:
        fmt = []
        for v in row:
            if isinstance(v, int):
                fmt.append(f"{v:>16d}")
            elif isinstance(v, str):
                fmt.append(f"{v:>16}")
            elif abs(v) < 1e-4 and v != 0:
                fmt.append(f"{v:>16.6e}")
            else:
                fmt.append(f"{v:>16.10f}")
        print("".join(fmt))


def main():
    num_steps = 5000

    # Discrete: beta_start=1e-7, beta_end=2e-3 (from config)
    discrete = TSDiffNoiseScheduler(
        beta_start=1e-7,
        beta_end=2e-3,
        num_diffusion_timesteps=num_steps,
        schedule_type="sigmoid",
    )

    # Continuous: beta_start=1e-7, beta_end=10 (calibrated default)
    continuous = SigmoidDiffusionScheduler(
        c=12,
        beta_start=1e-7,
        beta_end=10,
    )

    key_steps = [0, 10, 50, 100, 150, 300, 500, 1000, 1500, 2000, 2500, 3000, 4000, 4999]

    # ---- discrete alpha_bar = cumprod(1-beta) ----
    disc_timesteps = torch.arange(num_steps)
    alpha_bar = discrete.alphas  # shape [5000], already cumprod(1-beta)

    # ---- continuous alpha(t) at t = i/N ----
    t_cont = disc_timesteps.float() / num_steps
    alpha_cont = continuous.get_alpha(t_cont)
    sigma_cont = continuous.get_sigma(t_cont)

    # ---- TEST 1: alpha_bar vs alpha(t) ----
    print("=" * 120)
    print("TEST 1: alpha_bar (discrete) vs alpha(t) (continuous)")
    print("  discrete: cumprod(1-beta), beta from sigmoid(-6..6) * (2e-3 - 1e-7) + 1e-7")
    print("  continuous: exp(-0.5*integral(beta)), beta from sigmoid(12*(t-0.5)) * (10 - 1e-7) + 1e-7")
    print("=" * 120)

    rows = []
    for t in key_steps:
        a_d = alpha_bar[t].item()
        a_c = alpha_cont[t].item()
        rel_err = abs(a_d - a_c) / (abs(a_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", a_d, a_c, abs(a_d - a_c), f"{rel_err:.4f}%"])

    print_table("alpha_bar vs alpha(t)",
                ["t_disc", "t_cont", "alpha_bar", "alpha(t)", "abs_err", "rel_err"],
                rows)

    err1 = (alpha_bar - alpha_cont).abs() / (alpha_bar.abs() + 1e-30) * 100
    print(f"\n  Max relative error:  {err1.max().item():.4f}%")
    print(f"  Mean relative error: {err1.mean().item():.4f}%")

    # ---- TEST 2: alpha_bar vs alpha(t)^2 ----
    print("\n\n")
    print("=" * 120)
    print("TEST 2: alpha_bar (discrete) vs alpha(t)^2 (continuous)")
    print("  From user's notebook: alpha_bar_sq (discrete) matches alpha_sq (continuous)")
    print("  So alpha_bar might match alpha(t)^2 ?")
    print("=" * 120)

    alpha_cont_sq = alpha_cont ** 2

    rows = []
    for t in key_steps:
        a_d = alpha_bar[t].item()
        a_c_sq = alpha_cont_sq[t].item()
        rel_err = abs(a_d - a_c_sq) / (abs(a_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", a_d, a_c_sq, abs(a_d - a_c_sq), f"{rel_err:.4f}%"])

    print_table("alpha_bar vs alpha(t)^2",
                ["t_disc", "t_cont", "alpha_bar", "alpha(t)^2", "abs_err", "rel_err"],
                rows)

    err2 = (alpha_bar - alpha_cont_sq).abs() / (alpha_bar.abs() + 1e-30) * 100
    print(f"\n  Max relative error:  {err2.max().item():.4f}%")
    print(f"  Mean relative error: {err2.mean().item():.4f}%")

    # ---- TEST 3: alpha_bar^2 vs alpha(t)^2 ----
    print("\n\n")
    print("=" * 120)
    print("TEST 3: alpha_bar^2 (discrete) vs alpha(t)^2 (continuous)")
    print("  Alternative: discrete alpha_bar_sq = alpha_bar^2, continuous alpha_sq = alpha(t)^2")
    print("=" * 120)

    alpha_bar_sq = alpha_bar ** 2

    rows = []
    for t in key_steps:
        a_d_sq = alpha_bar_sq[t].item()
        a_c_sq = alpha_cont_sq[t].item()
        rel_err = abs(a_d_sq - a_c_sq) / (abs(a_d_sq) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", a_d_sq, a_c_sq, abs(a_d_sq - a_c_sq), f"{rel_err:.4f}%"])

    print_table("alpha_bar^2 vs alpha(t)^2",
                ["t_disc", "t_cont", "alpha_bar^2", "alpha(t)^2", "abs_err", "rel_err"],
                rows)

    err3 = (alpha_bar_sq - alpha_cont_sq).abs() / (alpha_bar_sq.abs() + 1e-30) * 100
    print(f"\n  Max relative error:  {err3.max().item():.4f}%")
    print(f"  Mean relative error: {err3.mean().item():.4f}%")

    # ---- TEST 4: sigma comparison ----
    print("\n\n")
    print("=" * 120)
    print("TEST 4: sigma comparison")
    print("  discrete sigma = sqrt(1 - alpha_bar^2)")
    print("  continuous sigma = sqrt(1 - alpha(t)^2)  [from get_sigma]")
    print("=" * 120)

    sigma_disc = torch.sqrt(1 - alpha_bar ** 2)

    rows = []
    for t in key_steps:
        s_d = sigma_disc[t].item()
        s_c = sigma_cont[t].item()
        rel_err = abs(s_d - s_c) / (abs(s_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", s_d, s_c, abs(s_d - s_c), f"{rel_err:.4f}%"])

    print_table("sigma: sqrt(1-alpha_bar^2) vs sigma(t)",
                ["t_disc", "t_cont", "sigma_disc", "sigma(t)", "abs_err", "rel_err"],
                rows)

    err4 = (sigma_disc - sigma_cont).abs() / (sigma_disc.abs() + 1e-30) * 100
    print(f"\n  Max relative error:  {err4.max().item():.4f}%")
    print(f"  Mean relative error: {err4.mean().item():.4f}%")

    # ---- TEST 5: Key conversion point: t_disc=150 ↔ t_cont=0.03 ----
    print("\n\n")
    print("=" * 120)
    print("TEST 5: Key conversion — t_disc=150 → t_cont=0.03")
    print("=" * 120)

    t_disc_150 = 150
    t_cont_003 = torch.tensor([0.03])

    a_d_150 = alpha_bar[t_disc_150].item()
    a_c_003 = continuous.get_alpha(t_cont_003).item()
    s_d_150 = sigma_disc[t_disc_150].item()
    s_c_003 = continuous.get_sigma(t_cont_003).item()

    print(f"\n  alpha_bar[150]  = {a_d_150:.10f}")
    print(f"  alpha(0.03)     = {a_c_003:.10f}")
    print(f"  rel error       = {abs(a_d_150-a_c_003)/abs(a_d_150)*100:.4f}%")
    print()
    print(f"  sigma_disc[150] = {s_d_150:.10f}")
    print(f"  sigma(0.03)     = {s_c_003:.10f}")
    print(f"  rel error       = {abs(s_d_150-s_c_003)/abs(s_d_150)*100:.4f}%")

    # ---- SUMMARY ----
    print("\n\n")
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()
    tests = [
        ("alpha_bar vs alpha(t)", err1.max().item(), err1.mean().item()),
        ("alpha_bar vs alpha(t)^2", err2.max().item(), err2.mean().item()),
        ("alpha_bar^2 vs alpha(t)^2", err3.max().item(), err3.mean().item()),
        ("sigma_disc vs sigma(t)", err4.max().item(), err4.mean().item()),
    ]
    print(f"  {'Test':<35} {'Max Error':>12} {'Mean Error':>12} {'< 1%?':>8}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8}")
    for name, mx, mn in tests:
        verdict = "YES" if mx < 1.0 else "NO"
        print(f"  {name:<35} {mx:>11.4f}% {mn:>11.4f}% {verdict:>8}")

    print()
    best_test = min(tests, key=lambda x: x[1])
    print(f"  Best match: '{best_test[0]}' — max error = {best_test[1]:.4f}%")
    print()
    if best_test[1] < 1.0:
        print("  CONCLUSION: EQUIVALENT — continuous scheduler with beta_end=10 is calibrated")
        print("  correctly to match discrete scheduler with beta_end=2e-3, N=5000.")
        print("  Safe to proceed with full continuous time conversion.")
        print("  DECISION: ADOPT")
    else:
        print("  CONCLUSION: NOT EQUIVALENT — further calibration needed.")
        print("  DECISION: INVESTIGATE FURTHER")


if __name__ == "__main__":
    main()
