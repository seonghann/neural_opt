"""
Analysis: Debug Langevin sampling coefficient mismatch
Date: 2026-03-21
Related progress log: docs/progress/260321_port_experiment_branches.md

Problem:
    Langevin sampling produces RMSD=1.54Å instead of expected ~0.03Å.
    Paper shows Diffusion SDE achieves 0.0329Å on same checkpoint.
    Suspect coefficient computation in predict_score() is wrong.

    Paper formula (VE-SDE on manifold):
        dx = -dσ²/dt · f_θ/σ² · dt + sqrt(dσ²/dt) · dB

    Code formula:
        coeff1 = β · exp(-2·log_σ) · dt = β/σ²_VP · dt
        g_VE = sqrt(β) · exp(-log_α) = sqrt(β)/α

    Need to verify: does β/σ²_VP match dσ²/dt / σ² in the paper?

Judgment Criteria:
    Coefficients should match paper's formula at all time points.
    Total integral of coefficient over [0, start_time] should give O(1) displacement.

Conclusion:
    <filled after running>

Usage: python analyze/260321_langevin_coefficient_debug.py
"""
import sys
sys.path.insert(0, "/home/seonghwan/RxnExpPipe/neural_opt")

import torch
import numpy as np
from src.diffusion.continuous_scheduler import SigmoidDiffusionScheduler


def main():
    sched = SigmoidDiffusionScheduler(c=12, beta_start=1e-7, beta_end=10)

    # Fine time grid
    t = torch.linspace(0.001, 0.03, 1000)  # avoid t=0 singularity
    dt_val = 0.03 / 128  # actual dt used in sampling

    alpha = sched.get_alpha(t)
    alpha_sq = alpha ** 2
    sigma_sq_VP = 1 - alpha_sq  # σ²_VP = 1 - α²
    sigma_sq_VE = sigma_sq_VP / alpha_sq  # σ²_VE = (1-α²)/α²
    beta = sched.get_beta(t)
    log_alpha = sched._log_alpha(t)
    log_sigma = sched._log_sigma(t)

    # Paper's dσ² depends on which σ convention:
    # If σ²_VP = 1-α²: dσ²_VP/dt = 2α·(0.5β·α) = α²·β
    dsigma_sq_VP_dt = alpha_sq * beta

    # If σ²_VE = (1-α²)/α²: dσ²_VE/dt = β/α² (can be shown)
    dsigma_sq_VE_dt = beta / alpha_sq

    print("=" * 100)
    print("COEFFICIENT COMPARISON AT KEY TIME POINTS")
    print("=" * 100)

    key_t = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

    # Paper formula interpretation 1: σ = σ_VP
    # drift = dσ²_VP/dt / σ²_VP = α²β / (1-α²)
    print("\n--- Interpretation 1: Paper uses σ_VP = sqrt(1-α²) ---")
    print(f"{'t':>8} {'α':>10} {'σ_VP':>10} {'β':>12} {'paper_coeff':>14} {'code_coeff':>14} {'ratio':>10}")
    for tv in key_t:
        tt = torch.tensor([tv])
        a = sched.get_alpha(tt).item()
        s = sched.get_sigma(tt).item()
        b = sched.get_beta(tt).item()
        la = sched._log_alpha(tt).item()
        ls = sched._log_sigma(tt).item()

        paper_coeff1 = a**2 * b / s**2  # dσ²_VP/dt / σ²_VP
        code_coeff1 = b * np.exp(-2 * ls)  # β / σ²_VP = β * exp(-2*log_σ)

        print(f"{tv:>8.3f} {a:>10.6f} {s:>10.6f} {b:>12.6f} {paper_coeff1:>14.4f} {code_coeff1:>14.4f} {paper_coeff1/code_coeff1:>10.6f}")

    # Paper formula interpretation 2: σ = σ_VE
    # drift = dσ²_VE/dt / σ²_VE = (β/α²) / ((1-α²)/α²) = β/(1-α²) = β/σ²_VP
    print("\n--- Interpretation 2: Paper uses σ_VE = sqrt((1-α²)/α²) ---")
    print(f"{'t':>8} {'σ_VE':>10} {'paper_coeff':>14} {'code_coeff':>14} {'ratio':>10}")
    for tv in key_t:
        tt = torch.tensor([tv])
        a = sched.get_alpha(tt).item()
        s = sched.get_sigma(tt).item()
        b = sched.get_beta(tt).item()
        ls = sched._log_sigma(tt).item()

        sigma_VE = s / a
        paper_coeff2 = b / s**2  # = dσ²_VE/dt / σ²_VE = β/(1-α²) = β/σ²_VP
        code_coeff1 = b * np.exp(-2 * ls)

        print(f"{tv:>8.3f} {sigma_VE:>10.6f} {paper_coeff2:>14.4f} {code_coeff1:>14.4f} {paper_coeff2/code_coeff1:>10.6f}")

    print("\n→ Interpretation 2 matches! Paper uses σ_VE. Code's coeff1 = β/σ²_VP = paper's dσ²_VE/dt / σ²_VE")

    # Now check the TOTAL integral of coeff over [0, start_time]
    # For FM-ODE: ∫ f/t dt from eps to T → log(T/eps) ~ f * something ~ O(1)
    # For Diff ODE: ∫ 0.5 * β/σ² dt from 0 to start_time
    print("\n" + "=" * 100)
    print("TOTAL INTEGRAL OF COEFFICIENTS OVER [0, start_time=0.03]")
    print("=" * 100)

    n = 10000
    t_int = torch.linspace(1e-5, 0.03, n)
    dt_int = 0.03 / n

    beta_vals = sched.get_beta(t_int)
    sigma_vals = sched.get_sigma(t_int)
    alpha_vals = sched.get_alpha(t_int)

    # Diffusion ODE: ∫ 0.5 * β/σ² dt
    diff_ode_integrand = 0.5 * beta_vals / sigma_vals**2
    diff_ode_integral = (diff_ode_integrand * dt_int).sum().item()

    # FM-ODE: ∫ 1/t dt = log(start_time / eps) → ∫ 1/(1-τ) dτ from 0 to 1-eps
    fm_ode_integrand = 1.0 / t_int
    fm_ode_integral = (fm_ode_integrand * dt_int).sum().item()

    print(f"\n  Diffusion ODE: ∫ 0.5·β/σ² dt = {diff_ode_integral:.4f}")
    print(f"  FM ODE:        ∫ 1/t dt       = {fm_ode_integral:.4f}")
    print()
    print(f"  If f_θ ≈ Δx (displacement), then total dx ≈ f_θ * integral")
    print(f"  For reasonable results, integral should be O(1).")
    print()

    if diff_ode_integral > 10:
        print(f"  ⚠ Diffusion integral = {diff_ode_integral:.1f} >> 1. Coefficients blow up!")
        print(f"    This means: 128 steps × large coeff = excessive displacement")
    elif diff_ode_integral < 0.1:
        print(f"  ⚠ Diffusion integral = {diff_ode_integral:.4f} << 1. Undershoot!")
    else:
        print(f"  ✓ Diffusion integral is O(1). Coefficients seem reasonable.")

    # Per-step analysis at first and last steps
    print("\n" + "=" * 100)
    print("PER-STEP COEFFICIENT VALUES (128 steps, start_time=0.03)")
    print("=" * 100)

    nfe = 128
    dt_step = 0.03 / nfe
    t_steps = torch.linspace(0.03, dt_step, nfe)  # from start_time down to ~0

    beta_steps = sched.get_beta(t_steps)
    sigma_steps = sched.get_sigma(t_steps)
    log_sigma_steps = sched._log_sigma(t_steps)

    coeff1_steps = beta_steps * (-2 * log_sigma_steps).exp() * dt_step
    coeff1_ode = coeff1_steps * 0.5  # ODE has 0.5 factor

    print(f"\n  {'step':>6} {'t':>8} {'β':>12} {'σ_VP':>12} {'coeff1*dt':>12} {'coeff_ODE':>12}")
    for i in [0, 1, 2, 10, 50, 100, 126, 127]:
        if i >= nfe:
            continue
        print(f"  {i:>6d} {t_steps[i].item():>8.5f} {beta_steps[i].item():>12.6f} "
              f"{sigma_steps[i].item():>12.8f} {coeff1_steps[i].item():>12.6f} {coeff1_ode[i].item():>12.6f}")

    print(f"\n  Sum of coeff_ODE over all steps: {coeff1_ode.sum().item():.4f}")
    print(f"  Mean coeff_ODE per step: {coeff1_ode.mean().item():.6f}")
    print(f"  Max coeff_ODE: {coeff1_ode.max().item():.6f} (at step {coeff1_ode.argmax().item()}, t={t_steps[coeff1_ode.argmax()].item():.5f})")

    # Compare with FM-ODE coefficient
    print("\n  FM-ODE comparison:")
    t_fm = torch.linspace(1.0, 1.0/128, 128)  # FM goes from t=1 to t~0
    dt_fm = 1.0 / 128
    coeff_fm = dt_fm / t_fm
    print(f"  Sum of FM coeff: {coeff_fm.sum().item():.4f}")
    print(f"  Max FM coeff: {coeff_fm.max().item():.6f}")


if __name__ == "__main__":
    main()
