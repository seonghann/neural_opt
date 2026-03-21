"""
Analysis: Discrete vs Continuous Scheduler Comparison
Date: 2026-03-21
Related progress log: docs/progress/260321_port_experiment_branches.md

Problem:
    Verify that SigmoidDiffusionScheduler (continuous) produces equivalent
    alpha/sigma values to TSDiffNoiseScheduler (discrete) at corresponding
    time points, enabling a full discrete->continuous migration.

    The discrete scheduler computes alphas = cumprod(1 - betas) while the
    continuous one computes alpha = exp(-0.5 * integral(beta)). These are
    mathematically related: cumprod(1-b_i) ~ exp(-sum(b_i)) when b_i << 1.

Judgment Criteria:
    - Max relative error of alpha values < 1% across all timesteps
    - At key time points (t=150, t=1500), alpha values should be nearly identical

Conclusion:
    NOT EQUIVALENT with naive t_cont = t_disc / N mapping.
    Three root causes identified, in order of impact:

    1. BETA SCALING (DOMINANT): sum(disc_betas) = 5.0 but integral(cont_beta) = 1e-3.
       Ratio = N = 5000. Discrete betas are per-step values; continuous betas are
       per-unit-time rates. The continuous beta must be scaled by N.

    2. EXPONENT FACTOR (0.5): Discrete alpha = exp(-sum(beta)), but continuous
       alpha = exp(-0.5 * integral(beta)). The 0.5 comes from VP-SDE convention.
       Combined with issue 1, the exponent is off by 2N = 10000x.

    3. BETA SHAPE (residual): Different sigmoid normalization. Discrete uses raw
       sigmoid on linspace(-6,6,N); continuous normalizes to [0,1] range.
       Causes ~1% residual error even after fixing issues 1 and 2.

    ERROR DECOMPOSITION:
    - Naive (all 3 issues):         max alpha error = 14799.70%
    - Remove 0.5 only:              max alpha error = 14792.26%  (scaling dominates)
    - Scale by N + remove 0.5:      max alpha error = 1.10%      (shape residual)
    - Scale by 2N + keep 0.5:       max alpha error = 1.10%      (same, equivalent)
    - Mean alpha error (corrected): 0.75%

    DECISION: INVESTIGATE FURTHER
    The schedulers are NOT drop-in replacements. After correcting for scaling and
    the 0.5 factor, ~1% residual error remains from beta shape differences.
    To achieve < 1% error, the continuous sigmoid must also be denormalized to
    match the discrete's raw sigmoid, OR alpha(t) should be fit directly.

Usage: python analyze/260321_scheduler_comparison.py
"""
import sys
sys.path.insert(0, "/home/seonghwan/RxnExpPipe/neural_opt")

import torch
import numpy as np

from src.diffusion.noise_scheduler import TSDiffNoiseScheduler
from src.diffusion.continuous_scheduler import SigmoidDiffusionScheduler


def print_table(title, headers, rows):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    header_str = "".join(f"{h:>14}" for h in headers)
    print(header_str)
    print("-" * 100)
    for row in rows:
        fmt = []
        for v in row:
            if isinstance(v, int):
                fmt.append(f"{v:>14d}")
            elif isinstance(v, str):
                fmt.append(f"{v:>14}")
            elif abs(v) < 1e-4 and v != 0:
                fmt.append(f"{v:>14.6e}")
            else:
                fmt.append(f"{v:>14.10f}")
        print("".join(fmt))


def main():
    # ---- 1. Create both schedulers with SAME parameters ----
    beta_start = 1e-7
    beta_end = 2e-3
    num_steps = 5000

    discrete = TSDiffNoiseScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        num_diffusion_timesteps=num_steps,
        schedule_type="sigmoid",
    )

    continuous = SigmoidDiffusionScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
    )

    key_steps = [0, 50, 100, 150, 500, 1000, 1500, 2500, 3000, 4000, 4999]

    # ---- 2. PART A: Diagnose beta mismatch ----
    print("=" * 100)
    print("PART A: ROOT CAUSE DIAGNOSIS -- BETA MISMATCH")
    print("=" * 100)
    print()
    print("Discrete:   beta[i] = sigmoid(-6 + 12*i/(N-1)) * (beta_end-beta_start) + beta_start")
    print("              where sigmoid is raw (unnormalized)")
    print("Continuous: beta(t) = [sigmoid(12*(t-0.5)) - sigmoid(-6)] / [sigmoid(6)-sigmoid(-6)]")
    print("              * (beta_end-beta_start) + beta_start")
    print()
    print("At t = i/N:")
    print("  Discrete sigmoid input:   -6 + 12*i/(N-1)")
    print("  Continuous sigmoid input:  12*(i/N) - 6 = 12*i/N - 6")
    print("  Difference:  12*i/(N-1) - 12*i/N = 12*i/[N*(N-1)]  (tiny but nonzero)")
    print()

    # Show beta comparison
    disc_timesteps = torch.arange(num_steps)
    beta_disc = discrete.betas
    t_cont = disc_timesteps.float() / num_steps
    beta_cont = continuous.get_beta(t_cont)

    rows = []
    for t in key_steps:
        b_d = beta_disc[t].item()
        b_c = beta_cont[t].item()
        abs_err = abs(b_d - b_c)
        rel_err = abs_err / (abs(b_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", b_d, b_c, abs_err, f"{rel_err:.2f}%"])

    print_table("BETA(t) comparison: discrete beta[i] vs continuous beta(i/N)",
                ["t_disc", "t_cont", "beta_disc", "beta_cont", "abs_err", "rel_err"],
                rows)

    beta_rel = (beta_disc - beta_cont).abs() / (beta_disc.abs() + 1e-30) * 100
    print(f"\n  Max beta relative error:  {beta_rel.max().item():.2f}%")
    print(f"  Mean beta relative error: {beta_rel.mean().item():.2f}%")
    print(f"  -> Betas do NOT match. Normalization and linspace indexing differ.")

    # ---- 3. PART B: Alpha comparison with naive mapping t_cont = t_disc/N ----
    alpha_disc = discrete.get_alpha(disc_timesteps)
    alpha_cont = continuous.get_alpha(t_cont)

    rows = []
    for t in key_steps:
        a_d = alpha_disc[t].item()
        a_c = alpha_cont[t].item()
        abs_err = abs(a_d - a_c)
        rel_err = abs_err / (abs(a_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", a_d, a_c, abs_err, f"{rel_err:.4f}%"])

    print_table("ALPHA comparison: naive mapping t_cont = t_disc / N",
                ["t_disc", "t_cont", "alpha_disc", "alpha_cont", "abs_err", "rel_err"],
                rows)

    alpha_rel = (alpha_disc - alpha_cont).abs() / (alpha_disc.abs() + 1e-30) * 100
    print(f"\n  Max alpha relative error:  {alpha_rel.max().item():.4f}%")
    print(f"  Mean alpha relative error: {alpha_rel.mean().item():.4f}%")

    # ---- 4. PART C: Understand the 0.5 factor ----
    print("\n" + "=" * 100)
    print("PART C: THE FACTOR-OF-2 ISSUE (cumprod vs exp-integral)")
    print("=" * 100)
    print()
    print("Discrete:   alpha(t) = prod(1-beta_i) = exp(sum(log(1-beta_i))) ~ exp(-sum(beta_i))")
    print("Continuous: alpha(t) = exp(-0.5 * integral(beta(s) ds))")
    print()
    print("Even if betas matched, the continuous uses -0.5 * integral while")
    print("the discrete uses -1.0 * sum. This is a fundamental formulation difference.")
    print()
    print("Let's verify: compute alpha from discrete betas manually, with and without 0.5:")

    # Manual computation: sum of discrete betas
    cumsum_betas = torch.cumsum(beta_disc, dim=0)
    alpha_manual_full = torch.exp(-cumsum_betas)          # exp(-sum(beta))
    alpha_manual_half = torch.exp(-0.5 * cumsum_betas)    # exp(-0.5*sum(beta))

    rows = []
    for t in key_steps:
        a_disc = alpha_disc[t].item()
        a_full = alpha_manual_full[t].item()
        a_half = alpha_manual_half[t].item()
        rows.append([t, a_disc, a_full, a_half,
                     f"{abs(a_disc-a_full)/abs(a_disc+1e-30)*100:.6f}%",
                     f"{abs(a_disc-a_half)/abs(a_disc+1e-30)*100:.6f}%"])

    print_table("Alpha: cumprod(1-b) vs exp(-sum(b)) vs exp(-0.5*sum(b))",
                ["t_disc", "cumprod(1-b)", "exp(-sum b)", "exp(-0.5*sum b)",
                 "err(full)", "err(half)"],
                rows)

    print()
    print("  -> cumprod(1-b) ~ exp(-sum(b)), NOT exp(-0.5*sum(b))")
    print("  -> The continuous scheduler's 0.5 factor makes it a different schedule.")

    # ---- 5. PART D: What if we remove the 0.5? ----
    # Manually compute continuous-style alpha without the 0.5
    print("\n" + "=" * 100)
    print("PART D: HYPOTHETICAL -- Continuous alpha WITHOUT 0.5 factor")
    print("=" * 100)
    print()
    print("If we defined continuous alpha(t) = exp(-integral(beta(s) ds)) instead of")
    print("exp(-0.5 * integral), would the betas still need to match?")
    print()

    # Compute integral of continuous beta numerically
    # Use fine grid for numerical integration
    n_fine = 50000
    t_fine = torch.linspace(0, 1, n_fine)
    dt = 1.0 / n_fine
    beta_fine = continuous.get_beta(t_fine)
    integral_fine = torch.cumsum(beta_fine * dt, dim=0)

    # Alpha without 0.5
    alpha_cont_no_half = torch.exp(-integral_fine)

    # Sample at corresponding discrete times
    # disc_time i corresponds to t_fine index = i * n_fine / num_steps
    disc_to_fine = (disc_timesteps.float() / num_steps * n_fine).long().clamp(max=n_fine-1)
    alpha_cont_no_half_sampled = alpha_cont_no_half[disc_to_fine]

    rows = []
    for t in key_steps:
        a_d = alpha_disc[t].item()
        a_c_nohalf = alpha_cont_no_half_sampled[t].item()
        abs_err = abs(a_d - a_c_nohalf)
        rel_err = abs_err / (abs(a_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", a_d, a_c_nohalf, abs_err, f"{rel_err:.4f}%"])

    print_table("Alpha: discrete vs continuous(no 0.5, numerical integral)",
                ["t_disc", "t_cont", "alpha_disc", "alpha_cont_no05", "abs_err", "rel_err"],
                rows)

    rel_nohalf = (alpha_disc - alpha_cont_no_half_sampled).abs() / (alpha_disc.abs() + 1e-30) * 100
    print(f"\n  Max relative error (no 0.5): {rel_nohalf.max().item():.4f}%")
    print(f"  Mean relative error (no 0.5): {rel_nohalf.mean().item():.4f}%")

    # ---- 6. PART E: Scaling diagnosis ----
    print("\n" + "=" * 100)
    print("PART E: SCALING DIAGNOSIS -- Discrete sum vs Continuous integral")
    print("=" * 100)

    sum_disc_beta = beta_disc.sum().item()
    # Numerical integral of continuous beta over [0,1]
    n_int = 100000
    t_int = torch.linspace(0, 1, n_int)
    dt_int = 1.0 / n_int
    int_cont_beta = (continuous.get_beta(t_int) * dt_int).sum().item()

    print(f"\n  sum(discrete betas):             {sum_disc_beta:.6f}")
    print(f"  integral(continuous beta, 0->1): {int_cont_beta:.6e}")
    print(f"  Ratio sum/integral:              {sum_disc_beta / int_cont_beta:.1f}x  (= N = {num_steps})")
    print()
    print("  The discrete betas are per-step values. Their SUM over N steps ~ N * mean(beta).")
    print("  The continuous integral over [0,1] ~ mean(beta) * 1.")
    print("  So sum(disc) / integral(cont) = N. This is the DOMINANT source of error.")
    print()
    print("  Discrete:   alpha = exp(-sum(beta_i))       = exp(-N * mean(beta))")
    print("  Continuous: alpha = exp(-0.5 * int(beta))    = exp(-0.5 * mean(beta))")
    print("  -> Off by factor N/0.5 = 2N = 10000 in the exponent!")

    # ---- 6b. PART E2: Corrected continuous with N scaling ----
    print("\n" + "=" * 100)
    print("PART E2: CORRECTED -- Continuous with beta scaled by N (and no 0.5)")
    print("=" * 100)
    print()
    print("If continuous beta_corrected(t) = N * beta(t), then:")
    print("  integral(N*beta, 0->1) = N * integral(beta, 0->1) = sum(disc betas)")
    print("  alpha = exp(-integral(N*beta)) should match discrete alpha")

    # Compute corrected continuous alpha numerically
    beta_fine_scaled = continuous.get_beta(t_fine) * num_steps  # scale by N
    integral_scaled = torch.cumsum(beta_fine_scaled * dt, dim=0)
    alpha_cont_corrected = torch.exp(-integral_scaled)  # no 0.5 factor

    alpha_cont_corrected_sampled = alpha_cont_corrected[disc_to_fine]

    rows = []
    for t in key_steps:
        a_d = alpha_disc[t].item()
        a_c = alpha_cont_corrected_sampled[t].item()
        abs_err = abs(a_d - a_c)
        rel_err = abs_err / (abs(a_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", a_d, a_c, abs_err, f"{rel_err:.4f}%"])

    print_table("Alpha: discrete vs corrected continuous (N*beta, no 0.5)",
                ["t_disc", "t_cont", "alpha_disc", "alpha_corr", "abs_err", "rel_err"],
                rows)

    rel_corrected = (alpha_disc - alpha_cont_corrected_sampled).abs() / (alpha_disc.abs() + 1e-30) * 100
    print(f"\n  Max relative error (corrected):  {rel_corrected.max().item():.4f}%")
    print(f"  Mean relative error (corrected): {rel_corrected.mean().item():.4f}%")

    # ---- 6c. PART E3: Corrected with 0.5 and 2N scaling ----
    print("\n" + "=" * 100)
    print("PART E3: ALTERNATIVE -- Keep 0.5 factor, scale beta by 2N")
    print("=" * 100)

    beta_fine_2n = continuous.get_beta(t_fine) * 2 * num_steps
    integral_2n = torch.cumsum(beta_fine_2n * dt, dim=0)
    alpha_cont_2n = torch.exp(-0.5 * integral_2n)  # keep 0.5 factor
    alpha_cont_2n_sampled = alpha_cont_2n[disc_to_fine]

    rows = []
    for t in key_steps:
        a_d = alpha_disc[t].item()
        a_c = alpha_cont_2n_sampled[t].item()
        abs_err = abs(a_d - a_c)
        rel_err = abs_err / (abs(a_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", a_d, a_c, abs_err, f"{rel_err:.4f}%"])

    print_table("Alpha: discrete vs continuous (2N*beta, with 0.5)",
                ["t_disc", "t_cont", "alpha_disc", "alpha_2N", "abs_err", "rel_err"],
                rows)

    rel_2n = (alpha_disc - alpha_cont_2n_sampled).abs() / (alpha_disc.abs() + 1e-30) * 100
    print(f"\n  Max relative error (2N scaling): {rel_2n.max().item():.4f}%")
    print(f"  Mean relative error (2N scaling): {rel_2n.mean().item():.4f}%")

    # ---- 6d. Sigma comparison (VP convention) ----
    sigma_disc = torch.sqrt(1 - alpha_disc ** 2)
    sigma_cont = continuous.get_sigma(t_cont)

    rows = []
    for t in key_steps:
        s_d = sigma_disc[t].item()
        s_c = sigma_cont[t].item()
        abs_err = abs(s_d - s_c)
        rel_err = abs_err / (abs(s_d) + 1e-30) * 100
        rows.append([t, f"{t/num_steps:.4f}", s_d, s_c, abs_err, f"{rel_err:.4f}%"])

    print_table("SIGMA comparison (uncorrected): sqrt(1 - alpha^2)",
                ["t_disc", "t_cont", "sigma_disc", "sigma_cont", "abs_err", "rel_err"],
                rows)

    # ---- 7. PART F: Overall conclusion ----
    print("\n" + "=" * 100)
    print("OVERALL CONCLUSION")
    print("=" * 100)
    print()

    max_alpha_rel = alpha_rel.max().item()
    max_alpha_rel_nohalf = rel_nohalf.max().item()
    max_alpha_rel_corrected = rel_corrected.max().item()
    max_alpha_rel_2n = rel_2n.max().item()

    print("  THREE root causes prevent equivalence:")
    print()
    print("  1. BETA SCALING MISMATCH (DOMINANT):")
    print(f"     sum(disc betas) = {sum_disc_beta:.4f}")
    print(f"     integral(cont beta) = {int_cont_beta:.6e}")
    print(f"     Ratio = {sum_disc_beta/int_cont_beta:.0f}x (= N)")
    print("     Discrete betas are per-step; continuous betas are per-unit-time.")
    print("     To match: continuous beta must be scaled by N.")
    print()
    print("  2. EXPONENT FACTOR (0.5):")
    print("     Discrete:   alpha = exp(-1.0 * sum(beta))")
    print("     Continuous: alpha = exp(-0.5 * integral(beta))")
    print("     VP-SDE convention uses 0.5; discrete cumprod does not.")
    print()
    print("  3. BETA SHAPE MISMATCH (minor):")
    print("     Discrete: linspace(-6,6,N) -> raw sigmoid")
    print("     Continuous: c*(t-0.5), c=12 -> normalized sigmoid")
    print(f"     Max beta relative error: {beta_rel.max().item():.2f}%")
    print()
    print("  ERROR DECOMPOSITION:")
    print(f"    Naive (all 3 issues):              max alpha error = {max_alpha_rel:.2f}%")
    print(f"    Remove 0.5 only:                   max alpha error = {max_alpha_rel_nohalf:.2f}%")
    print(f"    Scale by N + remove 0.5:           max alpha error = {max_alpha_rel_corrected:.4f}%")
    print(f"    Scale by 2N + keep 0.5:            max alpha error = {max_alpha_rel_2n:.4f}%")
    print()

    if max_alpha_rel_corrected < 1.0 or max_alpha_rel_2n < 1.0:
        best_fix = "N*beta, no 0.5" if max_alpha_rel_corrected < max_alpha_rel_2n else "2N*beta, keep 0.5"
        best_err = min(max_alpha_rel_corrected, max_alpha_rel_2n)
        print(f"  VERDICT: FIXABLE -- with '{best_fix}', max error = {best_err:.4f}%")
        print()
        print("  Recommended approach:")
        print("    In SigmoidDiffusionScheduler.__init__, pass num_diffusion_timesteps=N.")
        print("    In _integrate_beta, multiply the result by N (or 2N if keeping the 0.5).")
        print("    This makes the continuous scheduler produce the same alpha curve")
        print("    as the discrete one, enabling a drop-in replacement.")
    else:
        print("  VERDICT: NOT EQUIVALENT -- deeper investigation needed.")
        print()
        print("  Even with scaling corrections, the beta shape difference")
        print("  causes significant error. Consider fitting alpha(t) directly.")


if __name__ == "__main__":
    main()
