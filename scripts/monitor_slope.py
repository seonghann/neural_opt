"""Monitor log-log slope of valid_pred_target_norm_perr from training log.
Prints current slope and convergence judgment."""

import numpy as np
import sys

def ema_smooth(x, alpha=0.05):
    s = np.zeros_like(x)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
    return s

def local_slope(x, y, window=30):
    slopes = []
    for i in range(window, len(x)):
        xi = x[i-window:i]
        yi = y[i-window:i]
        A = np.vstack([xi, np.ones(len(xi))]).T
        slope, _ = np.linalg.lstsq(A, yi, rcond=None)[0]
        slopes.append(slope)
    return np.array(slopes)

logfile = sys.argv[1] if len(sys.argv) > 1 else "logs/stage2_rdsm.log"

import subprocess
result = subprocess.run(["grep", "valid_epoch/pred_target_norm_perr", logfile], capture_output=True, text=True)
lines = result.stdout.strip().split("\n")
perr = np.array([float(l.split()[-1]) for l in lines if l.strip()])

if len(perr) < 40:
    print(f"Only {len(perr)} validation points, need >=40. Too early to judge.")
    sys.exit(0)

steps = np.arange(1, len(perr)+1) * 3 * 47
log_s = np.log10(steps)
log_p = np.log10(perr)
log_p_smooth = ema_smooth(log_p, alpha=0.05)
slopes = local_slope(log_s, log_p_smooth, window=30)

recent = slopes[-30:]
epoch_now = len(perr) * 3
step_now = epoch_now * 47

print(f"=== Stage 2 R-DSM Monitor ===")
print(f"Epoch: {epoch_now}/3000 | Step: {step_now}")
print(f"Current perr: {perr[-1]:.6f} (raw), {10**log_p_smooth[-1]:.6f} (smoothed)")
print(f"Recent slope (last 30 vals): {recent.mean():.4f} ± {recent.std():.4f}")
print(f"Threshold: -0.05")

if recent.mean() > -0.05:
    print(">>> CONVERGED — slope near zero, recommend STOP")
elif recent.mean() > -0.10:
    print(">>> SLOWING — approaching convergence")
else:
    print(">>> LEARNING — still improving steadily")
