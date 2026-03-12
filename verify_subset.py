# DEPRECATED: Use `python sample.py <config> --batch_idx_end 4` directly instead.
# Scheduled for removal after paper reproduction is confirmed.
"""Quick subset sampling verification.
Runs sampling on ~5% of test data (4 batches) to verify code works after refactoring.

Usage:
    python verify_subset.py
    # or equivalently:
    python sample.py configs/sampling.qm9.rdsm.yaml --batch_idx_end 4 --save_dynamic ./save_dynamic.qm9.rdsm.finetuned.subset.pt
"""
import sys
import os

if __name__ == "__main__":
    # Dispatch to sample.py with subset args
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "sample.py"),
        "configs/sampling.qm9.rdsm.yaml",
        "--batch_idx_end", "4",
        "--save_dynamic", "./save_dynamic.qm9.rdsm.finetuned.subset.pt",
    ]
    print(f"Running: {' '.join(cmd)}")
    os.execv(sys.executable, cmd)
