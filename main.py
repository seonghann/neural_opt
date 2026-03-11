"""
Entry point that dispatches to train.py or sample.py based on config.

This file replaces the old PL-based main.py.  For direct use, prefer:
    python train.py  <config.yaml>          # training
    python sample.py <config.yaml>          # sampling / testing

This dispatcher is kept for backward-compatible invocation:
    python main.py <config.yaml>
"""

import sys
import os


def main():
    if len(sys.argv) == 1:
        config_file = "configs/config.yaml"
    else:
        config_file = sys.argv[-1]

    from omegaconf import OmegaConf
    config = OmegaConf.load(config_file)

    if config.general.test_only:
        # Sampling / testing mode  ->  delegate to sample.py
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "sample.py"), config_file]
        print(f"Dispatching to sample.py: {' '.join(cmd)}")
        os.execv(sys.executable, cmd)
    else:
        # Training mode  ->  delegate to train.py
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "train.py"), config_file]
        if config.general.resume:
            cmd += ["--resume_pl", config.general.resume]
        print(f"Dispatching to train.py: {' '.join(cmd)}")
        os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
