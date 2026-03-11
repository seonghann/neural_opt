"""
Standalone sampling script (replaces PL trainer.test workflow).

Usage:
    python sample.py configs/sampling.qm9.rdsm.yaml
    python sample.py configs/sampling.qm9.rdsm.yaml --batch_idx_end 4

Loads a trained checkpoint, runs sampling on test data, computes metrics,
and saves results (dynamic graph objects).
"""

import sys
import time
import argparse

import torch
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from dataset.data_module import load_datamodule
from diffusion.model import DiffusionModel
from diffusion.sampling import sample_batch_simple, sample_batch_diffusion
from metrics.metrics import SamplingMetrics
from utils.wandb_utils import setup_wandb


def load_checkpoint(model, ckpt_path, device="cpu"):
    """
    Load a PL checkpoint into the plain DiffusionModel.

    PL checkpoints store state_dict with keys like 'NeuralNet.atom_embedding.weight'.
    Since DiffusionModel keeps self.NeuralNet, the keys already match for the
    learnable parameters. We just need to extract the state_dict from the PL wrapper.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # PL stores all model params under ckpt['state_dict']
    pl_state = ckpt["state_dict"]

    # Filter to only NeuralNet keys (the only learnable parameters)
    model_state = {}
    for k, v in pl_state.items():
        if k.startswith("NeuralNet."):
            model_state[k] = v

    # Load into model (DiffusionModel has self.NeuralNet)
    missing, unexpected = model.load_state_dict(model_state, strict=False)

    # The "missing" keys should be zero -- NeuralNet covers all learnable params
    # "unexpected" should be empty since we filtered
    if missing:
        # Check that missing keys are all non-learnable (noise_schedule, geodesic_solver, etc.)
        learnable_missing = [k for k in missing if k.startswith("NeuralNet.")]
        if learnable_missing:
            raise RuntimeError(f"Missing learnable keys: {learnable_missing}")
        print(f"  Non-learnable keys (expected): {len(missing)} skipped")
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected}")

    print(f"  Loaded {len(model_state)} parameter tensors from checkpoint")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling script")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument("--batch_idx_end", type=int, default=None,
                        help="Stop after this many batches (for quick testing)")
    parser.add_argument("--save_dynamic", type=str, default=None,
                        help="Override save_dynamic path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Override from CLI
    if args.batch_idx_end is not None:
        config.sampling.batch_idx_end = args.batch_idx_end
    if args.save_dynamic is not None:
        config.debug.save_dynamic = args.save_dynamic

    torch.set_default_dtype(torch.float32)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup wandb if configured
    if getattr(config.general, 'use_wandb', False):
        setup_wandb(config)

    # Load data
    datamodule = load_datamodule(config)
    test_dl = datamodule.test_dataloader()
    print(f"Test dataloader: {len(test_dl)} batches")

    # Load model
    model = DiffusionModel(config)
    if config.general.test_only:
        model = load_checkpoint(model, config.general.test_only, device="cpu")
    model = model.to(device)
    model.eval()

    # Metrics (move to device so torchmetrics states match input tensors)
    sampling_metrics = SamplingMetrics(model.geodesic_solver, name='test').to(device)

    # Sampling
    dynamic_graph_list = []
    samples = []
    stochastic = config.sampling.stochastic
    batch_idx_end = getattr(config.sampling, 'batch_idx_end', None)

    start = time.time()
    for i, batch in enumerate(tqdm(test_dl, total=len(test_dl))):
        if batch_idx_end is not None and i >= batch_idx_end:
            break

        batch = batch.to(device)
        if config.sampling.score_type == "cfm":
            batch_out = sample_batch_simple(
                model,
                batch,
                config,
                stochastic=stochastic,
                num_cycles=config.sampling.num_cycles,
                dynamic_graph_list=dynamic_graph_list,
            )
        elif config.sampling.score_type == "diffusion":
            batch_out = sample_batch_diffusion(
                model,
                batch,
                config,
                stochastic=stochastic,
                start_from_time=getattr(config.sampling, 'start_from_time', None),
                dynamic_graph_list=dynamic_graph_list,
            )
        else:
            raise NotImplementedError(f"Unsupported score_type: {config.sampling.score_type}")
        samples.extend(batch_out)

    elapsed = time.time() - start
    print(f"Done. Sampling took {elapsed:0.1f}s for {len(samples)} molecules")

    # Compute metrics
    name = config.general.name + f"_sample"
    sampling_metrics(samples, name, current_epoch=0, valid_counter=-1, test=True, local_rank=0)

    # Save dynamic graph objects
    if config.debug.save_dynamic:
        torch.save(dynamic_graph_list, config.debug.save_dynamic)
        print(f"Saved dynamic graph to {config.debug.save_dynamic}")

    print(f"\nSampling complete.")


if __name__ == "__main__":
    main()
