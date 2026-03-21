"""
Standalone sampling script (replaces PL trainer.test workflow).

Usage:
    python sample.py configs/sampling.qm9.rdsm.yaml
    python sample.py configs/sampling.qm9.rdsm.yaml --batch_idx_end 4

Loads a trained checkpoint, runs sampling on test data, computes metrics,
and saves results (dynamic graph objects).
"""

import sys
import os
import time
import argparse

# Ensure neural_opt/ is on the path when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.dataset.data_module import load_datamodule
from src.diffusion.model import DiffusionModel
from src.diffusion.sampling import sample_batch_simple, sample_batch_diffusion
from src.metrics.metrics import SamplingMetrics
from src.utils.wandb_utils import setup_wandb


def load_checkpoint(model, ckpt_path, device="cpu"):
    """
    Load a checkpoint into the plain DiffusionModel.

    Supports two formats:
    - PL checkpoint (.ckpt): keys under ckpt['state_dict']
    - Accelerate checkpoint (directory with model.safetensors): keys directly in state_dict
    """
    print(f"Loading checkpoint: {ckpt_path}")

    # Detect format: directory with safetensors = Accelerate, else PL .ckpt
    safetensors_path = os.path.join(ckpt_path, "model.safetensors") if os.path.isdir(ckpt_path) else None

    if safetensors_path and os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        all_state = load_file(safetensors_path, device=str(device))
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        all_state = ckpt["state_dict"]

    # Filter to only NeuralNet keys (the only learnable parameters)
    model_state = {}
    for k, v in all_state.items():
        if k.startswith("NeuralNet."):
            model_state[k] = v

    # Remap legacy PL checkpoint keys (named attributes → layers.N aliases)
    # GeoDiffEncoder registers submodules both as named attrs and in a ModuleList:
    #   layers.0 = atom_embedding, layers.1 = edge_encoder,
    #   layers.2 = encoder, layers.3 = score_mlp, layers.4 = atom_feat_embedding
    # PL checkpoints used layers.N naming; Accelerate checkpoints use named attrs.
    # Since state_dict() exposes both paths (they alias the same tensors),
    # we normalize everything to the layers.N convention for compatibility.
    _NAMED_TO_LAYERS = {
        "NeuralNet.atom_embedding.": "NeuralNet.layers.0.",
        "NeuralNet.edge_encoder.": "NeuralNet.layers.1.",
        "NeuralNet.encoder.": "NeuralNet.layers.2.",
        "NeuralNet.score_mlp.": "NeuralNet.layers.3.",
        "NeuralNet.atom_feat_embedding.": "NeuralNet.layers.4.",
    }
    remapped_state = {}
    num_remapped = 0
    for k, v in model_state.items():
        new_k = k
        for named_prefix, layers_prefix in _NAMED_TO_LAYERS.items():
            if k.startswith(named_prefix):
                new_k = layers_prefix + k[len(named_prefix):]
                num_remapped += 1
                break
        remapped_state[new_k] = v
    if num_remapped > 0:
        print(f"  Remapped {num_remapped} legacy key(s) to layers.N convention")
    model_state = remapped_state

    # Load into model (DiffusionModel has self.NeuralNet)
    missing, unexpected = model.load_state_dict(model_state, strict=False)

    # Report missing and unexpected keys
    # GeoDiffEncoder exposes each parameter via two paths (named attr and ModuleList alias).
    # Filter out alias-path keys that are already loaded via the other path.
    _ALIAS_PREFIXES = list(_NAMED_TO_LAYERS.keys()) + list(_NAMED_TO_LAYERS.values())
    if missing:
        # A key is truly missing only if neither its alias nor itself was loaded
        loaded_suffixes = set()
        for k in model_state:
            # Strip the first two path components (e.g., "NeuralNet.layers.0.") to get suffix
            for pfx in _ALIAS_PREFIXES:
                if k.startswith(pfx):
                    loaded_suffixes.add(k[len(pfx):])
                    break
        truly_missing = []
        for k in missing:
            if not k.startswith("NeuralNet."):
                continue
            suffix = None
            for pfx in _ALIAS_PREFIXES:
                if k.startswith(pfx):
                    suffix = k[len(pfx):]
                    break
            if suffix is not None and suffix in loaded_suffixes:
                continue  # alias of an already-loaded key
            truly_missing.append(k)
        non_learnable_missing = len([k for k in missing if not k.startswith("NeuralNet.")])
        if truly_missing:
            print(f"  Learnable keys not in checkpoint (randomly initialized): {len(truly_missing)}")
            for k in truly_missing[:5]:
                print(f"    {k}")
            if len(truly_missing) > 5:
                print(f"    ... and {len(truly_missing) - 5} more")
        if non_learnable_missing:
            print(f"  Non-learnable keys (expected): {non_learnable_missing} skipped")
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
