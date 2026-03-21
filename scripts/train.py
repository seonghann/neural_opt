"""
Accelerate-based training script (replaces PL Trainer + BridgeDiffusion hooks).

Usage:
    # Single GPU
    python train.py configs/training.qm9.rdsm.yaml

    # Multi-GPU via accelerate
    accelerate launch train.py configs/training.qm9.rdsm.yaml

    # Resume from checkpoint
    python train.py configs/training.qm9.rdsm.yaml --resume checkpoints/last
"""

import sys
import os
import time
import argparse
import logging

# Ensure neural_opt/ is on the path when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import wandb
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.dataset.data_module import load_datamodule
from src.diffusion.model import DiffusionModel
from src.diffusion.sampling import sample_batch_simple, sample_batch_diffusion
from src.metrics.metrics import LossFunction, TrainMetrics, ValidMetrics, SamplingMetrics
from src.model import get_optimizer, get_scheduler
from src.utils.wandb_utils import setup_wandb
from src.utils.ema import EMA
from scripts.sample import load_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script (Accelerate)")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Accelerate checkpoint directory to resume from")
    parser.add_argument("--resume_pl", type=str, default=None,
                        help="Path to PL .ckpt file (for model weights only, no optimizer state)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (for smoke testing)")
    return parser.parse_args()


def save_checkpoint(accelerator, model, optimizer, scheduler, epoch, best_valid_loss, save_dir, ema=None):
    """Save Accelerate-style checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    # Save accelerator state (model, optimizer, scheduler, RNG)
    accelerator.save_state(save_dir)

    # Save extra metadata + EMA state
    if accelerator.is_main_process:
        meta = {"epoch": epoch, "best_valid_loss": best_valid_loss}
        if ema is not None:
            meta["ema"] = ema.state_dict()
        torch.save(meta, os.path.join(save_dir, "meta.pt"))

    accelerator.print(f"Checkpoint saved to {save_dir}")


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    # CLI overrides
    if args.epochs is not None:
        config.train.epochs = args.epochs

    # Accelerator setup
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="wandb" if config.general.use_wandb else None,
    )

    set_seed(config.train.seed)
    torch.set_default_dtype(torch.float32)

    # Logging
    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO)
        if config.general.use_wandb:
            setup_wandb(config)

    # Data
    accelerator.print("Loading data...")
    datamodule = load_datamodule(config)

    # Setup DDP data sharding if multi-GPU
    if accelerator.num_processes > 1:
        datamodule.setup_distributed(
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
        )
        accelerator.print(
            f"DDP: {accelerator.num_processes} processes, "
            f"effective batch_size={config.train.batch_size * accelerator.num_processes}"
        )

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    accelerator.print(f"Train: {len(train_dl)} batches/gpu, Val: {len(val_dl)} batches/gpu")

    # Model
    accelerator.print("Building model...")
    model = DiffusionModel(config)

    # Optionally load PL checkpoint weights
    if args.resume_pl:
        model = load_checkpoint(model, args.resume_pl)

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    # Prepare with accelerator
    # NOTE: PyG DataLoaders use special batching (Batch.from_data_list), so we
    # do NOT pass them to accelerator.prepare(). Instead, we use manual
    # DistributedSampler in data_module and handle device transfer ourselves.
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # EMA
    ema_decay = getattr(config.train, 'ema_decay', 0.999)
    ema = EMA(accelerator.unwrap_model(model), decay=ema_decay)
    accelerator.print(f"EMA enabled with decay={ema_decay}")

    # Metrics (kept on main process only for logging)
    lambda_x_train = config.train.lambda_x_train
    lambda_q_train = config.train.lambda_q_train
    lambda_x_valid = config.train.lambda_x_valid
    lambda_q_valid = config.train.lambda_q_valid

    train_loss_fn = LossFunction(lambda_x_train, lambda_q_train, "train").to(accelerator.device)
    valid_loss_fn = LossFunction(lambda_x_valid, lambda_q_valid, "valid").to(accelerator.device)
    train_metrics = TrainMetrics(name='train').to(accelerator.device)
    valid_metrics = ValidMetrics(
        accelerator.unwrap_model(model).geodesic_solver,
        'valid', lambda_x_valid, lambda_q_valid,
    ).to(accelerator.device)
    valid_sampling_metrics = SamplingMetrics(
        accelerator.unwrap_model(model).geodesic_solver, name='valid'
    )

    # State
    start_epoch = 0
    best_valid_loss = 1e9
    val_counter = 0
    name = config.general.name + time.strftime(":%d-%m-%y:%H-%M-%S")

    ckpt_dir = f"checkpoints/{config.general.name}"

    # Resume from accelerate checkpoint
    if args.resume:
        accelerator.print(f"Resuming from {args.resume}")
        accelerator.load_state(args.resume)
        meta_path = os.path.join(args.resume, "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location=accelerator.device, weights_only=False)
            start_epoch = meta.get("epoch", 0) + 1
            best_valid_loss = meta.get("best_valid_loss", 1e9)
            if torch.is_tensor(best_valid_loss):
                best_valid_loss = best_valid_loss.item()
            if "ema" in meta:
                ema.load_state_dict(meta["ema"], device=accelerator.device)
                accelerator.print("EMA state restored from checkpoint")
            accelerator.print(f"Resuming from epoch {start_epoch}, best_valid_loss={best_valid_loss:.6f}")

    # =====================================================================
    # Training loop
    # =====================================================================
    accelerator.print("Training Start")

    for epoch in range(start_epoch, config.train.epochs):
        start_epoch_time = time.time()

        # ----- Train -----
        datamodule.set_epoch(epoch)  # DDP: reshuffle data per epoch
        model.train()
        train_loss_fn.reset()
        train_metrics.reset()

        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        num_steps = 0

        for step, data in enumerate(train_dl):
            data = data.to(accelerator.device)

            with accelerator.accumulate(model):
                raw_model = accelerator.unwrap_model(model)
                graph, target_x, target_q = raw_model.noise_sampling(data)
                pred_x, pred_q, edge_index, node2graph, edge2graph = model(graph)

                loss = train_loss_fn(
                    pred_x=pred_x,
                    pred_q=pred_q,
                    true_x=target_x,
                    true_q=target_q,
                    merge_edge=edge2graph,
                    merge_node=node2graph,
                    weight=raw_model.get_loss_weight(graph.t),
                )

                accelerator.backward(loss)
                if config.train.clip_grad:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                optimizer.step()
                optimizer.zero_grad()
                ema.update(accelerator.unwrap_model(model))

            gn = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            grad_norm_sum += gn
            grad_norm_max = max(grad_norm_max, gn)
            num_steps += 1

            train_metrics(
                pred_x, pred_q, target_x, target_q,
                edge2graph, node2graph, log=True,
            )

        # Log training metrics
        to_log = train_loss_fn.log_epoch_metrics()
        train_loss_val = list(to_log.values())[0]
        grad_norm_mean = grad_norm_sum / max(num_steps, 1)

        if wandb.run and accelerator.is_main_process:
            wandb.log({
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/grad_norm_mean": grad_norm_mean,
                "train/grad_norm_max": grad_norm_max,
                "epoch": epoch,
            })

        msg = f"Epoch {epoch} [train]"
        for k, v in to_log.items():
            msg += f"\n\t{k}: {v: 0.6f}"
        msg += f"\n\tgrad_norm: mean={grad_norm_mean:.4f}, max={grad_norm_max:.4f}"
        accelerator.print(msg + f"\n -- {time.time() - start_epoch_time:0.1f}s")

        # ----- Validation (with EMA weights) -----
        if (epoch + 1) % config.train.check_val_every_n_epoch == 0:
            raw_model = accelerator.unwrap_model(model)
            ema.apply_shadow(raw_model)  # swap in EMA weights

            model.eval()
            valid_loss_fn.reset()
            valid_metrics.reset()
            valid_sampling_metrics.reset()

            with torch.no_grad():
                for step, data in enumerate(val_dl):
                    data = data.to(accelerator.device)
                    graph, target_x, target_q = raw_model.noise_sampling(data)
                    pred_x, pred_q, edge_index, node2graph, edge2graph = model(graph)

                    valid_loss_fn(
                        pred_x=pred_x,
                        pred_q=pred_q,
                        true_x=target_x,
                        true_q=target_q,
                        merge_edge=edge2graph,
                        merge_node=node2graph,
                        weight=raw_model.get_loss_weight(graph.t),
                    )
                    valid_metrics(
                        pred_x, pred_q, target_x, target_q,
                        edge2graph, node2graph,
                        edge_index=edge_index, pos=graph.pos, log=True,
                    )

            # Log validation metrics
            val_to_log = valid_metrics.log_epoch_metrics()
            val_loss = val_to_log["valid_epoch/loss"]

            msg = f"Epoch {epoch} [valid]"
            for k, v in val_to_log.items():
                msg += f"\n\t{k}: {v: 0.6f}"
            accelerator.print(msg)

            # Step scheduler on validation loss
            scheduler.step(val_loss)

            # Checkpointing (save with EMA weights applied)
            if val_loss < best_valid_loss and accelerator.is_main_process:
                best_valid_loss = val_loss
                save_checkpoint(
                    accelerator, model, optimizer, scheduler,
                    epoch, best_valid_loss,
                    os.path.join(ckpt_dir, "best"),
                    ema=ema,
                )

            val_counter += 1

            # Periodic sampling during validation (using EMA weights)
            if val_counter % config.train.sample_every_n_valid == 0:
                stochastic = config.sampling.stochastic
                val_samples = []

                sample_start = time.time()
                for si, sbatch in enumerate(val_dl):
                    if si % config.train.sample_every_n_batch == 0:
                        sbatch = sbatch.to(accelerator.device)
                        if config.sampling.score_type == "diffusion":
                            batch_out = sample_batch_diffusion(
                                raw_model, sbatch, config,
                                stochastic=stochastic,
                                start_from_time=getattr(config.sampling, 'start_from_time', None),
                            )
                        else:
                            batch_out = sample_batch_simple(
                                raw_model, sbatch, config,
                                stochastic=stochastic,
                                num_cycles=getattr(config.sampling, 'num_cycles', 1),
                            )
                        val_samples.extend(batch_out)
                accelerator.print(f"Sampling took {time.time() - sample_start:0.1f}s")

                valid_sampling_metrics(
                    val_samples, name, epoch,
                    valid_counter=-1, test=True, local_rank=accelerator.local_process_index,
                )

            ema.restore(raw_model)  # restore training weights

        # Save last checkpoint every epoch
        if config.general.save_model and accelerator.is_main_process:
            save_checkpoint(
                accelerator, model, optimizer, scheduler,
                epoch, best_valid_loss,
                os.path.join(ckpt_dir, "last"),
                ema=ema,
            )

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
