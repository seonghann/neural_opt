"""Exponential Moving Average for model parameters."""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """Maintains exponential moving average of model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # After each optimizer step:
        ema.update()
        # For validation:
        ema.apply_shadow()   # swap in EMA weights
        validate(model)
        ema.restore()        # swap back training weights
        # For saving:
        ema.apply_shadow()
        save(model)
        ema.restore()

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (0.999 = slow update, 0.99 = fast update).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Swap model params with EMA shadow params (for eval)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params (after eval)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict, device=None):
        """Load EMA state from checkpoint, moving to device if specified."""
        self.decay = state_dict["decay"]
        if device is not None:
            self.shadow = {k: v.to(device) for k, v in state_dict["shadow"].items()}
        else:
            self.shadow = state_dict["shadow"]
