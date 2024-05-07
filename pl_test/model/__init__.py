import torch
from .graph_encoder.schnet import SchNetEncoder as SchNet
from .graph_encoder.leftnet import LEFTNet as LeftNet

EncoderDict = {
    "schnet": SchNet,
    "leftnet": LeftNet,
}


def get_optimizer(cfg, model):
    if cfg.type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(
                cfg.beta1,
                cfg.beta2,
            ),
        )
    elif cfg.type == "adamw":
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            amsgrad=True,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer not supported: {optim_type}")


def get_scheduler(cfg, optimizer):
    if cfg.type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.t,
                eta_min=cfg.min_lr,
                #last_epoch=cfg.last_epoch
                )
        return scheduler
    elif cfg.type == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg.t,
                T_mult=cfg.mult,
                eta_min=cfg.min_lr,
                #last_epoch=cfg.last_epoch
                )
        return scheduler
    else:
        raise NotImplementedError("Scheduler not supported: %s" % cfg.type)
