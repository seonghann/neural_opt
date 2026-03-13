import omegaconf
import wandb

def setup_wandb(config):
    if config.general.use_wandb:
        config_dict = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        kwargs = {
            'name': config.general.name,
            'project': config.dataset.name,
            'config': config_dict,
            # 'settings': wandb.Settings(_disable_stats=True),
            'reinit': True,
            'mode': config.general.wandb
        }
        wandb.init(**kwargs)
        wandb.save("*.txt")
