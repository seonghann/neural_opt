def seed_everything(seed: int = 42):
    import random
    import os
    import torch
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

if __name__ == '__main__':
    import os
    import sys
    import pathlib

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from omegaconf import OmegaConf

    from dataset.data_module import load_datamodule
    from diffusion.diffusion_model import BridgeDiffusion

    torch.set_default_dtype(torch.float32)
    # torch.set_default_dtype(torch.float64)


    # Test data load
    if len(sys.argv) == 1:
        config_file = "configs/config.yaml"
    else:
        config_file = sys.argv[-1]
    config = OmegaConf.load(config_file)
    print(f"Debug: config=\n{config}")

    seed_everything(config.train.seed)
    use_gpu = config.general.gpus > 0 and torch.cuda.is_available()
    devices = config.general.gpus if use_gpu else 1
    strategy = config.general.strategy
    name = config.general.name


    ## Load data module
    datamodule = load_datamodule(config)
    print(f"data load success: {datamodule}")
    print("-----------------------------------------------------")

    ## Load model
    model = BridgeDiffusion(config)
    if hasattr(config.general, 'init_from_ckpt'):
        state_dict = torch.load(config.general.init_from_ckpt, map_location='cpu')['state_dict']
        model.load_state_dict(state_dict)
        print(f"model loaded from {config.general.init_from_ckpt}")
    callbacks = []

    if config.general.save_model:
        # You can remove empty directories named by '{epoch}-valid'
        # $ find . -type d -empty -exec rm -r {} +
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{config.general.name}",
            # filename='{epoch}',
            filename='{epoch}-{valid/loss:.5f}',
            monitor='valid/loss',
            save_top_k=5,
            mode='min',
            every_n_epochs=1
        )
        checkpoint_callback_perr = ModelCheckpoint(
            dirpath=f"checkpoints/{config.general.name}",
            filename='{epoch}-{valid/norm_perr:.5f}',
            monitor="valid/norm_perr",
            save_top_k=5,
            mode='min',
            every_n_epochs=1,
        )
        # checkpoint_callback_perr = ModelCheckpoint(
        #     dirpath=f"checkpoints/{config.general.name}",
        #     filename='{epoch}-{valid/rmsd_perr:.5f}',
        #     monitor="valid/rmsd_perr",
        #     save_top_k=5,
        #     mode='min',
        #     every_n_epochs=1,
        # )
        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints/{config.general.name}",
            filename='last',
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(checkpoint_callback_perr)
        callbacks.append(last_ckpt_save)

    trainer = Trainer(
        gradient_clip_val=config.train.clip_grad,
        strategy=strategy,  # 'ddp',
        accelerator='gpu' if use_gpu else 'cpu',
        devices=devices,
        max_epochs=config.train.epochs,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        fast_dev_run=10 if name == 'debug' else False,
        enable_progress_bar=True,
        callbacks=callbacks,
        log_every_n_steps=50 if name != 'debug' else 1,
        logger=[],
    )

    if not config.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=config.general.resume)
        # if config.general.name not in ['debug', 'test']:
        #     trainer.test(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule, ckpt_path=config.general.test_only)
        if config.general.evaluate_all_checkpoints:
            directory = pathlib.Path(config.general.test_only).parents[0]
            print(f"directory: {directory}")
            files = os.listdir(directory)
            for file in files:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == config.general.test_only:
                        continue
                    print(f"ckpt_path: {ckpt_path}")
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
