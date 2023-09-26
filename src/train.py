from typing import Optional
from pathlib import Path

from argparse import args

import torch
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb

from dataset import LOCODataModule, ImageNetTransforms
from method import MAEMethod
from common import ImageLogCallback, prepare_model




def train(args: args, **kwargs):
            
    imgnet_transforms = ImageNetTransforms(input_resolution=args.resolution)
    loco_datamodule = LOCODataModule(
        data_root=args.data_root,
        category=args.category,
        input_resolution=args.resolution,
        img_transforms=imgnet_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **kwargs,
    )
    
    mae = prepare_model(
        ckpt_dir=args.ckpt_dir,
        arch=args.arch,
    )
    
    method = MAEMethod(
        mae=mae,
        datamodule=loco_datamodule,
        args=args
    )
    
    method.load_state_dict(torch.load(args.pretrained_model_ckpt)['state_dict'], strict=False)
    
    logger_name = "mae-loco"
    logger = pl_loggers.WandbLogger(
        project="mae-loco",
        name=logger_name,
    )
    
    wandb_run = wandb.init(
        project="mae-loco",
        dir="wandb",
        name=logger_name, 
    )
    
    log_dir = Path(wandb.run.dir).parent
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
        dirpath=str(ckpt_dir),
        save_last=True,
        every_n_epochs=args.ckpt_every_n_epoch,
    )

    wandb.run.summary['ckpt_dir'] = str(ckpt_dir)
    wandb.run.summary['run_id'] = wandb.run.id
    
    trainer = Trainer(
        logger=logger,
        accelerator="cuda" if args.gpus > 1 else None,
        num_sanity_val_steps=args.num_sanity_val_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        devices=args.gpus,
        max_epochs=args.max_epochs,
        log_every_n_steps=50,
        callbacks=[
            LearningRateMonitor("step"),
            ImageLogCallback() if args.is_logger_enabled else None,
            ckpt_callback,
        ] 
    )
    trainer.fit(method, loco_datamodule)
    