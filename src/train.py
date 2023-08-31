from typing import Optional
from pathlib import Path

import torch
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms

import wandb

from dataset import LOCODataModule, ImageNetTransforms
from models_mae import MaskedAutoencoderViT
from method import MAEMethod
from params import MAEParams
from common import ImageLogCallback, prepare_model

import cv2



def main(params: Optional[MAEParams] = None, **kwargs):
    if params is None:
        params = MAEParams()
            
    imgnet_transforms = ImageNetTransforms(input_resolution=params.resolution)
    
    loco_datamodule = LOCODataModule(
        data_root=params.data_root,
        category=params.category,
        input_resolution=params.resolution,
        img_transforms=imgnet_transforms,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        **kwargs,
    )
    
    mae = prepare_model(
        ckpt_dir=params.ckpt_dir,
        arch=params.arch,
    )
    
    method = MAEMethod(
        mae=mae,
        datamodule=loco_datamodule,
        params=params
    )
    
    method.load_state_dict(torch.load(params.pretrained_model_ckpt)['state_dict'], strict=False)
    
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
        save_top_k=-1, # TODO: ,
        dirpath=str(ckpt_dir),
        save_last=True,
        every_n_epochs=params.ckpt_every_n_epoch,
    )

    wandb.run.summary['ckpt_dir'] = str(ckpt_dir)
    wandb.run.summary['run_id'] = wandb.run.id
    
    trainer = Trainer(
        logger=logger,
        accelerator="cuda" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        devices=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        callbacks=[
            LearningRateMonitor("step"),
            # ImageLogCallback() if params.is_logger_enabled else None,
            ckpt_callback,
        ] 
    )
    trainer.fit(method, loco_datamodule)

if __name__ == "__main__":
    main()
    