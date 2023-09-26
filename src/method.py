import random
from typing import Any

import pytorch_lightning as pl
import numpy as np
import torch
from torch import Tensor
from torch import optim
from torchvision import utils as vutils

from common import to_rgb_from_tensor, get_patches_containing_mask
from models_mae import MaskedAutoencoderViT

class MAEMethod(pl.LightningModule):
    def __init__(self, mae: MaskedAutoencoderViT, datamodule: pl.LightningDataModule, args: Any):
        super(MAEMethod, self).__init__()
        self.model = mae
        self.datamodule = datamodule
        self.args = args
    
    def forward(self, inputs: Tensor, mask_indices=[], mask_ratio=0.75, **kwargs) -> Tensor:
        return self.model(inputs, mask_indices=mask_indices, mask_ratio=mask_ratio, **kwargs)
    
    def training_step(self, inputs, batch_idx, optimizer_idx=0):
        
        loss, pred, mask = self.forward(inputs)
        self.log("train_loss", loss.mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def sample_images(self, mask_ratio: float = 0.75):
        """Sample images from validation set. """
        valid_dataloader = self.datamodule.val_dataloader()
        
        # sample images randomly in a batch from validation set.
        perm = torch.randperm(self.args.batch_size)
        indices = perm[:self.args.num_samples]
        
        inputs = next(iter(valid_dataloader))
        img_batch = inputs["images"][indices].to(self.device)
        files = [file for idx, file in enumerate(inputs["files"]) if idx in indices]

        _, pred, mask = self.forward(inputs)  # pred: (B, L, p*p*3), mask: (B, L)
        pred = pred[indices]
        mask = mask[indices]
        
        patch_size = self.model.patch_embed.patch_size[0]
        mask = mask.detach()  # -> (B, L)
        mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 * 3)  # -> (B, L, p*p*3)
        mask = self.model.unpatchify(mask)  # -> (B, 3, H, W)
        
        pred = self.model.unpatchify(pred)  # -> (B, 3, H, W)
        
        # mask input images.
        im_masked = img_batch * (1 - mask) # -> (B, 3, H, W)
        im_paste = im_masked + pred * mask # -> (B, 3, H, W)
        
        # convert tensor to rgb format.
        images = to_rgb_from_tensor(img_batch.cpu()) # -> (B, 3, H, W)
        im_masked = to_rgb_from_tensor(im_masked.cpu())  # -> (B, 3, H, W)
        im_paste = to_rgb_from_tensor(im_paste.cpu())  # -> (B, 3, H, W)
        
        # combine images in a way so we can display all outputs in one grid.
        out = torch.cat([images.unsqueeze(1), im_masked.unsqueeze(1), im_paste.unsqueeze(1)], dim=1)  # -> (B, 3, 3, H, W)
        out = out.view(-1, *out.shape[2:])  # -> (3*B, 3, H, W)
        
        images = vutils.make_grid(
            out.cpu(),
            nrows=out.shape[0] // 3,
            ncols=3,
            normalize=False, 
        )
        # images: (3, H, W)
        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss, pred, mask = self.forward(batch)
        val_loss = val_loss.mean().item()  # mean loss across batch
        logs = {"val_loss": val_loss}
        return logs
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(outputs, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, \
            weight_decay=self.args.weight_decay)
        
        warmup_steps_pct = self.args.warmup_steps_pct
        decay_steps_pct = self.args.decay_steps_pct
        total_steps = self.args.max_epochs * len(self.datamodule.train_dataloader())
        
        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.args.scheduler_gamma ** (step / decay_steps)
            return factor
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, \
            lr_lambda=warm_and_decay_lr_scheduler)
        
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", }]
        )

