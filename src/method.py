import random
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import numpy as np
import torch
from torch import Tensor
from torch import optim
from torchvision import utils as vutils

from common import to_rgb_from_tensor, apply_to_batch, tensor2hash, get_patches_containing_mask
import models_vit 
from models_mae import MaskedAutoencoderViT
from fastflow import FastFlow

import wandb

class MAEMethod(pl.LightningModule):
    def __init__(self, mae: MaskedAutoencoderViT, datamodule: pl.LightningDataModule, params: Any, mask_generator: Any = None):
        super(MAEMethod, self).__init__()
        self.model = mae
        self.datamodule = datamodule
        self.params = params
        self.mask_generator = mask_generator
        
        self.seg_mask_cache = {}
    
    def inference(self, inputs: dict, mask_ratio: float = 0.75, **kwargs) -> Tensor:
        """Inference the model.

        Args:
            inputs (dict): batch of images and image files.
            mask_ratio (float, optional): _description_. Defaults to 0.75.

        Returns:
            Tensor: _description_
        """
        files = inputs["files"]
        batch = inputs["images"]
        
        if self.params.gpus > 0:
            batch = batch.to(self.device)
        seg_masks = []
        if self.mask_generator is not None:
            # check if the segmentation masks are already in the cache.
            self.check_cache(batch, files)
            seg_masks = [self.seg_mask_cache[f] for f in files]
        loss, pred, mask = self.forward(batch, seg_masks=seg_masks, **kwargs)
        
        return loss, pred, mask
    
    def forward(self, inputs: Tensor, seg_masks: list = [], mask_indices=[], **kwargs) -> Tensor:
        if seg_masks:
            # use given segmentation masks
            mask_indices = self.mask_selection(seg_masks, self.params.top_k, self.params.mask_range)  # length: B, each element is a list of indices.
            # seg_masks: (B, L), each element is a binary mask which indicates the index of the masked patch.
            # warn: masking is a bit different from the original implementation. because we use the different mask for each element in a batch.
            
            return self.model(inputs, mask_indices=mask_indices, **kwargs)
        else: 
            return self.model(inputs, mask_indices=mask_indices, **kwargs)
    
    def training_step(self, inputs, batch_idx, optimizer_idx=0):
        
        loss, pred, mask = self.inference(inputs)
        self.log("train_loss", loss.mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def sample_images(self, mask_ratio: float = 0.75):
        """Sample images from validation set. """
        valid_dataloader = self.datamodule.val_dataloader()
        
        # sample images randomly in a batch from validation set.
        perm = torch.randperm(self.params.batch_size)
        indices = perm[:self.params.num_samples]
        
        inputs = next(iter(valid_dataloader))
        img_batch = inputs["images"][indices].to(self.device)
        files = [file for idx, file in enumerate(inputs["files"]) if idx in indices]

        _, pred, mask = self.inference(inputs)  # pred: (B, L, p*p*3), mask: (B, L)
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
        val_loss, pred, mask = self.inference(batch)
        val_loss = val_loss.mean().item()  # mean loss across batch
        logs = {"val_loss": val_loss}
        return logs
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(outputs, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params.lr, \
            weight_decay=self.params.weight_decay)
        
        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())
        
        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, \
            lr_lambda=warm_and_decay_lr_scheduler)
        
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", }]
        )
    
    def check_cache(self, batch: Tensor, files: dict):
        """Check if the hash value is already in the cache. If so, return the corresponding segmentation masks.
        If not, predict segmentation masks and save into cache.

        Args:
            batch (Tensor): (B, 3, H, W) tensor.
            files (List): list of file paths (length: B)
        Returns:
            hashes (List[str]): list of hash values (length: B)
        """
        # check if the segmentation masks are already in the cache.
        missing_indices = [i for i, f in enumerate(files) if f not in self.seg_mask_cache]
        if missing_indices:
            # predict segmentation masks
            seg_masks = self.predict_seg_masks(batch[missing_indices])
            
            # save into cache
            for i, idx in enumerate(missing_indices):
                f = files[idx]
                self.seg_mask_cache[f] = seg_masks[i]
        return 
    
    def predict_seg_masks(self, batch: Tensor):
        """Predict segmentation masks for the given batch.

        Args:
            batch (Tensor): (B, 3, H, W) tensor.
        Returns:
            seg_masks (List[dict]): list of dictionaly that contain segmentation masks (length: B)
        """
        # convert batch to numpy array
        batch_arr = batch.permute(0, 2, 3, 1).cpu().numpy()
        batch_arr = (batch_arr * 255).astype(np.uint8)
        
        # generate segmentation masks
        seg_masks = [self.mask_generator.generate(image) for image in batch_arr]
        return seg_masks
    
    def mask_selection(self, seg_masks: list, top_k: int, area_range: list) -> list:
        """Mask selection.

        Args:
            seg_masks (list): list of dictionaly that contain segmentation masks (length: B)
            top_k (int): number of top masks to select
            area_range (list): valid area range of masks, e.g. [0, 1000]

        Returns:
            list: list of segmentation masks (length: B)
        """
        masks = [select_masks_by_area(seg_mask, area_range) for seg_mask in seg_masks]
        topk_masks = [select_topk_masks(mask, top_k) for mask in masks] # length: B, each element is a dict
        
        # randomly select one of the selected masks for each element in a batch.
        selected_masks = [random.choice(masks)["segmentation"] for masks in topk_masks]  # length: B, each element is a np.ndarray(np.bool)
        
        l = [get_patches_containing_mask(selected_mask, self.model.patch_embed.patch_size, self.params.resolution) for \
            selected_mask in selected_masks]  # length: B, each element is a list of np.ndarray
        patches, patch_masks = [list(item) for item in zip(*l)]
        
        return patches


class MAEFlowMethod(pl.LightningModule):
    def __init__(self, flow: FastFlow, mae: MaskedAutoencoderViT, datamodule: pl.LightningDataModule, params: Any, mask_generator: Any = None):
        super(MAEFlowMethod, self).__init__()
        self.backbone = mae
        self.flow = flow
        self.datamodule = datamodule
        self.params = params
        self.mask_generator = mask_generator
        self.seg_mask_cache = {}
        
        # froze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def forward(self, batch: dict, seg_masks: list = [], mask_indices=[], **kwargs) -> Tensor:
        inputs = batch["images"]
        # inputs: (B, 3, H, W)
        B, _, H, W = inputs.shape
        assert H == W, "H and W should be the same."
        P = self.backbone.patch_embed.patch_size[0]
        N = H // P
        
        with torch.no_grad():
            enc_out, _, _ = self.backbone.forward_encoder(inputs, mask_ratio=0.)
            # enc_out: (B, N**2 + 1, D)
        
        # extract patch latents w/o the cls token
        enc_out = enc_out[:, 1:, :]
        # transform enc_out to (B, D, N, N)
        enc_out = enc_out.view(B, N, N, -1).permute(0, 3, 1, 2).contiguous()
        
        # apply flow
        flow_out = self.flow(enc_out)
        # flow_out(dict): "loss": (B, ), "outputs": (B, D, N, N), "anomaly_maps": (B, 1, P, P), "heatmaps": (B, 1, P, P)
        loss = flow_out["loss"]
        
        return loss
    
    def inference(self, batch: dict):
        inputs = batch["images"]
        labels = batch["labels"]
        # inputs: (B, 3, H, W)
        inputs = inputs.to(self.device)
        
        B, _, H, W = inputs.shape
        assert H == W, "H and W should be the same."
        P = self.backbone.patch_embed.patch_size[0]
        N = H // P
        
        with torch.no_grad():
            enc_out, _, _ = self.backbone.forward_encoder(inputs, mask_ratio=0.)
            # extract patch latents w/o the cls token
            enc_out = enc_out[:, 1:, :]
            # transform enc_out to (B, D, N, N)
            enc_out = enc_out.view(B, N, N, -1).permute(0, 3, 1, 2).contiguous()
            flow_out = self.flow(enc_out)
            
        return flow_out
        
    def training_step(self, inputs, batch_idx, optimizer_idx=0):
        loss = self.forward(inputs)
        self.log("train_loss", loss.mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.forward(batch)
        val_loss = val_loss.mean().item()  # mean loss across batch
        logs = {"val_loss": val_loss}
        return logs

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(outputs, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.flow.parameters(), lr=self.params.lr, \
            weight_decay=self.params.weight_decay)
        
        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())
        
        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, \
            lr_lambda=warm_and_decay_lr_scheduler)
        
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", }]
        )
    
    

    
        

def select_topk_masks(mask_list: list, top_k: int):
    """Select top k masks from mask_list based on the mask area.
    Args:
        mask_list (List[dict]): list of masks, each mask is a dict with keys: 'mask', 'area'
        top_k (int): number of top masks to select
    Returns:
        List[dict]: list of selected masks
    """
    assert top_k <= len(mask_list), "top_k should be less than the number of masks"
    assert mask_list[0].get('area') is not None, "mask_list should contain 'area' key"
    
    areas = [mask['area'] for mask in mask_list]
    sort_indices = np.argsort(areas)[::-1]  # descending order
    topk_indices = sort_indices[:top_k]
    
    masks = [mask_list[i] for i in topk_indices]
    return masks

def select_masks_by_area(mask_list, thresh_range: list):
    """Remove bigger masks from the list of masks.
    Args:
        mask_list (list): list of masks (np.ndarray)
        thresh_range (list): valid area range of masks, e.g. [0, 1000]
    """
    return [mask for mask in mask_list if mask["area"] < thresh_range[1] and mask["area"] > thresh_range[0]]
