from typing import Any
import hashlib
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import optim, Tensor
from torchvision import utils as vutils

from typing import Tuple, Union

import numpy as np

from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
from pytorch_lightning import Callback

import wandb
import timm

import models_mae



imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def unnormalize(x: torch.Tensor) -> torch.Tensor:
    return x * imagenet_std + imagenet_mean

def to_rgb_from_tensor(x: torch.Tensor) -> torch.Tensor:
    """Converts a 4D tensor to an RGB [0, 1] image.
    Args:
        x: (B, 3, H, W) tensor
    Returns:
        (B, 3, H, W) tensor
    """
    x = unnormalize(x)
    return x.clamp(0, 1)

class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        "Called when the train epoch ends."
        
        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.log({
                    "images": [wandb.Image(images)
                ]}, commit=False)
                
def prepare_model(ckpt_dir: str, arch="mae_vit_base_patch16"):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    ckpt = torch.load(ckpt_dir, map_location="cpu")
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(msg)
    return model

def get_patches_containing_bbox_1D(bbox: list, patch_size: list, img_size: list):
    """Get patche indices containing the bbox.

    Args:
        bbox (_type_): bbox in the format [xmin, ymin, xmax, ymax]
        patch_size (_type_): patch size in the format [width, height]
        img_size (_type_): image size in the format [width, height]

    Returns:
        list of patch indices that contain the bbox
    """
    xmin, ymin, xmax, ymax = bbox
    patch_width, patch_height = patch_size
    img_width, img_height = img_size
    
    patches = []
    
    # Calculate number of patches in width and height
    n_patches_width = img_width // patch_width
    n_patches_height = img_height // patch_height

    # Iterate over patches and check if the bbox intersects with the patch
    for i in range(n_patches_height):
        for j in range(n_patches_width):
            patch_xmin = j * patch_width
            patch_ymin = i * patch_height
            patch_xmax = patch_xmin + patch_width
            patch_ymax = patch_ymin + patch_height

            # Check for intersection using the separating axis theorem
            if (xmin < patch_xmax and xmax > patch_xmin and
                ymin < patch_ymax and ymax > patch_ymin):
                patch_index_1D = i * n_patches_width + j
                patches.append(patch_index_1D)
    
    return patches

def get_patches_containing_mask(mask, patch_size: list, img_size: list):
    """Get patche indices containing the mask region.

    Args:
        mask (np.ndarray): np.array of mask. format: (H, W), dtype: np.bool
        patch_size (list): patch size in the format [width, height]
        img_size (list): image size in the format [width, height]

    Returns:
        patches (list): list of patch indices that contain the mask region. format: (N, )
        patch_masks (list): list of patch masks that indicate masked patches. format: (N, )
    """
    patch_width, patch_height = patch_size
    img_width, img_height = img_size
    
    patches = []
    
    # Calculate number of patches in width and height
    n_patches_width = img_width // patch_width
    n_patches_height = img_height // patch_height

    # Iterate over patches and check if the mask region intersects with the patch
    for i in range(n_patches_height):
        for j in range(n_patches_width):
            patch_xmin = j * patch_width
            patch_ymin = i * patch_height
            patch_xmax = patch_xmin + patch_width
            patch_ymax = patch_ymin + patch_height

            # Check if any pixel in the patch falls within the mask region
            if np.any(mask[patch_ymin:patch_ymax, patch_xmin:patch_xmax] == 1):
                patch_index_1D = i * n_patches_width + j
                patches.append(patch_index_1D)
    
    patches = list(set(patches))
    patch_masks = [1 if i in patches else 0 for i in range(n_patches_width * n_patches_height)]
    return patches, patch_masks

def apply_to_batch(batch: Tensor, func):
    """Apply a function to a batch.
    Args:
        batch: (B, ...)
        func: a function that takes a tensor and returns a tensor.
    Returns:
        (B, ...)
    """
    results = [func(item) for item in batch]
    return results

def tensor2hash(tensor):
    tensor = tensor.detach().cpu()
    rounded_tensor = torch.round(tensor * 1000) / 1000
    tensor_str = str(rounded_tensor.tolist())
    hash_val = hashlib.md5(tensor_str.encode()).hexdigest()
    return hash_val
    # byte_data = tensor_str.cpu().numpy().tobytes()
    # return hashlib.md5(byte_data).hexdigest()