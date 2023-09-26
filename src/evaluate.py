import os
import sys
import random
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Any, List, Tuple, Dict, Union

import cv2
import hashlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import roc_auc_score

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn as nn
from torchvision import utils as vutils
from torchvision.transforms import ToPILImage
from IPython.display import display
from ipywidgets import interactive, IntSlider, HBox, VBox
import torch.nn.functional as F

sys.path.append('../src')
import models_mae
from dataset import LOCODataModule, ImageNetTransforms
from models_mae import MaskedAutoencoderViT
from method import MAEMethod, MAEFlowMethod
from params import MAEParams
from common import ImageLogCallback, prepare_model, to_rgb_from_tensor, get_patches_containing_bbox_1D, get_patches_containing_mask
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from fastflow import FastFlow


def main(params: Optional[MAEParams] = None, **kwargs):
    if params is None:
        params = MAEParams()
    
    params = MAEParams()
    img_transforms = ImageNetTransforms(input_resolution=params.resolution)
    datamodule = LOCODataModule(
        data_root=params.data_root,
        category=params.category,
        input_resolution=params.resolution,
        img_transforms=img_transforms,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
    )
    flow = FastFlow(
        feature_dims=[780],
        out_resolutions=[(14, 14)],
        input_resolution=params.resolution,
    )

    mae = getattr(models_mae, params.arch)()

    method = MAEMethod(
        mae=mae,
        datamodule=datamodule,
        params=params,
    )

    method.load_state_dict(torch.load(params.pretrained_model_ckpt)['state_dict'], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    method.eval()
    method.to(device)
    
    
    # load dataset
    datamodule.setup(stage='test')
    train_dataset = datamodule.train_dataset
    test_dataset = datamodule.test_dataset
    valid_dataset = datamodule.valid_dataset
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    valid_dataloader = datamodule.val_dataloader()

    train_files = datamodule.train_dataset.files
    test_files = datamodule.test_dataset.files

    labels = test_dataloader.dataset.labels
    abnormal_indices = [idx for idx, label in enumerate(labels) if label == 1]
    normal_indices = [idx for idx, label in enumerate(labels) if label == 0]

    # indices of the logical anomalies and structural anomalies
    log_indices = [idx for idx, file in enumerate(test_dataloader.dataset.files) if 'logical' in str(file)]
    str_indices = [idx for idx, file in enumerate(test_dataloader.dataset.files) if 'structural' in str(file)]
    error_arr = np.zeros((len(test_dataloader.dataset), params.eval_iter))
    
    for i in range(params.eval_iter):
        error_list = []
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                _, predicts, masks = method.inference(batch, mask_ratio=0.80)  # (B, N, C)
                _, targets, _ = method.inference(batch, mask_ratio=0.0)  # (B, N, C)
            
            fsz = predicts.shape[-1]
            patch_size = method.model.patch_embed.patch_size[0]
            num_patches = params.resolution[0] // patch_size
            mask_indices = [torch.where(m == 1.)[0] for m in masks]  # [(Nm, ), xB]
            pred_masked = torch.stack([predicts[i, m_ind, :] for i, m_ind in enumerate(mask_indices)])  # (B, Nm, C)
            tar_masked = torch.stack([targets[i, m_ind, :] for i, m_ind in enumerate(mask_indices)])  # (B, Nm, C)
            
            # compute error map
            error_masked = torch.sum((pred_masked - tar_masked) ** 2, dim=2, keepdim=False)  # (B, Nm)
            error_masked = torch.mean(error_masked, dim=1, keepdim=False).cpu().tolist()  # [B, ]
            error_list += error_masked 
        error_arr[:, i] = error_list # (N, )
    
    error_arr = np.mean(error_arr, axis=1)  # (N, )
    output_dir = Path("/home/dl/takamagahara/hutodama/MAE/output")
    
    # compute Image-level AUROC
    im_scores = list(error_arr)
    im_auroc = compute_image_auroc(im_scores, test_dataset.labels)
    print(f"Image-level AUROC: {im_auroc}")
    
def save_image(tensor: Tensor, save_path: Union[str, Path]):
    """Save tensor as image.

    Args:
        tensor (Tensor): tensor to save, shape (H, W)
        save_path (Union[str, Path]): save path
    """
    cv2.imwrite(str(save_path), tensor.numpy()*255)
            
def compute_error_map(inputs: Tensor, reconst: Tensor):
    """Compute error map between inputs and reconstructed images.

    Args:
        inputs (Tensor): input images, shape (B, C, H, W)
        reconst (Tensor): reconstructed images, shape (B, C, H, W)
    Returns:
        error_map (Tensor): error map, shape (B, H, W)
    """
    assert inputs.shape == reconst.shape
    error_map = torch.mean((inputs - reconst) ** 2, dim=1, keepdim=False)
    return error_map

def compute_image_auroc(scores: list, labels: list):
    """Compute image-level AUROC.

    Args:
        scores (list): list of scores, shape (N,)
        labels (list): list of labels, shape (N,)
    Returns:
        auroc (float): image-level AUROC
    """
    assert len(scores) == len(labels)
    scores = np.array(scores)
    labels = np.array(labels)
    auroc = roc_auc_score(labels, scores)
    return auroc

if __name__ == "__main__":
    main()