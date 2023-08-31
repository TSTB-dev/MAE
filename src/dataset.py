import os
import json
import h5py

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import timm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule


class ImageNetTransforms():
    def __init__(self, input_resolution: Tuple[int, int]):
        
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        self.img_transform = transforms.Compose([
            transforms.Resize(input_resolution, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img: Image) -> torch.Tensor:
        return self.img_transform(img)
    
    def inverse_affine(self, img: torch.Tensor) -> torch.Tensor:
        return img * self.std + self.mean
    
class MVTecLOCODataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, category: str, input_resolution: Tuple[int, int], split: str, \
        img_transforms: transforms, mask_config: dict = None, is_gtmask=False, color='rgb'):
        """Dataset for MVTec LOCO.
        Args:
            data_root: Root directory of MVTecAD dataset. It should contain the data directories for each class under this directory.
            category: Class name. Ex. 'juice_bottle'
            input_resolution: Input resolution of the model.
            split: 'train' or 'test'
            img_transforms: Image transforms applied to input image.
            mask_config(dict): Mask config dict.
            is_gtmask: If True, return the mask image as the target. Otherwise, return the label.
            color: rgb or grayscale
        """
        self.data_root = data_root
        self.category = category
        self.input_resolution = input_resolution
        self.split = split
        self.img_transforms = img_transforms
        self.mask_config = mask_config
        self.is_gtmask = is_gtmask
        self.color = color
        
        if mask_config is not None:
            self.mask_dir = mask_config['mask_dir']
        
        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'valid' or self.split == 'test'
        
        self.files = self.get_files()
        if self.split == 'test':
            # 異常画像の正解に対する変換処理
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(input_resolution),
                    transforms.ToTensor(),
                ]
            )
            # 正常・異常のラベル{0, 1}を作成
            self.labels = []
            for file in self.files:
                status = str(file).split(os.path.sep)[-2]
                if status == 'good':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
    
    def __getitem__(self, index):

        img_file = self.files[index]
        img = Image.open(img_file)
        
        if self.color == 'gray':
            img = img.convert('L')
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        
        if self.mask_config is not None:        
            masks = self.load_masks_from_hdf(img_file)
            masks = masks.astype(np.float32)
            masks = torch.from_numpy(masks)
        
        sample = {"images": img, "files": str(img_file)}
        if self.split == 'train' or self.split == 'valid':
            return sample

        else:
            sample["labels"] = self.labels[index]
            if not self.is_gtmask:
                return sample
            
            # 正常画像はマスクを0で埋める
            if os.path.dirname(img_file).endswith("good"):
                mask = torch.zeros([1, img.shape[-2], img.shape[-1]])

            # 異常画像のマスクを読み込む
            else:
                sep = os.path.sep
                mask = Image.open(
                    img_file.replace(f"{sep}test{sep}", f"{sep}ground_truth{sep}").replace(
                        ".png", "_mask.png"
                    )
                )
                mask = self.mask_transform(mask)
            
            sample["gt_masks"] = mask
            return sample
    
    def _preprocess(self, img: Image, masks: dict) -> np.ndarray:
        img = np.array(img)
        crop_images = [crop_image(mask['crop_box'], img) for mask in masks]
        crop_images = [cv2.resize(crop_image, self.input_resolution) for crop_image in crop_images]
        crop_images = np.array(crop_images)  # -> (K, H, W, C)
        return torch.from_numpy(crop_images)
    
    def __len__(self):
        return len(self.files)
    
    def get_files(self):
        if self.split == 'train':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'train', 'good')).glob('*.png'))
        elif self.split == 'valid':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'validation', 'good')).glob('*.png'))
        elif self.split == 'test':
            normal_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'good')).glob('*.png'))
            logical_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'logical_anomalies')).glob('*.png'))
            struct_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'structural_anomalies')).glob('*.png'))
            files = normal_img_files + logical_img_files + struct_img_files
            
        return files

    def load_masks_from_hdf(self, img_file: Path) -> dict:
        """load correcponding mask file from hdf5 file.
        Args:
            img_file (Path): path to image file
        Returns:
            masks(np.ndarray): (K, H, W)  
        """
        with h5py.File(os.path.join(self.mask_dir, self.category, self.split + '.h5'), 'r') as f:
            dataset_name = img_file.parent.stem + '/' + img_file.stem
            masks = f[dataset_name][:]
        return masks

class LOCODataModule(LightningDataModule): 
    def __init__(self, 
                 data_root: str,
                 category: str,
                 input_resolution: Tuple[int, int],
                 img_transforms: transforms,
                 batch_size: int,
                 num_workers: int,
                 mask_config: dict = None, 
                 **kwargs
                 ) -> None:
        super().__init__()
        
        self.data_root = data_root
        self.category = category
        self.input_resolution = input_resolution
        self.img_transforms = img_transforms
        self.mask_config = mask_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        
    def prepare_data(self):
        pass
    
    def setup(self, stage:str= None):
        self.train_dataset = MVTecLOCODataset(self.data_root, self.category, self.input_resolution, 'train', self.img_transforms, self.mask_config, **self.kwargs)
        self.valid_dataset = MVTecLOCODataset(self.data_root, self.category, self.input_resolution, 'valid', self.img_transforms, self.mask_config,**self.kwargs)
        self.test_dataset = MVTecLOCODataset(self.data_root, self.category, self.input_resolution, 'test', self.img_transforms, self.mask_config, **self.kwargs)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        

if __name__ == '__main__':
    data_root = './data/mvtec_loco'
    category = 'pushpins'
    mask_config = {'top_k': 5, 'mask_dir': './mask_data/mvtec_loco'}
    
    dataset = MVTecLOCODataset(
        data_root=data_root,
        category=category,
        input_resolution=(256, 256),
        split='test',
        img_transforms=None,
        mask_config=mask_config,
    )
    datamodule = LOCODataModule(
        data_root,
        category,
        (256, 256),
        ImageNetTransforms((256, 256)), 
        4,
        4,
        mask_config=mask_config,
    )
    datamodule.setup()
    print(next(iter(datamodule.val_dataloader()))['masks'].shape)
    # crop_boxes: (B, K, 4)
    # crop_images: (B, K, H, W, C)
    # masks: (B, K, H, W)