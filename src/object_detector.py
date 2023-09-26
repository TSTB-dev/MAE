import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import cv2
import mmcv


from mmdet.apis import DetInferencer
import torch
import itertools
from typing import List, Optional, Tuple, Union


def xy_to_flat(x: int, y: int, img_size: int) -> int:
    """Convert (x, y) to flat index.
    Args:
        x (int): x coordinate.
        y (int): y coordinate.
        img_size (int): Image size.
    Returns:
        int: Flat index.
    """
    return y * img_size + x

def flat_to_xy(flat: int, img_size: int) -> Tuple[int, int]:
    """Convert flat index to (x, y).
    Args:
        flat (int): Flat index.
        img_size (int): Image size.
    Returns:
        Tuple[int, int]: (x, y) coordinate.
    """
    return flat % img_size, flat // img_size

def convert_bbox_to_tokens(bbox: list[int], patch_size: int, img_size: int) -> list[int]:
    """Convert bbox to token indices.
    Args:
        bbox (list[int]): Bounding box (left, top, right, bottom) of the object.
        patch_size (int): Patch size.
        img_size (int): Image size.
    Returns:
        list[int]: Token indices.
    """
    # Check bbox format.
    assert len(bbox) == 4, f"Invalid bbox format: {bbox}"
    assert bbox[2] < img_size and bbox[3] < img_size, f"Invalid bbox: {bbox}"
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3], f"Invalid bbox: {bbox}"
    
    # Convert bbox to token indices.
    x_min, y_min, x_max, y_max = [int(pos) for pos in bbox]  
    x_min_p, y_min_p, x_max_p, y_max_p = x_min // patch_size, y_min // patch_size, x_max // patch_size, y_max // patch_size
    
    # Check bbox size.
    # in_bbox_tokens: list[tuple[int, int]], list of token indices in the bbox.
    in_bbox_tokens = list(itertools.product(range(x_min_p, x_max_p + 1), range(y_min_p, y_max_p + 1)))
    in_bbox_tokens = [xy_to_flat(x, y, patch_size) for x, y in in_bbox_tokens]  # convert (x, y) to flat index. list[int]
    
    return in_bbox_tokens

def bbox_token_mask(bbox: list[int], patch_size: int, img_size: int) -> list[int]:
    """Get mask tokens of bbox.
    Args:
        bbox (list[int]): Bounding box (left, top, right, bottom) of the object.
        patch_size (int): Patch size.
        img_size (int): Image size.
    Returns:
        tensor: Mask tokens. (num_tokens, )
    """
    # Check bbox format. -> list[int]. Token indices.
    in_bbox_tokens = convert_bbox_to_tokens(bbox, patch_size, img_size)
    
    # Mask tokens. 1: in bbox, 0: out of bbox.
    mask_tokens = torch.zeros(patch_size * patch_size)
    mask_tokens[in_bbox_tokens] = 1
    
    return mask_tokens
    
def mask_img(img, img_size, patch_size, mask_tokens: list):
    """Mask image with mask tokens.
    Args:
        img (np.ndarray): Image.
        mask_tokens (list[int]): Mask tokens.
    Returns:
        np.ndarray: Masked image.
    """
    img = img.copy()
    for token in mask_tokens:
        x, y = token
        img[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size] = 0
    return img

def top_k_results(result, top_k=10) -> tuple:
    """Get top k results from detection result.

    Args:
        result (dict): detection result.
        top_k (int, optional): the number of top-k selection. Defaults to 10.

    Returns:
        tuple: (bboxes, scores, labels)
    """
    bboxes = np.array(result['predictions'][0]['bboxes'])
    scores = np.array(result['predictions'][0]['scores'])
    labels = np.array(result['predictions'][0]['labels'])
    top_k = list(sorted(range(len(scores)), key=lambda k: scores[k]))[::-1][:top_k]
    return bboxes[top_k], scores[top_k], labels[top_k]

def apply_nms(boxes, scores, threshold=0.5):
    """Apply non-maximum suppression to remove overlapping bounding boxes.
    Args:
        boxes (list): List of bounding boxes.
        scores (list): List of scores.
        threshold (float, optional): Threshold. Defaults to 0.5.
    """
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=threshold)
    return indices

def remove_big_boxes(boxes: list, img_size: int, threshold=0.5):
    """Remove big boxes.

    Args:
        boxes (list): List of bounding boxes.
        img_size (int): image size.
        threshold (float, optional): The threshold ratio of bbox area. Defaults to 0.5.

    Returns:
        list: List of bounding boxes.
    """
    bbox_areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in boxes]
    filtered_indices = [i for i, area in enumerate(bbox_areas) if area < img_size * img_size * threshold]
    return [boxes[i] for i in filtered_indices]
    
class ObjectDetector(object):
    def __init__(self, config_path: str, ckpt_path: str, img_size: int, patch_size: int, device: str = 'cpu'):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.img_size = img_size
        self.patch_size = patch_size
        self.inferencer = DetInferencer(config_path, ckpt_path, device=device)
        
    def predict(self, image_path: str, top_k: int = 10, nms_threshold: float = 0.99, num_objects: int = 6) -> tuple:
        """Detect objects in an image.
        Args:
            image_path (str): Path to the image.
            top_k (int, optional): The number of top-k selection. Defaults to 10.
            nms_threshold (float, optional): The threshold of non-maximum suppression. Defaults to 0.99.
            num_objects (int): The number of objects to detect.
        Returns:
            tuple: (bbox_token_mask, token_indices, filtered_boxes)
        """
        
        # preprocess
        image = self._preprocess(image_path)
        
        # inference
        result = self.inferencer(image)
        
        # get top k results
        bboxes, scores, labels = top_k_results(result, top_k=top_k)
        
        # NMS
        filtered_indices = apply_nms(bboxes, scores, threshold=0.99)
        filtered_boxes = [bboxes[i] for i in filtered_indices]
        
        # remove_big_boxes
        filtered_boxes = remove_big_boxes(filtered_boxes, self.img_size, threshold=0.5)
        
        # restrict hte number of bboxes
        filtered_boxes = filtered_boxes[:num_objects]
        
        # convert to token indices
        token_indices = [convert_bbox_to_tokens(bbox, self.patch_size, self.img_size) for bbox in filtered_boxes]   # List[List[int]]
        token_masks = [bbox_token_mask(bbox, self.patch_size, self.img_size) for bbox in filtered_boxes]   # List[Tensor]
        token_mask = torch.stack(token_masks, dim=0)   # Tensor (num_boxes, num_tokens)
        
        return token_mask, token_indices, filtered_boxes
    
    def _preprocess(self, image_path: str) -> tuple:
        """Preprocess image.
        Args:
            image_path (str): Path to the image.
        Returns:
            tuple: (image, image_pil, image_cv)
        """
        image = cv2.imread(image_path)
        resized_img = cv2.resize(image, (self.img_size, self.img_size))
        return resized_img
