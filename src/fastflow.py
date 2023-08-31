from ast import literal_eval
from typing import *

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class FastFlow(nn.Module):
    def __init__(
        self,
        feature_dims: List[int],
        out_resolutions: List[Tuple[int, int]],
        input_resolution: Tuple[int, int],
        **kwargs
    ):
        """
        Args:
            feature_dims: the dimensions of the feature maps.
            out_resolutions: the resolutions of the feature maps.
            input_resolution: the input image resolution. format: (H, W)
        """
        self.feature_dims = feature_dims
        self.out_resolutions = out_resolutions
        self.input_resolution = input_resolution
        
        super(FastFlow, self).__init__()

        self.flows = nn.ModuleList()

        # create Normalizing Flow
        for idx, (dim, resolution) in enumerate(zip(self.feature_dims, self.out_resolutions)):
            self.flows.append(
                fast_flow(
                    feature_shape=[768, 14, 14],
                    kernel_size=3,
                    use_3x3_only=False,
                    hidden_ratio=1.0,
                    num_flow_steps=8,
                )
            )
        self.input_resolution = input_resolution
        

    def forward(self, inputs: Union[List, Tensor]) -> Tensor: 

        
        loss = 0
        outputs = []

        # output: (B, D, H', W')
        # log_jac_dets: (B, )
        output, log_jac_dets = self.flows[0](inputs)
        loss += self.negative_log_likelihood(output, log_jac_dets)
        outputs.append(output)
            
        ret = {"loss": loss}
        
        anomaly_maps = []
        prob_maps = []
        
        if not self.training:
            ret['outputs'] = outputs
            for output in outputs:
                log_prob_map = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob_map = torch.exp(log_prob_map)
                anomaly_map = F.interpolate(
                    -1 * prob_map,
                    size=[self.input_resolution[0], self.input_resolution[1]],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_maps.append(anomaly_map)  # -> (M], B, 1, H, W)
                prob_maps.append(prob_map)  # -> (M], B, 1, H, W)
            
            # -> (B, 1, P, P, M)
            anomaly_maps = 1 + torch.stack(anomaly_maps, dim=-1)
            ret["anomaly_maps"] = anomaly_maps
            ret["prob_maps"] = prob_maps

            # take the average of the anomaly maps
            # -> (B, 1, P, P)
            heatmaps = torch.mean(anomaly_maps, dim=-1)
            ret["heatmaps"] = heatmaps
        
        return ret
    
    def negative_log_likelihood(self, x: Tensor, log_jac_dets: Tensor) -> Tensor:
        """Calculates the negative log-likelihood of the input.

        Args:
            x (Tensor): The input tensor. shape: (B, D, H, W)
            log_jac_dets (Tensor): The log-determinant of the Jacobian. shape: (B, )

        Returns:
            Tensor: The negative log-likelihood of the input.
        """
        return torch.mean(
                0.5 * torch.sum(x**2, dim=(1, 2, 3)) - log_jac_dets
        )

def subnet_conv_func(kernel_size: int, hidden_ratio: float):
    """Returns a function that returns a two-layer convolutional layer with the specified kernel size and number of kernels.
    Args:
        kernel_size: The kernel size of the convolutional layer.
        hidden_ratio: The number of channels in the hidden layer relative to the input channel.

    Returns:
        subnet_conv: A function that takes input and output channel numbers and returns a two-layer convolutional layer.
    """
    def subnet_conv(in_channels: int, out_channels: int):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def fast_flow(feature_shape: list, kernel_size: int, use_3x3_only: bool, \
    hidden_ratio: float, num_flow_steps: int, pos_encode: bool = False, clamp: float = 2.0):
    """2D-Normalizing Flow in FastFlow
    Args:
        feature_shape: A shape of feature map, format: (C, H, W)
        kernel_size: The kernel size of the convolutional layer.
        use_3x3_only: In the internal convolutional layers of the Flow, whether to use only 3x3 convolutional layers or alternate between 1x1 and 3x3 convolutional layers.        hidden_ratio: 特徴マップのチャンネル数の拡張率.
        num_flow_steps: The number of flows.
        pos_encode(bool): whether to use positional encoding.
        clamp: The clamping parameter [s] for scaling terms in the affine coupling layer.

    Returns:
        nodes: A list of nodes that make up the Normalizing Flow.
    """
    nodes = Ff.SequenceINN(*feature_shape)
    for i in range(num_flow_steps):
        if i % 2 == 1 and not use_3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
                Fm.AllInOneBlock,
                subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
                affine_clamping=clamp,
                permute_soft=False,
            )  
    return nodes


def build_fastflow(backbone_name: str, flow_arch: str, input_resolution: Tuple[int, int], block_indices: List[int], \
    flow_config: dict, pre_concat: bool, \
        **kwargs) -> torch.nn.Module:
    """Builds a FastFlow model.

    Args:
        backbone_name (str): pre-trained model name.
        flow_arch (str): the architecture of the Normalizing Flow model.
        input_resolution (Tuple[int, int]): tuple of input image resolution. format: (H, W)
        block_indices (List[int]): the indices of the blocks to be used as feature maps.
        flow_config (Dict[str, Any]): the configuration of the Normalizing Flow model.
        pre_concat (bool): Boolean value indicating whether to concatenate the input to the output of the first convolutional layer within the Normalizing Flow model.

    Returns:
        torch.nn.Module: FastFlow model instance.
    """

    model = FastFlow(
        backbone_name,
        flow_arch,
        input_resolution,
        block_indices,
        flow_config,
        pre_concat,
        **kwargs
    )

    # Get the number of parameters of the model
    print(
        "Total model Param#: {}[MB]".format(
            sum(p.numel() for p in model.parameters())/1e+6
        )
    )
    return model


def build_optimizer(model) -> torch.optim.Optimizer:
    """build optimizer
    """
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )