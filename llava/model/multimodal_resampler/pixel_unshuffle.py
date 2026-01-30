import torch
import torch.nn as nn
import math
import functools


class PixelUnShuffle(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()
       

        assert hasattr(vision_tower.config, "merge_kernel_size"), "vision_tower.config must have attribute 'merge_kernel_size'"
        
        spacial_reduction = functools.reduce(lambda x, y: x*y, vision_tower.config.merge_kernel_size)
        
        self.out_channels = vision_tower.hidden_size * spacial_reduction


    def forward(self, image_features, images, *args, **kwargs):
            return image_features.flatten(-2, -1)  # flattens the second-to-last dim into the last one

    @property
    def config(self):
        return {
            "mm_resampler_type": "pixel_unfshuffle",
            "mm_spatial_pool_out_channels": self.out_channels,
        }

    @property
    def hidden_size(self):
        return self.out_channels
