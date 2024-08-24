import math
import os

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.functional import interpolate
from torchvision import transforms
from torchvision.io import read_video, read_video_timestamps
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from transformers import VideoMAEFeatureExtractor, VideoMAEModel

from .videogpt import load_vqvae
from .videogpt.data import preprocess

# img_path="00_002.jpg"


class VideoGPTFT(torch.nn.Module):
    def __init__(
        self,
        model_id="kinetics_stride2x4x4",
        output="dense",
        time_step=250,
        layer=1,
        return_multilayer=True,
    ):
        super().__init__()
        assert output in ["gap", "dense"], "Only supports gap or dense output"
        self.features = {}
        self.output = output
        self.checkpoint_name = model_id + f"_noise-{time_step}"
        self.patch_size = 16
        self.timestep = time_step
        assert layer in [-1, 0, 1, 2, 3]
        self.sequence_length = 1
        self.resolution = 128
        self.device = torch.device("cuda")

        self.model = load_vqvae("kinetics_stride2x4x4").to(self.device)

        # feat_dims = [1280, 1280, 640, 320]
        multilayers = [0, 1, 2, 3]
        self.multilayers = multilayers[:4]
        # self.feat_dim=[self.model.config.hidden_size]*4
        self.feat_dim = [128] * 4
        # if return_multilayer:
        #     self.feat_dim = feat_dims
        #     self.multilayers = multilayers
        # else:
        #     layer = multilayers[-1] if layer == -1 else layer
        #     self.feat_dim = feat_dims[layer]
        #     self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # print(name,output.shape)
                self.features[name] = output.detach()

        return hook

    def process_images(self, batch_tensor, k=16):
        # Ensure input is a 4D tensor: (batch_size, channels, height, width)
        if batch_tensor.ndimension() != 4:
            raise ValueError(
                "Input tensor must be 4-dimensional (batch_size, channels, height, width)"
            )

        # Get device of the input tensor
        device = batch_tensor.device
        # Resize each image to 224x224
        resized_images = F.interpolate(
            batch_tensor, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Duplicate each image k times
        batch_size, channels, height, width = resized_images.shape
        duplicated_images = resized_images.unsqueeze(1).expand(
            batch_size, k, channels, height, width
        )

        # Ensure the final tensor is on the same device as the input tensor
        duplicated_images = duplicated_images.to(device)
        # print("frames",duplicated_images.shape)
        return duplicated_images

    def forward(self, images, categories=None, prompts=None):
        spatial = []
        batch_size = images.shape[0]

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (128, 128)
                ),  # Resize the image to match the resolution
                # transforms.ToTensor(),  # Convert the image to a tensor
            ]
        )

        img = images[0]
        img_tensor = (
            transform(img).unsqueeze(0).to(self.device)
        )  # Add batch dimension and move to GPU
        img_tensor = img_tensor.permute(0, 2, 3, 1)
        pixels = preprocess(img_tensor, self.resolution, self.sequence_length).to(
            self.device
        )
        # Register hooks for each layer
        for name, layer in self.model.named_modules():
            # print(name)
            layer.register_forward_hook(self.hook_fn(name))

        pixels = pixels.repeat(1, 4, 1, 1).unsqueeze(0)
        # print(pixels.shape)
        pixels = pixels.to(self.device)
        encodings = self.model.encode(pixels)

        # Print the shape of the output features from each layer
        feat = []
        for name, feature in self.features.items():
            # print(name)
            # print(feature.shape)
            feature = reshape_and_interpolate_to_fixed_shape(feature)
            if feature is not None:
                feat.append(feature)
                # print(feature.shape)

        feat = torch.cat(feat, dim=0)
        # print(feat.shape)
        # print("Num feat",len(feat))
        # print(f"Layer: {name}, Feature shape: {feature.shape}")
        # Calculate the size in bytes
        # size_in_bytes = feat.element_size() * feat.nelement()

        # Convert bytes to megabytes (1 MB = 1,048,576 bytes)
        # size_in_mb = size_in_bytes / (1024 * 1024)
        # print(size_in_mb)
        return feat


import operator
from functools import reduce

import torch.nn.functional as F


def reshape_and_interpolate_to_fixed_shape(tensor, target_shape=(480, 64, 64)):
    target_total_size = reduce(operator.mul, target_shape)

    # Reshape to a near shape (merge dimensions)
    current_size = tensor.numel()
    # if current_size == target_total_size:
    #     tensor_reshaped = tensor.view(*target_shape)
    # elif current_size > target_total_size:
    #     factor = current_size // target_total_size
    #     tensor_reshaped = tensor.view(factor, *target_shape).view(*target_shape)
    # else:
    #     # return None
    #     # if len(tensor.shape) > 5:  # Assuming we have more than (N, C, H, W, D)
    #     #     tensor = tensor.view(-1, *tensor.shape[-3:])

    #     # # Ensure the tensor has the required format: (N, C, H, W) or (N, C, D1, D2, D3)
    #     # if len(tensor.shape) == 3:
    #     #     tensor = tensor.unsqueeze(1)  # Adding a channel dimension if necessary
    #     # print(tensor.shape)
    #     # # The tensor needs to be interpolated to match the target shape
    #     # tensor_reshaped = tensor.view(1, 1, *tensor.shape).float()  # Add channels for interpolation
    #     # print(tensor_reshaped.shape)
    #     # tensor_reshaped = F.interpolate(tensor_reshaped, size=target_shape, mode='trilinear', align_corners=False)
    #     # tensor_reshaped = tensor_reshaped.view(*target_shape)
    height = tensor.size(-2)
    width = tensor.size(-1)
    if height != width:
        return None
    tensor = tensor.view(-1, tensor.size(-2), tensor.size(-1))
    grid_size = int(math.sqrt(tensor.shape[0]))
    channels = tensor.shape[0]
    grid_size_w = grid_size
    grid_size_h = grid_size
    # print(tensor.shape)
    if grid_size * grid_size != tensor.shape[0]:
        # print(grid_size,tensor.shape[0])
        grid_size_h = pow(2, int(math.log2(grid_size)) + 1)
        grid_size_w = channels // grid_size_h

    # Step 1: Reshape the tensor into a 4D tensor (grid_size, grid_size, height, width)
    tensor = tensor.view(
        grid_size_h, grid_size_w, height, width
    )  # Shape: (grid_size, grid_size, height, width)

    # Step 2: Permute the tensor to move the grid dimensions next to each other
    tensor = tensor.permute(2, 0, 3, 1)  # Shape: (height, grid_size, width, grid_size)

    # Step 3: Reshape the tensor to combine the grid into the height and width dimensions
    tensor = tensor.contiguous().view(
        height * grid_size_h, width * grid_size_w
    )  # Shape: (height * grid_size, width * grid_size)

    # Add a batch dimension if necessary
    tensor = tensor.unsqueeze(0)  # Shape: (1, height * grid_size, width * grid_size)

    # Now the tensor shape is [480, 32, 32]
    # Step 2: Interpolate to the target shape [480, 64, 64]
    tensor_reshaped = F.interpolate(
        tensor.unsqueeze(0), size=(512, 480), mode="bilinear", align_corners=False
    )
    # tensor_reshaped=tensor
    # print(tensor_reshaped.shape)
    return tensor_reshaped
