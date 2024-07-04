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
        pixels = pixels.repeat(1, 4, 1, 1).unsqueeze(0)
        # print(pixels.shape)
        pixels = pixels.to(self.device)
        encodings = self.model.encode(pixels)
        return encodings
