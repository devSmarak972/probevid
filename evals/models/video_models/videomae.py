import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from transformers import VideoMAEFeatureExtractor, VideoMAEModel


class VideoMaeFT(torch.nn.Module):
    def __init__(
        self,
        model_id="MCG-NJU/videomae-base",
        output="dense",
        time_step=250,
        layer=1,
        return_multilayer=True,
    ):
        super().__init__()
        assert output in ["gap", "dense"], "Only supports gap or dense output"

        self.output = output
        self.checkpoint_name = model_id.split("/")[1] + f"_noise-{time_step}"
        self.patch_size = 16
        self.timestep = time_step
        self.up_ft_index = [0, 1, 2, 3]  # keep all the upblock feats
        assert layer in [-1, 0, 1, 2, 3]
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(
            "MCG-NJU/videomae-base"
        )

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

        pixels = self.process_images(images, 16)
        # List to hold the series of PIL images for each batch
        # batch_of_series_of_pil_images = []
        # frames=torch.empty((len(images),16,3,224,224))
        i = 0

        # frames_final=np.array(frames_final)

        # print(images.shape)

        # features = self.feature_extractor.preprocess(images,return_tensors="pt")
        # print(type(features),len(features))
        # handle prompts

        spatial = self.model(pixel_values=pixels, output_hidden_states=True)
        del pixels
        spatial = spatial.hidden_states
        # Remove the extra dimensions and transpose

        h, w = images.shape[2] // self.patch_size, images.shape[3] // self.patch_size
        # spatial = [spatial[i].squeeze().transpose(0, 1) for i in self.multilayers]
        spatial = [spatial[i] for i in self.multilayers]
        spatial = torch.stack(spatial)
        spatial = spatial.detach().cpu()
        # spatial = spatial.permute(0, 3, 1, 2)
        spatial = spatial.reshape(len(self.multilayers), batch_size, 128, 64, 147)

        # print(spatial.shape,"spatial")
        # spatial = spatial.squeeze(1).squeeze(1)

        return spatial[0] if len(spatial) == 1 else spatial
