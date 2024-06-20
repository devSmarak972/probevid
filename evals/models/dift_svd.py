""" Code taken from the DIFT repo: github:Tsingularity/dift"""

import gc
from typing import Optional, Union

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline,
)
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import export_to_video, load_image
from loguru import logger
from torchvision import transforms


def reshape_tensor(tensor):
    # Combine the first two dimensions (28 and 1280) into one dimension (28 * 1280 = 35840)
    # combined_dim = tensor.shape[2] * tensor.shape[3]*tensor.shape[0]
    # reshaped_tensor = tensor.view(combined_dim, tensor.shape[1])
    # print("Previous shape",tensor.shape)
    # Reshape to the target shape [1, 128, any, any]
    new_shape = (
        1,
        tensor.shape[1],
        tensor.shape[2] * int(tensor.shape[0]),
        tensor.shape[3],
    )
    reshaped_tensor = tensor.view(*new_shape)
    return reshaped_tensor


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
    ):
        r"""
		Args:
			sample (`torch.FloatTensor`):
				(batch, channel, height, width) noisy inputs tensor
			timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
			encoder_hidden_states (`torch.FloatTensor`):
				(batch, sequence_length, feature_dim) encoder hidden states
		"""
        output = super().forward(*args, **kwargs)
        up_ft = {}
        # for i in up_ft_indices:
        # 	up_ft[i] = sample

        output = {}
        output["up_ft"] = up_ft
        return output


class OneStepSVDPipeline(StableVideoDiffusionPipeline):
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        device = self._execution_device

        scale_factor = self.vae.config.scaling_factor
        latents = scale_factor * self.vae.encode(img_tensor).latent_dist.mode()

        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(
            latents_noisy, t, up_ft_indices, encoder_hidden_states=prompt_embeds
        )
        return unet_output


class SVDFeaturizer(torch.nn.Module):
    def __init__(self, sd_id="stabilityai/stable-video-diffusion-img2vid-xt"):
        super().__init__()

        breakpoint()

        # unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = StableVideoDiffusionPipeline.from_pretrained(
            sd_id, safety_checker=None, variant="fp16"
        )

        # onestep_pipe.vae.decoder = None
        # onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
        # 	sd_id, subfolder="scheduler"
        # )
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        # onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe
        self.activations = []
        # self.tokenizer = onestep_pipe.tokenizer
        # self.text_encoder = onestep_pipe.text_encoder
        self.unet = onestep_pipe.unet
        self.vae = onestep_pipe.vae
        self.scheduler = onestep_pipe.scheduler
        # Indices of the up_blocks from which to get activations
        # self.up_ft_indices = [1,4,7]
        # Dictionary to store the activations
        self.layer_outputs = {}
        # Attach hooks
        self.generator = torch.manual_seed(42)

        for name, param in self.named_parameters():
            if name.split(".")[0] != "unet":
                param.requires_grad = False

    def attach_hooks(self):
        for idx in self.up_ft_indices:
            handle = self.pipe.unet.up_blocks[idx].register_forward_hook(self.hook_fn)
            block = self.pipe.unet.up_blocks[idx]
            # print(f"Upsample Block {idx}/{len(self.pipe.unet.up_blocks)}: {block.__class__.__name__}")

            setattr(self, f"hook_handle_{idx}", handle)

    def hook_fn(self, module, input, output):
        self.layer_outputs[module] = output

    def remove_hooks(self):
        for idx in self.up_ft_indices:
            handle = getattr(self, f"hook_handle_{idx}")
            handle.remove()

    def forward(self, images, t=1, up_ft_index=[1, 4, 7]):
        """
		Args:
			img_tensor: should be a single tensor of shape [1, C, H, W] or [C, H, W]
			prompt: the prompt to use, a string
			t: the time step to use, should be an int in the range of [0, 1000]
			up_ft_index: upsampling block of the U-Net for feat. extract. [0, 1, 2, 3]
		Return:
			unet_ft: a torch tensor in the shape of [1, c, h, w]
		"""
        self.up_ft_indices = up_ft_index
        self.activations = []

        self.attach_hooks()

        device = images.device
        # print(type(images))
        # Ensure input is a batch of images
        # if images.dim() == 3:  # If single image, add batch dimension
        # images = images.unsqueeze(0)
        # with torch.no_grad():
        # 	# prompt_embeds = self.encode_prompt(
        # 	# 	prompt=prompts, device=device
        # 	# )  # [1, 77, dim]

        # 	# what was happening in the pipeline
        # 	scale_factor = self.vae.config.scaling_factor
        # 	latents = scale_factor * self.vae.encode(images).latent_dist.mode()

        # 	t = torch.tensor(t, dtype=torch.long, device=device)
        # 	noise = torch.randn_like(latents).to(device)
        # 	latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # print(images.shape)
        # Define the resize transform
        # resize_transform = transforms.Resize((1024, 576))

        # Apply the resize transform to each image in the batch
        # images = (images - images.min()) / (images.max() - images.min())
        # Define the transform to convert tensors to PIL images
        # to_pil = transforms.ToPILImage()

        # Convert each tensor in the batch to a PIL image
        transform = transforms.Compose(
            [transforms.Resize((576, 1024)), transforms.ToPILImage()]
        )
        images = [transform(img) for img in images]
        # pil_images = [transform(tensor) for tensor in images]
        # tensor = torch.randn(3, 576, 1024)
        # pil_image = transform(tensor)
        # image = load_image(
        #     "https://images.pexels.com/photos/139303/pexels-photo-139303.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
        # )
        # tensor = torch.rand(3, 576, 1024)
        # to_pil_image = transforms.ToPILImage()
        # pil_image = to_pil_image(tensor)
        # print(np.array(pil_image).shape)
        frames = self.pipe(
            images, decode_chunk_size=2, generator=self.generator, num_frames=14
        ).frames[0]
        # print("this is the num layers-------------------",len(self.layer_outputs.values()))
        # print(self.up_ft_indices)
        # spatial = torch.stack(list(self.layer_outputs.values()))
        for id, tsr in self.layer_outputs.items():
            self.layer_outputs[id] = reshape_tensor(tsr)
            # print( "new shape",self.layer_outputs[id].shape)
        self.remove_hooks()
        return list(self.layer_outputs.values())
