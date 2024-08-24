"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import gc
import operator as op
import os
from collections import defaultdict
from datetime import datetime
from functools import reduce
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.autograd import Variable
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.losses import DepthLoss
from evals.utils.metrics import evaluate_depth, match_scale_and_shift
from evals.utils.optim import cosine_decay_linear_warmup


def get_tensor_memory_usage(tensor):
    return tensor.element_size() * tensor.numel()


def interpolate_to_fixed_4d_list(tensors, target_shape=(1, 64, 512, 480)):
    """
    Interpolates 2D, 3D, or 4D tensors in a list to a fixed list of 4 4D tensors of shape (1, 128, 64, 64).

    Args:
    - tensors (list of torch.Tensor): List of input tensors to interpolate.
    - target_shape (tuple): Desired shape for the output tensors (1, 128, 64, 64).

    Returns:
    - list of torch.Tensor: Interpolated tensors, a list of 4 tensors each with shape (1, 128, 64, 64).
    """
    # Ensure tensors is a list of tensors
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    elif isinstance(tensors, list):
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise ValueError("If tensors is a list, all elements must be torch.Tensor.")
    else:
        raise TypeError(
            "Input tensors must be a torch.Tensor or a list of torch.Tensor."
        )

    interpolated_tensors = []
    # print(len(tensors))
    for tensor in tensors:
        # Determine number of spatial dimensions (2D, 3D, or 4D)
        input_dims = tensor.dim()
        spatial_dims = input_dims - 1
        # print("tensor shape",tensor.shape)
        # Validate input dimensions and target shape

        # Reshape tensor to combine batch and channel dimensions for interpolation

        if input_dims < 3 or input_dims > 5:
            raise ValueError("Input tensors must have 2, 3, or 4 dimensions.")
        # if len(target_shape) != 3:
        #     raise ValueError("Target shape must be (1, 128, 64, 64) for interpolation.")

        if spatial_dims == 2:
            # 2D to 4D (bilinear interpolation)
            # print(tensor.unsqueeze(0).unsqueeze(0).shape)
            tsr = []
            for batch in tensor:
                interpolated_tensor = F.interpolate(
                    batch.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    size=target_shape,
                    mode="trilinear",
                    align_corners=False,
                )
                tsr.append(interpolated_tensor)
            interpolated_tensor = torch.stack(tsr)

            # interpolated_tensor = interpolated_tensor

        elif spatial_dims == 3:
            # 3D to 4D (trilinear interpolation)
            tsr = []
            for batch in tensor:
                interpolated_tensor = F.interpolate(
                    batch.unsqueeze(0).unsqueeze(0),
                    size=target_shape,
                    mode="trilinear",
                    align_corners=False,
                )
                tsr.append(interpolated_tensor)
            interpolated_tensor = torch.stack(tsr)
            # interpolated_tensor = interpolated_tensor

        elif spatial_dims == 4:
            # print(tensor.shape,"here")
            tsr = []
            for batch in tensor:
                # Already 4D, resize using trilinear interpolation

                interpolated_tensor = F.interpolate(
                    batch.unsqueeze(0),
                    size=target_shape,
                    mode="trilinear",
                    align_corners=False,
                )
                tsr.append(interpolated_tensor)
            interpolated_tensor = torch.stack(tsr)

        else:
            raise ValueError("Input tensors must have 2, 3, or 4 dimensions.")

        interpolated_tensors.append(interpolated_tensor)

    # If there are fewer than 4 tensors, repeat the existing tensors until there are 4
    while len(interpolated_tensors) < 4:
        interpolated_tensors.append(interpolated_tensors[-1].clone())

    # If there are more than 4 tensors, take only the first 4
    interpolated_tensors = interpolated_tensors[:4]
    if interpolated_tensors[0].dim() > 4:
        interpolated_tensors = [f.squeeze(1).squeeze(1) for f in interpolated_tensors]
    return interpolated_tensors


def print_memory_usage():
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / (1024 ** 2)} MB")
    cnt = 0
    shape_counts = defaultdict(int)

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                # print(obj.size())
                # if memory_usage ==4816896:
                cnt += 1
                shape_counts[obj.size()] += 1
                memory_usage = get_tensor_memory_usage(obj)
                print(
                    f"Tensor size: {obj.size()}, Memory usage: {memory_usage/(1e6)} MB"
                )
                # Example: Delete tensors of shape (2, 3)
                if obj.size() == (1, 128, 64, 147):
                    # print("deleting......")
                    obj = None  # This removes the reference to the object
                    del obj
                # print(reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
        except:
            pass
            # print("restricted shared file")
    # gc.collect()  # Optional: Force garbage collection to reclaim memory
    print("Num Tensors:", cnt)
    for shape, count in shape_counts.items():
        print(f"Shape: {shape}, Count: {count}")
    # stats = torch.cuda.memory_stats()
    # for stat in stats:
    #     print(f"{stat}: {stats[stat]}")


def train_probe(probe, scale_invariant, loss_fn, target, feats):
    pred = probe(feats)
    pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

    if scale_invariant:
        pred = match_scale_and_shift(pred, target)
        pred = pred.clamp(min=0.001, max=10.0)
    # print(pred.shape,"predictions shape",target.shape)
    # target=target.detach()
    # pred=pred.detach()
    # print(pred.shape,target.shape)
    # pred.detach()
    loss = loss_fn(pred, target)
    # loss.detach()
    # loss = Variable(loss.detach().data, requires_grad=True)

    loss.backward()
    return loss.item()


def ddp_setup(rank: int, world_size: int, port: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def extract(
    model,
    input_folder,
    output_folder,
    detach_model,
    rank=0,
    world_size=1,
    hidden_dim=128,
    feat_size=64,
):
    # feats=torch.empty()
    transform = transforms.Compose(
        [transforms.ToTensor()]  # Convert the image to a tensor
    )  # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_folder = os.path.abspath(input_folder)
    # phase_output_folder = os.path.abspath(output_folder)
    # phase_input_folder = os.path.join(phase_input_folder, phase)
    # phase_output_folder = os.path.join(phase_output_folder, phase)
    # print(phase_output_folder)

    for dir in tqdm(os.listdir(input_folder), desc=f"Processing images"):
        subfolder_path = os.path.join(input_folder, dir)
        # print(dir)
        for tdir in tqdm(os.listdir(subfolder_path), desc="subfolder"):
            if "3d_scan" in tdir:
                continue
            spath = os.path.join(subfolder_path, tdir, "images")
            if not os.path.exists(spath):
                continue
            for file in tqdm(os.listdir(spath), desc="files"):

                # print(file)
                if file.endswith(".jpg"):
                    img_path = os.path.join(spath, file)

                    img = Image.open(img_path).convert("RGB")
                    img = transform(img).unsqueeze(0)
                    # print(img.shape)
                    img = img.to(rank)

                    if detach_model:
                        with torch.no_grad():
                            feats = model(img)
                            if isinstance(feats, (tuple, list)):
                                feats = [_f.detach() for _f in feats]
                            else:
                                feats = feats.detach()
                    else:
                        feats = model(img)
                    # print(len(feats))
                    feats = [f.to(rank).float() for f in feats]
                    print("Before: ", len(feats), feats[0].shape)
                    feats = interpolate_to_fixed_4d_list(
                        feats, (hidden_dim, feat_size, feat_size)
                    )
                    print("After: ", feats[0].shape)

                    # print(output.shape)
                    # Create corresponding output subfolder structure
                    # relative_path = os.path.relpath(subfolder_path, phase_input_folder)
                    output_subfolder = os.path.join(output_folder, "test", dir, tdir)
                    # print(output_subfolder)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)

                    output_filename = os.path.splitext(file)[0] + ".pt"
                    output_filepath = os.path.join(output_subfolder, output_filename)
                    torch.save(feats, output_filepath)
    print(f"Processed and saved")


def extract_old(model, input_folder, output_folder, detach_model, rank=0, world_size=1):

    # feats=torch.empty()
    transform = transforms.Compose(
        [transforms.ToTensor()]  # Convert the image to a tensor
    )  # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for phase in ["train", "val"]:
        phase_input_folder = os.path.abspath(input_folder)
        phase_output_folder = os.path.abspath(output_folder)
        phase_input_folder = os.path.join(phase_input_folder, phase)
        phase_output_folder = os.path.join(phase_output_folder, phase)
        print(phase_output_folder)

        for dir in tqdm(
            os.listdir(phase_input_folder), desc=f"Processing {phase} images"
        ):
            subfolder_path = os.path.join(phase_input_folder, dir)
            # print(dir)
            for file in os.listdir(subfolder_path):

                # print(file)
                if file.endswith(".jpg"):
                    img_path = os.path.join(subfolder_path, file)

                    img = Image.open(img_path).convert("RGB")
                    img = transform(img).unsqueeze(0)
                    # print(img.shape)
                    img = img.to(rank)

                    if detach_model:
                        with torch.no_grad():
                            feats = model(img)
                            if isinstance(feats, (tuple, list)):
                                feats = [_f.detach() for _f in feats]
                            else:
                                feats = feats.detach()
                    else:
                        feats = model(img)
                    # print(output.shape)
                    # Create corresponding output subfolder structure
                    # relative_path = os.path.relpath(subfolder_path, phase_input_folder)
                    output_subfolder = os.path.join(phase_output_folder, dir)
                    # print(output_subfolder)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)

                    output_filename = os.path.splitext(file)[0] + ".pt"
                    output_filepath = os.path.join(output_subfolder, output_filename)
                    torch.save(feats, output_filepath)
        print(f"Processed and saved: {phase}")


def validate(
    model, probe, loader, loss_fn, verbose=True, scale_invariant=False, aggregate=True
):
    total_loss = 0.0
    metrics = None
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluation") if verbose else loader
        for batch in pbar:
            images = batch["image"].cuda()
            target = batch["depth"].cuda()

            feat = model(images)
            pred = probe(feat).detach()
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            loss = loss_fn(pred, target)
            total_loss += loss.item()

            batch_metrics = evaluate_depth(
                pred, target, scale_invariant=scale_invariant
            )
            if metrics is None:
                metrics = {key: [value] for key, value in batch_metrics.items()}
            else:
                for key, value in batch_metrics.items():
                    metrics[key].append(value)

    # aggregate
    total_loss = total_loss / len(loader)
    for key in metrics:
        metric_key = torch.cat(metrics[key], dim=0)
        metrics[key] = metric_key.mean() if aggregate else metric_key

    return total_loss, metrics


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ===== GET DATA LOADERS =====
    # validate and test on single gpu
    # trainval_loader = build_loader(cfg.dataset, "trainval", cfg.batch_size, world_size)
    # test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    # trainval_loader.dataset.__getitem__(0)

    # ===== Get models =====
    model = instantiate(cfg.backbone)
    # print("feat dim:", model.feat_dim)

    # setup experiment name
    # === job info
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")

    model_info = [
        f"{model.checkpoint_name:40s}",
        f"{model.patch_size:2d}",
        f"{str(model.layer):5s}",
        f"{model.output:10s}",
    ]
    # train_info = [
    #     f"{cfg.optimizer.n_epochs:3d}",
    #     f"{cfg.optimizer.warmup_epochs:4.2f}",
    #     f"{str(cfg.optimizer.probe_lr):>10s}",
    #     f"{str(cfg.optimizer.model_lr):>10s}",
    #     f"{batch_size:4d}",
    #     f"{train_dset:10s}",
    #     f"{test_dset:10s}",
    # ]

    # define exp_name
    exp_name = "_".join([timestamp] + model_info + ["extraction"])
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")  # remove spaces

    # ===== SETUP LOGGING =====
    if rank == 0:
        exp_path = Path(__file__).parent / f"depth_exps/{exp_name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "extraction.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(rank)

    # very hacky ... SAM gets some issues with DDP finetuning
    print(cfg)
    model_name = cfg.data.model_name
    # if "sam" in model_name or "vit-mae" in model_name:
    #     h, w = trainval_loader.dataset.__getitem__(0)["image"].shape[-2:]
    #     model.resize_pos_embed(image_size=(h, w))

    # move to DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # probe = DDP(probe, device_ids=[rank])

    extract(
        model,
        cfg.data.input_folder,
        cfg.data.output_folder + "/" + str(model_name),
        detach_model=True,
        rank=rank,
        world_size=world_size,
        hidden_dim=cfg.data.hidden_dim,
        feat_size=cfg.data.feat_size,
        # valid_loader=test_loader,
    )

    if world_size > 1:
        destroy_process_group()


@hydra.main(
    config_name="feature_extraction", config_path="./configs", version_base=None
)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
