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
import numpy as np
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.autograd import Variable
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.losses import DepthLoss
from evals.utils.metrics import evaluate_depth, match_scale_and_shift
from evals.utils.optim import cosine_decay_linear_warmup


def get_tensor_memory_usage(tensor):
    return tensor.element_size() * tensor.numel()


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


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    loss_fn,
    rank=0,
    world_size=1,
    valid_loader=None,
    scale_invariant=False,
    writer=None,
):
    dataset = train_loader.dataset

    # Calculate the number of samples for 20% of the data
    num_samples = int(0.2 * len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    dataset = Subset(dataset, indices)
    train_loader = DataLoader(
        dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
    )

    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        torch.autograd.set_detect_anomaly(True)

        # feats=torch.empty()
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        running_loss = 0.0

        for i, batch in enumerate(pbar):
            images = batch["image"].to(rank)
            target = batch["depth"].to(rank)

            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if isinstance(feats, (tuple, list)):
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()
            else:
                feats = model(images)
            #    ----probe trained here--------------
            # print("feats",len(feats),feats[0].shape)
            # feats = feats.to(rank)

            loss = train_probe(probe, scale_invariant, loss_fn, target, feats)
            # feats=feats.detach()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            # loss = loss.detach()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(
                    f"{ep} | loss: {loss:.4f} ({_loss:.4f}) probe_lr: {pr_lr:.2e}"
                )
            # feats = feats.detach().cpu()
            running_loss += loss
            if (i % 100) == 0:
                # print("Memory: ")

                if writer is not None:
                    writer.add_scalar(
                        "training loss", running_loss / 100, ep * len(train_loader) + i
                    )
                    running_loss = 0.0

                # print(target.numel(),"|",feats.numel(),"|",pred.numel())
                # print_memory_usage()
                # print("After:")
                # print_memory_usage()
                gc.collect()
                # breakpoint()
                torch.cuda.empty_cache()

                # breakpoint()
            del target, images, feats, loss
            # torch.cuda.empty_cache()

        train_loss /= len(train_loader)

        if rank == 0:
            logger.info(f"train loss {ep}   | {train_loss:.4f}")
            if valid_loader is not None:
                val_loss, val_metrics = validate(
                    model, probe, valid_loader, loss_fn, scale_invariant=scale_invariant
                )
                if writer is not None:
                    writer.add_scalar("val loss", val_loss, ep)
                logger.info(f"valid loss {ep}   | {val_loss:.4f}")
                for metric in val_metrics:
                    logger.info(f"valid SA {metric:10s} | {val_metrics[metric]:.4f}")


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
    trainval_loader = build_loader(cfg.dataset, "trainval", cfg.batch_size, world_size)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    trainval_loader.dataset.__getitem__(0)

    # ===== Get models =====
    model = instantiate(cfg.backbone)
    # print("feat dim:", model.feat_dim)
    if type(model.feat_dim) is not list:
        feat_dim = [model.feat_dim] * 4
    else:
        feat_dim = model.feat_dim
    # print(feat_dim)
    probe = instantiate(
        cfg.probe, feat_dim=feat_dim, max_depth=trainval_loader.dataset.max_depth
    )

    # setup experiment name
    # === job info
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    train_dset = trainval_loader.dataset.name
    test_dset = test_loader.dataset.name
    model_info = [
        f"{model.checkpoint_name:40s}",
        f"{model.patch_size:2d}",
        f"{str(model.layer):5s}",
        f"{model.output:10s}",
    ]
    probe_info = [f"{probe.name:25s}"]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{str(cfg.optimizer.probe_lr):>10s}",
        f"{str(cfg.optimizer.model_lr):>10s}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]
    # writer = SummaryWriter(f"runs/experiment_{model_name}_{test_dset}_{train_dset}")

    # define exp_name
    exp_name = "_".join([timestamp] + model_info + probe_info + train_info)
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")  # remove spaces

    # ===== SETUP LOGGING =====
    if rank == 0:
        exp_path = Path(__file__).parent / f"depth_exps/{exp_name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(rank)
    probe = probe.to(rank)

    # very hacky ... SAM gets some issues with DDP finetuning
    model_name = model.checkpoint_name
    if "sam" in model_name or "vit-mae" in model_name:
        h, w = trainval_loader.dataset.e__getitem__(0)["image"].shape[-2:]
        model.resize_pos_embed(image_size=(h, w))

    writer = SummaryWriter(f"runs/experiment_{model_name}_{test_dset}_{train_dset}")

    # move to DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        probe = DDP(probe, device_ids=[rank])

    if cfg.optimizer.model_lr == 0:
        optimizer = torch.optim.AdamW(
            [{"params": probe.parameters(), "lr": cfg.optimizer.probe_lr}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.optimizer.probe_lr},
                {"params": model.parameters(), "lr": cfg.optimizer.model_lr},
            ]
        )

    lambda_fn = lambda epoch: cosine_decay_linear_warmup(  # noqa: E731
        epoch,
        cfg.optimizer.n_epochs * len(trainval_loader),
        cfg.optimizer.warmup_epochs * len(trainval_loader),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    loss_fn = DepthLoss()

    train(
        model,
        probe,
        trainval_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        loss_fn=loss_fn,
        rank=rank,
        world_size=world_size,
        writer=writer
        # valid_loader=test_loader,
    )
    writer.close()
    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        test_sa_loss, test_sa_metrics = validate(model, probe, test_loader, loss_fn)
        logger.info(f"Scale-Aware Final test loss       | {test_sa_loss:.4f}")
        for metric in test_sa_metrics:
            logger.info(f"Final test SA {metric:10s} | {test_sa_metrics[metric]:.4f}")
        results_sa = ", ".join([f"{test_sa_metrics[_m]:.4f}" for _m in test_sa_metrics])

        # get scale invariant
        test_si_loss, test_si_metrics = validate(
            model, probe, test_loader, loss_fn, scale_invariant=True
        )
        logger.info(f"Scale-Invariant Final test loss       | {test_si_loss:.4f}")
        for metric in test_si_metrics:
            logger.info(f"Final test SI {metric:10s} | {test_si_metrics[metric]:.4f}")
        results_si = ", ".join([f"{test_si_metrics[_m]:.4f}" for _m in test_si_metrics])

        # log experiments
        exp_info = ", ".join(model_info + probe_info + train_info)
        log = f"{timestamp}, {exp_info}, SA_results(d1,d2,d3,rmse): {results_sa}, SI_results(d1,d2,d3,rmse) {results_si} \n"
        with open(f"depth_results_{test_dset}.log", "a") as f:
            f.write(log)

        # save final model
        ckpt_path = exp_path / "ckpt.pth"
        checkpoint = {
            "cfg": cfg,
            "model": model.state_dict(),
            "probe": probe.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint at {ckpt_path}")

    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="depth_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
