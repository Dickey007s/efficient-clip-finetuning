# -*- coding: utf-8 -*-
"""Small performance helpers shared by training scripts."""
import contextlib

import torch


def configure_torch_runtime(device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def make_grad_scaler(device):
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


def dataloader_kwargs(num_workers, pin_memory):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def move_to_device(*tensors, device):
    non_blocking = device.type == "cuda"
    moved = [tensor.to(device, non_blocking=non_blocking) for tensor in tensors]
    return moved[0] if len(moved) == 1 else tuple(moved)
