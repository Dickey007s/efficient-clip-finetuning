# -*- coding: utf-8 -*-
"""Checkpoint loading helpers for reusable experiment runs."""
from pathlib import Path

import torch


def load_lora_checkpoint(model, checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"LoRA checkpoint must be a state dict: {path}")

    named_params = dict(model.named_parameters())
    loaded = []
    skipped = []
    with torch.no_grad():
        for name, tensor in state.items():
            param = named_params.get(name)
            if param is None:
                skipped.append((name, "missing parameter"))
                continue
            if tuple(param.shape) != tuple(tensor.shape):
                skipped.append((name, f"shape {tuple(tensor.shape)} != {tuple(param.shape)}"))
                continue
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded.append(name)

    if not loaded:
        details = "; ".join(f"{name}: {reason}" for name, reason in skipped[:5])
        raise RuntimeError(f"No LoRA tensors loaded from {path}. {details}")

    trainable_lora = [name for name, p in named_params.items() if p.requires_grad and "lora_" in name]
    missing_trainable = [name for name in trainable_lora if name not in loaded]
    if missing_trainable:
        sample = ", ".join(missing_trainable[:5])
        raise RuntimeError(
            f"LoRA checkpoint {path} is incomplete: missing {len(missing_trainable)} trainable tensors. "
            f"Examples: {sample}"
        )

    return len(loaded), len(skipped)
