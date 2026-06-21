# -*- coding: utf-8 -*-
"""
feature_cache.py
预计算 CLIP image features 的通用组件。

原则：凡是 CLIP image encoder 在训练过程中冻结不更新的方法，
都应该训练前预计算 image features，避免每个 epoch 重复跑 ViT。
"""
import contextlib
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


@torch.no_grad()
def precompute_image_features(clip_model, image_loader, device, name="train", use_amp=True):
    clip_model.eval()
    all_features = []
    all_labels = []
    print(f"[Precompute] Encoding {name} image features...")
    t0 = time.time()
    amp_enabled = use_amp and device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(image_loader):
        images = images.to(device, non_blocking=True)
        ctx = torch.amp.autocast("cuda") if amp_enabled else contextlib.nullcontext()
        with ctx:
            features = clip_model.encode_image(images)
            features = F.normalize(features, dim=-1)
        all_features.append(features.float().cpu())
        all_labels.append(labels.cpu().long())
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(image_loader):
            print(f"[Precompute] {name}: {batch_idx + 1}/{len(image_loader)} batches")

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(
        f"[Precompute] {name} done: "
        f"features={tuple(features.shape)}, labels={tuple(labels.shape)}, "
        f"time={time.time() - t0:.1f}s"
    )
    return features, labels


def make_feature_loaders(train_features, train_labels, test_features, test_labels, batch_size):
    train_loader = DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        FeatureDataset(test_features, test_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, test_loader
