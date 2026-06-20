# -*- coding: utf-8 -*-
"""
data_utils.py
统一的数据加载与采样工具。

支持：
- 标准 GTSRB 加载（train / test）
- Few-shot 分层采样：按类别随机取 K 张
- 类别不均衡处理：可选加权采样
- CLIP 预处理统一封装
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision
from torchvision.datasets import GTSRB

ROOT = Path(__file__).resolve().parent.parent / "data"


def get_clip_preprocess():
    """返回 CLIP 标准预处理（Resize 224, Normalize ImageNet mean/std）。"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def create_few_shot_subset(dataset, shots_per_class, seed=42):
    """
    从 dataset 中按类别分层采样，每类取 shots_per_class 张。

    dataset 需支持 __getitem__(i) -> (image, label)
    返回：Subset
    """
    rng = random.Random(seed)
    # 收集每类的索引
    class_to_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_to_indices.setdefault(label, []).append(i)

    selected = []
    for cls_id, indices in sorted(class_to_indices.items()):
        n = min(shots_per_class, len(indices))
        selected.extend(rng.sample(indices, n))

    return Subset(dataset, selected)


def get_class_weights(dataset):
    """计算每类的逆频率权重，用于 WeightedRandomSampler。"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counter = Counter(labels)
    num_classes = len(counter)
    weights = [1.0 / counter[label] for label in labels]
    return weights, num_classes


def get_gtsrb_dataloaders(
    shots_per_class=None,      # None = 全量; int = few-shot 每类取 N 张
    batch_size=64,
    num_workers=0,             # Windows 下建议 0 避免 multiprocessing 问题
    use_weighted_sampler=False, # 处理类别不均衡
    seed=42,
):
    """
    返回 (train_loader, test_loader, num_classes)

    shots_per_class: few-shot 设置，如 1, 2, 4, 8, 16
    """
    preprocess = get_clip_preprocess()

    # 训练集：先加载原始数据，再套 transform
    train_raw = GTSRB(root=str(ROOT), split="train", download=False)
    test_raw = GTSRB(root=str(ROOT), split="test", download=False)

    # 包装成带 transform 的 dataset
    class _Wrapped(Dataset):
        def __init__(self, base, tfm):
            self.base = base
            self.tfm = tfm
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            img, label = self.base[i]
            return self.tfm(img), label

    if shots_per_class is not None:
        train_raw = create_few_shot_subset(train_raw, shots_per_class, seed=seed)
        print(f"[Data] Few-shot mode: {shots_per_class} shots per class, "
              f"total {len(train_raw)} training samples.")
    else:
        print(f"[Data] Full training set: {len(train_raw)} samples.")

    train_set = _Wrapped(train_raw, preprocess)
    test_set = _Wrapped(test_raw, preprocess)

    # Sampler（可选，处理不均衡）
    sampler = None
    if use_weighted_sampler and shots_per_class is None:
        weights, _ = get_class_weights(train_raw)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader, 43


if __name__ == "__main__":
    # 简单测试
    print("测试全量数据加载...")
    tl, vl, nc = get_gtsrb_dataloaders()
    print(f"  训练 batch 数: {len(tl)}, 测试 batch 数: {len(vl)}, 类别: {nc}")

    print("测试 4-shot 数据加载...")
    tl, vl, nc = get_gtsrb_dataloaders(shots_per_class=4)
    print(f"  训练 batch 数: {len(tl)}, 测试 batch 数: {len(vl)}, 类别: {nc}")
    # 检查类别分布
    labels = []
    for _, lb in tl:
        labels.extend(lb.tolist())
    from collections import Counter
    print(f"  训练集类别分布: {dict(Counter(labels))}")
