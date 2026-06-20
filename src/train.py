# -*- coding: utf-8 -*-
"""
train.py
统一训练入口，支持多种 CLIP 高效微调方法。

当前实现：
  M0: zero-shot    -- 零样本推理，无需训练
  M1: linear       -- 冻结 CLIP，训练线性分类头

待实现：
  M2: coop         -- 学习连续 prompt 向量
  M3: adapter      -- CLIP-Adapter
  M4: lora         -- LoRA 微调
  M5: coop-lora    -- 两阶段：CoOp 预热 + LoRA 微调

用法：
  python src/train.py --method zero-shot --batch_size 64
  python src/train.py --method linear --epochs 10 --lr 1e-3 --batch_size 64
  python src/train.py --method linear --shots 16 --epochs 20 --lr 1e-3
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import open_clip
from tqdm import tqdm

from data_utils import get_gtsrb_dataloaders
from class_names import all_prompts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_model(model_name="ViT-B-32", pretrained="openai"):
    """加载 CLIP 模型并返回 (model, preprocess_fn, tokenizer)。"""
    print(f"[Model] Loading {model_name} (pretrained={pretrained}) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, quick_gelu=True,
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {n_params/1e6:.1f}M")
    return model, tokenizer


def build_text_features(model, tokenizer, prompts=None):
    """
    编码所有类别的文本特征，用于 zero-shot / linear probe 评估。
    prompts: 自定义 prompt 列表，默认用 "a photo of a {class_name}"
    """
    if prompts is None:
        prompts = all_prompts("a photo of a {}")
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features  # [num_classes, embed_dim]


@torch.no_grad()
def evaluate_zero_shot(model, test_loader, text_features):
    """M0: Zero-shot CLIP 评估。"""
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(test_loader, desc="Zero-shot eval"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        with autocast():
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ text_features.t()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = 100.0 * correct / total
    return acc


class LinearProbe(nn.Module):
    """M1: 线性探测头，冻结 CLIP 主干。"""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, image_features):
        return self.head(image_features)


def train_linear_probe(model, train_loader, test_loader, text_features,
                        epochs=10, lr=1e-3, weight_decay=1e-4):
    """M1: 训练线性分类头，CLIP 主干冻结。"""
    model.eval()  # 冻结 CLIP
    for p in model.parameters():
        p.requires_grad = False

    # 获取 embedding 维度
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
        embed_dim = model.encode_image(dummy).shape[-1]

    probe = LinearProbe(embed_dim, num_classes=43).to(DEVICE)
    n_trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"[Linear Probe] Trainable params: {n_trainable:,} ({n_trainable/1e3:.1f}K)")

    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
                logits = probe(image_features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # 评估
        acc = evaluate_linear_probe(model, probe, test_loader)
        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc

    return probe, best_acc


@torch.no_grad()
def evaluate_linear_probe(model, probe, test_loader):
    """评估 Linear Probe。"""
    model.eval()
    probe.eval()
    correct, total = 0, 0
    for images, labels in tqdm(test_loader, desc="Linear eval"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with autocast():
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = probe(image_features)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="CLIP Fine-Tuning on GTSRB")
    parser.add_argument("--method", type=str, required=True,
                        choices=["zero-shot", "linear", "coop", "adapter", "lora", "coop-lora"],
                        help="Fine-tuning method")
    parser.add_argument("--shots", type=int, default=None,
                        help="Few-shot: shots per class (None = full dataset)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print(f"Method: {args.method} | Shots: {args.shots} | Batch: {args.batch_size}")
    print("=" * 60)

    # 加载数据
    train_loader, test_loader, num_classes = get_gtsrb_dataloaders(
        shots_per_class=args.shots,
        batch_size=args.batch_size,
    )

    # 加载 CLIP
    model, tokenizer = load_clip_model()
    text_features = build_text_features(model, tokenizer)

    # 执行对应方法
    t0 = time.time()
    if args.method == "zero-shot":
        acc = evaluate_zero_shot(model, test_loader, text_features)
        print(f"\n[Result] Zero-shot Top-1 Accuracy: {acc:.2f}%")

    elif args.method == "linear":
        probe, acc = train_linear_probe(
            model, train_loader, test_loader, text_features,
            epochs=args.epochs, lr=args.lr,
        )
        print(f"\n[Result] Linear Probe Best Accuracy: {acc:.2f}%")
        # 保存模型
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"linear_probe_shots{args.shots or 'full'}.pt"
        torch.save(probe.state_dict(), save_path)
        print(f"[Save] Model saved to {save_path}")

    else:
        print(f"[Error] Method '{args.method}' not yet implemented. "
              "Supported: zero-shot, linear")
        return

    dt = time.time() - t0
    print(f"[Time] Total elapsed: {dt:.1f}s")

    # 显存统计
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[Memory] Peak GPU memory: {alloc:.2f} GB")


if __name__ == "__main__":
    main()
