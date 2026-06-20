# -*- coding: utf-8 -*-
"""
train_m4.py
M4: LoRA (Low-Rank Adaptation) on CLIP image encoder
在 CLIP ViT 的 Q/V attention 投影注入低秩适配器，冻结原始权重。

参考: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

实现细节:
  - 对 ViT attention 的 q_proj 和 v_proj 注入 LoRA（标准做法）
  - k_proj 和 out_proj 保持冻结
  - MLP (c_fc/c_proj) 不注入 LoRA
  - rank=4, alpha=8 (scaling = alpha / rank = 2.0)
  - 只训练 LoRA 参数，原始 CLIP 权重冻结
  - 使用自定义 lora_utils 解决 peft 无法匹配 in_proj 的问题

用法:
  python src/train_m4.py --epochs 20 --lr 1e-3 --batch_size 64 --shots 16 --rank 4 --alpha 8
  python src/train_m4.py --epochs 10 --lr 1e-3 --batch_size 64 --rank 4 --alpha 8  # 全数据
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
import sys
from pathlib import Path
from io import StringIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import open_clip
from lora_utils import inject_lora_into_visual
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB
from torchvision import transforms

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

CLASS_NAMES = [
    "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60",
    "speed limit 70", "speed limit 80", "end of speed limit 80", "speed limit 100",
    "speed limit 120", "no passing", "no passing for vehicles over 3.5 metric tons",
    "right-of-way at the next intersection", "priority road", "yield", "stop",
    "no vehicles", "vehicles over 3.5 metric tons prohibited", "no entry",
    "general caution", "dangerous curve to the left", "dangerous curve to the right",
    "double curve", "bumpy road", "slippery road", "road narrows on the right",
    "road work", "traffic signals", "pedestrians", "children crossing",
    "bicycles crossing", "beware of ice or snow", "wild animals crossing",
    "end of all speed and passing limits", "turn right ahead", "turn left ahead",
    "ahead only", "go straight or right", "go straight or left", "keep right",
    "keep left", "roundabout mandatory", "end of no passing",
    "end of no passing by vehicles over 3.5 metric tons",
]


class Logger:
    """同时输出到控制台和内存缓冲区的日志记录器。"""
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()
        self.buffer.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def get_preprocess():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base, tfm):
        self.base = base
        self.tfm = tfm
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, label = self.base[i]
        return self.tfm(img), label


def create_few_shot_subset(dataset, shots_per_class, seed=42):
    import random
    rng = random.Random(seed)
    class_to_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_to_indices.setdefault(label, []).append(i)
    selected = []
    for cls_id, indices in sorted(class_to_indices.items()):
        n = min(shots_per_class, len(indices))
        selected.extend(rng.sample(indices, n))
    return torch.utils.data.Subset(dataset, selected)


def load_data(batch_size=64, shots_per_class=None):
    preprocess = get_preprocess()
    train_raw = GTSRB(root=str(DATA_ROOT), split="train", download=False)
    test_raw = GTSRB(root=str(DATA_ROOT), split="test", download=False)

    if shots_per_class is not None:
        train_raw = create_few_shot_subset(train_raw, shots_per_class)
        print(f"[Data] Few-shot mode: {shots_per_class} shots/class, total {len(train_raw)} samples")
    else:
        print(f"[Data] Full training set: {len(train_raw)} samples")
    print(f"[Data] Test set: {len(test_raw)} samples")

    train_set = WrappedDataset(train_raw, preprocess)
    test_set = WrappedDataset(test_raw, preprocess)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, test_loader


def load_clip_with_lora(rank=4, alpha=8):
    """加载 CLIP 并在 image encoder 上注入 LoRA。"""
    print(f"[Model] Loading CLIP ViT-B/32 (openai) on {DEVICE} ...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True,
    )
    model = model.to(DEVICE)

    # 冻结所有参数
    for p in model.parameters():
        p.requires_grad = False

    # 注入轻量 LoRA 到 image encoder: only Q/V attention
    # 用自定义 AttentionWithLoRA 替换 MHA，解决 in_proj 无法被 peft 匹配的问题
    print(f"[Model] Injecting LoRA (rank={rank}, alpha={alpha}) into ViT...")
    inject_lora_into_visual(model.visual, rank=rank, alpha=alpha)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] CLIP total params: {n_total/1e6:.1f}M")
    print(f"[Model] LoRA trainable params: {n_trainable:,} ({n_trainable/1e3:.1f}K)")

    return model


def build_text_features(model, tokenizer):
    """编码所有类别的文本特征（冻结）。"""
    prompts = [f"a photo of a {name}" for name in CLASS_NAMES]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features


@torch.no_grad()
def evaluate(model, text_features, test_loader):
    """评估：image encoder 输出与 text features 对比。"""
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast('cuda'):
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ text_features.t()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def evaluate_detailed(model, text_features, test_loader):
    """
    详细评估：返回准确率、混淆矩阵、每类准确率、失败案例。
    """
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100.0 * image_features @ text_features.t()
            preds = logits.argmax(dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            # 记录全局索引（近似）
            start_idx = batch_idx * test_loader.batch_size
            all_indices.extend(range(start_idx, start_idx + len(labels)))
    
    acc = 100.0 * correct / total
    
    # 混淆矩阵
    num_classes = 43
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1
    
    # 每类准确率
    per_class_acc = {}
    for c in range(num_classes):
        class_total = confusion[c].sum().item()
        class_correct = confusion[c, c].item()
        per_class_acc[c] = 100.0 * class_correct / class_total if class_total > 0 else 0.0
    
    # 失败案例
    failures = [(idx, t, p) for idx, t, p in zip(all_indices, all_labels, all_preds) if t != p]
    
    return acc, confusion, per_class_acc, failures


def save_metrics(metrics, save_path):
    """保存训练曲线数据：epoch, loss, test_acc。"""
    import csv
    csv_path = save_path.with_suffix('').with_name(save_path.stem + '_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'test_acc'])
        for row in metrics:
            writer.writerow(row)
    print(f"[Save] Metrics saved to {csv_path}")


def save_confusion(confusion, save_path):
    """保存混淆矩阵为 CSV。"""
    import csv
    csv_path = save_path.with_suffix('').with_name(save_path.stem + '_confusion.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class'] + [f'pred_{i}' for i in range(43)])
        for i in range(43):
            writer.writerow([i] + confusion[i].tolist())
    print(f"[Save] Confusion matrix saved to {csv_path}")


def save_per_class(per_class_acc, save_path):
    """保存每类准确率为 CSV。"""
    import csv
    csv_path = save_path.with_suffix('').with_name(save_path.stem + '_per_class.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'accuracy'])
        for c in range(43):
            writer.writerow([c, CLASS_NAMES[c], f"{per_class_acc[c]:.2f}"])
    print(f"[Save] Per-class accuracy saved to {csv_path}")


def save_failures(failures, save_path, max_samples=100):
    """保存失败案例为 CSV。"""
    import csv
    csv_path = save_path.with_suffix('').with_name(save_path.stem + '_failures.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_idx', 'true_label', 'true_name', 'pred_label', 'pred_name'])
        for idx, t, p in failures[:max_samples]:
            writer.writerow([idx, t, CLASS_NAMES[t], p, CLASS_NAMES[p]])
    print(f"[Save] Failures ({min(len(failures), max_samples)} samples) saved to {csv_path}")


def train_epoch(model, text_features, train_loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ text_features.t()
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shots", type=int, default=None, help="Few-shot: shots per class")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="outputs/m4_lora.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 重定向 stdout 到 Logger
    logger = Logger()
    sys.stdout = logger

    print("=" * 60)
    print("M4: LoRA (Low-Rank Adaptation) on CLIP Image Encoder")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size} | Shots: {args.shots or 'full'}")
    print(f"LoRA rank: {args.rank} | alpha: {args.alpha} | scaling: {args.alpha/args.rank:.2f}")
    print("=" * 60)

    # 加载数据
    train_loader, test_loader = load_data(args.batch_size, args.shots)

    # 加载 CLIP + LoRA
    clip_model = load_clip_with_lora(rank=args.rank, alpha=args.alpha)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # 预计算 text features（冻结）
    text_features = build_text_features(clip_model, tokenizer)
    print(f"[Model] Text features precomputed: {text_features.shape}")

    # 优化器（只优化 LoRA 参数）
    lora_params = [p for p in clip_model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0.0
    t_start = time.time()
    metrics = []  # [epoch, loss, test_acc]

    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_epoch(clip_model, text_features, train_loader, optimizer, scaler, criterion)
        
        # 评估（详细）
        acc, confusion, per_class_acc, failures = evaluate_detailed(clip_model, text_features, test_loader)
        epoch_time = time.time() - t0
        best_acc = max(best_acc, acc)
        metrics.append([epoch + 1, loss, acc])

        print(f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {loss:.4f} | "
              f"Test Acc: {acc:.2f}% | Best: {best_acc:.2f}% | Time: {epoch_time:.1f}s")

    total_time = time.time() - t_start
    print("=" * 60)
    print(f"Final Best Accuracy: {best_acc:.2f}%")
    print(f"Total time: {total_time:.1f}s")
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {alloc:.2f} GB")

    # 保存模型（只保存 LoRA 参数）
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 LoRA 状态
    lora_state = {}
    for name, param in clip_model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu()
    torch.save(lora_state, save_path)
    print(f"[Save] LoRA weights saved to {save_path}")

    # 保存日志
    log_path = save_path.with_suffix('.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(logger.getvalue())
    print(f"[Save] Log saved to {log_path}")

    # 保存详细数据
    save_metrics(metrics, save_path)
    save_confusion(confusion, save_path)
    save_per_class(per_class_acc, save_path)
    save_failures(failures, save_path)

    # 恢复 stdout
    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
