# -*- coding: utf-8 -*-
"""
train_m2.py
M2: CoOp (Context Optimization)
冻结 CLIP 主干，学习连续的 prompt context vectors。

参考: Zhou et al., "Learning to Prompt for Vision-Language Models", IJCV 2022
官方代码: https://github.com/KaiyangZhou/CoOp

实现细节:
  - Unified context (共享 across classes)，默认 16 个 context tokens
  - Context 随机初始化: nn.init.normal_(std=0.02)
  - Prompt 结构: [SOS] + [ctx_1] + ... + [ctx_M] + class_name + .
  - 手动过 CLIP text encoder (transformer + ln_final + text_projection)
  - 只优化 context vectors，CLIP 全部冻结

用法:
  python src/train_m2.py --epochs 20 --lr 0.002 --batch_size 64 --shots 16 --n_ctx 16
  python src/train_m2.py --epochs 50 --lr 0.002 --batch_size 64 --n_ctx 16  # 全数据
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
from torch.optim import SGD
from torch.cuda.amp import autocast, GradScaler
import open_clip
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



class PromptLearner(nn.Module):
    """
    CoOp 的 Prompt Learner。
    学习连续的 context vectors，与 class name 拼接成完整 prompt。
    """
    def __init__(self, n_ctx, ctx_dim, n_cls, classnames, clip_model):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        self.dtype = clip_model.token_embedding.weight.dtype

        # 1. 随机初始化 context vectors (unified context，共享)
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # [n_ctx, ctx_dim]

        # 2. 预计算 SOS 和 class_name 的 embedding (冻结)
        # 构造 prompts: "X X ... X class_name."
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in classnames]

        tokenized = open_clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(self.dtype)

        # SOS token: [n_cls, 1, ctx_dim]
        self.register_buffer("token_prefix", embedding[:, :1, :])
        # class_name + . : [n_cls, *, ctx_dim]
        # 注意: tokenized 长度可能不同，但 embedding 已经 padding 到相同长度
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.register_buffer("tokenized_prompts", tokenized)

    def forward(self):
        """
        构造完整的 prompts: [SOS] + ctx + class_name + .
        返回: [n_cls, L, ctx_dim]
        """
        # 扩展 context 到所有类别
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, ctx_dim]

        # 拼接: [SOS] + ctx + class_name + .
        prompts = torch.cat([
            self.token_prefix,  # [n_cls, 1, ctx_dim]
            ctx,                # [n_cls, n_ctx, ctx_dim]
            self.token_suffix,  # [n_cls, *, ctx_dim]
        ], dim=1)
        return prompts


class CoOpModel(nn.Module):
    """
    完整的 CoOp 模型: CLIP image encoder + learnable prompt text encoder。
    """
    def __init__(self, clip_model, prompt_learner):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.dtype = clip_model.token_embedding.weight.dtype

    def forward(self, image):
        """
        返回 logits: [batch_size, n_cls]
        """
        # Image features (冻结 CLIP)
        image_features = self.clip_model.encode_image(image.type(self.dtype))
        image_features = F.normalize(image_features, dim=-1)

        # Text features with learnable prompts
        prompts = self.prompt_learner()  # [n_cls, L, ctx_dim]

        # 手动过 text encoder
        # 1. 加 positional embedding
        x = prompts + self.clip_model.positional_embedding.type(self.dtype)

        # 2. Transformer: NLD -> LND
        x = x.permute(1, 0, 2)  # [L, n_cls, ctx_dim]
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # [n_cls, L, ctx_dim]

        # 3. LayerNorm
        x = self.clip_model.ln_final(x).type(self.dtype)

        # 4. 取 EOT token (end-of-text)
        # EOT token 的位置: tokenized_prompts 中最大的值对应的位置
        eot_indices = self.prompt_learner.tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices]  # [n_cls, ctx_dim]

        # 5. Text projection
        text_features = x @ self.clip_model.text_projection
        text_features = F.normalize(text_features, dim=-1)

        # 6. 相似度
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


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


def load_clip():
    print(f"[Model] Loading CLIP ViT-B/32 (openai) on {DEVICE} ...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True,
    )
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Model] CLIP total params: {n_total/1e6:.1f}M (frozen)")
    return model


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast('cuda'):
            logits = model(images)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def evaluate_detailed(model, test_loader):
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
                logits = model(images)
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


def train_epoch(model, train_loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.002, help="SGD learning rate (CoOp default)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shots", type=int, default=None, help="Few-shot: shots per class")
    parser.add_argument("--n_ctx", type=int, default=16, help="Number of context tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="outputs/m2_coop.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 重定向 stdout 到 Logger
    logger = Logger()
    sys.stdout = logger

    print("=" * 60)
    print("M2: CoOp (Context Optimization)")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size} | Shots: {args.shots or 'full'} | N_CTX: {args.n_ctx}")
    print("=" * 60)

    # 加载数据
    train_loader, test_loader = load_data(args.batch_size, args.shots)

    # 加载 CLIP
    clip_model = load_clip()

    # 获取 context dimension
    ctx_dim = clip_model.token_embedding.weight.shape[1]
    print(f"[Model] Context dimension: {ctx_dim}")

    # 创建 Prompt Learner
    prompt_learner = PromptLearner(
        n_ctx=args.n_ctx,
        ctx_dim=ctx_dim,
        n_cls=43,
        classnames=CLASS_NAMES,
        clip_model=clip_model,
    ).to(DEVICE)

    n_trainable = sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)
    print(f"[Model] PromptLearner params: {n_trainable:,} ({n_trainable/1e3:.1f}K)")

    # 创建完整模型
    model = CoOpModel(clip_model, prompt_learner).to(DEVICE)

    # 只优化 prompt_learner
    optimizer = SGD(prompt_learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0.0
    t_start = time.time()
    metrics = []  # [epoch, loss, test_acc]

    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, scaler, criterion)
        
        # 评估（详细）
        acc, confusion, per_class_acc, failures = evaluate_detailed(model, test_loader)
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

    # 保存模型
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_learner.state_dict(), save_path)
    print(f"[Save] Model saved to {save_path}")

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
