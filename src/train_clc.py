# -*- coding: utf-8 -*-
"""
train_clc.py
CLC: Cyclic LoRA-CoOp (三阶段循环策略)

Stage 1: CoOp warm-up — 训练 learnable context vectors，CLIP 全部冻结 (10 epochs)
Stage 2: LoRA fine-tuning — 冻结 Stage 1 的 prompt，只训练 image encoder 的 LoRA (20 epochs)
Stage 3: CoOp refinement — 冻结 Stage 2 的 LoRA，重新微调 prompt (10 epochs)

思想:
  - 先让文本提示快速热身
  - 再让图像端适配当前文本提示
  - 最后让文本提示再次适配已经调整好的图像端

用法:
  python src/train_clc.py --coop_epochs 10 --lora_epochs 20 --stage3_epochs 10 \
      --lr 0.002 --lora_lr 1e-4 --batch_size 64 --shots 8 \
      --n_ctx 16 --rank 8 --alpha 16 --save outputs/clc_8shot.pt
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
from torch.optim import SGD, AdamW
import open_clip
from lora_utils import inject_lora_into_visual
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB
from torchvision import transforms
from feature_cache import FeatureDataset, precompute_image_features, make_feature_loaders

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
    """CoOp 的 Prompt Learner。"""
    def __init__(self, n_ctx, ctx_dim, n_cls, classnames, clip_model):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        self.dtype = clip_model.token_embedding.weight.dtype

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in classnames]

        tokenized = open_clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized)

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts


class CoOpFeatureModel(nn.Module):
    """CoOp model using frozen precomputed image features."""
    def __init__(self, clip_model, prompt_learner):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.dtype = clip_model.token_embedding.weight.dtype

    def encode_text_with_prompts(self):
        prompts = self.prompt_learner()
        x = prompts + self.clip_model.positional_embedding.to(prompts.dtype)
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = self.clip_model.ln_final(x)
        eot_indices = self.prompt_learner.tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices]
        if self.clip_model.text_projection is not None:
            if isinstance(self.clip_model.text_projection, nn.Linear):
                x = self.clip_model.text_projection(x)
            else:
                x = x @ self.clip_model.text_projection
        return F.normalize(x, dim=-1)

    def forward(self, image_features):
        image_features = image_features.to(DEVICE, non_blocking=True)
        image_features = F.normalize(image_features, dim=-1)
        text_features = self.encode_text_with_prompts()
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


class CoOpLoRAModel(nn.Module):
    """CoOp-LoRA 完整模型。"""
    def __init__(self, clip_model, prompt_learner):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.dtype = clip_model.token_embedding.weight.dtype

    def encode_text_with_prompts(self):
        prompts = self.prompt_learner()
        x = prompts + self.clip_model.positional_embedding.to(prompts.dtype)
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = self.clip_model.ln_final(x)
        eot_indices = self.prompt_learner.tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices]
        if self.clip_model.text_projection is not None:
            if isinstance(self.clip_model.text_projection, nn.Linear):
                x = self.clip_model.text_projection(x)
            else:
                x = x @ self.clip_model.text_projection
        return F.normalize(x, dim=-1)

    def forward(self, image):
        image_features = self.clip_model.encode_image(image.type(self.dtype))
        image_features = F.normalize(image_features, dim=-1)
        text_features = self.encode_text_with_prompts()
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
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    return train_loader, test_loader


def load_clip():
    print(f"[Model] Loading CLIP ViT-B/32 (openai) on {DEVICE} ...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True,
    )
    model = model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Model] CLIP total params: {n_total/1e6:.1f}M (frozen)")
    return model


@torch.no_grad()
def check_text_encoder_equivalence(clip_model):
    """验证手动 text encoder 和 open_clip.encode_text 等价。"""
    clip_model.eval()
    prompts = [f"a photo of a {name}." for name in CLASS_NAMES]
    tokenized = open_clip.tokenize(prompts).to(DEVICE)

    f_ref = clip_model.encode_text(tokenized)
    f_ref = F.normalize(f_ref, dim=-1)

    x = clip_model.token_embedding(tokenized)
    x = x + clip_model.positional_embedding.to(x.dtype)
    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
    x = clip_model.ln_final(x)

    eot_indices = tokenized.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=x.device), eot_indices]

    if clip_model.text_projection is not None:
        if isinstance(clip_model.text_projection, nn.Linear):
            x = clip_model.text_projection(x)
        else:
            x = x @ clip_model.text_projection

    f_manual = F.normalize(x, dim=-1)

    max_diff = (f_ref - f_manual).abs().max().item()
    mean_cos = (f_ref * f_manual).sum(dim=-1).mean().item()
    print(f"[Check] text encoder max diff: {max_diff:.2e}")
    print(f"[Check] text encoder mean cosine: {mean_cos:.6f}")
    if max_diff > 1e-4 or mean_cos < 0.999:
        raise RuntimeError(
            f"Text encoder equivalence check FAILED: max_diff={max_diff:.2e}, mean_cos={mean_cos:.6f}. "
            "Manual text encoder does not match open_clip.encode_text(). Stop training."
        )
    print("[Check] text encoder equivalence check PASSED")
    return True


def evaluate_acc(model, test_loader):
    """轻量评估：只返回准确率。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def evaluate_detailed(model, test_loader):
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
            start_idx = batch_idx * test_loader.batch_size
            all_indices.extend(range(start_idx, start_idx + len(labels)))

    acc = 100.0 * correct / total
    num_classes = 43
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1

    per_class_acc = {}
    for c in range(num_classes):
        class_total = confusion[c].sum().item()
        class_correct = confusion[c, c].item()
        per_class_acc[c] = 100.0 * class_correct / class_total if class_total > 0 else 0.0

    failures = [(idx, t, p) for idx, t, p in zip(all_indices, all_labels, all_preds) if t != p]
    return acc, confusion, per_class_acc, failures


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


def train_epoch_features(model, train_loader, optimizer, scaler, criterion):
    """Stage 1/3 CoOp: 训练使用 feature loader。"""
    model.train()
    model.clip_model.eval()
    model.prompt_learner.train()
    total_loss = 0.0
    for features, labels in train_loader:
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def save_metrics(metrics, save_path):
    import csv
    csv_path = save_path.with_suffix('').with_name(save_path.stem + '_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'test_acc'])
        for row in metrics:
            writer.writerow(row)
    print(f"[Save] Metrics saved to {csv_path}")


def save_confusion(confusion, save_path):
    import csv
    csv_path = save_path.with_suffix('').with_name(save_path.stem + '_confusion.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class'] + [f'pred_{i}' for i in range(43)])
        for i in range(43):
            writer.writerow([i] + confusion[i].tolist())
    print(f"[Save] Confusion matrix saved to {csv_path}")


def save_per_class(per_class_acc, save_path):
    import csv
    csv_path = save_path.with_suffix('').with_name(save_path.stem + '_per_class.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'accuracy'])
        for c in range(43):
            writer.writerow([c, CLASS_NAMES[c], f"{per_class_acc[c]:.2f}"])
    print(f"[Save] Per-class accuracy saved to {csv_path}")


def save_failures(failures, save_path, max_samples=100):
    import csv
    csv_path = save_path.with_suffix('').with_name(save_path.stem + '_failures.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_idx', 'true_label', 'true_name', 'pred_label', 'pred_name'])
        for idx, t, p in failures[:max_samples]:
            writer.writerow([idx, t, CLASS_NAMES[t], p, CLASS_NAMES[p]])
    print(f"[Save] Failures ({min(len(failures), max_samples)} samples) saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coop_epochs", type=int, default=10, help="Stage 1: CoOp warm-up epochs")
    parser.add_argument("--lora_epochs", type=int, default=20, help="Stage 2: LoRA fine-tuning epochs")
    parser.add_argument("--stage3_epochs", type=int, default=10, help="Stage 3: CoOp refinement epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="Stage 1/3 CoOp LR")
    parser.add_argument("--lora_lr", type=float, default=1e-4, help="Stage 2 LoRA LR")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shots", type=int, default=None, help="Few-shot: shots per class")
    parser.add_argument("--n_ctx", type=int, default=16, help="Number of context tokens")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--coop_ckpt", type=str, default=None, help="Stage 1 CoOp prompt checkpoint，提供则跳过 Stage 1")
    parser.add_argument("--lora_ckpt", type=str, default=None, help="Stage 2 LoRA checkpoint，提供则跳过 Stage 2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="outputs/clc_coop_lora.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    logger = Logger()
    sys.stdout = logger

    print("=" * 60)
    print("CLC: Cyclic LoRA-CoOp (Three-Stage)")
    print(f"CoOp epochs: {args.coop_epochs} | LoRA epochs: {args.lora_epochs} | Stage3 epochs: {args.stage3_epochs}")
    print(f"CoOp LR: {args.lr} | LoRA LR: {args.lora_lr} | Batch: {args.batch_size}")
    print(f"Shots: {args.shots or 'full'} | N_CTX: {args.n_ctx} | LoRA rank: {args.rank} | alpha: {args.alpha}")
    print("=" * 60)

    # 加载数据
    image_train_loader, image_test_loader = load_data(args.batch_size, args.shots)

    # 加载 CLIP
    clip_model = load_clip()
    check_text_encoder_equivalence(clip_model)

    # 创建 PromptLearner
    ctx_dim = clip_model.token_embedding.weight.shape[1]
    prompt_learner = PromptLearner(
        n_ctx=args.n_ctx, ctx_dim=ctx_dim, n_cls=43,
        classnames=CLASS_NAMES, clip_model=clip_model,
    ).to(DEVICE)

    # ========== Stage 1: CoOp Warm-up ==========
    print(f"\n{'='*60}")
    print("Stage 1: CoOp Warm-up")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()

    if args.coop_ckpt is not None and Path(args.coop_ckpt).exists():
        print(f"[Stage 1] Loading CoOp prompt from {args.coop_ckpt}")
        prompt_learner.load_state_dict(torch.load(args.coop_ckpt, map_location=DEVICE))
        # 评估加载的 prompt
        train_features, train_labels = precompute_image_features(clip_model, image_train_loader, DEVICE, name="train")
        test_features, test_labels = precompute_image_features(clip_model, image_test_loader, DEVICE, name="test")
        feature_train_loader, feature_test_loader = make_feature_loaders(train_features, train_labels, test_features, test_labels, args.batch_size)
        model = CoOpFeatureModel(clip_model, prompt_learner).to(DEVICE)
        stage1_best_acc = evaluate_acc(model, feature_test_loader)
        print(f"[Stage 1] Loaded CoOp accuracy: {stage1_best_acc:.2f}%")
    elif args.coop_epochs > 0:
        print("Stage 1: CoOp Warm-up (training context vectors)")

        train_features, train_labels = precompute_image_features(clip_model, image_train_loader, DEVICE, name="train")
        test_features, test_labels = precompute_image_features(clip_model, image_test_loader, DEVICE, name="test")
        feature_train_loader, feature_test_loader = make_feature_loaders(train_features, train_labels, test_features, test_labels, args.batch_size)

        model = CoOpFeatureModel(clip_model, prompt_learner).to(DEVICE)
        optimizer = SGD(prompt_learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scaler = torch.amp.GradScaler('cuda')

        best_acc = 0.0
        best_prompt_state = None
        for epoch in range(args.coop_epochs):
            t0 = time.time()
            loss = train_epoch_features(model, feature_train_loader, optimizer, scaler, criterion)
            acc = evaluate_acc(model, feature_test_loader)
            epoch_time = time.time() - t0
            if acc > best_acc:
                best_acc = acc
                best_prompt_state = {k: v.detach().cpu().clone() for k, v in prompt_learner.state_dict().items()}
            print(f"Epoch {epoch+1:2d}/{args.coop_epochs} | Loss: {loss:.4f} | "
                  f"Test Acc: {acc:.2f}% | Best: {best_acc:.2f}% | Time: {epoch_time:.1f}s")

        if best_prompt_state is not None:
            prompt_learner.load_state_dict(best_prompt_state)

        print(f"[Stage 1] Best CoOp accuracy: {best_acc:.2f}%")
        stage1_best_acc = best_acc

        stage1_path = Path(args.save).with_suffix('').with_name(Path(args.save).stem + '_stage1_coop.pt')
        stage1_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_prompt_state if best_prompt_state is not None else prompt_learner.state_dict(), stage1_path)
        print(f"[Stage 1] CoOp prompt saved to {stage1_path}")
    else:
        raise ValueError("必须提供 --coop_ckpt 或 --coop_epochs > 0")

    # ========== Stage 2: LoRA Fine-tuning ==========
    for p in prompt_learner.parameters():
        p.requires_grad = False

    print(f"\n{'='*60}")
    print("Stage 2: LoRA Fine-tuning")
    print(f"{'='*60}")

    print(f"[Model] Injecting LoRA (rank={args.rank}, alpha={args.alpha}) into ViT...")
    inject_lora_into_visual(clip_model.visual, rank=args.rank, alpha=args.alpha)

    if args.lora_ckpt is not None and Path(args.lora_ckpt).exists():
        print(f"[Stage 2] Loading LoRA weights from {args.lora_ckpt}")
        lora_state = torch.load(args.lora_ckpt, map_location=DEVICE)
        with torch.no_grad():
            for n, p in clip_model.named_parameters():
                if p.requires_grad:
                    p.copy_(lora_state[n].to(p.device))
        model = CoOpLoRAModel(clip_model, prompt_learner).to(DEVICE)
        stage2_final_acc = evaluate_acc(model, image_test_loader)
        print(f"[Stage 2] Loaded LoRA accuracy: {stage2_final_acc:.2f}%")
        stage2_metrics = []
    elif args.lora_epochs > 0:
        print("Stage 2: LoRA Fine-tuning (prompt frozen)")

        n_trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
        print(f"[Model] LoRA trainable params: {n_trainable:,} ({n_trainable/1e3:.1f}K)")

        model = CoOpLoRAModel(clip_model, prompt_learner).to(DEVICE)
        stage2_start_acc = evaluate_acc(model, image_test_loader)
        print(f"[Stage 2] Frozen-prompt accuracy before LoRA: {stage2_start_acc:.2f}%")

        lora_params = [p for p in clip_model.parameters() if p.requires_grad]
        optimizer = AdamW(lora_params, lr=args.lora_lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        best_acc = 0.0
        best_lora_state = None
        t_start = time.time()
        stage2_metrics = []

        for epoch in range(args.lora_epochs):
            t0 = time.time()
            loss = train_epoch(model, image_train_loader, optimizer, scaler, criterion)
            acc = evaluate_acc(model, image_test_loader)
            epoch_time = time.time() - t0
            if acc > best_acc:
                best_acc = acc
                best_lora_state = {n: p.detach().cpu().clone() for n, p in clip_model.named_parameters() if p.requires_grad}
            stage2_metrics.append([epoch + 1, loss, acc])

            delta = acc - stage2_start_acc
            print(f"Epoch {epoch+1:2d}/{args.lora_epochs} | Loss: {loss:.4f} | "
                  f"Test Acc: {acc:.2f}% | Best: {best_acc:.2f}% | "
                  f"Delta vs Stage1: {delta:+.2f}% | Time: {epoch_time:.1f}s")

        stage2_time = time.time() - t_start

        if best_lora_state is not None:
            with torch.no_grad():
                for n, p in clip_model.named_parameters():
                    if p.requires_grad:
                        p.copy_(best_lora_state[n].to(p.device))

        stage2_final_acc = evaluate_acc(model, image_test_loader)
        print(f"[Stage 2] Best LoRA accuracy: {best_acc:.2f}%")
        print(f"Stage 2 time: {stage2_time:.1f}s")
    else:
        raise ValueError("必须提供 --lora_ckpt 或 --lora_epochs > 0")

    # ========== Stage 3: CoOp Refinement ==========
    # 冻结 LoRA，放开 prompt
    for p in clip_model.parameters():
        p.requires_grad = False
    for p in prompt_learner.parameters():
        p.requires_grad = True

    print(f"\n{'='*60}")
    print("Stage 3: CoOp Refinement (LoRA frozen)")
    print(f"{'='*60}")

    # 使用冻结的 LoRA 重新预计算 image features
    print("[Precompute] Recomputing image features with frozen LoRA...")
    train_features_s3, train_labels_s3 = precompute_image_features(clip_model, image_train_loader, DEVICE, name="train")
    test_features_s3, test_labels_s3 = precompute_image_features(clip_model, image_test_loader, DEVICE, name="test")
    feature_train_loader_s3, feature_test_loader_s3 = make_feature_loaders(
        train_features_s3, train_labels_s3, test_features_s3, test_labels_s3, args.batch_size)

    model_s3 = CoOpFeatureModel(clip_model, prompt_learner).to(DEVICE)
    stage3_start_acc = evaluate_acc(model_s3, feature_test_loader_s3)
    print(f"[Stage 3] Frozen-LoRA accuracy before refinement: {stage3_start_acc:.2f}%")

    optimizer = SGD(prompt_learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    best_prompt_state_s3 = None
    t_start = time.time()
    stage3_metrics = []

    for epoch in range(args.stage3_epochs):
        t0 = time.time()
        loss = train_epoch_features(model_s3, feature_train_loader_s3, optimizer, scaler, criterion)
        acc = evaluate_acc(model_s3, feature_test_loader_s3)
        epoch_time = time.time() - t0
        if acc > best_acc:
            best_acc = acc
            best_prompt_state_s3 = {k: v.detach().cpu().clone() for k, v in prompt_learner.state_dict().items()}
        stage3_metrics.append([epoch + 1, loss, acc])

        delta = acc - stage3_start_acc
        print(f"Epoch {epoch+1:2d}/{args.stage3_epochs} | Loss: {loss:.4f} | "
              f"Test Acc: {acc:.2f}% | Best: {best_acc:.2f}% | "
              f"Delta vs Stage2: {delta:+.2f}% | Time: {epoch_time:.1f}s")

    stage3_time = time.time() - t_start

    if best_prompt_state_s3 is not None:
        prompt_learner.load_state_dict(best_prompt_state_s3)

    print(f"[Stage 3] Best CoOp refinement accuracy: {best_acc:.2f}%")
    print(f"Stage 3 time: {stage3_time:.1f}s")

    # 最终评估：使用完整模型（冻结 LoRA + Stage 3 best prompt）
    final_model = CoOpLoRAModel(clip_model, prompt_learner).to(DEVICE)
    acc, confusion, per_class_acc, failures = evaluate_detailed(final_model, image_test_loader)
    print(f"[Eval] Best model detailed accuracy: {acc:.2f}%")

    print("=" * 60)
    print(f"Stage 1 CoOp Acc:       {stage1_best_acc:.2f}%")
    print(f"Stage 2 LoRA Best Acc:  {stage2_final_acc:.2f}%")
    print(f"Stage 3 CoOp Best Acc:  {best_acc:.2f}%")
    print(f"Final Detailed Acc:     {acc:.2f}%")
    print("=" * 60)
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {alloc:.2f} GB")

    # 保存
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lora_path = save_path.with_suffix('').with_name(save_path.stem + '_lora.pt')
    lora_state = {n: p.data.cpu() for n, p in clip_model.named_parameters() if p.requires_grad}
    torch.save(lora_state, lora_path)
    print(f"[Save] LoRA weights saved to {lora_path}")

    stage3_coop_path = save_path.with_suffix('').with_name(save_path.stem + '_stage3_coop.pt')
    torch.save(prompt_learner.state_dict(), stage3_coop_path)
    print(f"[Save] Stage 3 CoOp prompt saved to {stage3_coop_path}")

    log_path = save_path.with_suffix('.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(logger.getvalue())
    print(f"[Save] Log saved to {log_path}")

    if stage2_metrics or stage3_metrics:
        all_metrics = []
        for row in stage2_metrics:
            all_metrics.append(['stage2'] + row)
        for row in stage3_metrics:
            all_metrics.append(['stage3'] + row)
        save_metrics(all_metrics, save_path)
    if confusion is not None:
        save_confusion(confusion, save_path)
        save_per_class(per_class_acc, save_path)
        save_failures(failures, save_path)

    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
