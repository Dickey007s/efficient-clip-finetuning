# -*- coding: utf-8 -*-
"""
train_m2.py
M2: CoOp (Context Optimization), stable feature-precompute version.

核心设计：
- CLIP image encoder 完全冻结。
- 先预计算 train/test image features。
- 训练阶段只优化 learnable prompt context vectors。
- 避免每个 batch 重复跑 ViT image encoder。
- 适合 full-data、few-shot、后续 sweep。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
import sys
from pathlib import Path
from io import StringIO
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB
from torchvision import transforms
import open_clip

from feature_cache import FeatureDataset, precompute_image_features, make_feature_loaders
from feature_cache import FeatureDataset, precompute_image_features, make_feature_loaders


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
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        self.buffer = StringIO()
        self.file_handle = None

        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            self.file_handle = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)
        if self.file_handle is not None:
            self.file_handle.write(message)
            self.file_handle.flush()

    def flush(self):
        self.terminal.flush()
        self.buffer.flush()
        if self.file_handle is not None:
            self.file_handle.flush()

    def getvalue(self):
        return self.buffer.getvalue()

    def close(self):
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None


class PromptLearner(nn.Module):
    """
    CoOp prompt learner:
    [SOS] + [learnable ctx tokens] + class name + [.]
    """

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
        prompts = torch.cat(
            [
                self.token_prefix,
                ctx,
                self.token_suffix,
            ],
            dim=1,
        )
        return prompts


class CoOpFeatureModel(nn.Module):
    """
    CoOp model using frozen precomputed image features.
    Only prompt_learner.ctx receives gradients.
    """

    def __init__(self, clip_model, prompt_learner):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.dtype = clip_model.token_embedding.weight.dtype

    def encode_text_with_prompts(self):
        prompts = self.prompt_learner()

        x = prompts + self.clip_model.positional_embedding.to(prompts.dtype)
        x = self.clip_model.transformer(
            x,
            attn_mask=self.clip_model.attn_mask,
        )
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


def get_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


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


def load_image_data(batch_size=64, shots_per_class=None, seed=42):
    preprocess = get_preprocess()

    train_raw = GTSRB(root=str(DATA_ROOT), split="train", download=False)
    test_raw = GTSRB(root=str(DATA_ROOT), split="test", download=False)

    if shots_per_class is not None:
        train_raw = create_few_shot_subset(train_raw, shots_per_class, seed=seed)
        print(
            f"[Data] Few-shot mode: {shots_per_class} shots/class, "
            f"total {len(train_raw)} samples"
        )
    else:
        print(f"[Data] Full training set: {len(train_raw)} samples")

    print(f"[Data] Test set: {len(test_raw)} samples")

    train_set = WrappedDataset(train_raw, preprocess)
    test_set = WrappedDataset(test_raw, preprocess)

    # Windows stable: num_workers=0, pin_memory=False
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, test_loader


def make_feature_loaders(train_features, train_labels, test_features, test_labels, batch_size):
    train_set = FeatureDataset(train_features, train_labels)
    test_set = FeatureDataset(test_features, test_labels)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, test_loader


def load_clip():
    print(f"[Model] Loading CLIP ViT-B/32 (openai) on {DEVICE} ...")

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        force_quick_gelu=True,
    )

    model = model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad = False

    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Model] CLIP total params: {n_total / 1e6:.1f}M (frozen)")

    return model


@torch.no_grad()
def check_text_encoder_equivalence(clip_model):
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
            f"Text encoder equivalence check FAILED: "
            f"max_diff={max_diff:.2e}, mean_cos={mean_cos:.6f}"
        )

    print("[Check] text encoder equivalence check PASSED")


@torch.no_grad()
def precompute_image_features(clip_model, image_loader, name="train", use_amp=True):
    clip_model.eval()

    all_features = []
    all_labels = []

    print(f"[Precompute] Encoding {name} image features...")
    t0 = time.time()

    amp_enabled = use_amp and DEVICE.type == "cuda"

    for batch_idx, (images, labels) in enumerate(image_loader):
        images = images.to(DEVICE, non_blocking=True)

        ctx = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
        with ctx:
            features = clip_model.encode_image(
                images.type(clip_model.token_embedding.weight.dtype)
            )
            features = F.normalize(features, dim=-1)

        all_features.append(features.float().cpu())
        all_labels.append(labels.cpu().long())

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(image_loader):
            print(
                f"[Precompute] {name}: "
                f"{batch_idx + 1}/{len(image_loader)} batches"
            )

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    elapsed = time.time() - t0
    print(
        f"[Precompute] {name} done: "
        f"features={tuple(features.shape)}, labels={tuple(labels.shape)}, "
        f"time={elapsed:.1f}s"
    )

    return features, labels


@torch.no_grad()
def evaluate_acc(model, test_loader, use_amp=False):
    """轻量评估：只返回准确率。"""
    model.eval()
    correct, total = 0, 0
    amp_enabled = use_amp and DEVICE.type == "cuda"
    for features, labels in test_loader:
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        ctx = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
        with ctx:
            logits = model(features)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train_epoch(model, train_loader, optimizer, criterion, use_amp=False):
    model.train()
    model.clip_model.eval()
    model.prompt_learner.train()

    total_loss = 0.0

    amp_enabled = use_amp and DEVICE.type == "cuda"

    for features, labels in train_loader:
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        ctx = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
        with ctx:
            logits = model(features)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate_detailed(model, test_loader, use_amp=False):
    model.eval()

    correct, total = 0, 0
    all_preds = []
    all_labels = []
    all_indices = []

    amp_enabled = use_amp and DEVICE.type == "cuda"

    for batch_idx, (features, labels) in enumerate(test_loader):
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        ctx = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
        with ctx:
            logits = model(features)

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
        per_class_acc[c] = (
            100.0 * class_correct / class_total if class_total > 0 else 0.0
        )

    failures = [
        (idx, t, p)
        for idx, t, p in zip(all_indices, all_labels, all_preds)
        if t != p
    ]

    return acc, confusion, per_class_acc, failures


def save_metrics(metrics, save_path):
    import csv

    csv_path = save_path.with_suffix("").with_name(save_path.stem + "_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "test_acc"])
        for row in metrics:
            writer.writerow(row)

    print(f"[Save] Metrics saved to {csv_path}")


def save_confusion(confusion, save_path):
    import csv

    csv_path = save_path.with_suffix("").with_name(save_path.stem + "_confusion.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class"] + [f"pred_{i}" for i in range(43)])
        for i in range(43):
            writer.writerow([i] + confusion[i].tolist())

    print(f"[Save] Confusion matrix saved to {csv_path}")


def save_per_class(per_class_acc, save_path):
    import csv

    csv_path = save_path.with_suffix("").with_name(save_path.stem + "_per_class.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "accuracy"])
        for c in range(43):
            writer.writerow([c, CLASS_NAMES[c], f"{per_class_acc[c]:.2f}"])

    print(f"[Save] Per-class accuracy saved to {csv_path}")


def save_failures(failures, save_path, max_samples=100):
    import csv

    csv_path = save_path.with_suffix("").with_name(save_path.stem + "_failures.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_idx", "true_label", "true_name", "pred_label", "pred_name"]
        )
        for idx, t, p in failures[:max_samples]:
            writer.writerow([idx, t, CLASS_NAMES[t], p, CLASS_NAMES[p]])

    print(f"[Save] Failures ({min(len(failures), max_samples)} samples) saved to {csv_path}")


def build_optimizer(name, params, lr, weight_decay):
    name = name.lower()

    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="outputs/m2_coop.pt")

    parser.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--amp_precompute", action="store_true",
                        help="Use AMP when precomputing frozen image features.")
    parser.add_argument("--amp_train", action="store_true",
                        help="Use AMP during prompt training. Usually unnecessary.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_path = Path(args.save)
    log_path = save_path.with_suffix(".log")

    logger = Logger(log_file=str(log_path))
    sys.stdout = logger
    sys.stderr = logger

    try:
        print("=" * 60)
        print("M2: CoOp (Feature-Precompute Stable Version)")
        print(
            f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size} | "
            f"Shots: {args.shots or 'full'} | N_CTX: {args.n_ctx}"
        )
        print(f"Optimizer: {args.optim} | Weight decay: {args.weight_decay}")
        print(f"Device: {DEVICE}")
        print("=" * 60)

        image_train_loader, image_test_loader = load_image_data(
            batch_size=args.batch_size,
            shots_per_class=args.shots,
            seed=args.seed,
        )

        print("[Test] Testing image DataLoader...")
        test_images, test_labels = next(iter(image_train_loader))
        print(f"[Test] Image DataLoader OK: {tuple(test_images.shape)}, {tuple(test_labels.shape)}")

        clip_model = load_clip()
        check_text_encoder_equivalence(clip_model)

        ctx_dim = clip_model.token_embedding.weight.shape[1]
        print(f"[Model] Context dimension: {ctx_dim}")

        prompt_learner = PromptLearner(
            n_ctx=args.n_ctx,
            ctx_dim=ctx_dim,
            n_cls=43,
            classnames=CLASS_NAMES,
            clip_model=clip_model,
        ).to(DEVICE)

        n_trainable = sum(
            p.numel() for p in prompt_learner.parameters() if p.requires_grad
        )
        print(f"[Model] PromptLearner params: {n_trainable:,} ({n_trainable / 1e3:.1f}K)")

        train_features, train_labels = precompute_image_features(
            clip_model,
            image_train_loader,
            name="train",
            use_amp=args.amp_precompute,
        )

        test_features, test_labels = precompute_image_features(
            clip_model,
            image_test_loader,
            name="test",
            use_amp=args.amp_precompute,
        )

        train_loader, test_loader = make_feature_loaders(
            train_features,
            train_labels,
            test_features,
            test_labels,
            batch_size=args.batch_size,
        )

        print("[Precompute] Replaced image dataloaders with feature dataloaders.")
        print("[Test] Testing feature DataLoader...")
        test_feat, test_lab = next(iter(train_loader))
        print(f"[Test] Feature DataLoader OK: {tuple(test_feat.shape)}, {tuple(test_lab.shape)}")

        model = CoOpFeatureModel(clip_model, prompt_learner).to(DEVICE)

        optimizer = build_optimizer(
            args.optim,
            prompt_learner.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_state = None
        metrics = []
        confusion = None
        per_class_acc = None
        failures = []

        t_start = time.time()

        print(f"[Train] Starting training for {args.epochs} epochs...")

        for epoch in range(args.epochs):
            t0 = time.time()

            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                use_amp=args.amp_train,
            )

            acc = evaluate_acc(
                model,
                test_loader,
                use_amp=args.amp_train,
            )

            epoch_time = time.time() - t0

            if acc > best_acc:
                best_acc = acc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in prompt_learner.state_dict().items()
                }

            metrics.append([epoch + 1, loss, acc])

            print(
                f"Epoch {epoch + 1:2d}/{args.epochs} | "
                f"Loss: {loss:.4f} | Test Acc: {acc:.2f}% | "
                f"Best: {best_acc:.2f}% | Time: {epoch_time:.1f}s"
            )

        total_time = time.time() - t_start

        # 加载 best state 后只调用一次详细评估
        if best_state is not None:
            prompt_learner.load_state_dict(best_state)
        acc, confusion, per_class_acc, failures = evaluate_detailed(
            model,
            test_loader,
            use_amp=args.amp_train,
        )

        print("=" * 60)
        print(f"M2 CoOp Best Accuracy: {best_acc:.2f}%")
        print(f"Detailed eval accuracy: {acc:.2f}%")
        print(f"Total time: {total_time:.1f}s")

        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory: {alloc:.2f} GB")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if best_state is not None:
            torch.save(best_state, save_path)
            print(f"[Save] Best PromptLearner state saved to {save_path}")
        else:
            torch.save(prompt_learner.state_dict(), save_path)
            print(f"[Save] Final PromptLearner state saved to {save_path}")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(logger.getvalue())
        print(f"[Save] Log saved to {log_path}")

        save_metrics(metrics, save_path)

        if confusion is not None:
            save_confusion(confusion, save_path)
            save_per_class(per_class_acc, save_path)
            save_failures(failures, save_path)

    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()