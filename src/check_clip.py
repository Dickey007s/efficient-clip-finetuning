# -*- coding: utf-8 -*-
"""
check_clip.py
验证 CLIP 模型可加载、可推理，并跑一个最小零样本 sanity check。

作用：
1) 确认 open_clip + CLIP 权重在 4060 上能正常加载、显存占用合理
2) 确认 CLIP 的 image encoder 和 text encoder 都能 forward
3) 用 100 张 GTSRB 图片做一次零样本分类，初步看准确率

这是后续 5 种微调方法的"地基"，地基通了，后面才好做。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import time

import torch
import open_clip
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from class_names import all_prompts, GTSRB_CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip():
    """加载 CLIP ViT-B/32 (OpenAI 权重)。"""
    print("加载 CLIP ViT-B/32 (openai) ...")
    # OpenAI 权重使用 QuickGELU，需显式指定以消除配置不一致警告
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", quick_gelu=True
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params/1e6:.1f}M")
    print(f"  设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return model, preprocess, tokenizer


def zero_shot_check(model, preprocess, tokenizer, n_samples=100):
    """用 n 张 GTSRB 图做零样本分类，看初始准确率。"""
    print(f"\n零样本分类 sanity check (前 {n_samples} 张训练集图片) ...")
    test_set = GTSRB(root=str(DATA_ROOT), split="test", download=False)
    # 取前 n 张
    idxs = list(range(n_samples))
    subset = Subset(test_set, idxs)

    # CLIP 预处理已含 resize/normalize，但 GTSRB 返回 PIL，直接用
    # 注意：dataset 的 transform 默认 None，这里手动套 preprocess
    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, base, tfm):
            self.base = base
            self.tfm = tfm
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            img, label = self.base[i]
            return self.tfm(img), label

    ds = WrappedDataset(subset, preprocess)
    loader = DataLoader(ds, batch_size=32, num_workers=0, shuffle=False)

    # 编码所有类别 prompt
    prompts = all_prompts("a photo of a {}")
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f"  文本特征矩阵: {text_features.shape}")  # [43, 512]

    correct, total = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # 余弦相似度 -> logits
            logits = (image_features @ text_features.t()) * model.logit_scale.exp()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    dt = time.time() - t0
    acc = correct / total * 100
    print(f"  零样本 Top-1 准确率: {acc:.2f}%  ({correct}/{total})")
    print(f"  耗时: {dt:.1f}s  ({total/dt:.1f} img/s)")
    return acc


def check_memory(model):
    """看 CLIP 推理时的显存占用。"""
    if DEVICE != "cuda":
        print("  (CPU 模式，跳过显存检查)")
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n显存占用: 已分配 {allocated:.2f} GB / 已保留 {reserved:.2f} GB")


def main():
    print("=" * 60)
    print("CLIP 模型验证")
    print("=" * 60)
    model, preprocess, tokenizer = load_clip()
    check_memory(model)
    acc = zero_shot_check(model, preprocess, tokenizer, n_samples=200)
    print("\n✅ CLIP 验证通过。")
    print(f"   零样本基线准确率 ≈ {acc:.1f}%  (正式评估在完整测试集上做)")
    print("   这个数字偏低是预期的 —— 交通标志是符号化图像，远离 CLIP 训练分布。")
    print("   这正是我们要做高效微调的动机。")


if __name__ == "__main__":
    main()
