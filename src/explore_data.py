# -*- coding: utf-8 -*-
"""
explore_data.py
GTSRB 数据探查脚本：
1) 统计训练集 / 测试集类别分布
2) 统计图像尺寸分布（交通标志图像大小不一）
3) 可视化每个类别的样本示例
4) 输出类别分布柱状图

产出：figures/ 下的探查图，results/ 下的统计 CSV
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无窗口后端，便于脚本运行
import matplotlib.pyplot as plt

import torchvision
from torchvision.datasets import GTSRB

from class_names import GTSRB_CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


def load_datasets():
    print("加载数据集（不下载，仅读取本地）……")
    train = GTSRB(root=str(DATA_ROOT), split="train", download=False)
    test = GTSRB(root=str(DATA_ROOT), split="test", download=False)
    print(f"  训练集: {len(train)} 张")
    print(f"  测试集: {len(test)} 张")
    return train, test


def class_distribution(dataset, name):
    """统计每个类别的样本数。"""
    # GTSRB 训练集样本格式: (PIL_image, class_id)
    # GTSRB 测试集样本格式:   (PIL_image, class_id)
    labels = [dataset[i][1] for i in range(len(dataset))]
    counter = Counter(labels)
    df = pd.DataFrame(
        sorted(counter.items()),
        columns=["class_id", "count"],
    )
    df["class_name"] = df["class_id"].apply(lambda i: GTSRB_CLASS_NAMES[i])
    df.to_csv(RES_DIR / f"{name}_class_distribution.csv", index=False)
    print(f"  {name} 类别数: {len(counter)}")
    print(f"  {name} 每类最少: {min(counter.values())}  最多: {max(counter.values())}")
    return df


def image_size_stats(dataset, name, sample_n=2000):
    """抽样统计图像尺寸分布。"""
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(dataset), size=min(sample_n, len(dataset)), replace=False)
    widths, heights = [], []
    for i in idxs:
        img = dataset[int(i)][0]
        widths.append(img.size[0])
        heights.append(img.size[1])
    df = pd.DataFrame({"width": widths, "height": heights})
    stats = df.describe()
    stats.to_csv(RES_DIR / f"{name}_image_size_stats.csv")
    print(f"  {name} 图像尺寸 (抽样 {len(idxs)}): "
          f"宽 {df.width.min()}-{df.width.max()} (中位 {int(df.width.median())}), "
          f"高 {df.height.min()}-{df.height.max()} (中位 {int(df.height.median())})")
    return df


def plot_distribution(train_df, test_df):
    """画训练/测试类别分布对比柱状图。"""
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(train_df))
    width = 0.4
    ax.bar(x - width/2, train_df["count"], width, label="train", color="#4C72B0")
    if len(test_df) == len(train_df):
        ax.bar(x + width/2, test_df["count"], width, label="test", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(train_df["class_id"], fontsize=7, rotation=90)
    ax.set_xlabel("class id")
    ax.set_ylabel("number of images")
    ax.set_title("GTSRB class distribution (train vs test)")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  分布图已保存: {out}")


def plot_samples(dataset, n_per_class=1):
    """每个类别抽 1 张样本，拼成网格可视化。"""
    # 收集每个类别的第一张图
    seen = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in seen:
            seen[label] = i
        if len(seen) == 43:
            break

    cols, rows = 11, 4  # 11*4=44 >= 43
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6))
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx < 43:
            img, label = dataset[seen[idx]]
            ax.imshow(img)
            ax.set_title(f"{label}", fontsize=7)
        ax.axis("off")
    plt.suptitle("GTSRB — one sample per class (0-42)", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "samples_per_class.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  样本图已保存: {out}")


def main():
    print("=" * 60)
    print("GTSRB 数据探查")
    print("=" * 60)
    train, test = load_datasets()
    print("\n[1] 类别分布")
    train_df = class_distribution(train, "train")
    test_df = class_distribution(test, "test")
    print("\n[2] 图像尺寸统计")
    image_size_stats(train, "train")
    image_size_stats(test, "test")
    print("\n[3] 生成图表")
    plot_distribution(train_df, test_df)
    plot_samples(train)
    print("\n✅ 探查完成。输出目录:")
    print(f"   图表: {FIG_DIR}")
    print(f"   统计: {RES_DIR}")


if __name__ == "__main__":
    main()
