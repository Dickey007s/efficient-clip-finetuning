# -*- coding: utf-8 -*-
"""
download_data.py
下载 GTSRB (German Traffic Sign Recognition Benchmark) 数据集到 data/ 目录。

数据集信息：
- 43 个类别（德国交通标志）
- 训练集 ~39,209 张，测试集 ~12,630 张
- 单图分类任务

下载策略：
1) 优先 torchvision.datasets.GTSRB（官方源）
2) 若官方源不稳定，参考脚本末尾的手动下载说明
"""
import os
import sys
from pathlib import Path

# 让 OpenMP 多副本冲突不至于报错（Windows + miniconda 常见）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def download_via_torchvision():
    print(f"[1/2] 准备下载 GTSRB 到: {DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    import torchvision
    print(f"torchvision version: {torchvision.__version__}")

    print("下载训练集（train=True）……")
    train_set = torchvision.datasets.GTSRB(
        root=str(DATA_ROOT),
        split="train",
        download=True,
    )

    print("下载测试集（split='test'）……")
    test_set = torchvision.datasets.GTSRB(
        root=str(DATA_ROOT),
        split="test",
        download=True,
    )

    print("\n=== 下载完成 ===")
    print(f"训练集样本数: {len(train_set)}")
    print(f"测试集样本数: {len(test_set)}")
    print(f"数据根目录: {DATA_ROOT / 'gtsrb'}")
    return train_set, test_set


def main():
    try:
        download_via_torchvision()
        print("\n✅ GTSRB 数据集准备就绪。")
    except Exception as e:
        print(f"\n❌ 自动下载失败: {e}", file=sys.stderr)
        print("\n--- 手动下载备选方案 ---")
        print("1) Kaggle（需注册）:")
        print("   https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print("2) 官方源:")
        print("   https://benchmark.ini.rub.de/gtsrb_dataset.html")
        print("3) Zenodo（可引用）:")
        print("   https://zenodo.org/records/13741936")
        print("\n下载后把 Train/、Test/、Meta/、train.csv、test.csv 解压到:")
        print(f"  {DATA_ROOT / 'gtsrb' / 'GTSRB'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
