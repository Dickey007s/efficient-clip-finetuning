# CLIP 在交通标志识别上的高效微调对比研究

**语言：** [English](README.md) | 简体中文

---

## 摘要

CLIP 等视觉-语言模型在自然图像上具有出色的零样本识别能力，但在领域特定、符号化的视觉数据上表现会急剧下降。在 GTSRB 交通标志数据集上，我们观察到其零样本 Top-1 准确率仅为约 27.5%，远低于常见的自然图像基准。

本项目在 8 GB 消费级显卡的预算下，系统对比五种参数高效微调方法：零样本、线性探测、CoOp、CLIP-Adapter 与 LoRA。核心问题是：在将 CLIP 适配到交通标志识别这一任务时，哪种方法在准确率、可训练参数量与标注数据效率三者之间提供了最优权衡。

---

## 目录

1. [方法概览](#方法概览)
2. [仓库结构](#仓库结构)
3. [环境配置](#环境配置)
4. [快速开始](#快速开始)
5. [数据集](#数据集)
6. [实验结果](#实验结果)
7. [引用](#引用)
8. [许可证](#许可证)
9. [致谢](#致谢)

---

## 方法概览

| 编号 | 方法 | 可训练参数量 | 参考文献 |
|------|------|--------------|----------|
| M0 | 零样本 CLIP | 0 | Radford 等，2021 |
| M1 | 线性探测 | 约 22 K | Radford 等，2021 |
| M2 | CoOp | 约 8 K | Zhou 等，IJCV 2022 |
| M3 | CLIP-Adapter | 约 131 K | Gao 等，IJCV 2023 |
| M4 | CLIP 上的 LoRA | 约 147 K（r=4） | Hu 等，2022 |
| **M5** | **CoOp-LoRA（ours）** | **约 156 K（r=4）** | **本工作** |

已完成的消融实验：

- **主归档实验**：M1-M5 的全数据结果、M1-M5 的 16-shot 结果、M4/M5 的 8-shot 结果，以及 M4/M5 的 4-shot 结果。
- **待补低样本实验**：M1/M2/M3 的 4-shot 与 8-shot。没有收敛到可支撑结论的结果先留空。
- **LoRA 秩消融**：M4 的 r=1、4、8、16（8-shot，20 epochs）。
- **LoRA 学习率消融**：M4 的 1e-5、3e-5、5e-5、1e-4（8-shot，r=8，20 epochs）。
- **M5 prompt 长度消融**：n_ctx 取 4、8、16（8-shot，20 epochs）。
- **阶段顺序消融**：CoOp→LoRA（M5c） vs. LoRA→CoOp（M5d）。
- **多种子验证**：8-shot 下 M4 与 M5 的 seed0~3。
- 失败案例分析，含每类准确率与混淆矩阵（见 `outputs/`）。
- 额外的补充、探索性实验和欠收敛废弃实验也已单独归档，其中包括若干 M5 额外变体，以及两组 CLC 结果；后者保留了产物，但当前仓库快照中没有对应的 `train_clc.py`。

### CoOp-LoRA（M5）

**核心思想**：文本端优化（通过 CoOp 学习 prompt）与视觉端适配（LoRA）互为补充，可在固定主干下顺序训练。关键洞察在于**两组参数完全不重叠**：CoOp 只修改输入嵌入层，LoRA 只修改视觉编码器的中间注意力层。这种正交性使得顺序训练成为叠加两种互补偏置的合理方式。

**阶段一：CoOp 预热**。冻结整个 CLIP 模型，仅优化连续的 prompt 向量。利用小样本数据快速学到适配目标领域词汇的文本初始化，使类别原型对齐到交通标志语义。

**阶段二：LoRA 微调**。冻结 CLIP 与已收敛的 CoOp prompt，向视觉编码器注入 LoRA 层（仅 Q/V）。在已优化的文本初始化基础上适配视觉特征提取，将图像嵌入拉向已优化的文本原型。

**为什么有效**：CoOp 优化*查询*端（如何向 CLIP 提问），LoRA 优化*表征*端（如何看图像）。由于阶段一已经提供了强文本锚点，阶段二的视觉适配比从原始 CLIP 初始化训练 LoRA 有更清晰的方向。理论预期是**1+1≥1 混合**，但清理后的归档结果表明，这种互补性主要体现在低样本场景，并不是在所有数据规模下都稳定成立。

**消融变体**：
- M5a：纯 CoOp（同 M2，验证阶段一单独效果）。
- M5b：纯 LoRA（同 M4，验证阶段二单独效果）。
- M5c：CoOp 后接 LoRA（提出的方法，验证顺序叠加）。
- M5d：LoRA 后接 CoOp（反向顺序，验证阶段顺序的重要性；见 `train_m5d.py`）。

**如何解读 M5 结果**：核心判据仍然是 `M5_best > max(M2_best, M4_best)`，但应该按具体设置分别判断，而不是当成统一规律。最终 epoch 仍明显上升、轮数不足的结果不进入主结论。当前仓库快照中，M5 在三个已收敛的低样本设置（4-、8-、16-shot）以及匹配设置的 8-shot 多种子验证中都优于对应的 M4 基线，而全数据下仍是 M4 领先。

---

## 仓库结构

```
clip_traffic_sign/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── train_m1.py ... train_m5d.py
│   ├── class_names.py
│   ├── data_utils.py
│   ├── feature_cache.py
│   ├── lora_utils.py
│   ├── download_data.py
│   ├── explore_data.py
│   ├── check_clip.py
│   └── curate_outputs.py  归档实验产物并重建汇总表
├── logs/                  训练日志
├── outputs/
│   ├── main/             主结果归档
│   ├── ablations/        秩、学习率、prompt 长度、阶段顺序等消融
│   ├── validation/       多种子验证
│   ├── supplemental/     已完成但未纳入主表的补充实验
│   └── exploratory/      保留的探索性 CLC 结果
├── results/
│   ├── *.csv             数据集统计
│   └── tables/           自动生成的实验汇总表
├── figures/
└── notebooks/
```

`data/` 目录不纳入版本控制，由 `download_data.py` 创建。

---

## 环境配置

已验证的运行环境：

- Python 3.10+
- PyTorch 2.x，搭配支持 CUDA 的 GPU
- 8 GB 显存即可跑完全数据实验

创建全新的 conda 环境：

```cmd
conda create -n clip-tsn python=3.10 -y
conda activate clip-tsn
pip install torch torchvision
pip install -r requirements.txt
```

---

## 快速开始

```cmd
:: 1. 下载 GTSRB，约 276 MB
python src/download_data.py

:: 2. 探查数据集
python src/explore_data.py

:: 3. 验证 CLIP 并运行零样本检查
python src/check_clip.py

:: 4. 实验跑完后，整理 outputs 并重建所有汇总表
python src/curate_outputs.py
```

---

## 数据集

GTSRB，德国交通标志识别基准。

| 属性 | 取值 |
|------|------|
| 类别数 | 43 个德国交通标志 |
| 训练集图像数 | 26,640 |
| 测试集图像数 | 12,630 |
| 图像尺寸 | 不固定，中位数约 44 乘 43 像素 |
| 类别均衡性 | 不均衡，每类 150 到 1,500 张 |

当 torchvision 下载较慢时的备选来源：

- 官方：<https://benchmark.ini.rub.de/gtsrb_dataset.html>
- Kaggle：<https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>
- Zenodo：<https://zenodo.org/records/13741936>

---

## 实验结果

`results/tables/` 下的汇总表是当前仓库快照的结果真值来源。下面统一使用**归档 best checkpoint** 的指标，而不是最后一个 epoch 的数值。明显欠收敛的短轮次结果不进入主表，只保留在 `results/tables/excluded_underconverged_runs.csv` 中用于追溯。

### 16-shot（688 张训练样本）

| 方法 | Best Top-1 | 可训练参数量 | 说明 |
|------|-----------:|-------------:|------|
| M1 线性探测 | 48.23% | 22,059 | 仅头部适配 |
| M2 CoOp | 67.37% | 8,192 | 文本端优化 |
| M3 CLIP-Adapter | 62.49% | 131,072 | 冻结 CLIP + 图像适配器 |
| M4 LoRA (r=4) | 79.46% | 147,456 | 当前归档中很强的 16-shot LoRA 基线 |
| **M5 CoOp→LoRA** | **81.67%** | **155,648** | CoOp 预热 + 20-epoch LoRA，已收敛 |

### 全数据（26,640 张训练样本）

| 方法 | Best Top-1 | 可训练参数量 | 说明 |
|------|-----------:|-------------:|------|
| M1 线性探测 | 80.10% | 22,059 | 20 epochs |
| M2 CoOp | 82.95% | 8,192 | 最佳 checkpoint 出现在最后一个 epoch 之前 |
| M3 CLIP-Adapter | 86.28% | 131,072 | 10 epochs |
| **M4 LoRA (r=4)** | **97.25%** | **147,456** | 当前归档中最强的全数据结果 |
| M5 CoOp→LoRA | 96.48% | 155,648 | 20 CoOp + 20 LoRA |

**按当前归档产物能支持的结论**：在三个已收敛的低样本设置（4-、8-、16-shot）上，M5 都小幅超过参数匹配的 M4 LoRA 基线，且样本越少优势越大（8-shot +1.62 pp，16-shot +2.21 pp，4-shot +3.78 pp）。这种互补性在全数据上并**不**成立——全数据下 M4（97.25%）仍优于 M5（96.48%）。结论是：CoOp 预热恰恰在数据稀缺、文本锚点更关键时帮助最大，而当视觉编码器拥有足够样本可独立适配后，这种增益随之消失。

### 小样本覆盖矩阵

空白项表示实验还没补齐，先留空等待后续直接续填。8-shot 已填项采用当前归档中每个方法最强的已收敛结果；参数匹配的 r=4 结果保留在后面的消融表中。

| 设置 | M1 | M2 | M3 | M4 | M5 |
|------|----|----|----|----:|----:|
| full    | 80.10% | 82.95% | 86.28% | **97.25%** | 96.48% |
| 16-shot | 48.23% | 67.37% | 62.49% | 79.46% | **81.67%** |
| 8-shot  | 44.55% | 62.34% | 57.65% | 73.85% | **74.56%** |
| 4-shot  | 38.19% | 55.95% | 52.79% | 65.38% | **69.16%** |

待补实验列表见 `results/tables/pending_experiments.csv`。

### 可用低样本主结果对比（M4 vs. M5）

| 设置 | M4 LoRA | M5 CoOp→LoRA | 增益 / 状态 |
|------|--------:|-------------:|-----------|
| 4-shot  | 65.38% | **69.16%** | **+3.78 pp** |
| 8-shot  | 72.83% | **74.45%** | **+1.62 pp** |
| 16-shot | 79.46% | **81.67%** | **+2.21 pp** |

三个低样本行现在都使用已收敛的最强归档结果，且 M5 全部胜出。4-shot 的 M4 采用已收敛的 `m4_4shot_r4_a8_60ep_seed42`（最初的 10-epoch 短轮次仅 30.70%，已废弃并保留在 `excluded_underconverged_runs.csv`）；16-shot 的 M5 采用 `m5_16shot_lora40_r4_a8`。M5 的优势在样本最少的 4-shot 上最大，到 8-shot 收窄为小幅领先，和后面的多种子表一致。

### M4 LoRA 秩消融（8-shot，20 epochs）

| 秩 r | 缩放 α | Best Top-1 | 可训练参数量 |
|-----:|------:|-----------:|-------------:|
| 1    | 2     | 44.31% | 36,864 |
| 4    | 8     | 65.99% | 147,456 |
| 8    | 16    | 71.56% | 294,912 |
| 16   | 32    | **73.85%** | **589,824** |

准确率随秩单调上升，但从 r=8 到 r=16 的边际收益明显小于从 r=4 到 r=8。

### M4 学习率消融（8-shot，r=8，α=16，20 epochs）

| 学习率 | Best Top-1 |
|------:|-----------:|
| 1e-5  | 30.34% |
| 3e-5  | 42.57% |
| 5e-5  | 57.82% |
| **1e-4** | **71.56%** |

### M5 prompt 长度消融（8-shot，20 CoOp + 20 LoRA）

| n_ctx | Best Top-1 |
|------:|-----------:|
| 4     | 71.16% |
| 8     | 70.87% |
| **16** | **73.22%** |

### 阶段顺序消融（8-shot，匹配的 seed-42 归档）

| 变体 | 第一阶段 | 第二阶段 | Best Top-1 |
|------|----------|----------|-----------:|
| M5 CoOp→LoRA | CoOp 预热 | LoRA 微调 | **71.88%** |
| M5d LoRA→CoOp | LoRA 预热 | CoOp 微调 | 67.45% |

正向顺序比反向顺序高 **4.43 pp**。

### 多种子验证（8-shot，匹配的 20-epoch 研究）

| 方法 | Seed 0 | Seed 2 | Seed 3 | 均值 ± 标准差 |
|------|--------|--------|--------|--------------|
| M4 LoRA r=16 α=32 | 73.85% | 72.83% | 72.87% | 73.18 ± 0.58% |
| **M5 CoOp→LoRA r=8 α=16** | **74.56%** | 74.45% | 74.35% | **74.45 ± 0.11%** |

这是当前归档中最干净、也最能支持“M5 相对强 LoRA 基线存在小幅优势”这一说法的证据。

### 补充与探索性实验

- `results/tables/supplemental_runs.csv` 记录了额外完成、但未纳入主表的变体实验。
- `results/tables/excluded_underconverged_runs.csv` 记录了保留用于追溯、但不进入主结论的欠收敛短轮次实验。
- 两组探索性的 CLC 结果保存在 `outputs/exploratory/clc/`，并已登记到 `supplemental_runs.csv`。
- 这些 CLC 结果之所以保留，是因为产物完整；但由于当前仓库快照缺少 `src/train_clc.py`，因此只作为探索性归档，不作为完全可复现实验主结论。

### 数据探查产物

由 `explore_data.py` 生成：

- `figures/class_distribution.png`，训练集与测试集的类别分布。
- `figures/samples_per_class.png`，每个类别一张样本。
- `results/train_class_distribution.csv`，每个类别的图像数量。
- `results/train_image_size_stats.csv`，图像尺寸统计。

---

## 引用

若你在本工作的基础上开展研究，请引用以下核心方法。

```bibtex
@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle = {ICML},
  year      = {2021}
}

@article{zhou2022coop,
  title   = {Learning to Prompt for Vision-Language Models},
  author  = {Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  journal = {International Journal of Computer Vision},
  year    = {2022}
}

@article{gao2023clipadapter,
  title   = {CLIP-Adapter: Better Vision-Language Models with Feature Adapters},
  author  = {Gao, Peng and Geng, Shijie and others},
  journal = {International Journal of Computer Vision},
  year    = {2023}
}
```

---

## 许可证

代码基于 MIT 许可证发布，详见 [LICENSE](LICENSE)。

GTSRB 数据集由其作者按自有条款发布，相关授权请参见官方数据集页面。

---

## 致谢

本项目基于以下开源工作构建：

- [open_clip](https://github.com/mlfoundations/open_clip)，CLIP 模型库。
- [PEFT](https://github.com/huggingface/peft)，HuggingFace 参数高效微调库。
- [torchvision](https://github.com/pytorch/vision)，GTSRB 数据加载器。

课程：深度学习与计算机视觉。
