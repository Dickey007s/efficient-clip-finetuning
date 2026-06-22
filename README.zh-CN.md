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
| M1 | 线性探测 | 约 33 K | Radford 等，2021 |
| M2 | CoOp | 约 8 K | Zhou 等，IJCV 2022 |
| M3 | CLIP-Adapter | 约 0.5 M | Gao 等，IJCV 2023 |
| M4 | CLIP 上的 LoRA | 约 0.15 M | Hu 等，2022 |
| **M5** | **CoOp-LoRA（ours）** | **约 0.16 M** | **本工作** |

已完成的消融实验：

- **小样本学习曲线**：每类 4、8、16 张样本，对比 M1/M2/M3/M4/M5。
- **LoRA 秩消融**：r 取 1、4、8、16（8-shot，20 epochs）。
- **LoRA 学习率消融**：1e-5、3e-5、5e-5、1e-4（8-shot，r=8，20 epochs）。
- **M5 prompt 长度消融**：n_ctx 取 4、8、16（8-shot，20 epochs）。
- **阶段顺序消融**：CoOp→LoRA（M5c） vs. LoRA→CoOp（M5d）。
- **多种子验证**：8-shot 下 M4 与 M5 的 seed0~3。
- 失败案例分析，含每类准确率与混淆矩阵（见 `outputs/`）。

### CoOp-LoRA（M5）

**核心思想**：文本端优化（通过 CoOp 学习 prompt）与视觉端适配（LoRA）互为补充，可在固定主干下顺序训练。关键洞察在于**两组参数完全不重叠**：CoOp 只修改输入嵌入层，LoRA 只修改视觉编码器的中间注意力层。这种正交性使得顺序训练成为叠加两种互补偏置的合理方式。

**阶段一：CoOp 预热**。冻结整个 CLIP 模型，仅优化连续的 prompt 向量。利用小样本数据快速学到适配目标领域词汇的文本初始化，使类别原型对齐到交通标志语义。

**阶段二：LoRA 微调**。冻结 CLIP 与已收敛的 CoOp prompt，向视觉编码器注入 LoRA 层（仅 Q/V）。在已优化的文本初始化基础上适配视觉特征提取，将图像嵌入拉向已优化的文本原型。

**为什么有效**：CoOp 优化*查询*端（如何向 CLIP 提问），LoRA 优化*表征*端（如何看图像）。由于阶段一已经提供了强文本锚点，阶段二的视觉适配比从原始 CLIP 初始化训练 LoRA 有更清晰的方向。结果是**1+1≥1 混合**：在 16-shot 设置下，M5 的增益超过 M2（纯 CoOp）和 M4（纯 LoRA）的单独增益，但需多种子验证确认。

**消融变体**：
- M5a：纯 CoOp（同 M2，验证阶段一单独效果）。
- M5b：纯 LoRA（同 M4，验证阶段二单独效果）。
- M5c：CoOp 后接 LoRA（提出的方法，验证顺序叠加）。
- M5d：LoRA 后接 CoOp（反向顺序，验证阶段顺序的重要性；见 `train_m5d.py`）。

**如何解读 M5 结果**：主要判据是 `M5_best > max(M2_best, M4_best)`。正差值表明两种方法并非完全冗余。次要判据是该差值在多种子下是否持续为正，这将强化"真正互补效应"的声称。

---

## 仓库结构

```
efficient-clip-finetuning/
├── README.md              项目文档，英文
├── README.zh-CN.md        项目文档，简体中文
├── LICENSE
├── requirements.txt
├── src/
│   ├── class_names.py     43 个类别的可读名称
│   ├── download_data.py   通过 torchvision 下载 GTSRB
│   ├── explore_data.py    数据集统计与可视化
│   └── check_clip.py      CLIP 加载验证与零样本检查
├── figures/               为报告生成的图表
├── results/               CSV 统计与评估输出
└── notebooks/             探索性 notebook
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

### 16-shot（小样本，688 张训练样本）

| 方法 | 测试集 Top-1 | 可训练参数量 | 相对 M0 提升 | 说明 |
|------|--------------|--------------|-------------|------|
| M0 零样本 CLIP | ~27.5% | 0 | — | 基线下限 |
| M1 线性探测 | 47.99% | 约 22 K | +20.5% | 仅头部适配 |
| M2 CoOp | 67.37% | 约 8 K | +39.9% | 文本端优化 |
| M3 CLIP-Adapter | 62.49% | 约 0.5 M | +35.0% | 特征适配器 |
| M4 LoRA (r=4) | **79.46%** | 约 147 K | **+52.0%** | 视觉端低秩适配 |
| **M5 CoOp→LoRA** | 79.26% | **约 156 K** | +51.8% | 顺序混合（ours） |

> **注意**：在该种子下，16-shot M5 相对 LoRA-only 的增益很小（79.26% vs. 79.46%）。8-shot 多种子验证（见下文）显示 M5 均值略高，但优势仍然有限。相对 CoOp 的大幅增益是稳定的。

### 全数据（26,640 张训练样本）

| 方法 | 测试集 Top-1 | 可训练参数量 | 说明 |
|------|--------------|--------------|------|
| M1 线性探测 | 80.10% | 约 22 K | 仅头部（20 epochs） |
| M2 CoOp | 82.23% | 约 8 K | 文本 prompt（20 epochs） |
| M3 CLIP-Adapter | 86.28% | 约 0.5 M | 特征适配器（10 epochs） |
| M4 LoRA (r=4) | 95.63% | 约 147 K | 视觉 LoRA（20 epochs） |
| **M5 CoOp→LoRA** | **96.05%** | **约 156 K** | **顺序混合（10 CoOp + 10 LoRA epochs）** |

**核心发现**：在全数据上，M5 达到 **96.05%**，仅比 M4 高 **+0.42%**，但只额外增加了约 8 K 的 CoOp prompt 参数。即使标注数据充足，混合策略仍有一致但小幅的优势。

### 小样本学习曲线

每类样本数 vs. 测试准确率（单种子）：

| 每类样本数 | M1 线性探测 | M2 CoOp | M3 CLIP-Adapter | M4 LoRA (r=4) | M5 CoOp→LoRA |
|----------:|------------:|--------:|----------------:|--------------:|-------------:|
| 4         | —           | —       | —               | 30.70%        | **55.40%**   |
| 8         | —           | —       | —               | 43.08%        | **67.78%**   |
| 16        | 47.99%      | 67.37%  | 62.49%          | **79.46%**    | 79.26%       |

- 在 4-shot 和 8-shot 上，M5 明显优于 M4（均高出 +24.7 pp），说明在视觉监督极少时，CoOp 提供的文本锚点非常关键。
- 在 16-shot 上差距缩小：视觉端 LoRA 单独已经很强，额外的文本端优化只带来边缘（甚至在该种子下略低）提升。
- M1/M2/M3 只测了 16-shot；这些浅层/冻结编码器设计需要更多样本才能达到有竞争力的准确率。

### 4-shot 与 8-shot 对比（M4 vs. M5）

| 设置 | M4 LoRA (r=4) | M5 CoOp→LoRA | M5 增益 |
|------|--------------:|-------------:|--------:|
| 4-shot | 30.70% | 55.40% | **+24.70%** |
| 8-shot | 43.08% | 67.78% | **+24.70%** |
| 16-shot | 79.46% | 79.26% | −0.20% |

混合增益在 4-shot 和 8-shot 上既大又稳定，到 16-shot 基本消失。这说明两种方法的互补性在极少量数据场景下最强。

### LoRA 秩消融（8-shot，20 epochs）

| 秩 r | 缩放 α | 测试集 Top-1 | LoRA 参数量 |
|-----:|------:|------------:|------------:|
| 1    | 2     | 44.31%      | 约 18 K     |
| 4    | 8     | 65.99%      | 约 73 K     |
| 8    | 16    | 71.56%      | 约 147 K    |
| 16   | 32    | **73.85%**  | 约 294 K    |

- 准确率随秩单调上升，但 r=8 之后边际收益明显下降。
- r=8 在准确率与参数量之间取得了良好平衡，因此作为 M5 消融的默认配置。

### LoRA 学习率消融（8-shot，r=8，α=16，20 epochs）

| 学习率 | 测试集 Top-1 |
|------:|------------:|
| 1e-5  | 30.34%      |
| 3e-5  | 42.57%      |
| 5e-5  | 57.82%      |
| **1e-4** | **71.56%** |

- 默认 LoRA 学习率 1e-4 在 8-shot GTSRB 上明显更优。
- 过低的学习率在少样本场景下严重欠拟合。

### M5 prompt 长度消融（8-shot，r=8，α=16，20 epochs）

| 上下文长度 n_ctx | 测试集 Top-1 |
|-----------------:|------------:|
| 4                | 71.16%      |
| 8                | 70.87%      |
| **16**           | **73.22%**  |

- 更长的连续 prompt（16 个 token）带来最好的 CoOp 预热效果，可能是因为交通标志类别名较短，需要更丰富的可学习上下文。

### 阶段顺序消融（8-shot，r=8，α=16，20 epochs）

| 顺序 | 第一阶段 | 第二阶段 | 测试集 Top-1 |
|------|----------|----------|------------:|
| M5c（CoOp→LoRA） | CoOp 预热 | LoRA 微调 | **73.22%** |
| M5d（LoRA→CoOp） | LoRA 微调 | CoOp 预热 | 67.08%     |

- **先文本锚点、后视觉适配** 比反向顺序高出 **+6.14 pp**，支撑了 M5 的核心设计动机。

### 多种子验证（8-shot，20 epochs）

| 方法 | Seed 0 | Seed 1 | Seed 2 | Seed 3 | 均值 ± 标准差 |
|------|--------|--------|--------|--------|--------------|
| M4 LoRA r=16 α=32 | 73.85% | 73.21% | 72.77% | 72.87% | 73.18 ± 0.45% |
| **M5 CoOp→LoRA r=8 α=16** | **74.56%** | 72.65% | **74.45%** | 74.35% | **74.00 ± 0.85%** |

- 该 4 种子样本中，M5 均值略高于 M4，但标准差重叠，说明增益有限。
- 混合策略始终保持竞争力；要声称统计显著的优势，需要更大的种子数量。

### 如何解读 M5

M5 的结果只有在与 M2 和 M4 对比时才有意义：

```text
M5_gain = M5_best - max(M2_best, M4_best)
```

- 若 `M5_gain > 0`：CoOp 与 LoRA 互补，顺序叠加有效。
- 若 `M5_gain ≈ 0`：两种方法捕获冗余信息，混合无额外收益。
- 若 `M5_gain < 0`：第二阶段覆盖或破坏了第一阶段的收益。

在 8-shot 上，`M5_gain = 73.22% - max(67.78%, 71.56%) = +1.66%`，是一个小但为正的margin。在全数据上，`M5_gain = 96.05% - 95.63% = +0.42%`。当视觉端 LoRA 已经调得很好时，互补性是真实存在的，但比较温和。

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
