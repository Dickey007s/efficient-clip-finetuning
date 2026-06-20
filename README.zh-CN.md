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

计划开展的消融实验：

- 小样本学习曲线：每类 1、2、4、8、16 张样本。
- LoRA 秩消融：r 取 1、4、8、16、32。
- 失败案例分析，并给出归一化混淆矩阵。

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

- Windows 11
- Python 3.13
- CUDA 13.0
- NVIDIA RTX 4060 笔记本显卡，8 GB
- conda 环境，名称为 `QuantAI`

方式一，复用现有的 conda 环境：

```cmd
C:\Users\<你的用户名>\miniconda3\Scripts\activate QuantAI
```

方式二，创建全新的环境：

```cmd
conda create -n clip-tsn python=3.13 -y
conda activate clip-tsn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

在 Windows 上，若遇到 `OMP: Error #15`，请在运行脚本前设置：

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
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

实验仍在进行中，下表将随结果陆续更新。

| 方法 | 测试集 Top-1 | 可训练参数量 | 训练耗时 |
|------|--------------|--------------|----------|
| 零样本 CLIP | 27.5%，抽样检查 | 0 | 无 |
| 线性探测 | 待补充 | 约 33 K | 待补充 |
| CoOp | 待补充 | 约 8 K | 待补充 |
| CLIP-Adapter | 待补充 | 约 0.5 M | 待补充 |
| LoRA，r = 8 | 待补充 | 约 0.15 M | 待补充 |

由 `explore_data.py` 生成的数据探查产物：

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
