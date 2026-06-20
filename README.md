# Efficient Fine-Tuning of CLIP for Traffic Sign Recognition

**Languages:** English | [简体中文](README.zh-CN.md)

---

## Abstract

Vision-language models such as CLIP achieve strong zero-shot accuracy on natural
images, yet their performance drops sharply on domain-specific, symbolic visual
data. On the GTSRB traffic-sign benchmark we observe a zero-shot Top-1 accuracy of
only about 27.5 percent, far below typical natural-image benchmarks.

This project presents a systematic comparison of five parameter-efficient
fine-tuning methods — Zero-shot, Linear Probe, CoOp, CLIP-Adapter, and LoRA —
under a consumer-GPU budget of 8 GB. The central question is which method offers
the best trade-off between accuracy, trainable parameters, and labeled-data
efficiency for adapting CLIP to traffic-sign recognition.

---

## Contents

1. [Method Overview](#method-overview)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Quick Start](#quick-start)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Citation](#citation)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

---

## Method Overview

| ID | Method | Trainable Parameters | Reference |
|----|--------|----------------------|-----------|
| M0 | Zero-shot CLIP | 0 | Radford et al., 2021 |
| M1 | Linear Probe | about 33 K | Radford et al., 2021 |
| M2 | CoOp | about 8 K | Zhou et al., IJCV 2022 |
| M3 | CLIP-Adapter | about 0.5 M | Gao et al., IJCV 2023 |
| M4 | LoRA on CLIP | about 0.15 M | Hu et al., 2022 |
| **M5** | **CoOp-LoRA (ours)** | **about 0.16 M** | **This work** |

Planned ablation studies:

- Few-shot learning curves at 1, 2, 4, 8, and 16 shots per class.
- LoRA rank sweep at r equal to 1, 4, 8, 16, and 32.
- Failure-case analysis with a normalized confusion matrix.
- **CoOp-LoRA sequential ablation**: CoOp only, LoRA only, CoOp then LoRA, and LoRA then CoOp, to verify the contribution and ordering of each stage.

### CoOp-LoRA (M5)

**Core idea**: Text-side optimization (learned prompts via CoOp) and vision-side adaptation (LoRA) are complementary and can be trained sequentially under a fixed backbone. The key insight is that **the two parameter sets do not overlap**: CoOp only touches the input embedding layer, while LoRA only touches the intermediate attention layers of the vision encoder. This orthogonality makes sequential training a principled way to stack two complementary biases.

**Stage one -- CoOp warm-up**: Freeze the entire CLIP model and optimize only the continuous prompt vectors. This quickly learns a good text initialization from few-shot data, adapting the class prototypes to the target domain vocabulary.

**Stage two -- LoRA fine-tuning**: Freeze CLIP and the converged CoOp prompts, then inject LoRA layers into the vision encoder (Q/V only). This adapts visual feature extraction on top of the improved text initialization, pulling image embeddings closer to the already-optimized text prototypes.

**Why this works**: CoOp optimizes the *query* side (how to ask CLIP), while LoRA optimizes the *representation* side (how to see the image). Because stage one already provides a strong text anchor, stage two's visual adaptation has a clearer direction than training LoRA from the original CLIP initialization. The result is a **true 1+1>1 hybrid**: the gain of M5 should exceed the individual gains of M2 (CoOp only) and M4 (LoRA only).

**Ablation variants**:
- M5a: CoOp only (same as M2, verifies stage one alone).
- M5b: LoRA only (same as M4, verifies stage two alone).
- M5c: CoOp then LoRA (the proposed method, verifies sequential stacking).
- M5d: LoRA then CoOp (reversed order, verifies stage ordering matters).

**How to read the M5 result**: The critical metric is not absolute accuracy, but the **gain over Stage 1** (`M5_best - M2_best`). If this delta is positive and larger than `M4_best - M1_best`, the hybrid is genuinely additive. If the delta is near zero, LoRA is not helping on top of CoOp. If negative, LoRA is overfitting and destroying the CoOp initialization.

---

## Repository Structure

```
efficient-clip-finetuning/
├── README.md              Project documentation (English)
├── README.zh-CN.md        Project documentation (Simplified Chinese)
├── LICENSE
├── requirements.txt
├── src/
│   ├── class_names.py     Readable names for the 43 classes
│   ├── download_data.py   GTSRB download via torchvision
│   ├── explore_data.py    Dataset statistics and visualization
│   └── check_clip.py      CLIP load verification and zero-shot check
├── figures/               Generated plots for the report
├── results/               CSV statistics and evaluation outputs
└── notebooks/             Exploratory notebooks
```

The `data/` directory is excluded from version control and is created by
`download_data.py`.

---

## Environment Setup

Verified configuration:

- Windows 11
- Python 3.13
- CUDA 13.0
- NVIDIA RTX 4060 Laptop GPU, 8 GB
- conda environment named `QuantAI`

Option A. Reuse the existing conda environment:

```cmd
C:\Users\<you>\miniconda3\Scripts\activate QuantAI
```

Option B. Create a fresh environment:

```cmd
conda create -n clip-tsn python=3.13 -y
conda activate clip-tsn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

On Windows, if you encounter `OMP: Error #15`, set the following before running
any script:

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Quick Start

```cmd
:: 1. Download GTSRB, about 276 MB
python src/download_data.py

:: 2. Explore the dataset
python src/explore_data.py

:: 3. Verify CLIP and run the zero-shot sanity check
python src/check_clip.py
```

---

## Dataset

GTSRB, the German Traffic Sign Recognition Benchmark.

| Property | Value |
|----------|-------|
| Number of classes | 43 German traffic signs |
| Training images | 26,640 |
| Test images | 12,630 |
| Image size | Variable, median about 44 by 43 pixels |
| Class balance | Imbalanced, 150 to 1,500 images per class |

Alternative download sources when torchvision is slow:

- Official: <https://benchmark.ini.rub.de/gtsrb_dataset.html>
- Kaggle: <https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>
- Zenodo: <https://zenodo.org/records/13741936>

---

## Results

Experiments are in progress. The table below will be populated as results become
available.

| Method | Test Top-1 | Trainable Parameters | Training Time | Notes |
|--------|------------|----------------------|---------------|-------|
| M0 Zero-shot CLIP | 27.5% sanity check | 0 | none | Baseline lower bound |
| M1 Linear Probe | pending | ~33 K | pending | Head-only adaptation |
| M2 CoOp | pending | ~8 K | pending | Text-side optimization |
| M3 CLIP-Adapter | pending | ~0.5 M | pending | Residual feature adapters |
| M4 LoRA, r = 8 | pending | ~0.15 M | pending | Vision-side low-rank adaptation |
| **M5 CoOp→LoRA** | **pending** | **~0.16 M** | **pending** | **Sequential hybrid (ours)** |

### How to interpret M5

The M5 result is meaningful only when compared against M2 and M4:

```text
M5_gain = M5_best - max(M2_best, M4_best)
```

- If `M5_gain > 0`: CoOp and LoRA are complementary, sequential stacking works.
- If `M5_gain ≈ 0`: The two methods capture redundant information; no benefit from hybrid.
- If `M5_gain < 0`: The second stage overwrites or corrupts the first stage's gains.

The per-epoch log of M5 explicitly prints `Delta vs Stage1` so you can watch the
LoRA stage pull accuracy above the frozen CoOp prompt in real time.

Data exploration outputs generated by `explore_data.py`:

- `figures/class_distribution.png`, class distribution of train and test sets.
- `figures/samples_per_class.png`, one sample per class.
- `results/train_class_distribution.csv`, per-class image counts.
- `results/train_image_size_stats.csv`, image-size statistics.

---

## Citation

If you build on this work, please cite the core methods.

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

## License

The code is released under the MIT License. See [LICENSE](LICENSE).

The GTSRB dataset is distributed under its own license by its creators; see the
official dataset page for terms.

---

## Acknowledgements

This project builds on the following open-source projects:

- [open_clip](https://github.com/mlfoundations/open_clip), the CLIP model zoo.
- [PEFT](https://github.com/huggingface/peft), HuggingFace parameter-efficient fine-tuning.
- [torchvision](https://github.com/pytorch/vision), the GTSRB data loader.

Course: Deep Learning and Computer Vision.
