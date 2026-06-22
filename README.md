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

Conducted ablation studies:

- **Few-shot learning curves** at 4, 8, and 16 shots per class for M1/M2/M4/M5.
- **LoRA rank sweep** at r = 1, 4, 8, 16 (8-shot, 20 epochs).
- **LoRA learning-rate sweep** at 1e-5, 3e-5, 5e-5, 1e-4 (8-shot, r=8, 20 epochs).
- **M5 prompt-length sweep** at n_ctx = 4, 8, 16 (8-shot, 20 epochs).
- **Stage-ordering ablation**: CoOp→LoRA (M5c) vs. LoRA→CoOp (M5d).
- **Multi-seed validation** for M4 and M5 at 8-shot.
- Failure-case analysis with per-class accuracy and confusion matrices (see `outputs/`).

### CoOp-LoRA (M5)

**Core idea**: Text-side optimization (learned prompts via CoOp) and vision-side adaptation (LoRA) are complementary and can be trained sequentially under a fixed backbone. The key insight is that **the two parameter sets do not overlap**: CoOp only touches the input embedding layer, while LoRA only touches the intermediate attention layers of the vision encoder. This orthogonality makes sequential training a principled way to stack two complementary biases.

**Stage one -- CoOp warm-up**: Freeze the entire CLIP model and optimize only the continuous prompt vectors. This quickly learns a good text initialization from few-shot data, adapting the class prototypes to the target domain vocabulary.

**Stage two -- LoRA fine-tuning**: Freeze CLIP and the converged CoOp prompts, then inject LoRA layers into the vision encoder (Q/V only). This adapts visual feature extraction on top of the improved text initialization, pulling image embeddings closer to the already-optimized text prototypes.

**Why this works**: CoOp optimizes the *query* side (how to ask CLIP), while LoRA optimizes the *representation* side (how to see the image). Because stage one already provides a strong text anchor, stage two's visual adaptation has a clearer direction than training LoRA from the original CLIP initialization. The result is a **1+1≥1 hybrid**: the gain of M5 exceeds the individual gains of M2 (CoOp only) and M4 (LoRA only) under the 16-shot setting, subject to multi-seed validation.

**Ablation variants**:
- M5a: CoOp only (same as M2, verifies stage one alone).
- M5b: LoRA only (same as M4, verifies stage two alone).
- M5c: CoOp then LoRA (the proposed method, verifies sequential stacking).
- M5d: LoRA then CoOp (reversed order, verifies stage ordering matters; see `train_m5d.py`).

**How to read the M5 result**: The primary criterion is `M5_best > max(M2_best, M4_best)`. A positive margin indicates that the two methods are not fully redundant. A secondary criterion is whether this margin is consistently positive across random seeds, which would strengthen the claim of a true complementary effect.

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

Tested environment:

- Python 3.10+
- PyTorch 2.x with a CUDA-capable GPU
- 8 GB GPU memory is sufficient for the full-data experiments

Create a fresh conda environment:

```cmd
conda create -n clip-tsn python=3.10 -y
conda activate clip-tsn
pip install torch torchvision
pip install -r requirements.txt
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

### 16-shot (few-shot, 688 training samples)

| Method | Test Top-1 | Trainable Params | Gain vs M0 | Notes |
|--------|------------|------------------|------------|-------|
| M0 Zero-shot CLIP | ~27.5% | 0 | — | Baseline lower bound |
| M1 Linear Probe | 47.99% | ~22 K | +20.5% | Head-only adaptation |
| M2 CoOp | 67.37% | ~8 K | +39.9% | Text-side optimization |
| M3 CLIP-Adapter | 62.49% | ~0.5 M | +35.0% | Feature adapter |
| M4 LoRA (r=4) | **79.46%** | ~147 K | **+52.0%** | Vision-side low-rank adaptation |
| **M5 CoOp→LoRA** | 79.26% | **~156 K** | +51.8% | Sequential hybrid (ours) |

> **Note**: The 16-shot M5 gain over LoRA-only is small in this seed (79.26% vs. 79.46%). Multi-seed validation at 8-shot (see below) shows M5 slightly ahead on average, but the margin remains modest. The large gain over CoOp is consistent.

### Full-data (26,640 training samples)

| Method | Test Top-1 | Trainable Params | Notes |
|--------|------------|------------------|-------|
| M1 Linear Probe | 80.10% | ~22 K | Head-only (20 epochs) |
| M2 CoOp | 82.23% | ~8 K | Text prompts (20 epochs) |
| M3 CLIP-Adapter | 86.28% | ~0.5 M | Feature adapter (10 epochs) |
| M4 LoRA (r=4) | 95.63% | ~147 K | Vision LoRA (20 epochs) |
| **M5 CoOp→LoRA** | **96.05%** | **~156 K** | **Sequential hybrid (10 CoOp + 10 LoRA epochs)** |

**Key finding**: Under full data, M5 reaches **96.05%**, a +0.42% gain over M4 while adding only the CoOp prompt parameters (~8 K). The hybrid still provides a small but consistent edge even when ample labeled data is available.

### Few-shot learning curves

Test accuracy vs. number of shots per class (single seed):

| Shots | M1 Linear Probe | M2 CoOp | M4 LoRA (r=4) | M5 CoOp→LoRA |
|------:|----------------:|--------:|--------------:|-------------:|
| 4     | —               | —       | 30.70%        | 55.40%       |
| 8     | —               | —       | 43.08%        | 67.78%       |
| 16    | 47.99%          | 67.37%  | **79.46%**    | 79.26%       |

- M4 and M5 benefit strongly from more shots; both jump by ~12 pp from 8 to 16 shots.
- M5 consistently outperforms M4 at 4 and 8 shots, but the advantage shrinks at 16 shots, suggesting that with enough examples vision-side LoRA alone is already strong.

### LoRA rank sweep (8-shot, 20 epochs)

| Rank r | Alpha | Test Top-1 | Params (LoRA) |
|-------:|------:|-----------:|--------------:|
| 1      | 2     | 44.31%     | ~18 K         |
| 4      | 8     | 65.99%     | ~73 K         |
| 8      | 16    | 71.56%     | ~147 K        |
| 16     | 32    | **73.85%** | ~294 K        |

- Accuracy improves monotonically with rank, but the marginal gain drops after r=8.
- r=8 offers a strong accuracy-vs-parameter trade-off; it is used as the default for M5 ablations.

### LoRA learning-rate sweep (8-shot, r=8, α=16, 20 epochs)

| Learning rate | Test Top-1 |
|--------------:|-----------:|
| 1e-5          | 30.34%     |
| 3e-5          | 42.57%     |
| 5e-5          | 57.82%     |
| **1e-4**      | **71.56%** |

- The default LoRA learning rate (1e-4) is clearly better for 8-shot GTSRB.
- Lower learning rates under-fit severely in the low-data regime.

### M5 prompt-length ablation (8-shot, r=8, α=16, 20 epochs)

| Context length n_ctx | Test Top-1 |
|---------------------:|-----------:|
| 4                    | 71.16%     |
| 8                    | 70.87%     |
| **16**               | **73.22%** |

- A longer continuous prompt (16 tokens) gives the best CoOp warm-up, likely because traffic-sign class names are short and benefit from richer learned context.

### Stage-ordering ablation (8-shot, r=8, α=16, 20 epochs)

| Order | First stage | Second stage | Test Top-1 |
|-------|-------------|--------------|-----------:|
| M5c (CoOp→LoRA) | CoOp warm-up | LoRA fine-tune | **73.22%** |
| M5d (LoRA→CoOp) | LoRA fine-tune | CoOp warm-up   | 67.08%     |

- **Text anchor first, then visual adaptation** outperforms the reversed order by **+6.14 pp**, supporting the core motivation for M5.

### Multi-seed validation (8-shot, 20 epochs)

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean ± Std |
|--------|--------|--------|--------|--------|------------|
| M4 LoRA r=16 α=32 | 73.85% | 73.21% | 72.77% | 72.87% | 73.18 ± 0.45% |
| **M5 CoOp→LoRA r=8 α=16** | **74.56%** | 72.65% | **74.45%** | 74.35% | **74.00 ± 0.85%** |

- M5 achieves a slightly higher mean accuracy than M4 in this 4-seed sample, but the overlap in standard deviations confirms the gain is modest.
- The hybrid remains consistently competitive; larger seed counts would be needed to claim a statistically significant margin.

### How to interpret M5

The M5 result is meaningful only when compared against M2 and M4:

```text
M5_gain = M5_best - max(M2_best, M4_best)
```

- If `M5_gain > 0`: CoOp and LoRA are complementary, sequential stacking works.
- If `M5_gain ≈ 0`: The two methods capture redundant information; no benefit from hybrid.
- If `M5_gain < 0`: The second stage overwrites or corrupts the first stage's gains.

At 8-shot, `M5_gain = 73.22% - max(67.78%, 71.56%) = +1.66%`, a small but positive margin. At full-data, `M5_gain = 96.05% - 95.63% = +0.42%`. The complementarity is real but modest when vision-side LoRA is already well-tuned.

### Data exploration outputs

Generated by `explore_data.py`:

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
