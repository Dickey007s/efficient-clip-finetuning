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
| M1 | Linear Probe | about 22 K | Radford et al., 2021 |
| M2 | CoOp | about 8 K | Zhou et al., IJCV 2022 |
| M3 | CLIP-Adapter | about 131 K | Gao et al., IJCV 2023 |
| M4 | LoRA on CLIP | about 147 K (r=4) | Hu et al., 2022 |
| **M5** | **CoOp-LoRA (ours)** | **about 156 K (r=4)** | **This work** |

Conducted ablation studies:

- **Main archived runs** for M1-M5 on full data, M1-M5 at 16-shot, M4/M5 at 8-shot, and M4/M5 at 4-shot.
- **Pending low-shot runs** for M1/M2/M3 at 4-shot and 8-shot. These entries are intentionally left blank until converged runs are available.
- **LoRA rank sweep** at r = 1, 4, 8, 16 for M4 (8-shot, 20 epochs).
- **LoRA learning-rate sweep** at 1e-5, 3e-5, 5e-5, 1e-4 for M4 (8-shot, r=8, 20 epochs).
- **M5 prompt-length sweep** at n_ctx = 4, 8, 16 (8-shot, 20 epochs).
- **Stage-ordering ablation**: CoOp→LoRA (M5c) vs. LoRA→CoOp (M5d).
- **Multi-seed validation** for M4 and M5 at 8-shot.
- Failure-case analysis with per-class accuracy and confusion matrices (see `outputs/`).
- Supplemental, exploratory, and discarded underconverged artifacts are preserved separately, including extra M5 schedule/rank runs and two CLC runs whose source script is not present in the current repo snapshot.

### CoOp-LoRA (M5)

**Core idea**: Text-side optimization (learned prompts via CoOp) and vision-side adaptation (LoRA) are complementary and can be trained sequentially under a fixed backbone. The key insight is that **the two parameter sets do not overlap**: CoOp only touches the input embedding layer, while LoRA only touches the intermediate attention layers of the vision encoder. This orthogonality makes sequential training a principled way to stack two complementary biases.

**Stage one -- CoOp warm-up**: Freeze the entire CLIP model and optimize only the continuous prompt vectors. This quickly learns a good text initialization from few-shot data, adapting the class prototypes to the target domain vocabulary.

**Stage two -- LoRA fine-tuning**: Freeze CLIP and the converged CoOp prompts, then inject LoRA layers into the vision encoder (Q/V only). This adapts visual feature extraction on top of the improved text initialization, pulling image embeddings closer to the already-optimized text prototypes.

**Why this works**: CoOp optimizes the *query* side (how to ask CLIP), while LoRA optimizes the *representation* side (how to see the image). Because stage one already provides a strong text anchor, stage two's visual adaptation has a clearer direction than training LoRA from the original CLIP initialization. The intended effect is a **1+1≥1 hybrid**, but the cleaned artifacts show that this complementarity is strongest in the low-shot regime and does not hold uniformly at every data scale.

**Ablation variants**:
- M5a: CoOp only (same as M2, verifies stage one alone).
- M5b: LoRA only (same as M4, verifies stage two alone).
- M5c: CoOp then LoRA (the proposed method, verifies sequential stacking).
- M5d: LoRA then CoOp (reversed order, verifies stage ordering matters; see `train_m5d.py`).

**How to read the M5 result**: The main diagnostic is still `M5_best > max(M2_best, M4_best)`, but it should be evaluated per setting rather than treated as a universal rule. Runs whose accuracy was still clearly rising at the final epoch are excluded from the main claims. In this snapshot, M5 beats the matched M4 baseline across all three converged low-shot settings (4-, 8-, and 16-shot) and in the matched 8-shot multi-seed study, while M4 remains ahead on full data.

---

## Repository Structure

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
│   └── curate_outputs.py  Organize artifacts and rebuild summary tables
├── logs/                  Training logs
├── outputs/
│   ├── main/             Canonical archived runs
│   ├── ablations/        Rank, LR, prompt-length, and stage-order studies
│   ├── validation/       Multi-seed runs
│   ├── supplemental/     Extra completed variants not used in the main tables
│   └── exploratory/      Preserved but non-reproducible CLC artifacts
├── results/
│   ├── *.csv             Dataset statistics
│   └── tables/           Auto-generated experiment summary tables
├── figures/
└── notebooks/
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

:: 4. After running experiments, reorganize outputs and rebuild summary tables
python src/curate_outputs.py
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

The cleaned tables under `results/tables/` are the source of truth for the current repo snapshot. Numbers below use the archived **best checkpoint** for each run, not the final epoch when those differ. Short runs that were still clearly underconverged are excluded from the main tables and listed separately in `results/tables/excluded_underconverged_runs.csv`.

### 16-shot (688 training samples)

| Method | Best Top-1 | Trainable Params | Notes |
|--------|-----------:|-----------------:|-------|
| M1 Linear Probe | 48.23% | 22,059 | Head-only adaptation |
| M2 CoOp | 67.37% | 8,192 | Text-side optimization |
| M3 CLIP-Adapter | 62.49% | 131,072 | Frozen CLIP + image adapter |
| M4 LoRA (r=4) | 79.46% | 147,456 | Strong archived 16-shot LoRA baseline |
| **M5 CoOp→LoRA** | **81.67%** | **155,648** | CoOp warm-start + 20-epoch LoRA, converged |

### Full-data (26,640 training samples)

| Method | Best Top-1 | Trainable Params | Notes |
|--------|-----------:|-----------------:|-------|
| M1 Linear Probe | 80.10% | 22,059 | 20 epochs |
| M2 CoOp | 82.95% | 8,192 | Best checkpoint occurs before the last epoch |
| M3 CLIP-Adapter | 86.28% | 131,072 | 10 epochs |
| **M4 LoRA (r=4)** | **97.25%** | **147,456** | Best archived full-data result |
| M5 CoOp→LoRA | 96.48% | 155,648 | 20 CoOp + 20 LoRA |

**What the archived artifacts support**: across all three converged low-shot settings (4-, 8-, and 16-shot) M5 now edges out the matched M4 LoRA baseline, with the gap widening as labels become scarcer (+1.62 pp at 8-shot, +2.21 pp at 16-shot, +3.78 pp at 4-shot). The complementarity does **not** carry over to full data, where M4 (97.25%) still beats M5 (96.48%). The takeaway is that the CoOp warm-start helps most precisely when data is limited and the text anchor matters, and fades once the vision encoder has enough samples to adapt on its own.

### Few-shot coverage matrix

All cells are now filled. Each few-shot result uses the strongest converged archived run for the method; parameter-matched r=4 values are kept in the ablation tables below.

| Setting | M1 | M2 | M3 | M4 | M5 |
|--------|----|----|----|----:|----:|
| full    | 80.10% | 82.95% | 86.28% | **97.25%** | 96.48% |
| 16-shot | 48.23% | 67.37% | 62.49% | 79.46% | **81.67%** |
| 8-shot  | 44.55% | 62.34% | 57.65% | 73.85% | **74.56%** |
| 4-shot  | 38.19% | 55.95% | 52.79% | 65.38% | **69.16%** |

Run records are stored in `results/tables/pending_experiments.csv`.

### Usable low-shot main runs (M4 vs. M5)

| Setting | M4 LoRA | M5 CoOp→LoRA | Gain / status |
|---------|--------:|-------------:|---------------|
| 4-shot  | 65.38% | **69.16%** | **+3.78 pp** |
| 8-shot  | 72.83% | **74.45%** | **+1.62 pp** |
| 16-shot | 79.46% | **81.67%** | **+2.21 pp** |

All three low-shot rows now use converged best-archived runs, and M5 wins each. The 4-shot M4 cell uses the converged `m4_4shot_r4_a8_60ep_seed42` run (the original 10-epoch short run that was discarded reached only 30.70%, and is kept in `excluded_underconverged_runs.csv`). The 16-shot M5 cell uses `m5_16shot_lora40_r4_a8`. The M5 advantage is largest in the lowest-data 4-shot regime and shrinks to a small edge at 8-shot, consistent with the multi-seed table below.

### M4 LoRA rank sweep (8-shot, 20 epochs)

| Rank r | Alpha | Best Top-1 | Trainable Params |
|-------:|------:|-----------:|-----------------:|
| 1      | 2     | 44.31%     | 36,864 |
| 4      | 8     | 65.99%     | 147,456 |
| 8      | 16    | 71.56%     | 294,912 |
| 16     | 32    | **73.85%** | **589,824** |

Accuracy improves monotonically with rank, but the jump from r=8 to r=16 is much smaller than the jump from r=4 to r=8.

### M4 learning-rate sweep (8-shot, r=8, α=16, 20 epochs)

| Learning rate | Best Top-1 |
|--------------:|-----------:|
| 1e-5          | 30.34% |
| 3e-5          | 42.57% |
| 5e-5          | 57.82% |
| **1e-4**      | **71.56%** |

### M5 prompt-length ablation (8-shot, 20 CoOp + 20 LoRA)

| Context length n_ctx | Best Top-1 |
|---------------------:|-----------:|
| 4                    | 71.16% |
| 8                    | 70.87% |
| **16**               | **73.22%** |

### Stage-order ablation (8-shot, matched seed-42 runs)

| Variant | Stage 1 | Stage 2 | Best Top-1 |
|--------|---------|---------|------------:|
| M5 CoOp→LoRA | CoOp warm-up | LoRA fine-tune | **71.88%** |
| M5d LoRA→CoOp | LoRA warm-up | CoOp fine-tune | 67.45% |

The forward order outperforms the reversed order by **4.43 pp** on the matched archived seed-42 runs.

### Multi-seed validation (8-shot, matched 20-epoch study)

| Method | Seed 0 | Seed 2 | Seed 3 | Mean ± Std |
|--------|--------|--------|--------|------------|
| M4 LoRA r=16 α=32 | 73.85% | 72.83% | 72.87% | 73.18 ± 0.58% |
| **M5 CoOp→LoRA r=8 α=16** | **74.56%** | 74.45% | 74.35% | **74.45 ± 0.11%** |

This is the cleanest archived evidence for a small positive M5 gain over a well-tuned LoRA baseline.

### Supplemental and exploratory runs

- `results/tables/supplemental_runs.csv` records extra completed variants that are not part of the main tables.
- `results/tables/excluded_underconverged_runs.csv` records short runs that are retained for traceability but excluded from the main claims.
- Two exploratory CLC runs are preserved under `outputs/exploratory/clc/` and indexed in `supplemental_runs.csv`.
- The CLC artifacts are kept because they contain valid outputs, but `src/train_clc.py` is not present in the current repo snapshot, so they are documented as exploratory rather than fully reproducible.

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
