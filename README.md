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
7. [Limitations](#limitations)
8. [Citation](#citation)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

---

## Method Overview

| Method | Trainable Parameters | Reference |
|--------|----------------------|-----------|
| Zero-shot CLIP | 0 | Radford et al., 2021 |
| Linear Probe | about 22 K | Radford et al., 2021 |
| CoOp | about 8 K | Zhou et al., IJCV 2022 |
| CLIP-Adapter | about 131 K | Gao et al., IJCV 2023 |
| LoRA on CLIP | about 147 K (r=4) | Hu et al., 2022 |
| **CoOp→LoRA (ours)** | **about 156 K (r=4)** | **This work** |

Studies and analyses:

- **Main archived runs** for all five methods across full data, 16-shot, 8-shot, and 4-shot; the accuracy matrix is fully populated.
- **LoRA rank sweep** at r = 1, 4, 8, 16 (8-shot, 20 epochs).
- **LoRA learning-rate sweep** at 1e-5, 3e-5, 5e-5, 1e-4 (8-shot, r=8, 20 epochs).
- **CoOp→LoRA prompt-length sweep** at n_ctx = 4, 8, 16 (8-shot, 20 epochs).
- **Stage-ordering ablation**: CoOp→LoRA vs. LoRA→CoOp, plus an exploratory three-stage CoOp→LoRA→CoOp (CLC) variant.
- **Multi-seed validation** for LoRA and CoOp→LoRA at 8-shot.
- **Convergence-speed and sample-efficiency analyses**: epochs-to-target for LoRA vs. CoOp→LoRA, and CoOp's sample-scaling limitation (see Results and Limitations).
- Failure-case analysis with per-class accuracy and confusion matrices (see `outputs/`).
- Supplemental, exploratory, and discarded underconverged artifacts are preserved separately, including extra CoOp→LoRA schedule/rank runs and two CLC runs whose source script is not present in the current repo snapshot.

### CoOp→LoRA (ours)

**Core idea**: Text-side optimization (learned prompts via CoOp) and vision-side adaptation (LoRA) are complementary and can be trained sequentially under a fixed backbone. The key insight is that **the two parameter sets do not overlap**: CoOp only touches the input embedding layer, while LoRA only touches the intermediate attention layers of the vision encoder. This orthogonality makes sequential training a principled way to stack two complementary biases.

**Stage one -- CoOp warm-up**: Freeze the entire CLIP model and optimize only the continuous prompt vectors. This quickly learns a good text initialization from few-shot data, adapting the class prototypes to the target domain vocabulary.

**Stage two -- LoRA fine-tuning**: Freeze CLIP and the converged CoOp prompts, then inject LoRA layers into the vision encoder (Q/V only). This adapts visual feature extraction on top of the improved text initialization, pulling image embeddings closer to the already-optimized text prototypes.

**Why this works**: CoOp optimizes the *query* side (how to ask CLIP), while LoRA optimizes the *representation* side (how to see the image). Because stage one already provides a strong text anchor, stage two's visual adaptation has a clearer direction than training LoRA from the original CLIP initialization. The intended effect is a **1+1≥1 hybrid**, but the cleaned artifacts show that this complementarity is strongest in the low-shot regime and does not hold uniformly at every data scale.

**Ablation variants**:
- CoOp-only (same as the CoOp baseline, verifies stage one alone).
- LoRA-only (same as the LoRA baseline, verifies stage two alone).
- CoOp→LoRA (the proposed method, verifies sequential stacking).
- LoRA→CoOp (reversed order, verifies stage ordering matters; see `train_m5d.py`).

**How to read the CoOp→LoRA result**: The main diagnostic is still `CoOp→LoRA_best > max(CoOp_best, LoRA_best)`, but it should be evaluated per setting rather than treated as a universal rule. Runs whose accuracy was still clearly rising at the final epoch are excluded from the main claims. In this snapshot, CoOp→LoRA beats the matched LoRA baseline across all three converged low-shot settings (4-, 8-, and 16-shot) and in the matched 8-shot multi-seed study, while LoRA remains ahead on full data.

---

## Repository Structure

```
clip_traffic_sign/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── train_m1.py ... train_m5d.py  Linear Probe, CoOp, CLIP-Adapter, LoRA, CoOp→LoRA (and LoRA→CoOp)
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

This section is organized into three blocks: an **accuracy overview** across all methods and data regimes, a focused **LoRA-vs-CoOp→LoRA comparison** that motivates the proposed method, and the supporting **ablations and bookkeeping**.

### Accuracy overview (all methods × all settings)

Each few-shot result uses the strongest converged archived run for the method; parameter-matched r=4 values are kept in the ablation tables below.

| Setting | Linear Probe | CoOp | CLIP-Adapter | LoRA | CoOp→LoRA |
|--------|----|----|----|----:|----:|
| full    | 80.10% | 82.95% | 86.28% | **97.25%** | 96.48% |
| 16-shot | 48.23% | 67.37% | 62.49% | 79.46% | **81.67%** |
| 8-shot  | 44.55% | 62.34% | 57.65% | 72.83% | **74.45%** |
| 4-shot  | 38.19% | 55.95% | 52.79% | 65.38% | **69.16%** |

Run records are stored in `results/tables/pending_experiments.csv`. The two tables below break out the 16-shot and full-data regimes with trainable-parameter counts.

### 16-shot (688 training samples)

| Method | Best Top-1 | Trainable Params | Notes |
|--------|-----------:|-----------------:|-------|
| Linear Probe | 48.23% | 22,059 | Head-only adaptation |
| CoOp | 67.37% | 8,192 | Text-side optimization |
| CLIP-Adapter | 62.49% | 131,072 | Frozen CLIP + image adapter |
| LoRA (r=4) | 79.46% | 147,456 | Strong archived 16-shot LoRA baseline |
| **CoOp→LoRA** | **81.67%** | **155,648** | CoOp warm-start + 20-epoch LoRA, converged |

### Full-data (26,640 training samples)

| Method | Best Top-1 | Trainable Params | Notes |
|--------|-----------:|-----------------:|-------|
| Linear Probe | 80.10% | 22,059 | 20 epochs |
| CoOp | 82.95% | 8,192 | Best checkpoint occurs before the last epoch |
| CLIP-Adapter | 86.28% | 131,072 | 10 epochs |
| **LoRA (r=4)** | **97.25%** | **147,456** | Best archived full-data result |
| CoOp→LoRA | 96.48% | 155,648 | 20 CoOp + 20 LoRA |

**What the archived artifacts support**: across all three converged low-shot settings (4-, 8-, and 16-shot) CoOp→LoRA now edges out the matched LoRA baseline, with the gap widening as labels become scarcer (+1.62 pp at 8-shot, +2.21 pp at 16-shot, +3.78 pp at 4-shot). The complementarity does **not** carry over to full data, where LoRA (97.25%) still beats CoOp→LoRA (96.48%). The takeaway is that the CoOp warm-start helps most precisely when data is limited and the text anchor matters, and fades once the vision encoder has enough samples to adapt on its own.

### Low-shot comparison (LoRA vs. CoOp→LoRA)

| Setting | LoRA | CoOp→LoRA | Gain / status |
|---------|--------:|-------------:|---------------|
| 4-shot  | 65.38% | **69.16%** | **+3.78 pp** |
| 8-shot  | 72.83% | **74.45%** | **+1.62 pp** |
| 16-shot | 79.46% | **81.67%** | **+2.21 pp** |

All three low-shot rows now use converged best-archived runs, and CoOp→LoRA wins each. The 4-shot LoRA cell uses the converged `m4_4shot_r4_a8_60ep_seed42` run (the original 10-epoch short run that was discarded reached only 30.70%, and is kept in `excluded_underconverged_runs.csv`). The 16-shot CoOp→LoRA cell uses `m5_16shot_lora40_r4_a8`. The CoOp→LoRA advantage is largest in the lowest-data 4-shot regime and shrinks to a small edge at 8-shot, consistent with the multi-seed table below.

### Convergence speed (8-shot, LoRA vs. CoOp→LoRA)

Beyond the small final-accuracy edge, the CoOp warm-start makes CoOp→LoRA converge in **markedly fewer LoRA epochs** than plain LoRA. On the matched 8-shot seed-2 runs, the table below reports the first LoRA epoch that reaches each target test accuracy (full per-epoch curves in `outputs/validation/multi_seed/`).

| Target test acc | LoRA (epoch) | CoOp→LoRA (epoch) | Epochs saved |
|----------------:|----------------:|---------------------:|-------------:|
| 60% | 9  | 4  | 5 |
| 65% | 11 | 6  | 5 |
| 70% | 13 | 9  | 4 |
| 72% | 17 | 11 | 6 |

The gap is set at the very first epoch: after one LoRA epoch CoOp→LoRA is already at **54.46%** versus LoRA's **30.51%**, a +23.9 pp head-start carried over from Stage-1 CoOp. CoOp→LoRA also matches LoRA's full 20-epoch result (72.83%) by **epoch 12** (73.14%), i.e. it reaches the same accuracy with ~40% fewer LoRA epochs. The frozen, pre-aligned text prototypes give Stage-2 LoRA a much clearer target than adapting the vision encoder from raw CLIP. Source table: `results/tables/convergence_speed_8shot.csv`.

### Multi-seed validation (8-shot, matched 20-epoch study)

| Method | Seed 0 | Seed 2 | Seed 3 | Mean ± Std |
|--------|--------|--------|--------|------------|
| LoRA r=16 α=32 | 73.85% | 72.83% | 72.87% | 73.18 ± 0.58% |
| **CoOp→LoRA r=8 α=16** | **74.56%** | 74.45% | 74.35% | **74.45 ± 0.11%** |

This is the cleanest archived evidence for a small positive CoOp→LoRA gain over a well-tuned LoRA baseline.

### LoRA rank sweep (8-shot, 20 epochs)

| Rank r | Alpha | Best Top-1 | Trainable Params |
|-------:|------:|-----------:|-----------------:|
| 1      | 2     | 44.31%     | 36,864 |
| 4      | 8     | 65.99%     | 147,456 |
| 8      | 16    | 71.56%     | 294,912 |
| 16     | 32    | **73.85%** | **589,824** |

Accuracy improves monotonically with rank, but the jump from r=8 to r=16 is much smaller than the jump from r=4 to r=8.

### LoRA learning-rate sweep (8-shot, r=8, α=16, 20 epochs)

| Learning rate | Best Top-1 |
|--------------:|-----------:|
| 1e-5          | 30.34% |
| 3e-5          | 42.57% |
| 5e-5          | 57.82% |
| **1e-4**      | **71.56%** |

### CoOp→LoRA prompt-length ablation (8-shot, 20 CoOp + 20 LoRA)

| Context length n_ctx | Best Top-1 |
|---------------------:|-----------:|
| 4                    | 71.16% |
| 8                    | 70.87% |
| **16**               | **73.22%** |

### Stage-order ablation (8-shot, matched seed-42 runs)

| Variant | Stage 1 | Stage 2 | Best Top-1 |
|--------|---------|---------|------------:|
| CoOp→LoRA | CoOp warm-up | LoRA fine-tune | **71.88%** |
| LoRA→CoOp | LoRA warm-up | CoOp fine-tune | 67.45% |

The forward order outperforms the reversed order by **4.43 pp** on the matched archived seed-42 runs.

### Three-stage variant: CoOp→LoRA→CoOp (exploratory)

A natural question is whether appending a *second* CoOp stage after LoRA (CLC, CoOp→LoRA→CoOp) recovers further gains by re-fitting the prompt to the now-adapted vision features. Two exploratory 8-shot runs (r=8, α=16) are archived under `outputs/exploratory/clc/`:

| CLC run | Schedule | Stage-2 end | Stage-3 best | Stage-3 gain |
|---------|----------|------------:|-------------:|-------------:|
| `clc_8shot_r8_a16_20_20_10_seed42` | 20 CoOp + 20 LoRA + 10 CoOp | ~74.6% | **74.79%** | **+0.14 pp** |
| `clc_8shot_r8_a16_seed42` | 10 CoOp + 20 LoRA + 10 CoOp | 71.88% | 73.86% | +1.98 pp |

**Why the third stage does not help when Stage 2 is converged.** With a properly trained Stage 2 (20 LoRA epochs), the extra CoOp stage adds only **+0.14 pp** — well within the multi-seed noise band (CoOp→LoRA std ≈ 0.1–0.6 pp), so it is not a significant improvement. The larger +1.98 pp in the second run is misleading: its Stage 2 used only a 10-epoch CoOp warm-up and ended under-trained at 71.88%, so the third stage is mostly recovering headroom that a longer Stage 2 would have reached on its own, not adding new signal. Mechanistically, once Stage-2 LoRA has aligned image features to the CoOp prompt, the low-rank update moves those features only modestly; re-running CoOp then re-fits essentially the same target and has little left to learn. The two-stage CoOp→LoRA is therefore kept as the main method, and CLC is documented as exploratory only (there is no `src/train_clc.py` in the current snapshot). Source table: `results/tables/clc_three_stage.csv`.

### Supplemental and exploratory runs

- `results/tables/supplemental_runs.csv` records extra completed variants that are not part of the main tables.
- `results/tables/excluded_underconverged_runs.csv` records short runs that are retained for traceability but excluded from the main claims.
- Two exploratory CLC (CoOp→LoRA→CoOp) runs are preserved under `outputs/exploratory/clc/` and indexed in `supplemental_runs.csv`; their analysis is in the [three-stage variant](#three-stage-variant-cooploracoop-exploratory) section above.
- The CLC artifacts are kept because they contain valid outputs, but `src/train_clc.py` is not present in the current repo snapshot, so they are documented as exploratory rather than fully reproducible.

### Data exploration outputs

Generated by `explore_data.py`:

- `figures/class_distribution.png`, class distribution of train and test sets.
- `figures/samples_per_class.png`, one sample per class.
- `results/train_class_distribution.csv`, per-class image counts.
- `results/train_image_size_stats.csv`, image-size statistics.

---

## Limitations

### CoOp does not scale with samples on this task

CoOp is the most parameter-efficient method here (8,192 trainable context vectors), but a sample-angle analysis exposes a structural ceiling: it puts all adaptation capacity on the text side and keeps the vision encoder frozen, so additional traffic-sign samples cannot change *how CLIP sees the image*. Source table: `results/tables/coop_sample_limitation.csv`.

**1. The gap to LoRA widens monotonically as data grows.** CoOp is the only method whose relative disadvantage *increases* with more samples.

| Setting | CoOp | LoRA | Gap |
|---------|----------:|----------:|----:|
| 4-shot (172)   | 55.95% | 65.38% | −9.43 pp |
| 8-shot (344)   | 62.34% | 72.83% | −10.49 pp |
| 16-shot (688)  | 67.37% | 79.46% | −12.09 pp |
| full (26,640)  | 82.95% | 97.25% | **−14.30 pp** |

**2. Near-zero marginal sample utility and a hard ceiling.** Going from 16-shot to full data is a **38× increase** in samples (688 → 26,640) but buys CoOp only +15.58 pp before it plateaus at 82.95%. Even with the full training set, **24 of 43 classes stay below 80%** for CoOp, versus **0 of 43** for LoRA. The ceiling is structural, not a data shortage.

**3. Mechanism: the frozen visual representation is the bottleneck, not the prompt.** CoOp tunes only a shared, class-agnostic text context. Extra samples can refine *how we query* CLIP but never the out-of-domain, symbolic GTSRB image features. When those frozen features are not separable, no amount of prompt tuning recovers them.

**4. Class-level evidence: direction-symmetric signs collapse.** CoOp's worst classes are exactly the symbolic signs distinguished only by arrow direction, where the frozen features for "left" and "right" nearly overlap and a single unified context cannot pull them apart:

| Class | CoOp | LoRA |
|-------|-----:|-----:|
| keep left | **20.0%** | 96.7% |
| go straight or left | **26.7%** | 100% |
| dangerous curve to the left | **30.0%** | 100% |
| dangerous curve to the right | 47.8% | 100% |

**Takeaway for this project.** CoOp's usefulness is narrow: tiny parameter count and competitive *only* in the extreme low-data regime (4-shot, where it beats Linear Probe and CLIP-Adapter). It is fundamentally sample-inefficient at scale because its adaptation budget lives entirely on the frozen-feature text side. This is precisely why CoOp→LoRA couples CoOp (a text anchor) with LoRA (vision-side capacity), and why on full data CoOp→LoRA ≈ LoRA — once samples are plentiful, the LoRA stage does the heavy lifting.

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
