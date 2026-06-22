# Checkpoint Reuse Runbook

Use this file before launching any batch training run. The goal is to avoid rerunning stages that already have compatible checkpoints.

## Environment

Run training from the conda environment that has CUDA PyTorch:

```powershell
conda activate clip
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected current environment: `torch 2.12.1+cu126`, CUDA available, RTX 4060 Laptop GPU.

## Reuse Rules

- For M5, always look for an existing `*_stage1_coop.pt` or `*_coop.pt` before running Stage 1. Pass it with `--coop_ckpt`.
- For M4, M5, and M5d LoRA stages, look for a compatible `*_lora.pt` or `*_stage1_lora.pt`. Pass it with `--lora_ckpt`.
- Checkpoint compatibility is strict for LoRA tensor names and shapes. A different `rank`/`alpha` combination should be treated as a different checkpoint family.
- If a checkpoint exists and the goal is only evaluation/export, set the relevant epoch count to `0` and pass the checkpoint path.
- Keep `--num_workers 2` by default on this Windows laptop. If workers hang, rerun with `--num_workers 0`.
- Keep GPU feature cache enabled for M5/M5d by default. If VRAM becomes tight, add `--cpu_feature_cache`.

## Templates

Resume or extend M4 LoRA:

```powershell
python src/train_m4.py --shots 16 --epochs 20 --rank 4 --alpha 8 --lr 1e-4 --batch_size 64 --num_workers 2 --lora_ckpt outputs/main/m4/m4_16shot/m4_16shot.pt --save outputs/main/m4/m4_16shot_extend/m4_16shot_extend.pt
```

Run M5 16-shot while reusing an existing CoOp prompt:

```powershell
python src/train_m5.py --shots 16 --coop_epochs 0 --lora_epochs 40 --n_ctx 16 --rank 4 --alpha 8 --lr 0.002 --lora_lr 1e-4 --batch_size 64 --num_workers 2 --coop_ckpt outputs/main/m5/m5_16shot/m5_16shot_stage1_coop.pt --save outputs/main/m5/m5_16shot_lora40/m5_16shot_lora40.pt
```

Evaluate a loaded M5 LoRA checkpoint without additional LoRA training:

```powershell
python src/train_m5.py --shots 16 --coop_epochs 0 --lora_epochs 0 --n_ctx 16 --rank 4 --alpha 8 --batch_size 64 --num_workers 2 --coop_ckpt outputs/main/m5/m5_16shot/m5_16shot_stage1_coop.pt --lora_ckpt outputs/main/m5/m5_16shot/m5_16shot_lora.pt --save outputs/eval/m5_16shot_loaded/m5_16shot_loaded.pt
```

## Current Missing Priority

The highest-value missing result is a converged M5 16-shot run. The previous `m5_16shot` artifact was excluded because the 10-epoch LoRA stage was still climbing. Prefer reusing its Stage 1 CoOp prompt and rerunning only a longer LoRA stage.
