# -*- coding: utf-8 -*-
"""
run_all_experiments.py
串行调度所有未完成的实验，自动保存日志和结果。

用法:
    python run_all_experiments.py
"""
import subprocess
import sys
import time
from pathlib import Path

# 使用 clip 环境的 Python
PYTHON = r"C:/Users/Dinking/miniconda3/envs/clip/python.exe"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 所有需要跑的实验: (name, cmd_args, log_file)
EXPERIMENTS = [
    (
        "M2 CoOp full",
        ["src/train_m2.py", "--epochs", "20", "--lr", "0.002", "--batch_size", "64", "--n_ctx", "16", "--save", "outputs/m2_full.pt"],
        "m2_full.log"
    ),
    (
        "M4 LoRA full",
        ["src/train_m4.py", "--epochs", "20", "--lr", "1e-4", "--batch_size", "64", "--rank", "4", "--alpha", "8", "--save", "outputs/m4_full.pt"],
        "m4_full.log"
    ),
    (
        "M5c CoOp-LoRA full",
        ["src/train_m5.py", "--coop_epochs", "20", "--lora_epochs", "10", "--lr", "0.002", "--lora_lr", "1e-4", "--batch_size", "64", "--n_ctx", "16", "--rank", "4", "--alpha", "8", "--save", "outputs/m5c_full.pt"],
        "m5c_full.log"
    ),
    (
        "M3 CLIP-Adapter 16-shot",
        ["src/train_m3.py", "--epochs", "20", "--lr", "1e-3", "--batch_size", "64", "--shots", "16", "--save", "outputs/m3_16shot.pt"],
        "m3_16shot.log"
    ),
    (
        "M3 CLIP-Adapter full",
        ["src/train_m3.py", "--epochs", "20", "--lr", "1e-3", "--batch_size", "64", "--save", "outputs/m3_full.pt"],
        "m3_full.log"
    ),
    (
        "M5d LoRA-CoOp 16-shot",
        ["src/train_m5d.py", "--lora_epochs", "20", "--coop_epochs", "10", "--lr", "0.002", "--lora_lr", "1e-4", "--batch_size", "64", "--shots", "16", "--n_ctx", "16", "--rank", "4", "--alpha", "8", "--save", "outputs/m5d_16shot.pt"],
        "m5d_16shot.log"
    ),
    (
        "M5d LoRA-CoOp full",
        ["src/train_m5d.py", "--lora_epochs", "20", "--coop_epochs", "10", "--lr", "0.002", "--lora_lr", "1e-4", "--batch_size", "64", "--n_ctx", "16", "--rank", "4", "--alpha", "8", "--save", "outputs/m5d_full.pt"],
        "m5d_full.log"
    ),
]

def run_experiment(name, args, log_file):
    log_path = OUTPUT_DIR / log_file
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Starting: {name}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"# {name}\n")
        f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Command: {PYTHON} {' '.join(args)}\n")
        f.write(f"{'='*60}\n\n")
        f.flush()
        
        proc = subprocess.Popen(
            [PYTHON] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        for line in proc.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        proc.wait()
    
    elapsed = time.time() - start_time
    print(f"\n[{time.strftime('%H:%M:%S')}] Finished: {name} in {elapsed/60:.1f}min")
    
    if proc.returncode != 0:
        print(f"[ERROR] {name} failed with exit code {proc.returncode}")
        return False
    return True


def main():
    print("="*60)
    print("批量实验调度器")
    print(f"总计 {len(EXPERIMENTS)} 个实验")
    print("="*60)
    
    completed = 0
    failed = 0
    
    for i, (name, args, log_file) in enumerate(EXPERIMENTS, 1):
        print(f"\n\n[Experiment {i}/{len(EXPERIMENTS)}]")
        success = run_experiment(name, args, log_file)
        if success:
            completed += 1
        else:
            failed += 1
            print(f"[WARNING] Continuing to next experiment...")
    
    print(f"\n\n{'='*60}")
    print("所有实验完成！")
    print(f"成功: {completed}/{len(EXPERIMENTS)}")
    print(f"失败: {failed}/{len(EXPERIMENTS)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
