# -*- coding: utf-8 -*-
"""
run_all_experiments_direct.py
直接运行所有实验，每个实验的输出自动保存到日志文件。
用法: C:/Users/Dinking/miniconda3/envs/clip/python.exe run_all_experiments_direct.py
"""
import sys
import subprocess
import time
from pathlib import Path

PYTHON = sys.executable
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    ("M2 CoOp full", ["src/train_m2.py", "--epochs", "20", "--lr", "0.002", "--batch_size", "64", "--n_ctx", "16", "--save", "outputs/m2_full.pt"]),
    ("M3 CLIP-Adapter 16-shot", ["src/train_m3.py", "--epochs", "20", "--lr", "1e-3", "--batch_size", "64", "--shots", "16", "--save", "outputs/m3_16shot.pt"]),
    ("M3 CLIP-Adapter full", ["src/train_m3.py", "--epochs", "20", "--lr", "1e-3", "--batch_size", "64", "--save", "outputs/m3_full.pt"]),
    ("M5c CoOp-LoRA full", ["src/train_m5.py", "--coop_epochs", "20", "--lora_epochs", "10", "--lr", "0.002", "--lora_lr", "1e-4", "--batch_size", "64", "--n_ctx", "16", "--rank", "4", "--alpha", "8", "--save", "outputs/m5c_full.pt"]),
    ("M5d LoRA-CoOp 16-shot", ["src/train_m5d.py", "--lora_epochs", "20", "--coop_epochs", "10", "--lr", "0.002", "--lora_lr", "1e-4", "--batch_size", "64", "--shots", "16", "--n_ctx", "16", "--rank", "4", "--alpha", "8", "--save", "outputs/m5d_16shot.pt"]),
    ("M5d LoRA-CoOp full", ["src/train_m5d.py", "--lora_epochs", "20", "--coop_epochs", "10", "--lr", "0.002", "--lora_lr", "1e-4", "--batch_size", "64", "--n_ctx", "16", "--rank", "4", "--alpha", "8", "--save", "outputs/m5d_full.pt"]),
]

def run_experiment(name, args):
    log_file = OUTPUT_DIR / (Path(args[-1]).stem + ".log")
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Starting: {name}")
    print(f"Log: {log_file}")
    print(f"{'='*60}\n")
    
    start = time.time()
    
    # 使用 Popen + 实时读取，避免 Windows 缓冲问题
    proc = subprocess.Popen(
        [PYTHON] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # 行缓冲
    )
    
    stdout_lines = []
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"# {name}\n# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Command: {PYTHON} {' '.join(args)}\n{'='*60}\n\n")
        f.flush()
        
        for line in proc.stdout:
            stdout_lines.append(line)
            print(line, end='')  # 实时打印到控制台
            f.write(line)  # 实时写入文件
            f.flush()
    
    proc.wait()
    returncode = proc.returncode
    
    elapsed = (time.time() - start) / 60
    if returncode == 0:
        print(f"\n[OK] {name} completed in {elapsed:.1f} min")
        return True
    else:
        print(f"\n[FAIL] {name} failed (exit {returncode}) after {elapsed:.1f} min")
        return False

if __name__ == "__main__":
    print("="*60)
    print(f"批量实验调度器 | Python: {PYTHON}")
    print(f"总计 {len(EXPERIMENTS)} 个实验")
    print("="*60)
    
    completed = failed = 0
    for i, (name, args) in enumerate(EXPERIMENTS, 1):
        print(f"\n\n[Experiment {i}/{len(EXPERIMENTS)}]")
        if run_experiment(name, args):
            completed += 1
        else:
            failed += 1
    
    print(f"\n\n{'='*60}")
    print("所有实验完成！")
    print(f"成功: {completed}/{len(EXPERIMENTS)}")
    print(f"失败: {failed}/{len(EXPERIMENTS)}")
    print(f"{'='*60}")
