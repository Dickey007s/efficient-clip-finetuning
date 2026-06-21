import time
import subprocess

# 监控 M2 full 的进度
log_file = "outputs/m2_full.log"
pt_file = "outputs/m2_full.pt"

print("Monitoring M2 full experiment...")
print(f"Log: {log_file}")
print(f"Checkpoint: {pt_file}")
print("="*60)

for i in range(60):  # 监控最多 60 次（约 60 分钟）
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # 查找最后几行
        last_lines = lines[-10:] if len(lines) > 10 else lines
        
        # 查找 Epoch 信息
        epoch_info = [l for l in last_lines if 'Epoch' in l and 'Loss' in l]
        if epoch_info:
            print(f"[{i}] {epoch_info[-1].strip()}")
        else:
            print(f"[{i}] {len(lines)} lines in log...")
        
        # 检查 checkpoint 是否存在
        import os
        if os.path.exists(pt_file):
            print(f"[{i}] CHECKPOINT FOUND! Experiment completed.")
            break
            
    except Exception as e:
        print(f"[{i}] Error reading log: {e}")
    
    time.sleep(60)  # 每分钟检查一次

print("Monitoring complete.")
