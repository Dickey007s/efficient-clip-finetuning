@echo off
chcp 65001 >nul
set PYTHON=C:\Users\Dinking\miniconda3\envs\clip\python.exe
set OUTPUT_DIR=outputs

if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

echo ==========================================
echo 批量实验脚本：跑完所有未完成的实验
echo ==========================================
echo 开始时间: %date% %time%
echo.

:: ========== M2 full ==========
echo [1/7] M2 CoOp full data...
%PYTHON% src\train_m2.py --epochs 20 --lr 0.002 --batch_size 64 --n_ctx 16 --save %OUTPUT_DIR%/m2_full.pt > %OUTPUT_DIR%/m2_full.log 2>&1
echo [1/7] Done. Log: %OUTPUT_DIR%/m2_full.log
echo.

:: ========== M4 full ==========
echo [2/7] M4 LoRA full data...
%PYTHON% src\train_m4.py --epochs 20 --lr 1e-4 --batch_size 64 --rank 4 --alpha 8 --save %OUTPUT_DIR%/m4_full.pt > %OUTPUT_DIR%/m4_full.log 2>&1
echo [2/7] Done. Log: %OUTPUT_DIR%/m4_full.log
echo.

:: ========== M5c full ==========
echo [3/7] M5c CoOp-LoRA full data...
%PYTHON% src\train_m5.py --coop_epochs 20 --lora_epochs 10 --lr 0.002 --lora_lr 1e-4 --batch_size 64 --n_ctx 16 --rank 4 --alpha 8 --save %OUTPUT_DIR%/m5c_full.pt > %OUTPUT_DIR%/m5c_full.log 2>&1
echo [3/7] Done. Log: %OUTPUT_DIR%/m5c_full.log
echo.

:: ========== M3 16-shot ==========
echo [4/7] M3 CLIP-Adapter 16-shot...
%PYTHON% src\train_m3.py --epochs 20 --lr 1e-3 --batch_size 64 --shots 16 --save %OUTPUT_DIR%/m3_16shot.pt > %OUTPUT_DIR%/m3_16shot.log 2>&1
echo [4/7] Done. Log: %OUTPUT_DIR%/m3_16shot.log
echo.

:: ========== M3 full ==========
echo [5/7] M3 CLIP-Adapter full data...
%PYTHON% src\train_m3.py --epochs 20 --lr 1e-3 --batch_size 64 --save %OUTPUT_DIR%/m3_full.pt > %OUTPUT_DIR%/m3_full.log 2>&1
echo [5/7] Done. Log: %OUTPUT_DIR%/m3_full.log
echo.

:: ========== M5d 16-shot ==========
echo [6/7] M5d LoRA-CoOp 16-shot...
%PYTHON% src\train_m5d.py --lora_epochs 20 --coop_epochs 10 --lr 0.002 --lora_lr 1e-4 --batch_size 64 --shots 16 --n_ctx 16 --rank 4 --alpha 8 --save %OUTPUT_DIR%/m5d_16shot.pt > %OUTPUT_DIR%/m5d_16shot.log 2>&1
echo [6/7] Done. Log: %OUTPUT_DIR%/m5d_16shot.log
echo.

:: ========== M5d full ==========
echo [7/7] M5d LoRA-CoOp full data...
%PYTHON% src\train_m5d.py --lora_epochs 20 --coop_epochs 10 --lr 0.002 --lora_lr 1e-4 --batch_size 64 --n_ctx 16 --rank 4 --alpha 8 --save %OUTPUT_DIR%/m5d_full.pt > %OUTPUT_DIR%/m5d_full.log 2>&1
echo [7/7] Done. Log: %OUTPUT_DIR%/m5d_full.log
echo.

echo ==========================================
echo 所有实验完成！
echo 结束时间: %date% %time%
echo ==========================================
pause
