@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "PYTHON=C:\Users\Dinking\miniconda3\envs\clip\python.exe"
set "OUTPUT_DIR=outputs"
set "LOG_DIR=%OUTPUT_DIR%"

if not exist %LOG_DIR% mkdir %LOG_DIR%

echo ==========================================
echo 批量实验脚本：串行跑完所有未完成的实验
echo 开始时间: %date% %time%
echo ==========================================
echo.

:: 实验列表: 名称 "参数" 日志文件
set experiments[0]="M2 CoOp full" "src/train_m2.py --epochs 20 --lr 0.002 --batch_size 64 --n_ctx 16 --save outputs/m2_full.pt" "m2_full.log"
set experiments[1]="M4 LoRA full" "src/train_m4.py --epochs 20 --lr 1e-4 --batch_size 64 --rank 4 --alpha 8 --save outputs/m4_full.pt" "m4_full.log"
set experiments[2]="M5c CoOp-LoRA full" "src/train_m5.py --coop_epochs 20 --lora_epochs 10 --lr 0.002 --lora_lr 1e-4 --batch_size 64 --n_ctx 16 --rank 4 --alpha 8 --save outputs/m5c_full.pt" "m5c_full.log"
set experiments[3]="M3 CLIP-Adapter 16-shot" "src/train_m3.py --epochs 20 --lr 1e-3 --batch_size 64 --shots 16 --save outputs/m3_16shot.pt" "m3_16shot.log"
set experiments[4]="M3 CLIP-Adapter full" "src/train_m3.py --epochs 20 --lr 1e-3 --batch_size 64 --save outputs/m3_full.pt" "m3_full.log"
set experiments[5]="M5d LoRA-CoOp 16-shot" "src/train_m5d.py --lora_epochs 20 --coop_epochs 10 --lr 0.002 --lora_lr 1e-4 --batch_size 64 --shots 16 --n_ctx 16 --rank 4 --alpha 8 --save outputs/m5d_16shot.pt" "m5d_16shot.log"
set experiments[6]="M5d LoRA-CoOp full" "src/train_m5d.py --lora_epochs 20 --coop_epochs 10 --lr 0.002 --lora_lr 1e-4 --batch_size 64 --n_ctx 16 --rank 4 --alpha 8 --save outputs/m5d_full.pt" "m5d_full.log"

set /a total=7
set /a completed=0
set /a failed=0

for /L %%i in (0,1,6) do (
    for /f "tokens=1,*" %%a in (!experiments[%%i]!) do (
        set "name=%%~a"
        set "args=%%b"
        
        echo.
        echo ==========================================
        echo [%%i+1/%total%] Running: !name!
        echo 开始: %date% %time%
        echo ==========================================
        
        %PYTHON% !args!
        
        if !errorlevel! equ 0 (
            set /a completed+=1
            echo [OK] !name! 完成
        ) else (
            set /a failed+=1
            echo [FAIL] !name! 失败，继续下一个...
        )
        
        echo 结束: %date% %time%
    )
)

echo.
echo ==========================================
echo 所有实验完成！
echo 成功: %completed%/%total%
echo 失败: %failed%/%total%
echo 结束时间: %date% %time%
echo ==========================================
pause
