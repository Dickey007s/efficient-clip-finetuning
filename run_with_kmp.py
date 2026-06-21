import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import sys

# 运行 train_m2.py，传递所有参数
result = subprocess.run(
    [sys.executable, "-u", "src/train_m2.py"] + sys.argv[1:],
    stdout=None,  # 直接输出到终端
    stderr=None,
)

sys.exit(result.returncode)
