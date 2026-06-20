# -*- coding: utf-8 -*-
"""
class_names.py
GTSRB 43 个类别的可读英文名称。

这些名称将用于：
1) CLIP zero-shot 评估时构造 prompt（如 "a photo of a {class_name}"）
2) 混淆矩阵 / failure case 的可读标签
3) 报告里的类别说明表

名称来源：GTSRB 官方类别说明 + 交通标志语义化命名（便于 CLIP 理解）。
"""
# 类别 ID (0-42) -> 可读名称
GTSRB_CLASS_NAMES = [
    "speed limit 20",              # 0
    "speed limit 30",              # 1
    "speed limit 50",              # 2
    "speed limit 60",              # 3
    "speed limit 70",              # 4
    "speed limit 80",              # 5
    "end of speed limit 80",       # 6
    "speed limit 100",             # 7
    "speed limit 120",             # 8
    "no passing",                  # 9
    "no passing for vehicles over 3.5 metric tons",  # 10
    "right-of-way at the next intersection",          # 11
    "priority road",               # 12
    "yield",                       # 13
    "stop",                        # 14
    "no vehicles",                 # 15
    "vehicles over 3.5 metric tons prohibited",       # 16
    "no entry",                    # 17
    "general caution",             # 18
    "dangerous curve to the left", # 19
    "dangerous curve to the right",# 20
    "double curve",                # 21
    "bumpy road",                  # 22
    "slippery road",               # 23
    "road narrows on the right",   # 24
    "road work",                   # 25
    "traffic signals",             # 26
    "pedestrians",                 # 27
    "children crossing",           # 28
    "bicycles crossing",           # 29
    "beware of ice or snow",       # 30
    "wild animals crossing",       # 31
    "end of all speed and passing limits",            # 32
    "turn right ahead",            # 33
    "turn left ahead",             # 34
    "ahead only",                  # 35
    "go straight or right",        # 36
    "go straight or left",         # 37
    "keep right",                  # 38
    "keep left",                   # 39
    "roundabout mandatory",        # 40
    "end of no passing",           # 41
    "end of no passing by vehicles over 3.5 metric tons",  # 42
]

# 验证：必须是 43 个
assert len(GTSRB_CLASS_NAMES) == 43, f"Expected 43 classes, got {len(GTSRB_CLASS_NAMES)}"


def prompt_for_class(class_id: int, template: str = "a photo of a {}") -> str:
    """根据类别 ID 生成 CLIP prompt 文本。"""
    return template.format(GTSRB_CLASS_NAMES[class_id])


def all_prompts(template: str = "a photo of a {}") -> list:
    """生成所有 43 个类别的 prompt 列表。"""
    return [template.format(name) for name in GTSRB_CLASS_NAMES]


if __name__ == "__main__":
    for i, name in enumerate(GTSRB_CLASS_NAMES):
        print(f"{i:2d}: {name}")
