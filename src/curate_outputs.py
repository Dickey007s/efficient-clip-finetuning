"""
Curate experiment artifacts and regenerate result tables.

This script does two things:
1. Move flat files under outputs/ into a structured directory layout.
2. Regenerate CSV tables under results/tables/ from the discovered runs.

It is safe to run multiple times. Already-organized files are left in place.
"""

from __future__ import annotations

import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

ARTIFACT_SUFFIXES = [
    "_metrics.csv",
    "_confusion.csv",
    "_per_class.csv",
    "_failures.csv",
    "_stage1_coop.pt",
    "_stage3_coop.pt",
    "_stage1_lora.pt",
    "_coop.pt",
    "_lora.pt",
    ".log",
    ".pt",
]

METHOD_LABELS = {
    "m1": "M1 Linear Probe",
    "m2": "M2 CoOp",
    "m3": "M3 CLIP-Adapter",
    "m4": "M4 LoRA",
    "m5": "M5 CoOp->LoRA",
    "m5d": "M5d LoRA->CoOp",
    "clc": "CLC CoOp->LoRA->CoOp",
}

MAIN_RUN_LAYOUT = {
    "full": {
        "M1": "m1_full",
        "M2": "m2_full",
        "M3": "m3_full",
        "M4": "m4_full",
        "M5": "m5_full",
    },
    "16-shot": {
        "M1": "m1_16shot",
        "M2": "m2_16shot",
        "M3": "m3_16shot",
        "M4": "m4_16shot",
        "M5": "",
    },
    "8-shot": {
        "M1": "",
        "M2": "",
        "M3": "",
        "M4": "m4_8shot_r4_a8_20ep",
        "M5": "m5_8shot_base_20ep",
    },
    "4-shot": {
        "M1": "",
        "M2": "",
        "M3": "",
        "M4": "",
        "M5": "m5_4shot_r4_a8_lora20",
    },
}

EXCLUDED_RUNS = {
    "m4_4shot": "LoRA ran only 10 epochs and accuracy was still near its initial plateau; no converged replacement is archived.",
    "m4_8shot": "Superseded by m4_8shot_r4_a8_20ep; the 10-epoch run was still climbing.",
    "m5_4shot": "Superseded by m5_4shot_r4_a8_lora20; the 10-epoch LoRA stage was still climbing.",
    "m5_8shot": "Superseded by m5_8shot_base_20ep; the 10-epoch LoRA stage was still climbing.",
    "m5_16shot": "LoRA ran only 10 epochs and was still climbing at the final epoch; rerun with a longer LoRA stage before using as a main result.",
}


def strip_artifact_suffix(filename: str) -> str:
    for suffix in ARTIFACT_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def classify_run(run_name: str) -> tuple[str, str]:
    if run_name in EXCLUDED_RUNS:
        return "discarded/underconverged", "discarded"
    if run_name.startswith("clc_"):
        return "exploratory/clc", "exploratory"
    if run_name.startswith("m5d_"):
        return "ablations/stage_order", "completed"
    if "coop10_lora20" in run_name:
        return "ablations/stage_order", "completed"
    if run_name in {
        "m4_8shot_r16_a32_20ep",
        "m4_8shot_r16_a32_seed1_20ep",
        "m4_8shot_r16_a32_seed2_20ep",
        "m4_8shot_r16_a32_seed3_20ep",
        "m5_8shot_r8_a16_20ep",
        "m5_8shot_r8_a16_seed1_20ep",
        "m5_8shot_r8_a16_seed2_20ep",
        "m5_8shot_r8_a16_seed3_20ep",
    }:
        return "validation/multi_seed", "completed"
    if run_name.startswith("m4_8shot_r") and "_lr" in run_name:
        return "ablations/m4_learning_rate", "completed"
    if run_name.startswith("m4_8shot_r"):
        return "ablations/m4_rank", "completed"
    if run_name == "m5_4shot_r4_a8_lora20":
        return "main/m5", "completed"
    if run_name.startswith("m5_8shot_nctx") or run_name == "m5_8shot_base_20ep":
        return "ablations/m5_prompt_length", "completed"
    if run_name.startswith("m5_8shot_r1_a2_20ep"):
        return "supplemental/m5_rank", "completed"
    if run_name.startswith("m5_8shot_r4_lr"):
        return "supplemental/m5_schedule_lr", "completed"
    match = re.match(r"^(m[1-5])_", run_name)
    if match:
        return f"main/{match.group(1)}", "completed"
    return "supplemental/misc", "completed"


def parse_method(run_name: str) -> str:
    for key in ("m5d", "clc", "m1", "m2", "m3", "m4", "m5"):
        if run_name.startswith(key):
            return key
    return "unknown"


def parse_shots(run_name: str) -> str:
    if "_full" in run_name:
        return "full"
    match = re.search(r"_(\d+)shot", run_name)
    return f"{match.group(1)}-shot" if match else ""


def parse_seed(run_name: str) -> str:
    match = re.search(r"_seed(\d+)", run_name)
    return match.group(1) if match else "0"


def parse_rank(run_name: str) -> str:
    match = re.search(r"_r(\d+)_a(\d+)", run_name)
    return match.group(1) if match else ""


def infer_schedule(run_name: str) -> str:
    if run_name == "m5_4shot_r4_a8_lora20":
        return "20 CoOp + 40 LoRA"
    if run_name == "clc_8shot_r8_a16_seed42":
        return "10 CoOp + 20 LoRA + 10 CoOp"
    if run_name == "clc_8shot_r8_a16_20_20_10_seed42":
        return "20 CoOp + 20 LoRA + 10 CoOp"
    if run_name in {
        "m1_full",
        "m1_16shot",
        "m2_full",
        "m2_16shot",
        "m4_full",
        "m4_16shot",
        "m4_8shot_r1_a2_20ep",
        "m4_8shot_r4_a8_20ep",
        "m4_8shot_r8_a16_20ep",
        "m4_8shot_r16_a32_20ep",
        "m4_8shot_r16_a32_seed1_20ep",
        "m4_8shot_r16_a32_seed2_20ep",
        "m4_8shot_r16_a32_seed3_20ep",
        "m4_8shot_r8_a16_lr1e-5_20ep",
        "m4_8shot_r8_a16_lr3e-5_20ep",
        "m4_8shot_r8_a16_lr5e-5_20ep",
    }:
        return "20 epochs"
    if run_name in {"m4_4shot", "m4_8shot"}:
        return "10 epochs"
    if run_name == "m3_full":
        return "10 epochs"
    if run_name == "m3_16shot":
        return "20 epochs"
    if run_name.startswith("m5d_8shot_r8_a16_lora20_coop10"):
        return "20 LoRA + 10 CoOp"
    if run_name.startswith("m5_8shot_r8_a16_coop10_lora20"):
        return "10 CoOp + 20 LoRA"
    if run_name.startswith("m5_4shot") or run_name.startswith("m5_8shot") or run_name.startswith("m5_16shot") or run_name == "m5_full":
        if "base_20ep" in run_name or "nctx" in run_name or "_r1_a2_20ep" in run_name or "_r8_a16_20ep" in run_name:
            return "20 CoOp + 20 LoRA"
        return "20 CoOp + 10 LoRA"
    metric_rows = load_metric_rows_for_name(run_name)
    if metric_rows:
        return f"{len(metric_rows)} metric rows"
    return ""


def trainable_params(run_name: str) -> str:
    method = parse_method(run_name)
    if method == "m1":
        return "22059"
    if method == "m2":
        return "8192"
    if method == "m3":
        return "131072"
    if method in {"m4", "m5", "m5d", "clc"}:
        rank = parse_rank(run_name)
        if not rank:
            rank = "4"
        lora_params = 36864 * int(rank)
        if method == "m4":
            return str(lora_params)
        return str(lora_params + 8192)
    return ""


def organize_outputs() -> tuple[list[str], list[str]]:
    moved = []
    removed = []
    for path in list(OUTPUTS_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".log":
            target = LOGS_DIR / path.name
            if target.exists():
                path.unlink()
                removed.append(str(path.relative_to(ROOT)))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
                moved.append(f"{path.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
            continue

        run_name = strip_artifact_suffix(path.name)
        category, _ = classify_run(run_name)
        target_dir = OUTPUTS_DIR / category / run_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name
        if target_path.exists():
            if path.resolve() != target_path.resolve():
                path.unlink()
                removed.append(str(path.relative_to(ROOT)))
            continue
        shutil.move(str(path), str(target_path))
        moved.append(f"{path.relative_to(ROOT)} -> {target_path.relative_to(ROOT)}")
    return moved, removed


def discover_runs() -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    files_by_run: dict[str, list[Path]] = defaultdict(list)

    for path in OUTPUTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        run_name = strip_artifact_suffix(path.name)
        files_by_run[run_name].append(path)

    for run_name, files in files_by_run.items():
        category, status = classify_run(run_name)
        metrics_path = next((p for p in files if p.name.endswith("_metrics.csv")), None)
        confusion_path = next((p for p in files if p.name.endswith("_confusion.csv")), None)
        per_class_path = next((p for p in files if p.name.endswith("_per_class.csv")), None)
        failures_path = next((p for p in files if p.name.endswith("_failures.csv")), None)
        log_path = LOGS_DIR / f"{run_name}.log"

        best_acc = ""
        final_acc = ""
        metric_rows = 0
        stages = ""
        if metrics_path and metrics_path.exists():
            rows, best_acc_val, final_acc_val, stage_names = load_metrics(metrics_path)
            metric_rows = rows
            best_acc = f"{best_acc_val:.4f}"
            final_acc = f"{final_acc_val:.4f}"
            stages = "+".join(stage_names)

        artifact_dir = common_parent(files)
        grouped[run_name] = {
            "run_name": run_name,
            "method": parse_method(run_name),
            "method_label": METHOD_LABELS.get(parse_method(run_name), parse_method(run_name)),
            "shots": parse_shots(run_name),
            "seed": parse_seed(run_name),
            "rank": parse_rank(run_name),
            "category": category,
            "status": status,
            "schedule": infer_schedule(run_name),
            "trainable_params": trainable_params(run_name),
            "best_acc": best_acc,
            "final_acc": final_acc,
            "metric_rows": str(metric_rows),
            "stages": stages,
            "artifact_dir": str(artifact_dir.relative_to(ROOT)) if artifact_dir else "",
            "metrics_path": str(metrics_path.relative_to(ROOT)) if metrics_path else "",
            "confusion_path": str(confusion_path.relative_to(ROOT)) if confusion_path else "",
            "per_class_path": str(per_class_path.relative_to(ROOT)) if per_class_path else "",
            "failures_path": str(failures_path.relative_to(ROOT)) if failures_path else "",
            "log_path": str(log_path.relative_to(ROOT)) if log_path.exists() else "",
        }
    return grouped


def common_parent(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    parents = {path.parent for path in paths}
    if len(parents) == 1:
        return next(iter(parents))
    return None


def load_metrics(metrics_path: Path) -> tuple[int, float, float, list[str]]:
    stage_names = []
    accs = []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            accs.append(float(row[-1]))
            if len(row) == 4 and row[0] not in stage_names:
                stage_names.append(row[0])
    if not stage_names:
        stage_names = ["single_stage"]
    return len(accs), max(accs), accs[-1], stage_names


def load_metric_rows_for_name(run_name: str) -> list[list[str]]:
    for path in OUTPUTS_DIR.rglob(f"{run_name}_metrics.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return list(reader)
    return []


def best_acc_for_run(runs: dict[str, dict], run_name: str) -> str:
    info = runs.get(run_name)
    return info["best_acc"] if info else ""


def write_csv(filename: str, headers: list[str], rows: list[list[str]]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def generate_inventory_table(runs: dict[str, dict]) -> None:
    headers = [
        "run_name",
        "method",
        "shots",
        "seed",
        "rank",
        "category",
        "status",
        "schedule",
        "trainable_params",
        "best_acc",
        "final_acc",
        "metric_rows",
        "stages",
        "artifact_dir",
        "metrics_path",
        "confusion_path",
        "per_class_path",
        "failures_path",
        "log_path",
    ]
    rows = []
    for run_name in sorted(runs):
        info = runs[run_name]
        rows.append([info[key] for key in headers])
    write_csv("experiment_inventory.csv", headers, rows)


def generate_main_tables(runs: dict[str, dict]) -> None:
    rows_16 = []
    rows_full = []
    for method, run_name in MAIN_RUN_LAYOUT["16-shot"].items():
        info = runs.get(run_name, {})
        rows_16.append([
            method,
            run_name,
            info.get("best_acc", ""),
            info.get("trainable_params", ""),
            info.get("schedule", ""),
            "completed" if run_name else "pending",
        ])
    write_csv(
        "results_16shot.csv",
        ["method", "run_name", "best_acc", "trainable_params", "schedule", "status"],
        rows_16,
    )

    for method, run_name in MAIN_RUN_LAYOUT["full"].items():
        info = runs.get(run_name, {})
        rows_full.append([
            method,
            run_name,
            info.get("best_acc", ""),
            info.get("trainable_params", ""),
            info.get("schedule", ""),
            "completed" if run_name else "pending",
        ])
    write_csv(
        "results_full_data.csv",
        ["method", "run_name", "best_acc", "trainable_params", "schedule", "status"],
        rows_full,
    )

    curve_rows = []
    for setting, mapping in MAIN_RUN_LAYOUT.items():
        curve_rows.append(
            [
                setting,
                best_acc_for_run(runs, mapping["M1"]),
                best_acc_for_run(runs, mapping["M2"]),
                best_acc_for_run(runs, mapping["M3"]),
                best_acc_for_run(runs, mapping["M4"]),
                best_acc_for_run(runs, mapping["M5"]),
            ]
        )
    write_csv(
        "few_shot_curve.csv",
        ["setting", "M1", "M2", "M3", "M4", "M5"],
        curve_rows,
    )


def generate_m4_tables(runs: dict[str, dict]) -> None:
    rank_runs = [
        ("1", "2", "m4_8shot_r1_a2_20ep"),
        ("4", "8", "m4_8shot_r4_a8_20ep"),
        ("8", "16", "m4_8shot_r8_a16_20ep"),
        ("16", "32", "m4_8shot_r16_a32_20ep"),
    ]
    write_csv(
        "m4_rank_sweep.csv",
        ["rank", "alpha", "run_name", "best_acc", "trainable_params"],
        [
            [rank, alpha, run_name, best_acc_for_run(runs, run_name), runs.get(run_name, {}).get("trainable_params", "")]
            for rank, alpha, run_name in rank_runs
        ],
    )

    lr_runs = [
        ("1e-5", "m4_8shot_r8_a16_lr1e-5_20ep"),
        ("3e-5", "m4_8shot_r8_a16_lr3e-5_20ep"),
        ("5e-5", "m4_8shot_r8_a16_lr5e-5_20ep"),
        ("1e-4", "m4_8shot_r8_a16_20ep"),
    ]
    write_csv(
        "m4_learning_rate_sweep.csv",
        ["learning_rate", "run_name", "best_acc"],
        [[lr, run_name, best_acc_for_run(runs, run_name)] for lr, run_name in lr_runs],
    )


def generate_m5_tables(runs: dict[str, dict]) -> None:
    prompt_runs = [
        ("4", "m5_8shot_nctx4_20ep"),
        ("8", "m5_8shot_nctx8_20ep"),
        ("16", "m5_8shot_base_20ep"),
    ]
    write_csv(
        "m5_prompt_length_ablation.csv",
        ["n_ctx", "run_name", "best_acc"],
        [[n_ctx, run_name, best_acc_for_run(runs, run_name)] for n_ctx, run_name in prompt_runs],
    )

    stage_rows = [
        [
            "M5 CoOp->LoRA",
            "CoOp warm-up",
            "LoRA fine-tune",
            "m5_8shot_r8_a16_coop10_lora20_seed42",
            best_acc_for_run(runs, "m5_8shot_r8_a16_coop10_lora20_seed42"),
        ],
        [
            "M5d LoRA->CoOp",
            "LoRA warm-up",
            "CoOp fine-tune",
            "m5d_8shot_r8_a16_lora20_coop10_seed42",
            best_acc_for_run(runs, "m5d_8shot_r8_a16_lora20_coop10_seed42"),
        ],
    ]
    write_csv(
        "stage_order_ablation.csv",
        ["variant", "stage_1", "stage_2", "run_name", "best_acc"],
        stage_rows,
    )


def generate_multi_seed_table(runs: dict[str, dict]) -> None:
    m4_names = [
        "m4_8shot_r16_a32_20ep",
        "m4_8shot_r16_a32_seed1_20ep",
        "m4_8shot_r16_a32_seed2_20ep",
        "m4_8shot_r16_a32_seed3_20ep",
    ]
    m5_names = [
        "m5_8shot_r8_a16_20ep",
        "m5_8shot_r8_a16_seed1_20ep",
        "m5_8shot_r8_a16_seed2_20ep",
        "m5_8shot_r8_a16_seed3_20ep",
    ]
    m4_accs = [float(runs[name]["best_acc"]) for name in m4_names]
    m5_accs = [float(runs[name]["best_acc"]) for name in m5_names]

    rows = [
        [
            "M4 LoRA r=16 a=32",
            *[f"{acc:.4f}" for acc in m4_accs],
            f"{mean(m4_accs):.4f}",
            f"{pstdev(m4_accs):.4f}",
        ],
        [
            "M5 CoOp->LoRA r=8 a=16",
            *[f"{acc:.4f}" for acc in m5_accs],
            f"{mean(m5_accs):.4f}",
            f"{pstdev(m5_accs):.4f}",
        ],
    ]
    write_csv(
        "multi_seed_validation.csv",
        ["method", "seed0", "seed1", "seed2", "seed3", "mean", "std"],
        rows,
    )


def generate_pending_table(runs: dict[str, dict]) -> None:
    pending_rows = []
    for setting in ("8-shot", "4-shot"):
        for method in ("M1", "M2", "M3"):
            pending_rows.append(
                [
                    setting,
                    method,
                    "pending",
                    "",
                    "README few-shot matrix has no archived run yet.",
                ]
            )
    pending_rows.extend(
        [
            [
                "4-shot",
                "M4",
                "pending",
                "",
                "m4_4shot was discarded because the short run did not converge.",
            ],
            [
                "16-shot",
                "M5",
                "pending",
                "",
                "m5_16shot was discarded because the 10-epoch LoRA stage was still climbing.",
            ],
        ]
    )
    pending_rows.append(
        [
            "zero-shot",
            "M0",
            "not_archived",
            "",
            "check_clip.py provides a sanity check, but no full-test artifact is stored in outputs/.",
        ]
    )
    write_csv(
        "pending_experiments.csv",
        ["setting", "method", "status", "planned_run_name", "note"],
        pending_rows,
    )


def generate_excluded_table(runs: dict[str, dict]) -> None:
    rows = []
    for run_name, reason in EXCLUDED_RUNS.items():
        info = runs.get(run_name, {})
        rows.append(
            [
                run_name,
                info.get("method_label", ""),
                info.get("shots", ""),
                info.get("best_acc", ""),
                info.get("schedule", ""),
                reason,
            ]
        )
    write_csv(
        "excluded_underconverged_runs.csv",
        ["run_name", "method", "shots", "best_acc", "schedule", "reason"],
        rows,
    )


def generate_supplemental_table(runs: dict[str, dict]) -> None:
    keep = [
        "m5_8shot_r1_a2_20ep",
        "m5_8shot_r4_lr1e-5_20ep",
        "m5_8shot_r4_lr3e-5_20ep",
        "m5_8shot_r4_lr5e-5_20ep",
        "clc_8shot_r8_a16_seed42",
        "clc_8shot_r8_a16_20_20_10_seed42",
    ]
    rows = []
    for run_name in keep:
        if run_name not in runs:
            continue
        info = runs[run_name]
        rows.append(
            [
                run_name,
                info["method_label"],
                info["category"],
                info["best_acc"],
                info["schedule"],
                info["artifact_dir"],
                info["status"],
            ]
        )
    write_csv(
        "supplemental_runs.csv",
        ["run_name", "method", "category", "best_acc", "schedule", "artifact_dir", "status"],
        rows,
    )


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    moved, removed = organize_outputs()
    runs = discover_runs()
    generate_inventory_table(runs)
    generate_main_tables(runs)
    generate_m4_tables(runs)
    generate_m5_tables(runs)
    generate_multi_seed_table(runs)
    generate_pending_table(runs)
    generate_excluded_table(runs)
    generate_supplemental_table(runs)

    print(f"[Curate] Moved {len(moved)} files into structured outputs/.")
    print(f"[Curate] Removed {len(removed)} duplicate logs/files.")
    print(f"[Curate] Wrote tables to {TABLES_DIR}.")
    if moved:
        print("[Curate] Sample moves:")
        for line in moved[:8]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
