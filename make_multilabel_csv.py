"""
根据指定根目录下的五个子目录（watermark、glare、integrity、occlusion、source）
生成无表头的 train.csv 与 val.csv，同时生成配置文件便于训练脚本引用。

用法示例：
    uv run python make_multilabel_csv.py --data-root data_raw --val-ratio 0.2

默认输出位置：
    data_root 的同级目录生成 train.csv、val.csv 与 dataset_config.json

CSV 列（无表头）顺序固定：
    path, watermark, glare, integrity, occlusion
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# 每个子目录对应的标签模板
DIR_TO_LABELS: Dict[str, Dict[str, int]] = {
    "watermark": {"watermark": 1, "glare": 0, "integrity": 0, "occlusion": 0},
    "glare": {"watermark": 0, "glare": 1, "integrity": 0, "occlusion": 0},
    "integrity": {"watermark": 0, "glare": 0, "integrity": 1, "occlusion": 0},
    "occlusion": {"watermark": 0, "glare": 0, "integrity": 0, "occlusion": 1},
    "source": {"watermark": 0, "glare": 0, "integrity": 0, "occlusion": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成多标签 CSV")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="包含 watermark/glare/integrity/occlusion/source 的根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认 data-root 的同级目录）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例（0-1），默认 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于划分 train/val）",
    )
    parser.add_argument(
        "--train-name",
        type=str,
        default="train.csv",
        help="训练集标签文件名（默认 train.csv）",
    )
    parser.add_argument(
        "--val-name",
        type=str,
        default="val.csv",
        help="验证集标签文件名（默认 val.csv）",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="dataset_config.json",
        help="配置文件名（默认 dataset_config.json）",
    )
    return parser.parse_args()


def collect_samples(data_root: Path) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []

    if not data_root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")

    for dir_name, labels in DIR_TO_LABELS.items():
        subdir = data_root / dir_name
        if not subdir.exists():
            # 跳过缺失目录，便于逐步补齐数据
            continue

        for img_path in subdir.rglob("*"):
            if img_path.suffix.lower() not in ALLOWED_SUFFIXES or not img_path.is_file():
                continue

            rel_path = img_path.relative_to(data_root).as_posix()
            row = {
                "path": rel_path,
                "watermark": labels["watermark"],
                "glare": labels["glare"],
                "integrity": labels["integrity"],
                "occlusion": labels["occlusion"],
            }
            samples.append(row)

    if not samples:
        raise RuntimeError(f"在 {data_root} 下未找到任何有效图片（支持后缀: {ALLOWED_SUFFIXES}）")

    return samples


def apply_split(samples: List[Dict[str, str]], val_ratio: float, seed: int):
    if not (0 < val_ratio < 1):
        raise ValueError("val-ratio 需在 0-1 之间且不为 0")

    samples_copy = samples.copy()
    random.seed(seed)
    random.shuffle(samples_copy)

    val_count = max(1, int(len(samples_copy) * val_ratio))
    val_samples = samples_copy[:val_count]
    train_samples = samples_copy[val_count:]
    if not train_samples:
        raise ValueError("划分后训练集为空，请降低 val-ratio")
    return train_samples, val_samples


def write_csv(output: Path, samples: List[Dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    field_order = ["path", "watermark", "glare", "integrity", "occlusion"]

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in samples:
            writer.writerow([row[col] for col in field_order])


def write_config(output_dir: Path, config_name: str, args: argparse.Namespace) -> Path:
    config_path = output_dir / config_name
    config = {
        "images_root": str(args.data_root.resolve()),
        "train_label": str((output_dir / args.train_name).resolve()),
        "val_label": str((output_dir / args.val_name).resolve()),
        "label_columns": ["path", "watermark", "glare", "integrity", "occlusion"],
        "has_header": False,
        "tasks": {
            "watermark": {"type": "binary", "positive": 1},
            "glare": {"type": "binary", "positive": 1},
            "integrity": {"type": "binary", "positive": 1},
            "occlusion": {"type": "binary", "positive": 1},
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data_root.parent

    samples = collect_samples(args.data_root)
    train_samples, val_samples = apply_split(samples, args.val_ratio, args.seed)

    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    write_csv(train_path, train_samples)
    write_csv(val_path, val_samples)

    config_path = write_config(output_dir, args.config_name, args)

    print(f"已生成标签文件: {train_path}（{len(train_samples)} 行），{val_path}（{len(val_samples)} 行）")
    print(f"配置文件: {config_path}")


if __name__ == "__main__":
    main()
