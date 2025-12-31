"""
使用已训练好的多头/二分类模型做推理，可对单张图片或整个目录批量分类，
并按预测标签将图片复制到输出目录的子目录中，同时生成结果 CSV。

示例：
    uv run python infer.py \
        --config label-images/dataset_config.json \
        --checkpoint checkpoints/best.pth \
        --input some_image_or_dir \
        --output infer_output \
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn

from dataset import get_val_transform, load_dataset_config


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class MultiTaskEfficientNet(nn.Module):
    """与训练时一致的骨干+多头结构"""

    def __init__(self, backbone: str, task_names: List[str], pretrained: bool = False):
        super().__init__()
        self.task_names = task_names
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.heads = nn.ModuleDict({t: nn.Linear(feat_dim, 1) for t in task_names})

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {t: self.heads[t](feat).squeeze(1) for t in self.task_names}


def build_model(config: Dict, checkpoint_path: Path, device: torch.device, model_name: str | None):
    ckpt = torch.load(checkpoint_path, map_location=device)
    task_names = ckpt.get("task_names") or list(config["tasks"].keys())
    backbone = model_name or ckpt.get("model_name", "efficientnet_b0")
    model = MultiTaskEfficientNet(backbone=backbone, task_names=task_names, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, task_names


def iter_images(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() in ALLOWED_SUFFIXES:
        yield input_path
    elif input_path.is_dir():
        for p in input_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES:
                yield p
    else:
        raise FileNotFoundError(f"输入不存在或格式不支持: {input_path}")


def predict_image(
    img_path: Path,
    model: nn.Module,
    device: torch.device,
    transform,
    cfg: Dict,
    task_names: List[str],
    threshold: float,
) -> Dict[str, Dict[str, float]]:
    image = cv2.imread(str(img_path), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    results = {}
    for t in task_names:
        prob = torch.sigmoid(outputs[t]).item()
        task_cfg = cfg["tasks"][t]
        pos_name = str(task_cfg.get("positive", 1))
        neg_name = str(task_cfg.get("negative", 0))
        pred_label = pos_name if prob >= threshold else neg_name
        results[t] = {"prob": prob, "pred": pred_label}
    return results


def save_result(img_path: Path, results: Dict[str, Dict[str, float]], output_dir: Path, cfg: Dict):
    """
    将图片复制到 output_dir/task_name/pred_label/ 下。
    为避免重名，保持原文件名；若多任务会复制多份。
    """
    for task, info in results.items():
        dest_dir = output_dir / task / info["pred"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)


def write_csv(rows: List[List[str]], output_csv: Path, header: List[str]):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="多头/二分类模型推理与文件归类")
    ap.add_argument("--config", type=str, required=True, help="dataset_config.json 路径")
    ap.add_argument("--checkpoint", type=str, required=True, help="训练好的权重路径(best.pth)")
    ap.add_argument("--input", type=str, required=True, help="待预测的图片路径或目录")
    ap.add_argument("--output", type=str, default="infer_output", help="输出目录（默认 infer_output）")
    ap.add_argument("--model", type=str, default=None, help="可选，覆盖 checkpoint 中的模型名")
    ap.add_argument("--threshold", type=float, default=0.5, help="二分类阈值，默认 0.5")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu，不填自动检测")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_dataset_config(Path(args.config))

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"使用设备: {device}")

    model, task_names = build_model(cfg, Path(args.checkpoint), device, args.model)
    transform = get_val_transform()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    rows = []

    for img_path in iter_images(input_path):
        results = predict_image(img_path, model, device, transform, cfg, task_names, args.threshold)
        save_result(img_path, results, output_dir, cfg)
        row = [str(img_path)]
        for t in task_names:
            row.extend([results[t]["pred"], f"{results[t]['prob']:.6f}"])
        rows.append(row)

    header = ["path"]
    for t in task_names:
        header.extend([f"{t}_pred", f"{t}_prob"])
    write_csv(rows, output_dir / "predictions.csv", header)

    print(f"完成。结果 CSV: {output_dir / 'predictions.csv'}")
    print(f"分类后的文件位于: {output_dir}")


if __name__ == "__main__":
    main()
