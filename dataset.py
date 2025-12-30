import json
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


# =======================
# 配置与解析
# =======================

def load_dataset_config(config_path: Path) -> Dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    # 必需字段校验
    required_keys = ["images_root", "train_label", "val_label", "label_columns", "has_header", "tasks"]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"配置缺少字段: {k}")
    return cfg


def _parse_row_to_sample(row: List[str], cfg: Dict) -> Dict:
    """将一行 CSV 转换为样本 dict"""
    cols = cfg["label_columns"]
    if len(row) != len(cols):
        raise ValueError(f"CSV 列数 {len(row)} 与 label_columns {len(cols)} 不一致")
    item = {col: row[i] for i, col in enumerate(cols)}
    return item


# =======================
# 数据增强
# =======================

def get_train_transform(img_size=224):
    """训练集数据增强"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transform(img_size=224):
    """验证集数据增强（仅resize和normalize）"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


# =======================
# 数据集类
# =======================

class MultiTaskDataset(Dataset):
    """根据 dataset_config.json + 标签 CSV 读取多任务/二分类数据"""

    def __init__(self, cfg: Dict, split: str, transform=None, return_path: bool = False):
        """
        Args:
            cfg: load_dataset_config 返回的字典
            split: 'train' 或 'val'
            transform: albumentations transform
            return_path: 是否返回图片路径
        """
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.return_path = return_path

        images_root = Path(cfg["images_root"])
        label_file = Path(cfg["train_label"] if split == "train" else cfg["val_label"])
        if not label_file.exists():
            raise FileNotFoundError(f"{split} 标签文件不存在: {label_file}")

        self.samples: List[Dict] = []
        with label_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if idx == 0 and cfg.get("has_header", False):
                    # 跳过表头
                    continue
                if not line:
                    continue
                row = line.split(",")
                item = _parse_row_to_sample(row, cfg)

                img_path = images_root / item["path"]
                if not img_path.exists():
                    raise FileNotFoundError(f"图片不存在: {img_path}")

                # 生成标签 dict
                targets = {}
                for task_name, task_cfg in cfg["tasks"].items():
                    raw_value = item.get(task_name)
                    if raw_value is None:
                        # 允许 label 列名与 task 名不一致？目前要求一致
                        raise ValueError(f"标签缺少列: {task_name}")

                    # 二分类映射到 0/1
                    positive = task_cfg.get("positive", 1)
                    negative = task_cfg.get("negative", 0)
                    try:
                        # 优先尝试数字
                        num = float(raw_value)
                        targets[task_name] = 1.0 if num == float(positive) else 0.0
                    except ValueError:
                        # 字符串比较
                        targets[task_name] = 1.0 if raw_value == str(positive) else 0.0

                    # 如显式 negative 匹配，则置 0
                    if raw_value == str(negative):
                        targets[task_name] = 0.0

                self.samples.append({
                    "path": str(img_path),
                    "targets": targets,
                })

        if not self.samples:
            raise RuntimeError(f"{split} 标签为空: {label_file}")

        self.task_names = list(cfg["tasks"].keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["path"]
        targets = sample["targets"]

        image = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # 将 targets 转为 tensor，顺序固定为 self.task_names
        target_vec = torch.tensor([targets[t] for t in self.task_names], dtype=torch.float32)

        if self.return_path:
            return image, target_vec, img_path
        return image, target_vec


# =======================
# DataLoader 构建
# =======================

def create_dataloaders_from_config(
    config_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
):
    cfg = load_dataset_config(Path(config_path))

    train_dataset = MultiTaskDataset(
        cfg=cfg,
        split="train",
        transform=get_train_transform(img_size),
    )
    val_dataset = MultiTaskDataset(
        cfg=cfg,
        split="val",
        transform=get_val_transform(img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, train_dataset.task_names, cfg
