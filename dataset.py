import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class TradingCardDataset(Dataset):
    """eBay球星卡图片分类数据集"""

    def __init__(self, data_dir, transform=None, return_path=False):
        """
        Args:
            data_dir: 数据目录路径（train或val）
            transform: albumentations transform
            return_path: 是否在getitem返回图片路径，便于评测/导出
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_path = return_path
        self.classes = ['card','booklet']  # 按字母顺序
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 收集所有图片路径和标签
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            if not cls_dir.exists():
                continue

            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 使用cv2读取图片（albumentations需要），忽略iCCP警告
        image = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片: {img_path}（可能文件损坏或格式不被当前OpenCV支持）")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        if self.return_path:
            return image, label, img_path
        return image, label


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


def create_dataloaders(data_dir='dataset', batch_size=32, num_workers=4, img_size=224):
    """创建训练集和验证集的DataLoader"""

    # 创建数据集
    train_dataset = TradingCardDataset(
        data_dir=Path(data_dir) / 'train',
        transform=get_train_transform(img_size)
    )

    val_dataset = TradingCardDataset(
        data_dir=Path(data_dir) / 'val',
        transform=get_val_transform(img_size)
    )

    # 创建DataLoader
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

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"类别: {train_dataset.classes}")
    print(f"类别映射: {train_dataset.class_to_idx}")

    return train_loader, val_loader, train_dataset.classes
