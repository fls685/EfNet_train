import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare_dataset(source_dir='label-images', output_dir='dataset', train_ratio=0.9, seed=42):
    """
    将标注数据按9:1划分为训练集和验证集

    Args:
        source_dir: 源数据目录，包含front/back/detail/other四个子目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'

    classes = ['front', 'back', 'detail', 'other', 'lot']

    print(f"准备数据集，训练集比例: {train_ratio:.1%}")
    print("=" * 60)

    for cls in classes:
        # 获取所有图片文件
        cls_dir = source_path / cls
        if not cls_dir.exists():
            print(f"警告: {cls_dir} 不存在，跳过")
            continue

        # 获取所有图片文件（排除._开头的隐藏文件）
        image_files = [
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            and not f.name.startswith('._')
        ]

        if not image_files:
            print(f"警告: {cls} 类别没有找到图片文件")
            continue

        # 划分训练集和验证集
        train_files, val_files = train_test_split(
            image_files,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )

        # 创建目标目录
        train_cls_dir = train_dir / cls
        val_cls_dir = val_dir / cls
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)

        # 复制文件到训练集
        print(f"\n{cls}: {len(image_files)} 张图片 -> 训练集: {len(train_files)}, 验证集: {len(val_files)}")

        for src_file in tqdm(train_files, desc=f"  复制 {cls} 训练集"):
            dst_file = train_cls_dir / src_file.name
            shutil.copy2(src_file, dst_file)

        # 复制文件到验证集
        for src_file in tqdm(val_files, desc=f"  复制 {cls} 验证集"):
            dst_file = val_cls_dir / src_file.name
            shutil.copy2(src_file, dst_file)

    print("\n" + "=" * 60)
    print("数据集准备完成！")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")

    # 统计信息
    print("\n最终统计:")
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        split_dir = output_path / split
        for cls in classes:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                count = len([f for f in cls_dir.iterdir() if f.is_file()])
                print(f"  {cls}: {count} 张")

if __name__ == '__main__':
    prepare_dataset()
