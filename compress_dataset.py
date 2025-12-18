"""
预处理脚本：将数据集图片压缩到合适尺寸，加速训练（多线程版本）
"""
import cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def compress_single_image(args):
    """压缩单张图片"""
    img_path, target_path_img, max_size, quality = args

    try:
        # 读取图片
        img = cv2.imread(str(img_path), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)

        if img is None:
            return None, None, f"无法读取 {img_path}"

        # 原始大小
        h, w = img.shape[:2]
        size_before = img_path.stat().st_size

        # 计算缩放比例（如果图片大于max_size才缩放）
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 保存压缩图片
        cv2.imwrite(
            str(target_path_img),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        size_after = target_path_img.stat().st_size
        return size_before, size_after, None

    except Exception as e:
        return None, None, f"处理 {img_path} 失败: {str(e)}"


def compress_images(source_dir='dataset', target_dir='dataset_compressed', max_size=512, quality=95, num_workers=16):
    """
    压缩图片到指定最大尺寸（多线程版本）

    Args:
        source_dir: 源数据集目录
        target_dir: 压缩后的目标目录
        max_size: 最大边长（保持宽高比）
        quality: JPEG质量 (0-100)
        num_workers: 线程数
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    total_size_before = 0
    total_size_after = 0

    print(f"使用 {num_workers} 个线程并行压缩...")

    # 遍历train和val
    for split in ['train', 'val']:
        split_src = source_path / split
        split_dst = target_path / split

        if not split_src.exists():
            continue

        print(f"\n处理 {split} 集...")

        # 遍历所有类别
        for cls_dir in split_src.iterdir():
            if not cls_dir.is_dir():
                continue

            cls_name = cls_dir.name
            target_cls_dir = split_dst / cls_name
            target_cls_dir.mkdir(parents=True, exist_ok=True)

            # 获取所有图片
            images = [f for f in cls_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

            # 准备任务参数
            tasks = [
                (img_path, target_cls_dir / img_path.name, max_size, quality)
                for img_path in images
            ]

            # 多线程处理
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(compress_single_image, task): task for task in tasks}

                # 使用tqdm显示进度
                with tqdm(total=len(tasks), desc=f"  {cls_name}") as pbar:
                    for future in as_completed(futures):
                        size_before, size_after, error = future.result()

                        if error:
                            print(f"\n警告: {error}")
                        else:
                            total_size_before += size_before
                            total_size_after += size_after

                        pbar.update(1)

    # 统计
    print("\n" + "="*60)
    print("压缩完成！")
    print(f"原始大小: {total_size_before / 1024 / 1024:.2f} MB")
    print(f"压缩后大小: {total_size_after / 1024 / 1024:.2f} MB")
    print(f"压缩率: {(1 - total_size_after / total_size_before) * 100:.1f}%")
    print(f"目标目录: {target_path}")
    print("="*60)


if __name__ == '__main__':
    # 使用CPU核心数
    num_workers = min(os.cpu_count(), 32)  # 最多32线程

    compress_images(
        source_dir='dataset',
        target_dir='dataset_compressed',
        max_size=512,      # 最大边长512，训练时再resize到224
        quality=95,        # 高质量保存
        num_workers=num_workers
    )
