# EfficientNet 二分类训练项目

本项目基于 PyTorch + timm 的 EfficientNet 做图像二分类训练（普通卡 `card` vs. 折页卡 `booklet`），并提供数据准备、压缩与推理 API。

> 运行环境：Python >= 3.13；依赖通过 `uv` 管理。

## 文件与脚本速览
- `dataset.py`：数据集类、数据增强与 `create_dataloaders`。
- `prepare_data.py`：按 9:1 划分 train/val，输入 `label-images/` 输出 `dataset/`（card/booklet 二分类）。
- `compress_dataset.py`：多线程压缩 train/val 图像，输出 `dataset_compressed/`。
- `train.py`：训练主循环（timm EfficientNet + AdamW + CosineAnnealingLR + MLflow 记录，二分类）。
- `predict_api.py`：Flask 推理服务，读取 `checkpoints/best.pth`。
- `main.py`：占位入口，可按需扩展。
- `pyproject.toml` / `uv.lock`：依赖与锁定文件。
- `.gitignore` / `.python-version`：基础环境约定。

## 项目能力
- EfficientNet 训练（timm）
- 数据集切分（训练/验证）
- 数据集压缩加速训练
- MLflow 指标记录
- Flask 推理服务

## 依赖管理（uv）
本项目使用 `uv` 管理依赖与环境。

```bash
uv sync
```

## 数据准备
默认数据目录约定如下（仅两类）：

```text
label-images/
├── card/
└── booklet/
```

切分训练/验证集：

```bash
uv run python prepare_data.py
```

切分后目录：

```text
dataset/
├── train/
│   ├── card/ booklet/
└── val/
    ├── card/ booklet/
```

## 数据压缩（可选）
训练脚本默认使用压缩后的数据目录 `dataset_compressed/`。

```bash
uv run python compress_dataset.py
```

## 训练
```bash
uv run python train.py
```

训练产物：
- `checkpoints/latest.pth`
- `checkpoints/best.pth`

提示：`train.py` 默认连接 MLflow 地址 `http://127.0.0.1:5500`，如需记录请先确保服务可用。

## 推理 API
```bash
uv run python predict_api.py
```

接口：
- `GET /health`
- `POST /predict` 传入 `{ "url": "..." }`
- `POST /predict_batch` 传入 `{ "urls": ["...", "..."] }`

## 快速排错
- 如果训练/推理报找不到类别，确认 `dataset.py` 的类别列表（card/booklet）与数据目录一致。
- 预测报找不到 checkpoint，先运行训练或将权重放到 `checkpoints/best.pth`。
- MLflow 连接失败：检查 `train.py` 中 `mlflow.set_tracking_uri` 是否可达。

## 代码结构（当前）
- `train.py`：训练主脚本（EfficientNet + MLflow）
- `dataset.py`：数据集与数据增强定义
- `prepare_data.py`：数据集切分
- `compress_dataset.py`：图片压缩加速训练
- `predict_api.py`：推理服务（Flask）
- `pyproject.toml` / `uv.lock`：依赖管理
- `main.py`：占位入口（当前仅打印欢迎信息，可按需扩展）

## 约定与注意
- 类别固定为 `card/booklet`（如需改动，同步修改 `dataset.py` 与数据准备脚本）。
- 训练默认使用 `dataset_compressed/`，如未生成请修改 `train.py` 的 `data_dir`。
- 若调整目录结构或新增脚本，请同步更新本 README。
