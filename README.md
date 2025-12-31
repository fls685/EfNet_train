# EfficientNet 训练项目（支持二分类与多任务多头）

本项目基于 PyTorch + timm 的 EfficientNet，现已改为配置驱动：既可单任务二分类（如 card/booklet），也可多头多标签（如 watermark/glare/integrity/occlusion），并提供数据准备、压缩与推理 API。

> 运行环境：Python >= 3.13；依赖通过 `uv` 管理。

## 文件与脚本速览
- `dataset.py`：读取 `dataset_config.json`，基于 CSV 的多任务/二分类数据集与增强。
- `prepare_data.py`：按 9:1 划分 train/val，输入 `label-images/` 输出 `dataset/`（保留兼容的 card/booklet 二分类流程）。
- `compress_dataset.py`：多线程压缩 train/val 图像，输出 `dataset_compressed/`。
- `make_multilabel_csv.py`：从 `watermark/glare/integrity/occlusion/source` 五个子目录生成无表头的 `train.csv` / `val.csv`，并写出 `dataset_config.json`。
- `train.py`：训练主循环（timm EfficientNet + AdamW + CosineAnnealingLR + MLflow，配置驱动的单/多任务）。
- `predict_api.py`：Flask 推理服务，读取 `checkpoints/best.pth`。
- `main.py`：占位入口，可按需扩展。
- `pyproject.toml` / `uv.lock`：依赖与锁定文件。
- `.gitignore` / `.python-version`：基础环境约定。

## 项目能力
- EfficientNet 训练（timm，支持单/多任务）
- 数据集切分（训练/验证）
- 数据集压缩加速训练
- MLflow 指标记录
- Flask 推理服务
- 多标签 CSV 生成（watermark/glare/integrity/occlusion/source）

## 依赖管理（uv）
本项目使用 `uv` 管理依赖与环境。

```bash
uv sync
```

## 数据准备
- 二分类：可继续使用 `prepare_data.py`（输入 card/booklet 目录，输出 train/val 目录），再用脚本生成对应的无表头 `train.csv` / `val.csv`（列为 `path,label`）。
- 多标签：使用 `make_multilabel_csv.py` 生成 `train.csv` / `val.csv` 以及 `dataset_config.json`。

## 数据压缩（可选）
训练脚本默认使用压缩后的数据目录 `dataset_compressed/`。

```bash
uv run python compress_dataset.py
```

## 多标签 CSV 生成（watermark / glare / integrity / occlusion）
假设原始数据根目录为 `data_raw/`，其下包含五个子目录 `watermark`、`glare`、`integrity`、`occlusion`、`source`（source 为干净样本）。

```bash
uv run python make_multilabel_csv.py --data-root data_raw --val-ratio 0.2
```

输出（默认在 `data_raw` 的同级目录）：
- `train.csv`、`val.csv`：无表头，列顺序为 `path,watermark,glare,integrity,occlusion`
- `dataset_config.json`：记录图片根目录、标签文件路径、列顺序与是否有表头

如需自定义文件名或输出目录，可使用：
```bash
uv run python make_multilabel_csv.py \
  --data-root data_raw \
  --output-dir dataset \
  --train-name train.csv \
  --val-name val.csv \
  --config-name dataset_config.json \
  --val-ratio 0.2
```

## 训练（配置驱动）
1) 准备 `dataset_config.json`  
多头示例：
```json
{
  "images_root": "/abs/path/to/dataset",
  "train_label": "/abs/path/to/train.csv",
  "val_label": "/abs/path/to/val.csv",
  "label_columns": ["path", "watermark", "glare", "integrity", "occlusion"],
  "has_header": false,
  "tasks": {
    "watermark": { "type": "binary", "positive": 1 },
    "glare": { "type": "binary", "positive": 1 },
    "integrity": { "type": "binary", "positive": 1 },
    "occlusion": { "type": "binary", "positive": 1 }
  }
}
```

二分类示例（card/booklet）：
```json
{
  "images_root": "/abs/path/to/dataset",
  "train_label": "/abs/path/to/train.csv",
  "val_label": "/abs/path/to/val.csv",
  "label_columns": ["path", "label"],
  "has_header": false,
  "tasks": {
    "label": { "type": "binary", "positive": "card", "negative": "booklet" }
  }
}
```

2) 运行训练：
```bash
uv run python train.py \
  --config dataset_config.json \
  --model efficientnet_b0 \
  --epochs 30 \
  --batch-size 64 \
  --img-size 224 \
  --lr 1e-3 \
  --checkpoint-dir checkpoints
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
- `train.py`：训练主脚本（配置驱动的单/多任务 EfficientNet + MLflow）
- `dataset.py`：数据集与数据增强定义（读取 dataset_config.json）
- `prepare_data.py`：数据集切分（保留 card/booklet 流程）
- `compress_dataset.py`：图片压缩加速训练
- `make_multilabel_csv.py`：五类目录生成多标签 CSV 与 dataset_config.json
- `infer.py`：推理与文件归档（支持单张或目录，输出预测 CSV 与分标签子目录）
- `infer.py`：加载 checkpoint，对单张或目录做推理并按预测标签归档
- `predict_api.py`：推理服务（Flask）
- `pyproject.toml` / `uv.lock`：依赖管理
- `main.py`：占位入口（当前仅打印欢迎信息，可按需扩展）

## 约定与注意
- 训练数据与任务定义均以 `dataset_config.json` 为准；如切换任务，只需替换 CSV + 配置文件，无需改代码。
- 若使用压缩数据，请在生成 CSV/配置文件时写入压缩后路径，保持一致。
- 若调整目录结构或新增脚本，请同步更新本 README 与 `agent.md`。
