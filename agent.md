# 项目协作说明（给自动化助手/协作者）

## 基本要求
- 所有非代码说明使用中文。
- 依赖管理统一使用 `uv`。
- 目录结构或脚本有调整时，请同步更新 `README.md` 与本文件。

## 项目现状与约定
- 训练脚本：`train.py`（timm EfficientNet + MLflow）
- 数据准备：`prepare_data.py`（从 `label-images/` 切分到 `dataset/`）
- 数据压缩：`compress_dataset.py`（输出 `dataset_compressed/`）
- 推理服务：`predict_api.py`（Flask，使用 `checkpoints/best.pth`）
- 类别固定：`back/detail/front/lot/other`

## 目录约定
- 原始标注数据：`label-images/<class>/`
- 切分数据：`dataset/{train,val}/<class>/`
- 压缩数据：`dataset_compressed/{train,val}/<class>/`
- 模型权重：`checkpoints/{latest,best}.pth`

## 变更指引
- 若新增类别或调整标签顺序：
  - 同步修改 `dataset.py`、`prepare_data.py`、`train.py`、`predict_api.py`。
- 若训练数据目录变更：
  - 同步修改 `train.py` 的 `data_dir`，并更新 README。
- 若新增脚本或 API：
  - 请在 README 增补用途与调用方式。
