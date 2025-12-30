# 项目协作说明（给自动化助手/协作者）

## 基本要求
- 所有非代码说明使用中文。
- 依赖管理统一使用 `uv`。
- 目录结构或脚本有调整时，请同步更新 `README.md` 与本文件。

## 项目现状与约定
- 训练脚本：`train.py`（timm EfficientNet + MLflow，依据 `dataset_config.json` 支持单任务二分类或多头多标签）
- 数据准备：`prepare_data.py`（保留 card/booklet 目录切分流程，可为二分类生成目录数据）
- 数据压缩：`compress_dataset.py`（输出 `dataset_compressed/`）
- 多标签标签生成：`make_multilabel_csv.py`（从 watermark/glare/integrity/occlusion/source 目录生成 train.csv/val.csv 与 dataset_config.json，无表头）
- 推理服务：`predict_api.py`（Flask，使用 `checkpoints/best.pth`）
- 其他：`main.py` 为占位入口，可自行扩展

## 目录约定
- 原始标注数据：可为 `label-images/<class>/`（二分类旧流程）或多标签原始目录。
- 切分数据：`dataset/{train,val}/`（文件平铺或子目录均可，最终以 CSV 中的 path 为准）
- 压缩数据：`dataset_compressed/{train,val}/`（如使用压缩数据，生成 CSV 时路径需对应）
- 模型权重：`checkpoints/{latest,best}.pth`

## 变更指引
- 新增/调整任务或标签列：更新 `dataset_config.json` 与相应 CSV，无需改代码；同步更新 README/agent 说明。
- 若继续使用二分类目录流程：保持 `prepare_data.py` 输出，另行生成对应的 CSV 与配置。
- 若训练数据目录变更：确保 `dataset_config.json` 中的 `images_root` 与 CSV 路径一致。
- 若新增脚本或 API：请在 README 增补用途与调用方式。
- 若增加/删除文件或目录：在 README 的“文件与脚本速览”更新描述，保持与仓库一致。
