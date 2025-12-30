"""
多任务/二分类训练脚本

特性：
- 读取 dataset_config.json，按配置的列与任务加载标签
- 共享 EfficientNet 骨干，按任务名自动创建独立二分类头
- 适配单任务（二分类）与多任务（多头）统一流程
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import create_dataloaders_from_config, load_dataset_config


# =======================
# 模型定义
# =======================

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, backbone: str, task_names: List[str], pretrained: bool = True):
        super().__init__()
        self.task_names = task_names
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.heads = nn.ModuleDict({
            t: nn.Linear(feat_dim, 1) for t in task_names
        })

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        out = {t: self.heads[t](feat).squeeze(1) for t in self.task_names}
        return out


# =======================
# 训练器
# =======================

class Trainer:
    def __init__(
        self,
        config_path: str,
        model_name: str = "efficientnet_b0",
        img_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 30,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.cfg = load_dataset_config(Path(config_path))
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        # dataloader
        self.train_loader, self.val_loader, self.task_names, _ = create_dataloaders_from_config(
            config_path=config_path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
        )

        # model / loss / optim
        self.model = MultiTaskEfficientNet(
            backbone=model_name,
            task_names=self.task_names,
            pretrained=True,
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        self.best_val_score = 0.0  # 以平均 F1 作为早停指标

    # --------- metric helpers ----------
    @staticmethod
    def _collect_preds(outputs: Dict[str, torch.Tensor], targets: torch.Tensor, task_names: List[str]):
        """将张量转为 numpy 供 metric 计算"""
        probs = {}
        preds = {}
        labels = {}
        targets_np = targets.detach().cpu().numpy()
        for idx, t in enumerate(task_names):
            logit = outputs[t].detach().cpu()
            prob = torch.sigmoid(logit).numpy()
            pred = (prob >= 0.5).astype(np.int32)
            probs[t] = prob
            preds[t] = pred
            labels[t] = targets_np[:, idx]
        return probs, preds, labels

    @staticmethod
    def _compute_metrics(preds: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]) -> Dict[str, float]:
        task_metrics = {}
        f1_list = []
        for t in preds:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels[t], preds[t], average="binary", zero_division=0
            )
            acc = accuracy_score(labels[t], preds[t])
            task_metrics[f"{t}_precision"] = precision
            task_metrics[f"{t}_recall"] = recall
            task_metrics[f"{t}_f1"] = f1
            task_metrics[f"{t}_acc"] = acc
            f1_list.append(f1)
        task_metrics["f1_macro"] = float(np.mean(f1_list))
        return task_metrics

    # --------- epoch loops ----------
    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")

        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = 0.0
            for idx, t in enumerate(self.task_names):
                loss += self.criterion(outputs[t], targets[:, idx])

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # 收集指标
            _, preds_dict, labels_dict = self._collect_preds(outputs, targets, self.task_names)
            all_preds.append(preds_dict)
            all_labels.append(labels_dict)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 汇总指标
        merged_preds = {t: np.concatenate([p[t] for p in all_preds], axis=0) for t in self.task_names}
        merged_labels = {t: np.concatenate([l[t] for l in all_labels], axis=0) for t in self.task_names}
        metrics = self._compute_metrics(merged_preds, merged_labels)
        epoch_loss = running_loss / len(self.train_loader)

        return epoch_loss, metrics

    def validate(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")

        with torch.no_grad():
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)

                loss = 0.0
                for idx, t in enumerate(self.task_names):
                    loss += self.criterion(outputs[t], targets[:, idx])

                running_loss += loss.item()

                _, preds_dict, labels_dict = self._collect_preds(outputs, targets, self.task_names)
                all_preds.append(preds_dict)
                all_labels.append(labels_dict)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        merged_preds = {t: np.concatenate([p[t] for p in all_preds], axis=0) for t in self.task_names}
        merged_labels = {t: np.concatenate([l[t] for l in all_labels], axis=0) for t in self.task_names}
        metrics = self._compute_metrics(merged_preds, merged_labels)
        epoch_loss = running_loss / len(self.val_loader)

        return epoch_loss, metrics

    # --------- checkpoint ----------
    def save_checkpoint(self, epoch: int, val_score: float, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_score": val_score,
            "model_name": self.model_name,
            "task_names": self.task_names,
        }
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型，验证 f1_macro: {val_score:.4f}")

    # --------- main loop ----------
    def train(self):
        run_name = datetime.now().strftime("%Y-%m-%d %H:%M")
        mlflow.log_params({
            "model_name": self.model_name,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "scheduler": "CosineAnnealingLR",
            "optimizer": "AdamW",
            "tasks": ",".join(self.task_names),
        })

        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        for epoch in range(self.num_epochs):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # 记录到 MLflow
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "learning_rate": current_lr}
                | {f"train_{k}": v for k, v in train_metrics.items()}
                | {f"val_{k}": v for k, v in val_metrics.items()},
                step=epoch,
            )

            print(f"\nEpoch {epoch+1}/{self.num_epochs} 总结:")
            print(f"  训练 - Loss: {train_loss:.4f}, f1_macro: {train_metrics['f1_macro']:.4f}")
            print(f"  验证 - Loss: {val_loss:.4f}, f1_macro: {val_metrics['f1_macro']:.4f}")
            for t in self.task_names:
                print(f"    [{t}] P: {val_metrics[f'{t}_precision']:.3f} "
                      f"R: {val_metrics[f'{t}_recall']:.3f} F1: {val_metrics[f'{t}_f1']:.3f}")
            print(f"  学习率: {current_lr:.6f}")

            is_best = val_metrics["f1_macro"] > self.best_val_score
            if is_best:
                self.best_val_score = val_metrics["f1_macro"]
            self.save_checkpoint(epoch, val_metrics["f1_macro"], is_best)
            print("-" * 60)

        mlflow.log_metric("best_val_f1_macro", self.best_val_score)
        print("\n" + "=" * 60)
        print(f"训练完成！最佳验证 f1_macro: {self.best_val_score:.4f}")
        print("=" * 60)


# =======================
# CLI
# =======================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task / binary training with config")
    parser.add_argument("--config", type=str, default="dataset_config.json", help="dataset_config.json 路径")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="timm 模型名")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5500")
    parser.add_argument("--experiment", type=str, default="trading-card-classification")
    return parser.parse_args()


def main():
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=datetime.now().strftime("%Y-%m-%d %H:%M")):
        trainer = Trainer(
            config_path=args.config,
            model_name=args.model,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )
        trainer.train()


if __name__ == "__main__":
    main()
