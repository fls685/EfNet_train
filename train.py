import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from dataset import create_dataloaders
from datetime import datetime


class EfficientNetTrainer:
    """EfficientNet训练器，集成MLflow跟踪"""

    def __init__(
        self,
        model_name='efficientnet_b0',
        num_classes=4,
        img_size=224,
        batch_size=32,
        num_workers=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=50,
        device=None,
        data_dir='dataset',
        checkpoint_dir='checkpoints'
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.data_dir = data_dir
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 创建数据加载器
        self.train_loader, self.val_loader, self.classes = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size
        )

        # 创建模型
        self.model = self._create_model()

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )

        # 最佳验证准确率
        self.best_val_acc = 0.0

    def _create_model(self):
        """创建timm EfficientNet模型"""
        model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=self.num_classes
        )
        model = model.to(self.device)
        print(f"创建模型: {self.model_name}")
        return model

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算epoch指标
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]  ')

        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算指标
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        # 计算每个类别的精确率、召回率、F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'classes': self.classes,
        }

        # 保存最新模型
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # 如果是最佳模型，也保存一份
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型，验证准确率: {val_acc:.4f}")

    def train(self):
        """完整训练流程，集成MLflow"""

        # 生成运行名称：yyyy-MM-dd HH:mm
        run_name = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 开始MLflow运行
        with mlflow.start_run(run_name=run_name):
            # 记录超参数
            mlflow.log_params({
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'img_size': self.img_size,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR',
            })

            print("\n" + "="*60)
            print("开始训练")
            print("="*60)

            for epoch in range(self.num_epochs):
                # 训练
                train_loss, train_acc = self.train_epoch(epoch)

                # 验证
                val_metrics = self.validate(epoch)

                # 更新学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()

                # 记录到MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1'],
                    'learning_rate': current_lr,
                }, step=epoch)

                # 打印epoch总结
                print(f"\nEpoch {epoch+1}/{self.num_epochs} 总结:")
                print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                print(f"  验证 - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
                print(f"  学习率: {current_lr:.6f}")

                # 保存检查点
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']

                self.save_checkpoint(epoch, val_metrics['accuracy'], is_best)

                print("-" * 60)

            # 训练结束，记录最佳准确率
            mlflow.log_metric('best_val_accuracy', self.best_val_acc)

            print("\n" + "="*60)
            print(f"训练完成！最佳验证准确率: {self.best_val_acc:.4f}")
            print("="*60)


def main():
    """主函数"""
    # 设置MLflow跟踪URI
    mlflow.set_tracking_uri("http://127.0.0.1:5500")
    mlflow.set_experiment("trading-card-classification")

    # 创建训练器
    trainer = EfficientNetTrainer(
        model_name='efficientnet_b0',
        num_classes=5,  # 5分类: back, detail, front, lot, other
        img_size=224,
        batch_size=128,  # RTX 4090 有其他进程占用，保持128
        num_workers=8,   # 44核CPU，8个worker绰绰有余
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=50,
        data_dir='dataset_compressed',  # 使用压缩后的数据集
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
