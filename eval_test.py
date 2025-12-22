import torch, timm, shutil
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import TradingCardDataset, get_val_transform
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ckpt = torch.load("checkpoints/best.pth", map_location="cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TradingCardDataset("test", transform=get_val_transform(224), return_path=True)
# 单进程便于捕获具体报错；如需提速可把 num_workers 调回 >0
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# 可选：仅复制分类错误的图片到 output_tail/pred_<p>_true_<t>/ 下
SAVE_MISCLS = True
OUTPUT_DIR = Path("output_tail")
if SAVE_MISCLS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_labels, all_preds = [], []
with torch.no_grad():
    for images, labels, paths in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        all_labels += labels.cpu().tolist()
        all_preds += preds.cpu().tolist()

        if SAVE_MISCLS:
            for src_path, pred, true_label in zip(paths, preds, labels):
                if pred.item() == true_label.item():
                    continue
                pred_name = dataset.classes[pred.item()]
                true_name = dataset.classes[true_label.item()]
                src = Path(src_path)
                dst_dir = OUTPUT_DIR / f"pred_{pred_name}_true_{true_name}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"复制失败 {src} -> {dst}: {e}")

print("Test Acc:", accuracy_score(all_labels, all_preds))
print("Report:\n", classification_report(all_labels, all_preds, target_names=dataset.classes))
print("Confusion:\n", confusion_matrix(all_labels, all_preds))
