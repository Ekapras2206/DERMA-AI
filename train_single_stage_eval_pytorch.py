"""
train_single_stage_eval_pytorch_final.py
Single-stage EfficientNetB0 training for HAM10000 skin lesion classification (PyTorch, GPU-enabled).

✅ Features:
- Full GPU training (CUDA 12.x compatible)
- Balanced class weights
- Patient-wise split
- Data augmentation + normalization
- Early stopping + ReduceLROnPlateau
- Save best model automatically
- Training visualization (accuracy & loss)
- Evaluation with confusion matrix + classification report
- GPU memory usage display per epoch
"""

import os
import random
import copy
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =====================
# CONFIG
# =====================
DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
META_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.tab")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]
SEED = 42
NUM_WORKERS = 0  # 🔧 Set ke 0 jika di Windows untuk hindari multiprocessing error

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =====================
# DEVICE SETUP
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_info():
    print(f"🖥 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version (PyTorch build): {torch.version.cuda}")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
        print(f"   Memory Reserved: {torch.cuda.memory_reserved(0)/1024**2:.1f} MB")

print_device_info()

# =====================
# UTILS
# =====================
def find_image_filename(image_id):
    for ext in (".jpg", ".jpeg", ".png"):
        cand = os.path.join(IMG_DIR, image_id + ext)
        if os.path.exists(cand):
            return cand
    return None

def print_gpu_stats():
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        util = out.decode().strip().splitlines()[0].strip() + "%"
    except Exception:
        util = "N/A"
    print(f"   [GPU] Util: {util} | Alloc: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")

# =====================
# LOAD METADATA
# =====================
if META_PATH.endswith(".tab") or META_PATH.endswith(".tsv"):
    meta = pd.read_csv(META_PATH, sep="\t")
else:
    meta = pd.read_csv(META_PATH)

meta["filepath"] = meta["image_id"].apply(find_image_filename)
meta = meta.dropna(subset=["filepath"]).reset_index(drop=True)
meta = meta[meta["dx"].isin(CLASS_NAMES)].reset_index(drop=True)
label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
meta["label"] = meta["dx"].map(label_to_idx)

print("\n📊 Data per class:\n", meta["dx"].value_counts())

# =====================
# TRANSFORMS & DATASET
# =====================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.08, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])

# =====================
# MODEL SETUP
# =====================
try:
    effnet = models.efficientnet_b0(pretrained=True)
    in_features = effnet.classifier[1].in_features
    effnet.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, len(CLASS_NAMES))
    )
    model = effnet
except Exception as e:
    print("Fallback to timm EfficientNet:", e)
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(CLASS_NAMES))

model = model.to(device)
print("\n✅ Model loaded to:", device)

# =====================
# TRAINING FUNCTIONS
# =====================
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, current_loss, model):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total

def eval_model(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += inputs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
    return total_loss / max(total, 1), correct / max(total, 1), np.array(y_pred), np.array(y_true)

# =====================
# MAIN TRAINING
# =====================
if __name__ == "__main__":
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, test_idx = next(gss.split(meta, groups=meta["lesion_id"]))
    train_meta, test_meta = meta.iloc[train_idx].reset_index(drop=True), meta.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=SEED)
    train_idx2, val_idx2 = next(gss2.split(train_meta, groups=train_meta["lesion_id"]))
    train_meta, val_meta = train_meta.iloc[train_idx2].reset_index(drop=True), train_meta.iloc[val_idx2].reset_index(drop=True)

    print(f"\n📁 Split sizes → Train: {len(train_meta)}, Val: {len(val_meta)}, Test: {len(test_meta)}")

    train_loader = DataLoader(HAM10000Dataset(train_meta, train_transforms),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(HAM10000Dataset(val_meta, val_transforms),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(HAM10000Dataset(test_meta, val_transforms),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    class_weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=train_meta["label"].values)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopper = EarlyStopping(patience=7)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = np.inf
    best_path = os.path.join(MODEL_DIR, "best_model.pth")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n📘 Epoch {epoch}/{EPOCHS}")
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"⏱ Epoch time: {time.time() - start:.1f}s")

        print_gpu_stats()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, best_path)
            print("💾 Best model saved to", best_path)

        if early_stopper.step(val_loss, model):
            print("⏹ Early stopping triggered.")
            break

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("🔁 Loaded best model from", best_path)

    final_path = os.path.join(MODEL_DIR, "derma_ai_final.pt")
    torch.save(model.state_dict(), final_path)
    print("\n✅ Final model saved to", final_path)

    # =====================
    # TEST EVAL
    # =====================
    test_loss, test_acc, y_pred, y_true = eval_model(model, test_loader, criterion)
    print(f"\n📊 Test Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig("training_plot.png")
    plt.tight_layout()
    plt.show()
