import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import numpy as np
from tqdm import tqdm 

# ======================= CONFIG =======================
DATA_DIR = "data"           
BATCH_SIZE = 16
IMG_SIZE = 224
EPOCHS = 20
PATIENCE = 5
CHECKPOINT_PATH = "checkpoint.pt"
BEST_MODEL_PATH = "derma_ai_best.pt"
UNFREEZE_EPOCH = 5  
NUM_WORKERS = 0     

# Cek dan set device ke GPU (RTX Anda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Menggunakan device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ======================= TRANSFORMS =======================
# (Transformasi data tetap sama)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================= CUSTOM DATASET =======================
class CustomSkinDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform_map=None):
        self.samples = []
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform_map = transform_map if transform_map else {}
        
        for cls_name, idx in class_to_idx.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_folder, fname), idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        cls_name = self.idx_to_class[label]
        transform = self.transform_map.get(cls_name, val_transforms) 
        
        img = transform(img)
        return img, label

# ======================= TRAINING FUNCTION =======================
# (Fungsi training model tetap sama, karena logikanya sudah benar)
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                writer, num_epochs, patience, start_epoch=0, best_acc=0.0):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    is_unfrozen = any(p.requires_grad for p in model.features.parameters()) and (start_epoch >= UNFREEZE_EPOCH)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch+1}/{start_epoch + num_epochs}")
        print("-" * 20)
        
        if (epoch + 1) == UNFREEZE_EPOCH and not is_unfrozen:
            print(f"\n🔓 Unfreeze seluruh backbone mulai epoch {epoch+1}")
            for param in model.features.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            is_unfrozen = True
            print("🔧 Optimizer dan Scheduler diupdate untuk fine-tuning backbone")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"Phase {phase.upper()}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

            print(f"\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                target_names = dataloaders['train'].dataset.idx_to_class.values()
                
                print("Confusion Matrix:")
                print(confusion_matrix(all_labels, all_preds))
                print(classification_report(all_labels, all_preds, target_names=list(target_names), zero_division=0))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, BEST_MODEL_PATH)
                    print("✅ Model terbaik disimpan!")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Simpan checkpoint tiap akhir epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
            "is_unfrozen": is_unfrozen 
        }, CHECKPOINT_PATH)

        if epochs_no_improve >= patience:
            print("\n⏹️ Early stopping aktif!")
            break

    time_elapsed = time.time() - since
    print(f"\nTraining selesai dalam {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

# ======================= MAIN EXECUTION (FINAL DENGAN PENAMBAHAN KELAS) =======================
if __name__ == "__main__":
    writer = SummaryWriter(log_dir="runs/derma_ai_training")

    # 1. Ambil class_to_idx. NUM_CLASSES akan menjadi 10 (9 lama + scabies).
    tmp_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train')) 
    class_to_idx = tmp_dataset.class_to_idx
    NUM_CLASSES = len(class_to_idx)
    print(f"Ditemukan {NUM_CLASSES} kelas: {class_to_idx}")
    
    # 2. Inisialisasi Dataloaders (Transform Map tetap sama)
    train_transform_map = {k: train_transforms for k in class_to_idx.keys()}
    val_transform_map = {k: val_transforms for k in class_to_idx.keys()}
    
    full_train_dataset = CustomSkinDataset(os.path.join(DATA_DIR, 'train'), class_to_idx, train_transform_map)
    full_val_dataset = CustomSkinDataset(os.path.join(DATA_DIR, 'val'), class_to_idx, val_transform_map)
    
    print(f"✔️ Dataset TRAIN terdeteksi: {len(full_train_dataset)} gambar")
    print(f"✔️ Dataset VAL terdeteksi: {len(full_val_dataset)} gambar")

    # 3. Hitung Class Weight
    # (Logika ini sudah benar dan otomatis menangani 10 kelas)
    print("⏳ Menghitung Class Weights (Ini mungkin memakan waktu)...")
    temp_loader = DataLoader(full_train_dataset, batch_size=1024, shuffle=False, num_workers=NUM_WORKERS)
    
    labels_list = []
    for _, labels in tqdm(temp_loader, desc="Collecting labels"):
        labels_list.extend(labels.cpu().numpy())
    
    labels = np.array(labels_list)
    class_counts = np.bincount(labels)
    
    NUM_CLASSES_COUNTED = len(class_counts)
    if NUM_CLASSES_COUNTED < NUM_CLASSES:
         class_counts = np.pad(class_counts, (0, NUM_CLASSES - NUM_CLASSES_COUNTED), 'constant')

    safe_class_counts = np.where(class_counts > 0, class_counts, 1) 
    class_weights_array = len(labels) / (NUM_CLASSES * safe_class_counts)
    
    class_weights = torch.tensor(class_weights_array, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"⭐ Class Weights (di GPU): {class_weights.cpu().numpy()}")

    # 4. Inisialisasi Dataloaders utama
    dataloaders = {
        "train": DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        "val": DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    }
    dataset_sizes = {"train": len(full_train_dataset), "val": len(full_val_dataset)}


    # 5. Model, Freeze, dan Classifier
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze Backbone Awal
    for param in model.features.parameters():
        param.requires_grad = False

    # >>> BAGIAN KRITIS: Mengganti Classifier 9 -> 10 Kelas <<<
    # Lapisan classifier diganti dengan output NUM_CLASSES yang baru (10)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(device)

    # 6. Checkpoint dan Pemuatan Bobot Model Lama
    start_epoch = 0
    best_acc = 0.0
    is_unfrozen = False
    
    # Inisialisasi optimizer default untuk menjaga konsistensi
    # Menggunakan learning rate tahap fine-tuning karena kita ingin melanjutkan training
    initial_lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if os.path.exists(CHECKPOINT_PATH):
        print("🔄 Memuat bobot model lama dari checkpoint...")
        
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            
            # --- 1. Muat Bobot Model (Backbone) ---
            model_state = model.state_dict()
            old_state = checkpoint["model_state"]
            
            # Hapus bobot classifier lama (9 kelas) dari checkpoint
            if 'classifier.1.weight' in old_state:
                del old_state['classifier.1.weight']
            if 'classifier.1.bias' in old_state:
                del old_state['classifier.1.bias']

            # Muat sisa bobot (Backbone)
            model.load_state_dict(old_state, strict=False)
            
            # --- 2. Update Metadata ---
            # Kita TIDAK mengambil 'is_unfrozen' lama, karena kita mau fine-tuning ulang
            # Kita TIDAK mengambil 'start_epoch' lama, lebih baik mulai dari Epoch 1 untuk fase baru ini
            # start_epoch = checkpoint["epoch"] + 1  <-- JANGAN DIPAKAI
            # best_acc = checkpoint["best_acc"]      <-- JANGAN DIPAKAI (karena akurasi pasti drop dulu)
            
            print("✅ Berhasil memuat Backbone Model (EfficientNet).")
            print("⚠️  Optimizer lama & Epoch di-reset karena jumlah kelas berubah (9 -> 10).")
            print("🚀 Memulai Fine-Tuning baru dari Epoch 1...")

            # --- 3. JANGAN MUAT OPTIMIZER ---
            # optimizer.load_state_dict(checkpoint["optimizer_state"])  <-- HAPUS/COMMENT INI
            # scheduler.load_state_dict(checkpoint["scheduler_state"])  <-- HAPUS/COMMENT INI
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            exit()
    
    # 7. Mulai Training
    model = train_model(model, criterion, optimizer, scheduler, dataloaders,
                        dataset_sizes, writer, num_epochs=EPOCHS, patience=PATIENCE,
                        start_epoch=start_epoch, best_acc=best_acc)

    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print("\n💾 Model akhir disimpan ke derma_ai_best.pt")