"""
train_single_stage_eval.py
Single-stage EfficientNetB0 training for HAM10000 skin lesion classification.

✅ Features:
- Balanced class weights
- Patient-wise split
- Data augmentation + normalization
- Early stopping + ReduceLROnPlateau
- Save best model automatically
- Training visualization (accuracy & loss)
- Evaluation with confusion matrix + classification report
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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

# =====================
# LOAD METADATA
# =====================
if META_PATH.endswith(".tab") or META_PATH.endswith(".tsv"):
    meta = pd.read_csv(META_PATH, sep="\t")
else:
    meta = pd.read_csv(META_PATH)

def find_image_filename(image_id):
    for ext in (".jpg", ".jpeg", ".png"):
        cand = os.path.join(IMG_DIR, image_id + ext)
        if os.path.exists(cand):
            return cand
    return None

meta["filepath"] = meta["image_id"].apply(find_image_filename)
meta = meta.dropna(subset=["filepath"]).reset_index(drop=True)
meta = meta[meta["dx"].isin(CLASS_NAMES)].reset_index(drop=True)

label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
meta["label"] = meta["dx"].map(label_to_idx)

print("Data per class:\n", meta["dx"].value_counts())

# =====================
# SPLIT DATASET (patient-wise)
# =====================
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(meta, groups=meta["lesion_id"]))
train_meta, test_meta = meta.iloc[train_idx], meta.iloc[test_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=42)
train_idx2, val_idx2 = next(gss2.split(train_meta, groups=train_meta["lesion_id"]))
train_meta, val_meta = train_meta.iloc[train_idx2], train_meta.iloc[val_idx2]

print(f"Split sizes → Train: {len(train_meta)}, Val: {len(val_meta)}, Test: {len(test_meta)}")

# =====================
# DATASET PIPELINE
# =====================
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = preprocess_input(img)
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

def make_dataset(paths, labels, batch=32, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(2048)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_meta["filepath"].values, train_meta["label"].values, BATCH_SIZE, True)
val_ds = make_dataset(val_meta["filepath"].values, val_meta["label"].values, BATCH_SIZE, False)
test_ds = make_dataset(test_meta["filepath"].values, test_meta["label"].values, BATCH_SIZE, False)

# =====================
# CLASS WEIGHTS
# =====================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=train_meta["label"].values
)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
print("Class Weights:", class_weight_dict)

# =====================
# MODEL ARCHITECTURE
# =====================
base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = True  # langsung fine-tune seluruh model

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================
# CALLBACKS
# =====================
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model.h5"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
earlystop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=1)

# =====================
# TRAIN MODEL
# =====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# =====================
# SAVE FINAL MODEL
# =====================
model.save(os.path.join(MODEL_DIR, "derma_ai_single_stage_eval.h5"))
print("✅ Saved final model to model/derma_ai_single_stage_eval.h5")

# =====================
# EVALUATE
# =====================
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n📊 Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# =====================
# CONFUSION MATRIX + REPORT
# =====================
print("\n🔍 Evaluating per-class performance...")
y_true = []
y_pred = []

for imgs, labels in test_ds:
    preds = model.predict(imgs)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
print("\nClassification Report:\n", report)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =====================
# VISUALIZE TRAINING
# =====================
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("training_plot.png")
plt.tight_layout()
plt.show()
