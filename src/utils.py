# =============================================================
#  utils.py — Preprocessing, Augmentation & Label Helpers
#  Traffic Sign Recognition System | TrafficNet
# =============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ── Constants ────────────────────────────────────────────────
IMG_SIZE    = (48, 48)          # Model input resolution
NUM_CLASSES = 43                # GTSRB class count
BATCH_SIZE  = 64
EPOCHS      = 30

# ── GTSRB Class Labels ────────────────────────────────────────
CLASS_NAMES = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing (vehicles over 3.5t)",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve (left)",
    20: "Dangerous curve (right)",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows (right)",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing (3.5t)",
}

# ── Class Category Groups (for color coding) ─────────────────
CLASS_CATEGORIES = {
    "prohibitory": list(range(0, 10)) + [15, 16, 17],
    "mandatory":   list(range(33, 43)),
    "warning":     list(range(18, 32)),
    "priority":    [11, 12, 13, 14, 32],
}

CATEGORY_COLORS = {
    "prohibitory": "#FF4757",
    "mandatory":   "#2ED573",
    "warning":     "#FFA502",
    "priority":    "#1E90FF",
    "unknown":     "#A4B0BE",
}

def get_category(class_id: int) -> str:
    for cat, ids in CLASS_CATEGORIES.items():
        if class_id in ids:
            return cat
    return "unknown"

def get_category_color(class_id: int) -> str:
    return CATEGORY_COLORS[get_category(class_id)]

# ── Image Preprocessing ───────────────────────────────────────
def preprocess_image(img, target_size=IMG_SIZE):
    """Resize, convert BGR→RGB if needed, normalize to [0,1]."""
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None:
        raise ValueError("Image not found or unreadable.")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_for_model(img, target_size=IMG_SIZE):
    """Returns (1, H, W, 3) batch-ready tensor."""
    img = preprocess_image(img, target_size)
    return np.expand_dims(img, axis=0)

# ── Data Augmentation ─────────────────────────────────────────
def build_train_datagen():
    return ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.6, 1.4],
        horizontal_flip=False,          # Signs are NOT symmetric
        fill_mode='nearest',
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )

def build_val_datagen():
    return ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )

# ── GTSRB Dataset Loader ──────────────────────────────────────
def load_gtsrb_from_folder(data_dir, target_size=IMG_SIZE):
    """
    Load GTSRB images from a folder structure:
      data_dir/
        00000/  (class 0)
        00001/  (class 1)
        ...
    Returns X (N, H, W, 3) float32, y (N,) int arrays.
    """
    from tqdm import tqdm
    images, labels = [], []
    for class_id in tqdm(range(NUM_CLASSES), desc="Loading classes"):
        class_dir = os.path.join(data_dir, f"{class_id:05d}")
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.ppm', '.jpg', '.jpeg')):
                continue
            fpath = os.path.join(class_dir, fname)
            img = preprocess_image(fpath, target_size)
            if img is not None:
                images.append(img)
                labels.append(class_id)
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y

# ── Visualization Helpers ─────────────────────────────────────
def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0D1117')

    for ax in axes:
        ax.set_facecolor('#161B22')
        ax.tick_params(colors='#C9D1D9')
        ax.xaxis.label.set_color('#C9D1D9')
        ax.yaxis.label.set_color('#C9D1D9')
        ax.title.set_color('#F0F6FC')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363D')

    # Accuracy
    axes[0].plot(history.history['accuracy'],     color='#58A6FF', linewidth=2, label='Train')
    axes[0].plot(history.history['val_accuracy'], color='#3FB950', linewidth=2, label='Val',   linestyle='--')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(facecolor='#161B22', labelcolor='#C9D1D9')

    # Loss
    axes[1].plot(history.history['loss'],     color='#FF7B72', linewidth=2, label='Train')
    axes[1].plot(history.history['val_loss'], color='#FFA657', linewidth=2, label='Val',   linestyle='--')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(facecolor='#161B22', labelcolor='#C9D1D9')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 18))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.1, linecolor='#30363D',
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted Label', color='#C9D1D9', fontsize=12)
    ax.set_ylabel('True Label',      color='#C9D1D9', fontsize=12)
    ax.set_title('Confusion Matrix — TrafficNet', color='#F0F6FC', fontsize=16, fontweight='bold')
    ax.tick_params(colors='#C9D1D9')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0D1117')
    plt.show()

def show_sample_predictions(model, X, y, n=16, save_path=None):
    indices = np.random.choice(len(X), n, replace=False)
    preds = np.argmax(model.predict(X[indices]), axis=1)
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle('Sample Predictions — TrafficNet', color='#F0F6FC', fontsize=16, fontweight='bold')
    for ax, idx, pred in zip(axes.flat, indices, preds):
        ax.imshow(X[idx])
        correct = (pred == y[idx])
        color   = '#3FB950' if correct else '#FF7B72'
        ax.set_title(f"True: {CLASS_NAMES[y[idx]][:18]}\nPred: {CLASS_NAMES[pred][:18]}",
                     fontsize=7, color=color)
        ax.set_facecolor('#161B22')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0D1117')
    plt.show()
