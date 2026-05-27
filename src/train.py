# =============================================================
#  train.py — TrafficNet CNN Training Pipeline
#  Traffic Sign Recognition System
# =============================================================
#
#  Usage:
#    python src/train.py --data_dir dataset/train --epochs 30
#
#  Output:
#    models/trafficnet.h5
#    models/training_history.json
#    models/training_curves.png
# =============================================================

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

from src.utils import (
    IMG_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS,
    load_gtsrb_from_folder, build_train_datagen,
    plot_training_history, plot_confusion_matrix,
    show_sample_predictions
)

# ── GPU Config ────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU detected: {len(gpus)} device(s)")
else:
    print("⚠️  No GPU detected, using CPU")

# =============================================================
#  TrafficNet Architecture
#  Custom CNN with Residual Connections + Squeeze-and-Excitation
# =============================================================

def squeeze_excite_block(x, ratio=8):
    """Squeeze-and-Excitation block for channel attention."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def conv_block(x, filters, kernel_size=3, strides=1, use_bn=True):
    """Conv → BN → ReLU block."""
    x = layers.Conv2D(
        filters, kernel_size,
        strides=strides,
        padding='same',
        use_bias=not use_bn,
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer='he_normal'
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def residual_block(x, filters, downsample=False):
    """Residual block with optional downsampling."""
    strides = 2 if downsample else 1
    shortcut = x

    # Main path
    x = conv_block(x, filters, strides=strides)
    x = conv_block(x, filters)
    x = squeeze_excite_block(x)

    # Shortcut path
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=strides, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_trafficnet(input_shape=(48, 48, 3), num_classes=43):
    """
    TrafficNet — Custom CNN for Traffic Sign Recognition.
    Architecture:
      Stem → Block1 → Block2 → Block3 → Block4 → Block5
      → Global Average Pooling → Dense → Dropout → Softmax
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ── Stem ─────────────────────────────────────────────────
    x = conv_block(inputs, 32, kernel_size=3)
    x = conv_block(x, 32, kernel_size=3)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.2)(x)

    # ── Block 1: 64 filters ──────────────────────────────────
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.2)(x)

    # ── Block 2: 128 filters ─────────────────────────────────
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)

    # ── Block 3: 256 filters ─────────────────────────────────
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = layers.Dropout(0.3)(x)

    # ── Block 4: 512 filters ─────────────────────────────────
    x = residual_block(x, 512)
    x = layers.Dropout(0.4)(x)

    # ── Classification Head ──────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense_512")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense_256")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = Model(inputs, outputs, name="TrafficNet")
    return model


# =============================================================
#  Custom Callbacks
# =============================================================

class TrainingProgressCallback(keras.callbacks.Callback):
    """Prints a rich training summary after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc     = logs.get('accuracy', 0) * 100
        val_acc = logs.get('val_accuracy', 0) * 100
        loss    = logs.get('loss', 0)
        val_loss= logs.get('val_loss', 0)
        lr      = float(keras.backend.get_value(self.model.optimizer.lr))
        bar     = '█' * int(acc // 5) + '░' * (20 - int(acc // 5))
        print(f"\n  Epoch {epoch+1:03d} | [{bar}] {acc:.2f}% "
              f"| Val: {val_acc:.2f}% | Loss: {loss:.4f}/{val_loss:.4f} | LR: {lr:.2e}")


# =============================================================
#  Training Function
# =============================================================

def train(data_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, resume=False):
    print("\n" + "═"*60)
    print("  🚦 TrafficNet — Training Pipeline Started")
    print("═"*60)

    # ── Load Data ─────────────────────────────────────────────
    print(f"\n📂 Loading dataset from: {data_dir}")
    X, y = load_gtsrb_from_folder(data_dir)
    print(f"✅ Loaded {len(X)} images across {NUM_CLASSES} classes")
    print(f"   Shape: {X.shape} | dtype: {X.dtype}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n🔀 Split: Train={len(X_train)} | Val={len(X_val)}")

    # ── Class Weights ─────────────────────────────────────────
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}

    # ── Data Augmentation ─────────────────────────────────────
    datagen = build_train_datagen()
    datagen.fit(X_train)

    train_gen = datagen.flow(X_train, tf.keras.utils.to_categorical(y_train, NUM_CLASSES),
                             batch_size=batch_size)

    # ── Build Model ───────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "trafficnet.h5")
    if resume and os.path.exists(model_path):
        print(f"\n🔄 Resuming from: {model_path}")
        model = keras.models.load_model(model_path)
    else:
        model = build_trafficnet(
            input_shape=(*IMG_SIZE, 3),
            num_classes=NUM_CLASSES
        )

    # ── Compile ───────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')]
    )

    print(f"\n🧠 Model: {model.name}")
    print(f"   Parameters: {model.count_params():,}")
    model.summary(line_length=80)

    # ── Callbacks ─────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            model_path, monitor='val_accuracy',
            save_best_only=True, verbose=1, mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy', patience=8,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        ),
        TrainingProgressCallback(),
        TensorBoard(log_dir=os.path.join(MODELS_DIR, 'logs'), histogram_freq=1),
    ]

    # ── Train ─────────────────────────────────────────────────
    print("\n🚀 Starting training...\n")
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val, NUM_CLASSES)),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0,
    )

    # ── Evaluate ──────────────────────────────────────────────
    print("\n📊 Final Evaluation:")
    results = model.evaluate(
        X_val, tf.keras.utils.to_categorical(y_val, NUM_CLASSES), verbose=1
    )
    print(f"   Val Loss:     {results[0]:.4f}")
    print(f"   Val Accuracy: {results[1]*100:.2f}%")
    print(f"   Top-5 Acc:    {results[2]*100:.2f}%")

    # ── Save History ──────────────────────────────────────────
    hist_path = os.path.join(MODELS_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"\n💾 History saved → {hist_path}")

    # ── Save Plots ────────────────────────────────────────────
    curves_path = os.path.join(MODELS_DIR, "training_curves.png")
    plot_training_history(history, save_path=curves_path)

    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    y_pred  = np.argmax(model.predict(X_val, verbose=0), axis=1)
    plot_confusion_matrix(y_val, y_pred, save_path=cm_path)

    print("\n" + "═"*60)
    print(f"  ✅ Training Complete! Model saved → {model_path}")
    print("═"*60 + "\n")

    return model, history


# =============================================================
#  Entry Point
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrafficNet CNN")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(ROOT, "dataset", "train"),
                        help="Path to GTSRB training folder")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from existing checkpoint")
    args = parser.parse_args()

    train(
        data_dir   = args.data_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        resume     = args.resume,
    )
