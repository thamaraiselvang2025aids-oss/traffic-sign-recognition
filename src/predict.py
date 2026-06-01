# =============================================================
#  predict.py — Single Image Inference + Top-5 Predictions
#  Traffic Sign Recognition System | TrafficNet
# =============================================================
#
#  Usage:
#    python src/predict.py --image path/to/sign.jpg
#    python src/predict.py --image path/to/sign.jpg --gradcam
# =============================================================

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils import (
    CLASS_NAMES, IMG_SIZE, NUM_CLASSES,
    preprocess_image, preprocess_for_model,
    get_category, get_category_color
)

MODELS_DIR  = os.path.join(ROOT, "models")
HISTORY_DIR = os.path.join(ROOT, "models", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# =============================================================
#  Model Loader (singleton)
# =============================================================

_model_cache = {}

def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "trafficnet.h5")
    if model_path not in _model_cache:
        import tensorflow as tf
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Run `python src/train.py` first to train the model."
            )
        print(f"🔄 Loading model from: {model_path}")
        _model_cache[model_path] = tf.keras.models.load_model(model_path)
        print("✅ Model loaded.")
    return _model_cache[model_path]


# =============================================================
#  Core Prediction Function
# =============================================================

def predict(image_input, model=None, top_k=5, model_path=None):
    """
    Predict traffic sign class from image.

    Args:
        image_input: File path (str) or numpy array (BGR or RGB)
        model:       Pre-loaded Keras model (optional)
        top_k:       Number of top predictions to return
        model_path:  Path to model file (if model not provided)

    Returns:
        dict with keys:
          - top1_class_id   (int)
          - top1_class_name (str)
          - top1_confidence (float, 0–100)
          - category        (str)
          - category_color  (str hex)
          - top_k           (list of dicts)
          - timestamp       (str ISO)
          - input_shape     (tuple)
    """
    if model is None:
        model = load_model(model_path)

    # ── Preprocess ────────────────────────────────────────────
    if isinstance(image_input, str):
        raw_img = cv2.imread(image_input)
    else:
        raw_img = image_input.copy()

    batch   = preprocess_for_model(raw_img)
    probs   = model.predict(batch, verbose=0)[0]      # (43,)

    # ── Top-K ─────────────────────────────────────────────────
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_results = [
        {
            "rank":       int(i + 1),
            "class_id":   int(idx),
            "class_name": CLASS_NAMES[int(idx)],
            "confidence": float(round(probs[idx] * 100, 2)),
            "category":   get_category(int(idx)),
            "color":      get_category_color(int(idx)),
        }
        for i, idx in enumerate(top_indices)
    ]

    best = top_results[0]
    result = {
        "top1_class_id":   best["class_id"],
        "top1_class_name": best["class_name"],
        "top1_confidence": best["confidence"],
        "category":        best["category"],
        "category_color":  best["color"],
        "top_k":           top_results,
        "timestamp":       datetime.now().isoformat(),
        "input_shape":     tuple(raw_img.shape),
        "all_probs":       probs.tolist(),
    }
    return result


# =============================================================
#  Rich CLI Output
# =============================================================

def print_prediction(result):
    bar_len = 30
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  🚦 TrafficNet Prediction Result" + " "*25 + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Predicted : {result['top1_class_name'][:44]:<44} ║")
    print(f"║  Category  : {result['category'].upper():<44} ║")
    conf = result['top1_confidence']
    filled = int(bar_len * conf / 100)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"║  Confidence: [{bar}] {conf:5.1f}%      ║")
    print(f"║  Timestamp : {result['timestamp'][:19]:<44} ║")
    print("╠" + "═"*58 + "╣")
    print("║  Top-5 Predictions:                                      ║")
    for r in result['top_k']:
        filled_k = int(20 * r['confidence'] / 100)
        bar_k    = '█' * filled_k + '░' * (20 - filled_k)
        name_short = r['class_name'][:28]
        print(f"║  {r['rank']}. {name_short:<28} [{bar_k}] {r['confidence']:5.1f}% ║")
    print("╚" + "═"*58 + "╝\n")


# =============================================================
#  Matplotlib Visualization
# =============================================================

def visualize_prediction(image_input, result, save_path=None):
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
    else:
        img_bgr = image_input.copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(14, 6), facecolor='#0D1117')
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.05)

    # ── Left: Image ───────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_rgb)
    ax_img.set_facecolor('#0D1117')
    ax_img.axis('off')
    color = result['category_color']
    for spine in ax_img.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(3)
    ax_img.set_title(
        f"🚦 {result['top1_class_name']}\n"
        f"Category: {result['category'].upper()}  |  Confidence: {result['top1_confidence']:.1f}%",
        color='#F0F6FC', fontsize=11, fontweight='bold', pad=12
    )

    # ── Right: Top-5 Bar Chart ────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor('#161B22')
    names  = [r['class_name'][:30] for r in result['top_k']]
    confs  = [r['confidence']       for r in result['top_k']]
    colors = [r['color']            for r in result['top_k']]
    bars   = ax_bar.barh(range(len(names)), confs, color=colors,
                          height=0.55, edgecolor='none')
    for i, (bar_, conf) in enumerate(zip(bars, confs)):
        ax_bar.text(conf + 0.5, i, f"{conf:.1f}%",
                    va='center', ha='left', color='#C9D1D9', fontsize=9)
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(names, color='#C9D1D9', fontsize=9)
    ax_bar.set_xlabel("Confidence (%)", color='#C9D1D9', fontsize=10)
    ax_bar.set_title("Top-5 Predictions", color='#F0F6FC', fontsize=12, fontweight='bold')
    ax_bar.set_xlim(0, 110)
    ax_bar.tick_params(colors='#C9D1D9')
    ax_bar.invert_yaxis()
    for spine in ax_bar.spines.values():
        spine.set_edgecolor('#30363D')

    legend_patches = [
        mpatches.Patch(color='#FF4757', label='Prohibitory'),
        mpatches.Patch(color='#2ED573', label='Mandatory'),
        mpatches.Patch(color='#FFA502', label='Warning'),
        mpatches.Patch(color='#1E90FF', label='Priority'),
    ]
    ax_bar.legend(handles=legend_patches, loc='lower right',
                  facecolor='#161B22', labelcolor='#C9D1D9', fontsize=8)

    fig.suptitle("TrafficNet — AI Traffic Sign Recognition",
                 color='#58A6FF', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.show()
    return fig


# =============================================================
#  History Logger
# =============================================================

HISTORY_FILE = os.path.join(HISTORY_DIR, "predictions.json")

def log_prediction(result, image_filename=None):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    entry = {
        "timestamp":   result["timestamp"],
        "class_id":    result["top1_class_id"],
        "class_name":  result["top1_class_name"],
        "confidence":  result["top1_confidence"],
        "category":    result["category"],
        "image_file":  image_filename or "unknown",
    }
    history.insert(0, entry)
    history = history[:200]           # Keep last 200
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# =============================================================
#  CLI Entry Point
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficNet — Traffic Sign Predictor")
    parser.add_argument("--image",     type=str, required=True, help="Path to input image")
    parser.add_argument("--model",     type=str, default=None,  help="Path to .h5 model file")
    parser.add_argument("--top_k",    type=int, default=5,     help="Number of top predictions")
    parser.add_argument("--gradcam",   action="store_true",     help="Show Grad-CAM heatmap")
    parser.add_argument("--save",      type=str, default=None,  help="Save output image to path")
    parser.add_argument("--no_plot",   action="store_true",     help="Skip matplotlib plot")
    args = parser.parse_args()

    model  = load_model(args.model)
    result = predict(args.image, model=model, top_k=args.top_k)
    print_prediction(result)
    log_prediction(result, image_filename=os.path.basename(args.image))

    if not args.no_plot:
        visualize_prediction(args.image, result, save_path=args.save)

    if args.gradcam:
        from src.gradcam import generate_gradcam_overlay
        overlay = generate_gradcam_overlay(args.image, model)
        cv2.imshow("Grad-CAM", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
