# =============================================================
#  gradcam.py — Gradient-weighted Class Activation Mapping
#  Traffic Sign Recognition System | TrafficNet
# =============================================================
#
#  Generates heatmaps showing WHAT the model focuses on.
#
#  Usage:
#    python src/gradcam.py --image path/to/sign.jpg
# =============================================================

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.utils import IMG_SIZE, CLASS_NAMES, preprocess_image, preprocess_for_model
from src.predict import load_model

# =============================================================
#  Grad-CAM Core Implementation
# =============================================================

def get_gradcam_heatmap(model, img_array, layer_name=None, class_index=None):
    """
    Compute Grad-CAM heatmap for the given image.

    Args:
        model:       Keras model
        img_array:   (1, H, W, 3) float32 array
        layer_name:  Target conv layer name (auto-detects last Conv2D if None)
        class_index: Target class (uses top prediction if None)

    Returns:
        heatmap: (H, W) float32 array in [0, 1]
        class_index: int
    """
    # ── Auto-detect last Conv2D layer ─────────────────────────
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    if layer_name is None:
        raise ValueError("No Conv2D layer found in model.")

    # ── Build Grad-CAM sub-model ──────────────────────────────
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_index]

    # ── Gradients of class score w.r.t. conv output ──────────
    grads     = tape.gradient(loss, conv_outputs)                # (1, h, w, C)
    pooled    = tf.reduce_mean(grads, axis=(0, 1, 2))            # (C,)
    conv_out  = conv_outputs[0]                                   # (h, w, C)
    heatmap   = tf.reduce_sum(pooled * conv_out, axis=-1)        # (h, w)
    heatmap   = tf.nn.relu(heatmap)                              # ReLU
    heatmap   = heatmap / (tf.reduce_max(heatmap) + 1e-8)       # Normalize
    return heatmap.numpy(), class_index


def apply_heatmap(heatmap, original_img, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Resize heatmap and overlay on original image.

    Returns:
        superimposed: BGR uint8 image
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    elif original_img.shape[2] == 3 and original_img.dtype != np.uint8:
        original_img = np.uint8(255 * original_img)
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


# =============================================================
#  High-Level API
# =============================================================

def generate_gradcam_overlay(image_input, model=None, layer_name=None,
                              class_index=None, alpha=0.45):
    """
    Full pipeline: image → Grad-CAM overlay (BGR).

    Args:
        image_input: file path (str) or numpy array (BGR)
        model:       Keras model (loads default if None)
        ...
    Returns:
        overlay: BGR uint8 image
        heatmap: raw float32 heatmap
        pred_class: int
    """
    if model is None:
        model = load_model()

    if isinstance(image_input, str):
        original = cv2.imread(image_input)
    else:
        original = image_input.copy()

    img_array = preprocess_for_model(original)                   # (1, 48, 48, 3)
    heatmap, pred_class = get_gradcam_heatmap(
        model, img_array, layer_name=layer_name, class_index=class_index
    )
    overlay = apply_heatmap(heatmap, original, alpha=alpha)
    return overlay, heatmap, pred_class


def generate_gradcam_figure(image_input, model=None, save_path=None):
    """
    Creates a 3-panel matplotlib figure:
      [Original Image] | [Heatmap] | [Grad-CAM Overlay]
    """
    if model is None:
        model = load_model()

    if isinstance(image_input, str):
        original_bgr = cv2.imread(image_input)
    else:
        original_bgr = image_input.copy()

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay_bgr, heatmap, pred_class = generate_gradcam_overlay(
        original_bgr, model=model
    )
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Colorize heatmap for display
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_rgb     = cv2.resize(heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0]))

    class_name = CLASS_NAMES.get(pred_class, f"Class {pred_class}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle(f"Grad-CAM — Predicted: {class_name}",
                 color='#58A6FF', fontsize=14, fontweight='bold')

    titles = ["Original Image", "Attention Heatmap", "Grad-CAM Overlay"]
    imgs   = [original_rgb, heatmap_rgb, overlay_rgb]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img)
        ax.set_title(title, color='#C9D1D9', fontsize=11)
        ax.set_facecolor('#161B22')
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.show()
    return fig, pred_class


# =============================================================
#  Heatmap as PNG bytes (for Flask API)
# =============================================================

def gradcam_to_png_bytes(image_input, model=None):
    """Returns PNG bytes of the Grad-CAM overlay (for web serving)."""
    import io
    overlay_bgr, _, _ = generate_gradcam_overlay(image_input, model=model)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    pil_img = __import__('PIL.Image', fromlist=['Image']).Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# =============================================================
#  CLI Entry Point
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficNet — Grad-CAM Visualizer")
    parser.add_argument("--image",  type=str, required=True, help="Path to input image")
    parser.add_argument("--model",  type=str, default=None,  help="Path to .h5 model file")
    parser.add_argument("--layer",  type=str, default=None,  help="Target Conv2D layer name")
    parser.add_argument("--save",   type=str, default=None,  help="Save figure to path")
    args = parser.parse_args()

    model = load_model(args.model)
    generate_gradcam_figure(args.image, model=model, save_path=args.save)
