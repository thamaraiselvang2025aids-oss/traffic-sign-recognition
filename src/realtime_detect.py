# =============================================================
#  realtime_detect.py — Live Webcam Traffic Sign Detection
#  Traffic Sign Recognition System | TrafficNet
# =============================================================
#
#  Usage:
#    python src/realtime_detect.py
#    python src/realtime_detect.py --source video.mp4
#    python src/realtime_detect.py --source 0 --save output.avi
# =============================================================

import os
import sys
import time
import argparse
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils     import CLASS_NAMES, IMG_SIZE, preprocess_for_model, get_category_color
from src.predict   import load_model, predict
from src.gradcam   import generate_gradcam_overlay

# ── Color palette (BGR for OpenCV) ───────────────────────────
COLORS = {
    "prohibitory": (71, 71, 255),    # Red
    "mandatory":   (83, 214, 46),    # Green
    "warning":     (2, 165, 255),    # Orange
    "priority":    (235, 144, 30),   # Blue
    "unknown":     (160, 176, 164),  # Grey
}

# ── Overlay constants ─────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_LG   = 0.7
FONT_SCALE_SM   = 0.55
THICKNESS       = 2


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def draw_detection_overlay(frame, result, fps, detection_zone=None):
    """
    Draw a rich overlay on the webcam frame:
      - Top info bar with FPS and model name
      - Detection zone rectangle
      - Prediction badge with confidence bar
      - Category color coding
      - Top-3 predictions sidebar
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── 1. Top Header Bar ─────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 52), (13, 17, 23), -1)
    cv2.putText(overlay, "TrafficNet | AI Traffic Sign Recognition",
                (12, 22), FONT, FONT_SCALE_SM, (88, 166, 255), 1, cv2.LINE_AA)
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, FONT, FONT_SCALE_SM, 1)
    cv2.putText(overlay, fps_text, (w - tw - 12, 22),
                FONT, FONT_SCALE_SM, (63, 185, 80), 1, cv2.LINE_AA)

    if result is None:
        return cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

    # ── 2. Detection Zone (center crop) ──────────────────────
    zone_size = min(h, w) // 2
    x1 = (w - zone_size) // 2;  y1 = (h - zone_size) // 2
    x2 = x1 + zone_size;         y2 = y1 + zone_size
    color_bgr = hex_to_bgr(result['category_color'])

    # Animated corner brackets
    L = zone_size // 6
    thickness_bracket = 3
    for cx, cy in [(x1,y1), (x2,y1), (x1,y2), (x2,y2)]:
        dx_sign = 1 if cx == x1 else -1
        dy_sign = 1 if cy == y1 else -1
        cv2.line(overlay, (cx, cy), (cx + dx_sign * L, cy), color_bgr, thickness_bracket)
        cv2.line(overlay, (cx, cy), (cx, cy + dy_sign * L), color_bgr, thickness_bracket)

    # ── 3. Prediction Badge (bottom panel) ───────────────────
    panel_h = 130
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (13, 17, 23), -1)
    cv2.line(overlay, (0, h - panel_h), (w, h - panel_h), color_bgr, 2)

    class_name = result['top1_class_name']
    confidence = result['top1_confidence']
    category   = result['category'].upper()

    cv2.putText(overlay, class_name[:50],
                (14, h - panel_h + 30), FONT, FONT_SCALE_LG, (240, 246, 252), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"Category: {category}",
                (14, h - panel_h + 55), FONT, FONT_SCALE_SM, color_bgr, 1, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y = 14, h - panel_h + 70
    bar_w, bar_h_ = 260, 14
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_), (40, 40, 50), -1)
    fill_w = int(bar_w * confidence / 100)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_), color_bgr, -1)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_), (80, 80, 90), 1)
    cv2.putText(overlay, f"{confidence:.1f}%",
                (bar_x + bar_w + 8, bar_y + 11), FONT, FONT_SCALE_SM, (201, 209, 217), 1)

    # ── 4. Top-3 Sidebar ─────────────────────────────────────
    sidebar_x = w - 290
    cv2.rectangle(overlay, (sidebar_x - 8, h - panel_h + 4), (w - 6, h - 6), (22, 27, 34), -1)
    cv2.putText(overlay, "Top-3 Predictions:",
                (sidebar_x, h - panel_h + 22), FONT, 0.45, (88, 166, 255), 1)
    for i, pred in enumerate(result['top_k'][:3]):
        y_pos = h - panel_h + 40 + i * 30
        bar_fill = int(150 * pred['confidence'] / 100)
        c = hex_to_bgr(pred['color'])
        cv2.rectangle(overlay, (sidebar_x, y_pos + 2), (sidebar_x + bar_fill, y_pos + 16), c, -1)
        cv2.putText(overlay, f"{pred['class_name'][:26]}",
                    (sidebar_x, y_pos), FONT, 0.38, (201, 209, 217), 1)
        cv2.putText(overlay, f"{pred['confidence']:.1f}%",
                    (sidebar_x + bar_fill + 4, y_pos + 14), FONT, 0.38, c, 1)

    return cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)


def get_center_crop(frame):
    """Extract the center square crop for sign detection."""
    h, w = frame.shape[:2]
    size = min(h, w) // 2
    x1   = (w - size) // 2
    y1   = (h - size) // 2
    return frame[y1:y1+size, x1:x1+size]


def run_realtime(source=0, model_path=None, save_path=None,
                 show_gradcam=False, confidence_threshold=30.0):
    """
    Main real-time detection loop.

    Args:
        source:               Camera index (int) or video file path (str)
        model_path:           Path to .h5 model file
        save_path:            Path to save output video (optional)
        show_gradcam:         Show Grad-CAM overlay in separate window
        confidence_threshold: Minimum confidence % to display prediction
    """
    print("\n🚀 Starting TrafficNet Real-time Detection...")
    print(f"   Source: {source}  |  Grad-CAM: {show_gradcam}")
    print("   Press 'q' to quit | 'g' toggle Grad-CAM | 's' screenshot\n")

    model = load_model(model_path)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    # ── Video Writer ──────────────────────────────────────────
    writer = None
    if save_path:
        fps_out  = 20
        w_out    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_out    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc   = cv2.VideoWriter_fourcc(*'XVID')
        writer   = cv2.VideoWriter(save_path, fourcc, fps_out, (w_out, h_out))
        print(f"📹 Recording to: {save_path}")

    # ── State Variables ───────────────────────────────────────
    fps_counter   = 0
    fps_start     = time.time()
    current_fps   = 0.0
    current_result= None
    frame_skip    = 2       # Predict every N frames
    frame_count   = 0
    show_cam      = show_gradcam
    screenshot_n  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ── Inference (every frame_skip frames) ──────────────
        if frame_count % frame_skip == 0:
            crop = get_center_crop(frame)
            try:
                current_result = predict(crop, model=model, top_k=3)
                if current_result['top1_confidence'] < confidence_threshold:
                    current_result = None
            except Exception as e:
                current_result = None

        # ── FPS Calculation ───────────────────────────────────
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps  = fps_counter / (time.time() - fps_start)
            fps_counter  = 0
            fps_start    = time.time()

        # ── Draw Overlay ──────────────────────────────────────
        display_frame = draw_detection_overlay(frame.copy(), current_result, current_fps)

        # ── Grad-CAM Side Window ──────────────────────────────
        if show_cam and current_result is not None:
            crop = get_center_crop(frame)
            try:
                overlay_bgr, _, _ = generate_gradcam_overlay(crop, model=model)
                overlay_bgr = cv2.resize(overlay_bgr, (320, 320))
                cv2.putText(overlay_bgr, "Grad-CAM",
                            (8, 24), FONT, 0.7, (88, 166, 255), 1)
                cv2.imshow("TrafficNet — Grad-CAM", overlay_bgr)
            except Exception:
                pass

        cv2.imshow("TrafficNet — Live Detection", display_frame)
        if writer:
            writer.write(display_frame)

        # ── Key Controls ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            show_cam = not show_cam
            if not show_cam:
                cv2.destroyWindow("TrafficNet — Grad-CAM")
        elif key == ord('s'):
            screenshot_path = os.path.join(ROOT, f"screenshot_{screenshot_n:03d}.png")
            cv2.imwrite(screenshot_path, display_frame)
            print(f"📸 Screenshot saved: {screenshot_path}")
            screenshot_n += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\n✅ Detection stopped.")


# =============================================================
#  CLI Entry Point
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficNet — Real-time Detection")
    parser.add_argument("--source",    default=0,    help="Camera index or video path")
    parser.add_argument("--model",     default=None, help="Path to .h5 model file")
    parser.add_argument("--save",      default=None, help="Path to save output video")
    parser.add_argument("--gradcam",   action="store_true", help="Show Grad-CAM overlay")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Minimum confidence to show prediction (%)")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run_realtime(
        source=source,
        model_path=args.model,
        save_path=args.save,
        show_gradcam=args.gradcam,
        confidence_threshold=args.threshold,
    )
