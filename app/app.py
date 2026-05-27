# =============================================================
#  app.py — Flask Web Dashboard Backend
#  Traffic Sign Recognition System | TrafficNet
# =============================================================

import os
import sys
import json
import base64
import io
import time
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils   import CLASS_NAMES, NUM_CLASSES, preprocess_image, get_category, get_category_color
from src.predict import load_model, predict, log_prediction, HISTORY_FILE

# =============================================================
#  Flask App Config
# =============================================================

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024     # 16 MB upload limit
app.config['UPLOAD_FOLDER']      = os.path.join(ROOT, "models", "uploads")
ALLOWED_EXTENSIONS               = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'webp'}

MODELS_DIR   = os.path.join(ROOT, "models")
HISTORY_PATH = HISTORY_FILE

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Globals ───────────────────────────────────────────────────
_model      = None
_model_lock = False

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(MODELS_DIR, "trafficnet.h5")
        if os.path.exists(model_path):
            _model = load_model(model_path)
    return _model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(img_array_bgr):
    """Convert BGR numpy array to base64 PNG string for JSON response."""
    img_rgb = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================
#  Routes
# =============================================================

@app.route('/')
def index():
    return render_template('index.html')


# ── POST /predict ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Upload an image and get prediction results.
    Returns JSON with top-5 predictions, Grad-CAM overlay, and metadata.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Use: {ALLOWED_EXTENSIONS}'}), 400

    try:
        # ── Read image ────────────────────────────────────────
        file_bytes = file.read()
        np_arr     = np.frombuffer(file_bytes, np.uint8)
        img_bgr    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Cannot decode image'}), 400

        # ── Run prediction ────────────────────────────────────
        model  = get_model()
        if model is None:
            return jsonify({'error': 'Model not loaded. Run training first.'}), 503

        result = predict(img_bgr, model=model, top_k=5)

        # ── Original image → base64 ───────────────────────────
        img_b64 = image_to_base64(img_bgr)

        # ── Grad-CAM overlay → base64 ─────────────────────────
        gradcam_b64 = None
        try:
            from src.gradcam import generate_gradcam_overlay
            overlay_bgr, heatmap, _ = generate_gradcam_overlay(img_bgr, model=model)
            gradcam_b64 = image_to_base64(overlay_bgr)
        except Exception as e:
            app.logger.warning(f"Grad-CAM failed: {e}")

        # ── Log to history ────────────────────────────────────
        log_prediction(result, image_filename=secure_filename(file.filename))

        response = {
            'success':        True,
            'class_id':       result['top1_class_id'],
            'class_name':     result['top1_class_name'],
            'confidence':     result['top1_confidence'],
            'category':       result['category'],
            'category_color': result['category_color'],
            'top_k':          result['top_k'],
            'all_probs':      result['all_probs'],
            'timestamp':      result['timestamp'],
            'image_b64':      img_b64,
            'gradcam_b64':    gradcam_b64,
        }
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# ── GET /history ──────────────────────────────────────────────
@app.route('/history', methods=['GET'])
def get_history():
    """Return the last N prediction history entries."""
    n = int(request.args.get('n', 50))
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    return jsonify({'history': history[:n], 'total': len(history)})


# ── GET /model-stats ─────────────────────────────────────────
@app.route('/model-stats', methods=['GET'])
def model_stats():
    """Return training history and model info."""
    hist_path = os.path.join(MODELS_DIR, "training_history.json")
    if not os.path.exists(hist_path):
        return jsonify({'available': False, 'message': 'No training history found.'})

    with open(hist_path) as f:
        history = json.load(f)

    model = get_model()
    model_info = {}
    if model is not None:
        model_info = {
            'name':       model.name,
            'parameters': model.count_params(),
            'layers':     len(model.layers),
        }

    return jsonify({
        'available':  True,
        'history':    history,
        'model_info': model_info,
        'best_val_accuracy': max(history.get('val_accuracy', [0])) * 100,
        'best_val_loss':     min(history.get('val_loss', [999])),
        'epochs_trained':    len(history.get('accuracy', [])),
    })


# ── GET /class-info ───────────────────────────────────────────
@app.route('/class-info', methods=['GET'])
def class_info():
    """Return all 43 class names and categories."""
    classes = [
        {
            'id':       cid,
            'name':     name,
            'category': get_category(cid),
            'color':    get_category_color(cid),
        }
        for cid, name in CLASS_NAMES.items()
    ]
    return jsonify({'classes': classes, 'total': NUM_CLASSES})


# ── GET /model-status ────────────────────────────────────────
@app.route('/model-status', methods=['GET'])
def model_status():
    model_path = os.path.join(MODELS_DIR, "trafficnet.h5")
    model      = get_model()
    return jsonify({
        'model_loaded':  model is not None,
        'model_exists':  os.path.exists(model_path),
        'model_path':    model_path,
        'num_classes':   NUM_CLASSES,
        'input_size':    '48x48',
        'architecture':  'TrafficNet (Custom CNN + Residual + SE-Attention)',
    })


# ── POST /clear-history ──────────────────────────────────────
@app.route('/clear-history', methods=['POST'])
def clear_history():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    return jsonify({'success': True, 'message': 'History cleared.'})


# =============================================================
#  Main
# =============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  [TrafficNet] Web Dashboard")
    print("="*60)
    print(f"  URL:   http://127.0.0.1:5000")
    print(f"  Root:  {ROOT}")
    model_path = os.path.join(MODELS_DIR, "trafficnet.h5")
    if os.path.exists(model_path):
        print(f"  Model: FOUND -> {model_path}")
        get_model()
    else:
        print(f"  Model: NOT FOUND -- run training first")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
