# 🚦 TrafficNet — AI Traffic Sign Recognition System

> **CNN · TensorFlow · OpenCV · Flask · Python**  
> Recognizes 43 road sign categories in real-time with Grad-CAM explainability.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **TrafficNet CNN** | Custom architecture with Residual blocks + Squeeze-and-Excitation attention |
| 🔥 **Grad-CAM** | Visualizes *what* the model sees — heatmap overlays |
| 🎥 **Live Webcam** | Real-time OpenCV detection with FPS, confidence bars, category colors |
| 🌐 **Web Dashboard** | Glassmorphism dark-mode UI with drag-and-drop upload |
| 📊 **Analytics** | Training curves, confusion matrix, top-5 confidence chart |
| 📋 **History** | Timestamped log of all past predictions |
| ⚡ **One-Click Run** | `python run.py` — interactive menu for all features |

---

## 📂 Project Structure

```
traffic/
├── src/
│   ├── utils.py            # Preprocessing, class labels, visualization
│   ├── train.py            # TrafficNet CNN training pipeline
│   ├── predict.py          # Single image inference + Top-5
│   ├── realtime_detect.py  # OpenCV live webcam detection
│   └── gradcam.py          # Grad-CAM heatmap generation
├── app/
│   ├── app.py              # Flask web dashboard backend
│   ├── templates/
│   │   └── index.html      # Premium dark-mode UI
│   └── static/
│       ├── style.css       # Glassmorphism styles
│       └── main.js         # Interactive JavaScript
├── models/                 # Saved models + training artifacts
├── dataset/                # GTSRB dataset (see setup below)
├── requirements.txt
├── run.py                  # One-click launcher
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download GTSRB Dataset
```bash
python run.py --download
# OR manually: https://benchmark.ini.rub.de/gtsrb_dataset.html
```
Organize data as:
```
dataset/train/
  00000/   ← class 0 images
  00001/   ← class 1 images
  ...
  00042/   ← class 42 images
```

### 3. Train the Model
```bash
python src/train.py --epochs 30
# Model saved to: models/trafficnet.h5
```

### 4. Launch Web Dashboard
```bash
python run.py
# Opens: http://127.0.0.1:5000
```

### 5. Live Webcam Detection
```bash
python src/realtime_detect.py
# Press Q to quit | G for Grad-CAM | S for screenshot
```

### 6. Predict Single Image
```bash
python src/predict.py --image path/to/sign.jpg --gradcam
```

---

## 🧠 Model Architecture — TrafficNet

```
Input (48×48×3)
    ↓
Stem: Conv32 → Conv32 → MaxPool → Dropout
    ↓
ResBlock (64 + SE Attention) × 2 → MaxPool
    ↓
ResBlock (128 + SE Attention) × 2 → MaxPool
    ↓
ResBlock (256 + SE Attention) × 2
    ↓
ResBlock (512 + SE Attention) × 1
    ↓
Global Average Pooling
    ↓
Dense(512) → BN → Dropout(0.4)
    ↓
Dense(256) → Dropout(0.3)
    ↓
Softmax(43 classes)
```

**Key design choices:**
- **Residual connections** — prevent vanishing gradients in deep network
- **SE (Squeeze-and-Excitation) blocks** — learn channel importance weights
- **Batch Normalization** — faster convergence, stable training  
- **L2 regularization** — prevents overfitting
- **AdamW optimizer** — weight decay for better generalization
- **Class-weighted loss** — handles GTSRB class imbalance

---

## 🎯 Dataset — GTSRB

- **German Traffic Sign Recognition Benchmark**
- **43 classes**, ~50,000 training images
- Real-world variation: blur, occlusion, varying illumination
- Input resolution: **48×48 pixels**

### Sign Categories

| Category | Color | Classes |
|---|---|---|
| 🔴 Prohibitory | Red | Speed limits, No passing, No entry |
| 🟢 Mandatory | Green | Turn ahead, Keep right, Roundabout |
| 🟡 Warning | Orange | Caution, Curves, Crossings |
| 🔵 Priority | Blue | Yield, Stop, Right-of-way |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload image → get prediction + Grad-CAM |
| `GET` | `/history` | Fetch last N predictions |
| `GET` | `/model-stats` | Training curves + model info |
| `GET` | `/class-info` | All 43 class names + categories |
| `GET` | `/model-status` | Check if model is loaded |
| `POST` | `/clear-history` | Clear prediction history |

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.x, Keras
- **Computer Vision**: OpenCV 4.x
- **Web Framework**: Flask + Flask-CORS
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Chart.js
- **Augmentation**: Albumentations, Keras ImageDataGenerator
- **Frontend**: Vanilla CSS (glassmorphism), JavaScript (ES6+)

---

## 📈 Expected Performance

| Metric | Target |
|---|---|
| Validation Accuracy | ≥ 96% |
| Top-5 Accuracy | ≥ 99% |
| Training Time (GPU) | ~15 min / 30 epochs |
| Inference Time | < 10ms per image |

---

## 🙏 Credits

- **Dataset**: [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) by Ruhr-Universität Bochum
- **Architecture inspiration**: ResNet, SENet
- **Grad-CAM**: Selvaraju et al., 2017
