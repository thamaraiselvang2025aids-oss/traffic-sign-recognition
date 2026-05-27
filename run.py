# =============================================================
#  run.py — One-Click Launcher for TrafficNet
# =============================================================
#  Usage:
#    python run.py              → Launch web dashboard
#    python run.py --train      → Train the model
#    python run.py --detect     → Live webcam detection
#    python run.py --download   → Download GTSRB dataset
# =============================================================

import os, sys, argparse, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

BANNER = r"""
  ████████╗██████╗  █████╗ ███████╗███████╗██╗ ██████╗███╗   ██╗███████╗████████╗
     ██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝████╗  ██║██╔════╝╚══██╔══╝
     ██║   ██████╔╝███████║█████╗  █████╗  ██║██║     ██╔██╗ ██║█████╗     ██║   
     ██║   ██╔══██╗██╔══██║██╔══╝  ██╔══╝  ██║██║     ██║╚██╗██║██╔══╝     ██║   
     ██║   ██║  ██║██║  ██║██║     ██║     ██║╚██████╗██║ ╚████║███████╗   ██║   
     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═══╝╚══════╝   ╚═╝   
  
        🚦 AI-Powered Traffic Sign Recognition System
        CNN · TensorFlow · OpenCV · Flask · Python
"""

def print_menu():
    print(BANNER)
    print("  ┌────────────────────────────────────────┐")
    print("  │  1. Launch Web Dashboard               │")
    print("  │  2. Train TrafficNet CNN               │")
    print("  │  3. Live Webcam Detection              │")
    print("  │  4. Predict Single Image               │")
    print("  │  5. Download GTSRB Dataset             │")
    print("  │  6. Exit                               │")
    print("  └────────────────────────────────────────┘\n")

def launch_dashboard():
    print("\n🚀 Launching Web Dashboard at http://127.0.0.1:5000 ...\n")
    app_path = os.path.join(ROOT, "app", "app.py")
    subprocess.run([sys.executable, app_path])

def train_model():
    epochs = input("\n  Enter number of epochs [30]: ").strip() or "30"
    data   = input("  Dataset path [dataset/train]: ").strip() or os.path.join(ROOT, "dataset", "train")
    print(f"\n🚀 Training TrafficNet for {epochs} epochs...\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "train.py"),
                    "--epochs", epochs, "--data_dir", data])

def live_detection():
    src = input("\n  Camera index or video path [0]: ").strip() or "0"
    print("\n🎥 Starting live detection... (Press Q to quit)\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "realtime_detect.py"),
                    "--source", src])

def predict_image():
    path = input("\n  Image path: ").strip()
    if not os.path.exists(path):
        print(f"  ❌ File not found: {path}")
        return
    gradcam = input("  Show Grad-CAM? [y/N]: ").strip().lower() == 'y'
    cmd = [sys.executable, os.path.join(ROOT, "src", "predict.py"), "--image", path]
    if gradcam: cmd.append("--gradcam")
    subprocess.run(cmd)

def download_dataset():
    print("\n📦 Downloading GTSRB Dataset...")
    print("   This will download ~300 MB from Kaggle / direct source.\n")
    try:
        import urllib.request, zipfile, shutil
        url  = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        dest = os.path.join(ROOT, "dataset", "GTSRB.zip")
        os.makedirs(os.path.join(ROOT, "dataset"), exist_ok=True)
        print("  Downloading... (this may take a few minutes)")

        def progress(block, bsize, total):
            pct = min(int(block * bsize / total * 100), 100)
            print(f"\r  Progress: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print("\n  Extracting...")
        with zipfile.ZipFile(dest, 'r') as z:
            z.extractall(os.path.join(ROOT, "dataset"))
        os.remove(dest)
        print(f"  ✅ Dataset saved → {os.path.join(ROOT, 'dataset')}\n")
        print("  ℹ️  Organize into dataset/train/<class_id>/ folders before training.")
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        print("  Manual download: https://benchmark.ini.rub.de/gtsrb_dataset.html")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--detect",   action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--predict",  type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.train:    train_model();    return
    if args.detect:   live_detection(); return
    if args.download: download_dataset(); return
    if args.predict:
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "predict.py"),
                        "--image", args.predict])
        return

    # Interactive menu
    while True:
        print_menu()
        choice = input("  Select option [1-6]: ").strip()
        if   choice == '1': launch_dashboard()
        elif choice == '2': train_model()
        elif choice == '3': live_detection()
        elif choice == '4': predict_image()
        elif choice == '5': download_dataset()
        elif choice == '6': print("\n  👋 Goodbye!\n"); break
        else: print("  ⚠️  Invalid choice. Please enter 1-6.\n")

if __name__ == "__main__":
    main()
