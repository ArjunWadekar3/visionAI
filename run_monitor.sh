#!/bin/bash

# NeuralStream Vision AI - Monitor Launcher (Linux)
# Run: bash run_monitor.sh
# Or: chmod +x run_monitor.sh && ./run_monitor.sh

set -e

cd "$(dirname "$0")" || exit 1

echo "[INFO] Starting NeuralStream Vision AI Monitor..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating venv..."
    source venv/bin/activate
else
    echo "[WARN] venv not found. Make sure python -m venv venv && pip install -r requirements.txt"
fi

# Export detection settings (tune as needed)
export NSA_MODEL_PATH=/home/user/models/visdrone-s/best.pt
export NSA_CLASSES=0
export NSA_CONF=0.10
export NSA_TILED=1
export NSA_SLICE=512
export NSA_IMGSZ=1280
export NSA_MAX_DET=2000

echo "[INFO] Settings:"
echo "  Model: $NSA_MODEL_PATH"
echo "  Conf: $NSA_CONF"
echo "  TILED: $NSA_TILED, SLICE: $NSA_SLICE, IMGSZ: $NSA_IMGSZ"
echo ""
echo "[INFO] Running monitor.py..."
echo "  Select input: 1=Webcam, 2=HDMI, 3=Video folder"
echo ""

python src/vision/monitor.py

echo "[INFO] Monitor stopped."
