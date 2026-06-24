#!/bin/bash

# NeuralStream Vision AI - Monitor Launcher (Linux)
# Run: bash run_monitor.sh
# Or: chmod +x run_monitor.sh && ./run_monitor.sh

set -e

cd "$(dirname "$0")" || exit 1

# Activate virtual environment (silent)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Export detection settings (tune as needed)
export NSA_MODEL_PATH=/home/user/models/visdrone-s/best.pt
export NSA_CLASSES=0
export NSA_CONF=0.10
export NSA_TILED=1
export NSA_SLICE=512
export NSA_IMGSZ=1280
export NSA_MAX_DET=2000


python src/vision/monitor.py
