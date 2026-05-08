#!/bin/bash
# ============================================================
# AI Vision Assistant v2.0 — One-Command Environment Bootstrap
# ============================================================
# Usage:  chmod +x setup.sh && ./setup.sh
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs all pip dependencies from requirements.txt
#   3. Downloads YOLO-World x-variant model weights (primary)
#   4. Downloads YOLOv8n weights (fallback)
#   5. Pre-downloads Depth Anything V2 Metric Indoor model
#   6. Runs the full test suite to verify installation
# ============================================================

set -e  # Exit immediately on any error

echo "============================================"
echo "  AI Vision Assistant v2.0 — Setup"
echo "============================================"

# ── Step 1: Virtual Environment ─────────────────────────────
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/6] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/6] Virtual environment already exists — skipping."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

echo "  → Python: $(python --version)"
echo "  → Pip:    $(pip --version | awk '{print $2}')"

# ── Step 2: Install Dependencies ────────────────────────────
echo ""
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "  → All packages installed."

# ── Step 3: Download YOLOv8n Weights (fallback) ─────────────
echo ""
echo "[3/6] Downloading YOLOv8n fallback model weights..."
mkdir -p models
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('  → YOLOv8n weights downloaded successfully.')
"

# ── Step 4: Download YOLO-World x-variant (primary) ─────────
echo ""
echo "[4/6] Downloading YOLO-World x-variant (best accuracy)..."
python -c "
from ultralytics import YOLOWorld
model = YOLOWorld('yolov8x-worldv2.pt')
print('  → YOLO-World x-variant downloaded successfully.')
"

# ── Step 5: Pre-download Depth Anything V2 ──────────────────
echo ""
echo "[5/6] Pre-downloading Depth Anything V2 Metric Indoor model..."
echo "  (This may take a few minutes on first run)"
python -c "
from transformers import pipeline
p = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
print('  → Depth Anything V2 downloaded successfully.')
"

# ── Step 6: Run Tests ───────────────────────────────────────
echo ""
echo "[6/6] Running test suite..."
python -m pytest tests/ -v --tb=short

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup Complete! (v2.0 Rectified)"
echo "============================================"
echo ""
echo "  Activate environment:  source $VENV_DIR/bin/activate"
echo "  Run tests:             pytest tests/ -v"
echo "  Start system:          python main.py"
echo "  Start (headless):      python main.py --no-display"
echo "  Start (lite mode):     python main.py --lite"
echo ""
