@echo off
REM ============================================================
REM AI Vision Assistant v2.0 — Windows Environment Bootstrap
REM ============================================================
REM Usage:  setup.bat
REM
REM This script:
REM   1. Creates a Python virtual environment
REM   2. Installs all pip dependencies from requirements.txt
REM   3. Downloads YOLO-World x-variant model weights (primary)
REM   4. Downloads YOLOv8n weights (fallback)
REM   5. Pre-downloads Depth Anything V2 Metric Indoor model
REM   6. Runs the full test suite to verify installation
REM ============================================================

echo ============================================
echo   AI Vision Assistant v2.0 — Setup
echo ============================================

REM ── Step 1: Virtual Environment ─────────────────────────────
set VENV_DIR=venv
if not exist "%VENV_DIR%" (
    echo [1/6] Creating virtual environment...
    python -m venv %VENV_DIR%
) else (
    echo [1/6] Virtual environment already exists — skipping.
)

REM Activate virtual environment
call %VENV_DIR%\Scripts\activate.bat

echo   Python: 
python --version
echo   Pip:
pip --version

REM ── Step 2: Install Dependencies ────────────────────────────
echo.
echo [2/6] Installing Python dependencies...
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo   All packages installed.

REM ── Step 3: Download YOLOv8n Weights (fallback) ─────────────
echo.
echo [3/6] Downloading YOLOv8n fallback model weights...
if not exist "models" mkdir models
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('  YOLOv8n weights downloaded successfully.')"

REM ── Step 4: Download YOLO-World s-variant (primary) ─────────
echo.
echo [4/6] Downloading YOLO-World s-variant (fast + accurate)...
python -c "from ultralytics import YOLOWorld; model = YOLOWorld('yolov8s-worldv2.pt'); print('  YOLO-World s-variant downloaded successfully.')"

REM ── Step 5: Pre-download Depth Anything V2 ──────────────────
echo.
echo [5/6] Pre-downloading Depth Anything V2 Metric Indoor model...
echo   (This may take a few minutes on first run)
python -c "from transformers import pipeline; p = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'); print('  Depth Anything V2 downloaded successfully.')"

REM ── Step 6: Run Tests ───────────────────────────────────────
echo.
echo [6/6] Running test suite...
python -m pytest tests\ -v --tb=short
echo.

REM ── Done ─────────────────────────────────────────────────────
echo.
echo ============================================
echo   Setup Complete! (v2.0 Rectified)
echo ============================================
echo.
echo   Activate environment:  %VENV_DIR%\Scripts\activate
echo   Run tests:             pytest tests\ -v
echo   Start system:          python main.py
echo   Start (headless):      python main.py --no-display
echo   Start (lite mode):     python main.py --lite
echo.
pause
