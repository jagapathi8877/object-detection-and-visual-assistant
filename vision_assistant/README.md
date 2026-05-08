# AI Vision Assistant

> Real-time AI-based assistive system for visually impaired users — using object detection, depth estimation, direction calculation, and audio feedback.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-brightgreen)
![MiDaS](https://img.shields.io/badge/MiDaS-DPT--Small-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The AI Vision Assistant captures live camera frames, detects objects using YOLOv8n (80 COCO classes + 430 extended labels), estimates their distance via MiDaS DPT-Small monocular depth estimation, calculates their spatial direction (left / ahead / right), prioritises the most important objects using a 4-tier urgency system, and announces them as spoken audio via pyttsx3 — all in real time.

### Core Pipeline

```
Camera → Object Detection (YOLOv8n) → Depth Estimation (MiDaS DPT-Small)
       → Direction → Blind Assistance Engine → Audio (pyttsx3/gTTS)
```

### Example Output
```
"Warning! Person directly ahead, 2 metres"
"Chair on the left, 1 metre"
"Car approaching from ahead"
"Path is clear"
```

## Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd vision_assistant

# Linux/macOS
chmod +x setup.sh && ./setup.sh

# Windows
setup.bat
```

### 2. Run

```powershell
# Activate virtual environment first
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\activate         # Windows

# Normal mode (detection + depth + display + audio)
python main.py

# Lite mode (skip depth estimation — faster on CPU)
python main.py --lite

# Headless mode (Raspberry Pi / no monitor)
python main.py --no-display

# Custom camera index
python main.py --cam-index 1

# Custom config file
python main.py --config my_config.yaml

# Combine flags
python main.py --no-display --lite --cam-index 0
```

### 3. Test

```bash
pytest tests/ -v
```

### 4. Docker

```bash
docker build -t vision-assistant .
docker run --device /dev/video0 vision-assistant
docker run --device /dev/video0 vision-assistant --lite
```

## Project Structure

```
vision_assistant/
├── modules/                         # Core pipeline modules
│   ├── camera.py                    # Threaded camera capture (deque buffer)
│   ├── detector.py                  # YOLOv8n 80-class COCO detection
│   ├── depth_estimator.py           # MiDaS DPT-Small depth estimation
│   ├── direction.py                 # Spatial direction (left/ahead/right)
│   ├── prioritizer.py              # Priority scoring + cooldown engine
│   ├── blind_assistance.py         # 4-tier urgency navigation engine
│   └── audio.py                    # pyttsx3/gTTS non-blocking TTS
├── utils/                           # Shared utilities
│   ├── datatypes.py                # FrameData + DetectedObject contracts
│   ├── label_map.py                # 430+ object label mapping
│   ├── announcement_builder.py     # Context-aware speech generation
│   ├── logger.py                   # Centralised rotating-file logging
│   ├── helpers.py                  # Config loader, frame preprocessing
│   └── benchmark.py                # Per-frame latency tracker
├── tests/                           # pytest test suite
├── config.yaml                      # All runtime parameters
├── main.py                          # Full pipeline entry point
├── requirements.txt                 # Python dependencies
├── setup.sh / setup.bat             # One-command bootstrap
├── Dockerfile                       # Container deployment
├── MENTOR_PRESENTATION_GUIDE.md     # Presentation reference
└── README.md                        # This file
```

## Configuration Reference

All settings are in `config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| **camera** | `index` | `0` | Camera device index |
| | `width` | `640` | Frame width (pixels) |
| | `height` | `480` | Frame height (pixels) |
| | `fps` | `30` | Target capture FPS |
| **detection** | `model` | `yolov8n.pt` | YOLOv8 nano model |
| | `confidence` | `0.5` | Min confidence threshold |
| | `imgsz` | `320` | Input resolution (speed/accuracy) |
| | `max_objects` | `3` | Max announcements per cycle |
| **depth** | `model` | `DPT_Small` | MiDaS model variant |
| | `scale_factor` | `1000.0` | Depth → metres calibration |
| | `min_distance` | `0.3` | Min clamped distance (m) |
| | `max_distance` | `10.0` | Max clamped distance (m) |
| | `skip_frames` | `3` | Run depth every Nth frame |
| **direction** | `left_boundary` | `0.33` | Left zone threshold |
| | `right_boundary` | `0.66` | Right zone threshold |
| **audio** | `engine` | `pyttsx3` | TTS engine |
| | `rate` | `160` | Speech rate (wpm) |
| | `volume` | `0.9` | Volume [0.0–1.0] |
| | `cooldown_seconds` | `3.0` | Re-announce suppression |
| | `clear_path_interval` | `5.0` | "Path is clear" interval |
| **system** | `log_level` | `INFO` | Logging verbosity |
| | `headless` | `false` | No-display mode |
| | `target_latency_ms` | `500` | Latency target |

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Thread 1   │     │   Thread 2   │     │   Thread 3   │
│   Camera     │────▶│  Inference   │────▶│    Audio     │
│  (capture)   │     │  (main loop) │     │  (TTS worker)│
└──────────────┘     └──────────────┘     └──────────────┘
      │                     │
   deque(1)          FrameData flows:
   latest frame      detect → depth → direction → assist → announce
```

## Modes

| Mode | Flag | Depth | Use Case |
|------|------|-------|----------|
| **Full** | *(default)* | MiDaS DPT-Small | Laptop with GPU |
| **Lite** | `--lite` | Bbox heuristic | CPU / Raspberry Pi 4 |
| **Headless** | `--no-display` | Either | No monitor |

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Detection | YOLOv8n (Ultralytics) | 80-class COCO object detection |
| Depth | MiDaS DPT-Small (torch.hub) | Monocular depth estimation |
| Vision | OpenCV | Camera capture + display |
| Audio | pyttsx3 / gTTS | Offline / online TTS |
| Config | PyYAML | Runtime parameters |
| Testing | pytest | Full coverage |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **No camera found** | Check `camera.index` in config.yaml; try 1 or 2 |
| **Slow FPS** | Use `--lite`; set `imgsz: 320` in config.yaml |
| **No audio output** | Install `espeak` (Linux): `sudo apt install espeak` |
| **Import errors** | Re-run `setup.sh` / `setup.bat` |
| **CUDA out of memory** | `CUDA_VISIBLE_DEVICES="" python main.py` |
| **pyttsx3 error on Linux** | `sudo apt install libespeak1` |
| **First run slow** | Normal — YOLO model warmup takes ~30s on first frame |
| **Model download fails** | Check internet; MiDaS models cached after first download |

## License

MIT License — see [LICENSE](LICENSE) for details.
