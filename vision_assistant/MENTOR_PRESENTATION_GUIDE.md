# AI Vision Assistant — Mentor Presentation Guide

> **Project Name:** AI-Based Object Detection and Vision Assistant  
> **Purpose:** Help visually impaired individuals understand their surroundings in real time  
> **Version:** 1.0 · 2026  
> **Methodology:** Agile · 5 Sprints · ~12 Weeks  

---

## 1. Project Overview (2 min)

### What It Does
A real-time AI system that uses a camera to help blind/visually impaired users navigate safely by:
- **Detecting objects** in the camera feed (persons, cars, chairs, dogs, etc.)
- **Estimating distance** to each object in metres
- **Determining direction** (left / ahead / right)
- **Speaking alerts** aloud: *"Person ahead at 2 metres"*, *"Chair on the left at 1 metre"*

### Core Pipeline
```
Camera → Object Detection (YOLOv8n) → Depth Estimation (MiDaS DPT-Small)
       → Direction Calculation → Priority Engine → Audio Feedback (pyttsx3)
```

---

## 2. Technology Stack

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **Language** | Python 3.10+ | Rich AI/ML ecosystem |
| **Detection** | YOLOv8n (Ultralytics) | COCO 80-class, nano variant for CPU speed |
| **Depth** | MiDaS DPT-Small (torch.hub) | Monocular depth, no special hardware needed |
| **Vision** | OpenCV (cv2) | Camera capture, frame processing, display |
| **Audio** | pyttsx3 (offline) / gTTS (online fallback) | Works without internet |
| **Config** | PyYAML (config.yaml) | All settings externalized, no hardcoded values |
| **Testing** | pytest | Unit + integration tests |
| **Deployment** | Docker / Raspberry Pi 4 | Portable, edge-ready |

---

## 3. How to Run (Live Demo)

### Prerequisites
- Python 3.10+ installed
- Webcam connected
- Virtual environment set up

### Commands

```powershell
# Step 1: Navigate to project
cd vision_assistant

# Step 2: Activate virtual environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux/macOS

# Step 3: Run the assistant
python main.py                   # Full mode (detection + depth + audio)
python main.py --lite            # Lite mode (faster, skips depth model)
python main.py --no-display      # Headless mode (no GUI window)
python main.py --cam-index 1     # Use different camera
python main.py --lite --no-display  # Combined flags
```

### First Run Note
> The first run takes ~30 seconds for YOLOv8 model warmup/compilation.  
> Subsequent frames process much faster. MiDaS weights are downloaded  
> automatically on first use via torch.hub (~30MB).

---

## 4. Currently Active Features ✅

### 4.1 Object Detection
- **Model:** YOLOv8n (nano) — 80 COCO classes
- **430+ recognizable objects** via extended label mapping (synonyms/variants)
- Confidence-based filtering (threshold: 0.5)
- In-frame deduplication (IoU > 0.5 = keep highest confidence)
- Speech-friendly label normalization (e.g., "couch" → "sofa", "cell phone" → "phone")

### 4.2 Depth Estimation (Full Mode)
- **Model:** MiDaS DPT-Small loaded via `torch.hub.load('intel-isl/MiDaS', 'DPT_Small')`
- Multi-zone sampling per bounding box (centre, lower-centre, inset zones)
- EMA smoothing for stable distance readings across frames
- Approach velocity detection ("Car approaching" alerts)
- Frame skipping optimization (run depth every 3rd frame, reuse cache)
- Distance clamped to [0.3m, 10.0m] range

### 4.3 Lite Mode Distance (--lite flag)
- Bbox height heuristic (larger bbox = closer object)
- No depth model loaded — achieves higher FPS on low-power hardware

### 4.4 Direction Calculation
- 3-zone model: LEFT / AHEAD / RIGHT
- Configurable boundaries (default: 33% / 66% of frame width)
- Resolution-independent (uses fractions, not pixel values)

### 4.5 Blind Assistance Engine (Priority + Urgency)
- **4-tier urgency system:** CRITICAL → WARNING → INFO → CLEAR
- **5-zone spatial analysis:** DEAD_AHEAD, LEFT_NEAR, RIGHT_NEAR, LEFT_FAR, RIGHT_FAR
- Priority scoring: `base_weight × (1/distance)` — closer dangerous objects rank highest
- Per-label+direction cooldown to prevent repetitive alerts
- "Path is clear" reassurance after silence interval
- Scene summary every 15 seconds
- Staircase special handling (up/down detection)
- CRITICAL alerts always bypass cooldown (safety override)
- Max 2 announcements per frame (cognitive limit)

### 4.6 Audio Feedback
- **Engine:** pyttsx3 (offline) with gTTS (online) fallback
- Non-blocking: `speak()` returns immediately via PriorityQueue
- Background worker thread processes queue
- Priority ordering: CRITICAL (1) spoken before INFO (5)
- Flush capability for urgent alerts

### 4.7 Camera Module
- Threaded capture (daemon thread)
- `deque(maxlen=1)` buffer — always latest frame, no stale buildup
- BGR → RGB → resize → normalize preprocessing
- FPS logging every 5 seconds
- Clean resource release on shutdown

### 4.8 Display Overlay (when not headless)
- Bounding boxes with color-coding (person=red, vehicle=orange, furniture=blue)
- Distance + urgency labels on each detection
- Real-time FPS counter (color-coded: green/yellow/red)
- Inference time display
- Latest announcement text overlay

### 4.9 Pipeline Architecture
- **Thread 1:** Camera capture → frame_queue
- **Thread 2:** Inference worker (detect → depth → direction → assist)
- **Thread 3:** Audio TTS worker (asyncio)
- Frame-drop strategy: stale frames discarded, always process latest

### 4.10 Configuration System
- All parameters in `config.yaml` — zero hardcoded values
- CLI argument overrides (`--no-display`, `--lite`, `--cam-index`, `--config`)
- Graceful shutdown on Ctrl+C or Q key

---

## 5. Project Structure

```
vision_assistant/
├── modules/                         # Core pipeline modules
│   ├── camera.py                    # Threaded camera capture
│   ├── detector.py                  # YOLOv8n 80-class detection
│   ├── depth_estimator.py           # MiDaS DPT-Small depth
│   ├── direction.py                 # LEFT / AHEAD / RIGHT zones
│   ├── prioritizer.py              # Priority scoring + cooldown
│   ├── blind_assistance.py         # 4-tier urgency navigation engine
│   └── audio.py                    # pyttsx3/gTTS non-blocking TTS
├── utils/                           # Shared utilities
│   ├── datatypes.py                # FrameData + DetectedObject contracts
│   ├── label_map.py                # 430+ object label mapping
│   ├── announcement_builder.py     # Context-aware speech text
│   ├── logger.py                   # Centralized rotating-file logging
│   ├── helpers.py                  # Config loader, preprocessing
│   └── benchmark.py                # Latency tracker + JSON report
├── tests/                           # pytest test suite
├── config.yaml                      # All runtime parameters
├── main.py                          # Full pipeline entry point
├── requirements.txt                 # Python dependencies
├── setup.sh / setup.bat             # One-command bootstrap
├── Dockerfile                       # Container deployment
└── README.md                        # Project documentation
```

---

## 6. Sprint Summary (Agile Methodology)

| Sprint | Name | Deliverables | Status |
|--------|------|-------------|--------|
| **1** | Environment & Architecture | Project scaffold, config.yaml, datatypes, logger, setup scripts | ✅ Done |
| **2** | Camera Input Module | Threaded CameraStream, deque buffer, preprocessing | ✅ Done |
| **3A** | Object Detection | YOLOv8n detector, 80 COCO classes, whitelist filtering | ✅ Done |
| **3B** | Depth Estimation | MiDaS DPT-Small, multi-zone sampling, EMA smoothing | ✅ Done |
| **3C** | Direction Calculator | 3-zone left/ahead/right with configurable boundaries | ✅ Done |
| **4A** | Priority Engine | Scoring formula, cooldown, "Path is clear" | ✅ Done |
| **4B** | Audio Feedback | pyttsx3 + gTTS, PriorityQueue, non-blocking | ✅ Done |
| **5A** | Pipeline Integration | main.py, 3-thread async architecture, display overlay | ✅ Done |
| **5B** | Blind Assistance | 4-tier urgency, 5-zone spatial, approach detection | ✅ Done |
| **5C** | Optimization & Deploy | --lite mode, Dockerfile, README, config reference | ✅ Done |

---

## 7. Key Design Decisions (Mentor Q&A)

**Q: Why YOLOv8n and not a larger model?**  
A: Nano variant is optimized for CPU inference speed — essential for real-time on laptops/Raspberry Pi. Still detects 80 COCO classes with good accuracy at 320px input.

**Q: Why MiDaS and not a stereo camera?**  
A: MiDaS is monocular — works with any single camera. No special hardware needed. DPT-Small variant balances accuracy and speed.

**Q: How accurate is the distance estimation?**  
A: MiDaS outputs relative inverse-depth, calibrated via `scale_factor`. Multi-zone sampling (3 zones per bbox) + EMA smoothing + frame skipping provides stable readings. Accuracy is ±1-2m for objects within 5m — sufficient for navigation alerts.

**Q: How do you avoid bombarding the user with alerts?**  
A: Per-label+direction cooldown system, max 2 announcements per frame, 4-tier urgency (only CRITICAL bypasses cooldown), and "Path is clear" only after 10s silence.

**Q: Why pyttsx3 over edge-tts?**  
A: pyttsx3 is fully offline — critical for assistive devices that may not have internet. gTTS is available as online fallback when better voice quality is desired.

**Q: What about Raspberry Pi deployment?**  
A: Use `--lite` flag (skips depth model), set `imgsz: 320` and resolution to 320×240. Achieves usable FPS on RPi4.

---

## 8. Demo Script

1. **Start the system:** `python main.py --lite`
2. **Show camera feed** with bounding box overlays
3. **Walk in front of camera** — system announces "Person ahead at X metres"
4. **Hold up objects** (bottle, phone, book) — shows 80-class detection
5. **Move objects left/right** — direction changes in announcements
6. **Remove all objects** — "Path is clear" after cooldown
7. **Show headless mode:** `python main.py --lite --no-display`
8. **Show config.yaml** — all parameters tunable without code changes

---

## 9. Future Enhancements (Roadmap)

- [ ] GPU acceleration with FP16 half-precision
- [ ] Custom object training for regional obstacles
- [ ] Multi-language TTS support (Hindi, Tamil, etc.)
- [ ] Fall detection / sudden obstacle alerts
- [ ] Mobile app integration via WebSocket
- [ ] Battery-powered wearable device form factor

---

*AI Vision Assistant · Mentor Presentation Guide · 2026*
