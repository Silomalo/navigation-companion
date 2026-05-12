# Visually Impaired Navigation Assistant

A wearable navigation aid built on Raspberry Pi 5.  A chest-mounted Pi Camera streams to an on-device YOLOv8n model that detects obstacles and speaks navigation instructions through an earphone in real time.  The user can issue voice commands (Whisper STT) and the device learns familiar routes over time using a topological map.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Raspberry Pi 5                         │
│                                                          │
│  Pi Camera ──► camera.py                                │
│                    │  BGR frames                         │
│                    ▼                                     │
│             detector.py  ◄── YOLOv8n (6 MB, COCO)       │
│                    │  DetectionResult                    │
│                    ▼                                     │
│             navigator.py ◄── topo_map.py (route learn)  │
│                    │  speech phrases                     │
│                    ▼                                     │
│             audio_out.py ──► earphone  (pyttsx3/espeak)  │
│                                                          │
│  Microphone ──► audio_in.py ──► Whisper STT ──► nav cmd  │
└─────────────────────────────────────────────────────────┘
```

---

## Hardware

| Item | Part |
|------|------|
| Compute | Raspberry Pi 5 (4 GB+) |
| Camera  | Pi Camera Module v3 (12 MP autofocus) |
| Audio out | 3.5 mm earphone (Pi audio jack or USB audio) |
| Microphone | USB microphone dongle |
| Power | USB-C PD bank (20 000 mAh ≥ 27 W) |

**Physical setup:** Pi + camera mounted on chest harness. Thin coiled cable to ear (earphone). Thin gooseneck wire from ear to mouth (mic capsule).

---

## Models (all run locally, no internet needed)

| Model | Size | Speed on RPi 5 | Purpose |
|-------|------|-----------------|---------|
| YOLOv8n | 6 MB | ~15–20 fps | Obstacle / object detection (COCO 80 classes) |
| Whisper tiny | 39 MB | ~0.3× real-time | Speech recognition for voice commands |
| espeak-ng | built-in | real-time | Text-to-speech navigation instructions |

---

## Quick Start

### Option A — Simulation on any laptop (no RPi needed)

```bash
git clone <repo>
cd nav-assistant
pip install -r requirements.txt

# Run the visual simulator (opens an OpenCV window)
SIMULATE=1 python simulate.py

# With a video file instead of webcam:
SIMULATE=1 python simulate.py --video path/to/street.mp4

# No camera at all:
SIMULATE=1 python simulate.py --synthetic
```

**Keyboard shortcuts in the simulation window:**

| Key | Action |
|-----|--------|
| D | "Describe scene" |
| W | "Where am I?" |
| H | Help |
| Q | Quit |

### Option B — Install on Raspberry Pi 5

```bash
git clone <repo>
cd nav-assistant
chmod +x deploy/install.sh
./deploy/install.sh

# Test immediately
SIMULATE=1 python simulate.py --camera 0

# Run the real assistant
source venv/bin/activate
python src/main.py
```

### Option C — Auto-start on boot

```bash
sudo systemctl start nav-assistant
sudo systemctl status nav-assistant
journalctl -u nav-assistant -f
```

---

## Configuration (`src/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_WIDTH/HEIGHT` | 1280×720 | Capture resolution |
| `DETECT_FPS` | 10 | Inference frames per second |
| `YOLO_CONF_THRESH` | 0.45 | Minimum detection confidence |
| `FEEDBACK_MIN_INTERVAL_S` | 2.5 | Minimum seconds between spoken instructions |
| `WHISPER_MODEL` | `tiny` | STT model size |
| `TTS_RATE` | 165 | Speech rate (words/min) |
| `TOPO_SIMILARITY_THRESH` | 0.72 | Cosine similarity to match a known location |

---

## Voice Commands

| Say | Response |
|-----|----------|
| "describe scene" | Full description of all detected objects |
| "where am I" | Current topological location and visit count |
| "help" | Lists available commands |
| "stop" | Shuts down the assistant |

---

## Topological Map

The map is stored in `data/maps/routes.json` and survives restarts.  Every 5 seconds the system takes a "location fingerprint" (YOLOv8 class-confidence histogram) and either matches it to an existing node or creates a new one.  Over time the graph captures the user's regular routes and can report "you are at Location 3, which connects to Location 1 and Location 7."

To reset the learned map:  `rm data/maps/routes.json`

---

## Project Structure

```
src/
  main.py         — entry point, main loop
  config.py       — all constants (one place to change everything)
  camera.py       — Pi Camera / OpenCV capture
  detector.py     — YOLOv8n inference + navigation postprocessing
  audio_in.py     — microphone capture + Whisper STT
  audio_out.py    — pyttsx3 TTS speaker
  navigator.py    — navigation logic + voice command parser
  topo_map.py     — topological route learning
simulate.py       — visual simulation runner (no RPi needed)
requirements.txt
deploy/
  install.sh
  nav-assistant.service
data/
  models/         — model weight cache
  maps/           — learned topological maps
```
