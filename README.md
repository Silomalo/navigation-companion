# Visually Impaired Navigation Assistant

A wearable, fully-offline navigation aid built on Raspberry Pi 5. A chest-mounted
camera streams to an on-device YOLO11n model that detects obstacles and speaks
navigation instructions through an earphone in real time. The user can issue voice
commands (faster-whisper STT) and the device learns familiar routes using a topological map.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Raspberry Pi 5                          │
│                                                              │
│  Pi Camera ──► camera.py                                    │
│                    │  BGR frames @ 10 fps                    │
│                    ▼                                         │
│             detector.py  ◄── YOLO11n ONNX (10 MB, COCO-80) │
│                    │  DetectionResult (zone + proximity)     │
│                    ▼                                         │
│             navigator.py ◄── topo_map.py (route learning)   │
│                    │  speech phrases                         │
│                    ▼                                         │
│             audio_out.py ──► earphone (espeak on RPi)        │
│                                                              │
│  Microphone ──► audio_in.py ──► faster-whisper ──► nav cmd  │
└──────────────────────────────────────────────────────────────┘
```

---

## Hardware

| Item | Part |
|---|---|
| Compute | Raspberry Pi 5 (4 GB+) |
| Camera | Pi Camera Module v3 (12 MP autofocus) |
| Audio out | 3.5 mm earphone (Pi audio jack or USB audio) |
| Microphone | USB or Bluetooth microphone (16 kHz mono) |
| Power | USB-C PD bank (20 000 mAh, ≥ 27 W) |

**Physical setup:** Pi + camera mounted on a chest harness. Thin coiled cable to earphone. Gooseneck mic capsule near the mouth.

---

## Models

All models run **locally — no internet required after first download**.

| Model | Size | Speed on RPi 5 | Purpose |
|---|---|---|---|
| YOLO11n (**ONNX**) | 10 MB | ~30 fps | Obstacle detection — auto-exported on first run |
| faster-whisper tiny | 39 MB | ~200 ms / utterance | Voice command recognition (4× faster than Whisper) |
| espeak-ng (RPi) / `say` (macOS) | built-in | real-time | Text-to-speech navigation |

> **Everything is automatic.** On first run, `Detector.load()` downloads `yolo11n.pt`,
> exports `yolo11n.onnx` in ~1 s, and loads it. The faster-whisper model downloads
> from Hugging Face on first mic use. No manual steps needed.

---

## Detected Obstacles (36 classes)

| Category | Classes |
|---|---|
| People | person |
| Vehicles | bicycle, car, motorcycle, bus, truck, train |
| Animals — urgent | dog, horse, cow, sheep, elephant, bear, zebra, giraffe |
| Animals — caution | cat, bird |
| Road signs | traffic light, stop sign, parking meter |
| Ground hazards | fire hydrant, bench, chair, table, suitcase, skateboard, sports ball |
| Indoor | couch, bed, door, toilet, refrigerator, umbrella, backpack |
| Elevation | stairs *(requires custom-trained model — see `docs/model-improvements.md`)* |

---

## Quick Start

### Desktop simulation (no RPi needed)

```bash
git clone <repo>
cd navigation-companion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Webcam (grant camera + mic permission when prompted on macOS)
SIMULATE=1 python simulate.py --camera 0

# Video file
SIMULATE=1 python simulate.py --video path/to/street.mp4

# No camera — animated synthetic frames
SIMULATE=1 python simulate.py --synthetic
```

**On first run:**
1. `yolo11n.pt` downloads and is exported to `yolo11n.onnx` automatically (~1 s)
2. faster-whisper `tiny` model downloads from Hugging Face on first voice command

Subsequent starts are instant — both models are cached locally.

**Simulation window keyboard shortcuts:**

| Key | Action |
|---|---|
| `D` | Describe scene |
| `W` | Where am I? |
| `H` | Help |
| `Q` / `ESC` | Quit |

Voice commands work with any connected microphone — speak naturally, punctuation
and filler words are ignored by the command parser.

### Raspberry Pi 5

```bash
git clone <repo>
cd navigation-companion
chmod +x deploy/install.sh && ./deploy/install.sh

# Test in simulation mode first
SIMULATE=1 python simulate.py --camera 0

# Run the full assistant
python src/main.py
```

### Auto-start on boot

```bash
sudo systemctl enable nav-assistant
sudo systemctl start nav-assistant
journalctl -u nav-assistant -f   # live logs
```

---

## Configuration (`src/config.py`)

All tunable parameters live in one file — no magic numbers elsewhere.

| Setting | Default | Effect |
|---|---|---|
| `YOLO_MODEL_NAME` | `yolo11n.pt` | Swap to `yolo11s.pt` for +7.5 mAP |
| `DETECT_FPS` | `10` | Lower → less CPU; higher → more responsive |
| `YOLO_CONF_THRESH` | `0.45` | Raise to reduce false positives |
| `DIST_CLOSE_THRESH` | `0.55` | Fraction of frame height = "very close" |
| `FEEDBACK_MIN_INTERVAL_S` | `2.5` | Seconds between spoken instructions |
| `WHISPER_MODEL` | `tiny` | `base` for better accuracy on faster hardware |
| `MIC_SILENCE_THRESH` | `300.0` | Lower if mic is quiet or far from mouth |
| `TTS_RATE` | `165` | Words per minute — lower is clearer |
| `TOPO_SIMILARITY_THRESH` | `0.72` | Lower = more sensitive place-matching |

---

## Voice Commands

Commands are matched by keyword — punctuation and filler words are stripped,
so natural speech like *"can you describe the scene?"* works fine.

| Say | Response |
|---|---|
| *"describe"*, *"what do you see"*, *"surroundings"* | Full count of all detected objects + path status |
| *"where am I"*, *"location"* | Current topological location and visit count |
| *"help"*, *"commands"* | Lists available commands |
| *"stop"*, *"quit"*, *"silence"* | Shuts down the assistant |

---

## Topological Map

The map is stored in `data/maps/routes.json` and persists across restarts.
Every 5 seconds the system takes a visual fingerprint (YOLO11n class-confidence
histogram) and either matches it to a known node or creates a new one. Edges
are recorded between consecutive nodes to capture route order.

Over time the system can say:
> *"You are at Location 3. You have been here 7 times. Connected to: Location 1, Location 5."*

To reset the learned map:
```bash
rm data/maps/routes.json
```

---

## Platform Notes

| Feature | macOS (simulation) | Raspberry Pi 5 (production) |
|---|---|---|
| Camera | OpenCV webcam (`cv2.VideoCapture`) | `picamera2` |
| TTS | macOS built-in `say` command | `pyttsx3` + `espeak-ng` |
| STT | faster-whisper (CTranslate2 + int8) | faster-whisper (CTranslate2 + int8) |
| Microphone | Any USB / Bluetooth mic via `sounddevice` | USB mic via `sounddevice` |
| Detection | YOLO11n ONNX via `onnxruntime` CPU | YOLO11n ONNX via `onnxruntime` CPU |

> **Why different TTS on macOS?** `pyttsx3` uses `NSSpeechSynthesizer` which
> requires the AppKit main run loop and hangs silently in a background thread.
> The built-in `say` command is thread-safe and produces identical quality output.

---

## Project Structure

```
src/
  main.py           — RPi entry point, main perception-action loop
  config.py         — all constants (one place to change everything)
  camera.py         — Pi Camera / OpenCV capture abstraction
  detector.py       — YOLO11n ONNX inference + navigation postprocessing
  audio_in.py       — microphone capture + faster-whisper STT
  audio_out.py      — TTS speaker (macOS `say` / Linux espeak)
  navigator.py      — navigation logic + voice command parser
  topo_map.py       — topological route learning + persistence
simulate.py         — visual simulation runner (OpenCV window, no RPi needed)
requirements.txt
docs/
  model-improvements.md  — accuracy/speed upgrade guide + custom training
deploy/
  install.sh
  nav-assistant.service
data/
  models/           — yolo11n.pt + yolo11n.onnx (auto-generated on first run)
  maps/             — routes.json (learned topological map, persists across restarts)
```
