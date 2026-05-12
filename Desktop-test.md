# Navigation Companion — Project Documentation

> An AI-powered, offline assistive navigation system for visually impaired users,
> designed to run on a **Raspberry Pi 5** but fully testable on any laptop/desktop.

---

## Table of Contents

1. [What the System Does](#1-what-the-system-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Breakdown](#3-module-breakdown)
   - [config.py](#configpy--central-configuration)
   - [camera.py](#camerapy--camera-abstraction)
   - [detector.py](#detectorpy--yolov8n-obstacle-detection)
   - [navigator.py](#navigatorpy--navigation-logic)
   - [topo_map.py](#topo_mappy--topological-route-memory)
   - [audio_in.py](#audio_inpy--microphone--speech-to-text)
   - [audio_out.py](#audio_outpy--text-to-speech)
   - [main.py](#mainpy--hardware-entry-point)
   - [simulate.py](#simulatepy--desktop-visualisation)
4. [Data Flow](#4-data-flow)
5. [Detection & Spatial Logic](#5-detection--spatial-logic)
6. [Topological Map Explained](#6-topological-map-explained)
7. [Voice Commands](#7-voice-commands)
8. [Running the Project](#8-running-the-project)
9. [Key Configuration Values](#9-key-configuration-values)
10. [Dependencies](#10-dependencies)

---

## 1. What the System Does

The Navigation Companion is a **real-time, fully offline** assistive tool that:

- **Watches** the environment via a camera (RPi camera module or webcam)
- **Detects** obstacles using a YOLOv8n neural network (80 COCO classes)
- **Classifies** each obstacle by its spatial zone (left / ahead / right) and
  distance (very close / ahead / in the distance)
- **Speaks** navigation warnings through earphones via text-to-speech (no
  internet required)
- **Listens** for voice commands through a microphone, transcribed locally by
  OpenAI Whisper (tiny model, 39 MB)
- **Learns routes** by building a topological graph of familiar places, enabling
  "where am I?" awareness across sessions

---

## 2. Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                        Main Loop (10 fps)                  │
│                                                           │
│   Camera ──► Detector ──► Navigator ──► Speaker (TTS)     │
│                  │             │                           │
│              TopoMap      Microphone (STT)                │
└───────────────────────────────────────────────────────────┘
```

**Threading model:**

| Thread | Owned by | Responsibility |
|---|---|---|
| `camera-hw` / `camera-sim` | `Camera` | Continuous frame capture |
| `mic-capture` | `Microphone` | PCM audio blocks from mic |
| `mic-stt` | `Microphone` | Whisper transcription (VAD-gated) |
| `tts` | `Speaker` | Serialise pyttsx3 speech requests |
| **main** | `main.py` | Detect → Navigate → Sleep loop |

---

## 3. Module Breakdown

### `config.py` — Central Configuration

The **single source of truth** for every tunable parameter. No magic numbers
exist in any other module — everything imports from here.

Key sections:

| Section | Notable values |
|---|---|
| Camera | 1280 × 720 @ 30 fps capture, **10 fps** inference sub-sampling |
| YOLOv8 | `yolov8n.pt`, confidence ≥ 0.45, IOU ≤ 0.45, runs on `cpu` |
| Zones | Left edge at 33 %, right edge at 67 % of frame width |
| Proximity | Bounding-box height ≥ 55 % → *very close*; ≥ 30 % → *ahead* |
| Speech | Min 2.5 s between announcements (prevents audio fatigue) |
| Whisper | `tiny` model (39 MB), 16 kHz mono |
| TTS | 165 wpm, volume 0.95, offline via pyttsx3 |
| Topo map | Cosine similarity ≥ 0.72 to match a known place; new node every 5 s |

`OBSTACLE_MAP` maps COCO class names → `(nav_label, priority)` where
priority `1` = urgent (people, vehicles, stairs), `2` = normal (furniture,
signs), `3` = informational (toilet, fridge).

---

### `camera.py` — Camera Abstraction

Provides a **uniform `read()` → BGR ndarray** interface regardless of hardware.

| Environment | Backend used |
|---|---|
| `SIMULATE=0` on RPi | `picamera2` |
| `SIMULATE=1` or no picamera2 | `cv2.VideoCapture` (webcam) |
| No camera at all | Animated HSV gradient (synthetic frames) |

A background daemon thread continuously captures frames into a locked buffer.
Callers never block on I/O — they just call `camera.read()`.

---

### `detector.py` — YOLOv8n Obstacle Detection

Wraps Ultralytics YOLOv8n with navigation-specific post-processing.

**Detection pipeline per frame:**

1. Run YOLO inference on the BGR frame at 640 × 640
2. For each raw bounding box:
   - Discard classes not in `OBSTACLE_MAP`
   - Compute normalised centre-x → classify **Zone** (LEFT / CENTRE / RIGHT)
   - Compute normalised bbox height → classify **Proximity** (CLOSE / MEDIUM / FAR)
   - Build a `Detection` dataclass
3. Return a `DetectionResult` containing all detections + inference time

**Key properties on `Detection`:**

- `is_urgent` → `True` when priority = 1 **and** proximity = CLOSE
- `speech_phrase` → e.g. `"person ahead very close"`

**Feature vector (`feature_vector()`):** An 80-element float32 array of
per-class max-confidence scores. Used by the topological map to fingerprint
what the camera sees at each location.

---

### `navigator.py` — Navigation Logic

The **brain** of the system. Receives a `DetectionResult` every frame, decides
what to say, and respects rate limiting.

**Priority order each frame:**

```
1. Urgent obstacles (priority=1 AND very close AND centre)
   → Speaks immediately, bypasses rate limit, 1.5 s de-dup
2. Normal obstacle summary (rate-limited to FEEDBACK_MIN_INTERVAL_S)
   → "person ahead, car on the left"
3. Path-clear confirmation
   → Only every ~30 frames (~3 s) to avoid noise
```

**Phrase builders:**

| Function | Output example |
|---|---|
| `_build_urgent_phrase` | `"Warning! person ahead. also bicycle"` |
| `_build_normal_phrase` | `"chair ahead, in the distance. bench on the left"` |
| `_full_scene_description` | `"I can see 3 objects: 2 persons, 1 car. The path ahead has person very close."` |

---

### `topo_map.py` — Topological Route Memory

Builds a **persistent graph of recognisable places** the user visits regularly.

**How a node is created:**

1. Every 5 seconds (rate-limited), extract the 80-dim feature vector from the
   current `DetectionResult`
2. Compute cosine similarity against every stored node
3. If best similarity < 0.72 → **new node** created and saved
4. If best similarity ≥ 0.72 → **revisit** counted; node feature updated via
   exponential moving average (α = 0.10)
5. An **edge** is added between the previous node and the current one

**Persistence:** The graph is saved to `data/maps/routes.json` after every
observation and reloaded on the next startup — routes are remembered across
sessions.

**`describe_current_location()` example output:**
> "You are at Location 3. You have been here 7 times. Connected to: Location 2, Location 4."

---

### `audio_in.py` — Microphone & Speech-to-Text

**Hardware mode** (two threads):

| Thread | Role |
|---|---|
| `mic-capture` | `sounddevice.InputStream` → PCM blocks → `_pcm_queue` |
| `mic-stt` | VAD (RMS energy threshold) → collect speech → Whisper transcribe → `_utterance_queue` |

**Simulation mode** (one thread):  
`mic-keyboard` reads lines from `stdin` and places them directly into
`_utterance_queue`. The OpenCV window in `simulate.py` also injects preset
commands via keyboard shortcuts (D / W / H).

**VAD logic:** An utterance starts when RMS > 300 and ends after 1.2 s of
consecutive silence. The collected PCM is then passed to `whisper.transcribe()`.

---

### `audio_out.py` — Text-to-Speech

A single `tts` daemon thread owns the `pyttsx3` engine (which is not
thread-safe) and serialises all speech requests from a queue.

```
speaker.say("chair ahead")               # queued, non-blocking
speaker.say("Warning! person", urgent=True)  # flushes queue, speaks first
```

**Fallback chain:** pyttsx3 → print-only logging (if pyttsx3 unavailable).  
**Simulation:** all spoken text is also `print()`-ed to stdout with a `[TTS]`
prefix so the full pipeline is auditable without speakers.

---

### `main.py` — Hardware Entry Point

Wires all subsystems and runs the main **perception-action loop** at `DETECT_FPS` (10 fps):

```
while running:
    frame  = camera.read()
    result = detector.detect(frame, frame_id)
    nav.update(result)
    for text, urgent in nav.pending_speech():
        speaker.say(text, urgent=urgent)
    cmd = mic.pending_utterance()
    if cmd:
        nav.handle_command(cmd)
        ...
    sleep(remaining_frame_budget)
```

Handles `SIGINT` / `SIGTERM` for clean shutdown. Logs FPS telemetry every
10 seconds.

---

### `simulate.py` — Desktop Visualisation

A **rich OpenCV visualisation** of the full pipeline — no RPi needed.

**Window layout:**

```
┌─────────────────────────────┬────────────────────┐
│                             │ NAV ASSISTANT SIM  │
│   Camera/Webcam Frame       │ FPS / Inf time      │
│   with bounding boxes       │ Topological map     │
│   and zone dividers         │ Detection list      │
│                             │ Speech log          │
│                             │ Keyboard shortcuts  │
└─────────────────────────────┴────────────────────┘
```

**Bounding box colour coding:**

| Colour | Meaning |
|---|---|
| 🔴 Red | Urgent (priority 1 + very close) |
| 🟠 Orange | Normal priority 2 obstacle |
| 🟡 Yellow | Informational priority 3 |

**Keyboard shortcuts:**

| Key | Injected command |
|---|---|
| `D` | `"describe scene"` |
| `W` | `"where am I"` |
| `H` | `"help"` |
| `Q` / `ESC` | Quit |

---

## 4. Data Flow

```
Camera (BGR frame)
    │
    ▼
Detector.detect()
    │  returns DetectionResult
    │    ├─ list[Detection]  (zone, proximity, label, bbox)
    │    └─ feature_vector() (80-dim float32)
    │
    ├──► TopoMap.observe()       (route learning)
    │
    └──► Navigator.update()
              │
              ├─ Urgent?  → Speaker.say(urgent=True)
              ├─ Normal?  → Speaker.say()  [rate-limited]
              └─ Clear?   → Speaker.say()  [every ~3 s]

Microphone.pending_utterance()
    │  "describe scene"
    ▼
Navigator.handle_command()
    └──► Speaker.say(full_scene_description)
```

---

## 5. Detection & Spatial Logic

### Zone Classification

The frame is divided into three vertical columns:

```
0%        33%       67%       100%
│  LEFT   │  AHEAD  │  RIGHT  │
│ <0.33   │0.33–0.67│  >0.67  │
```

Classification is based on the **normalised centre-x** of the bounding box.

### Proximity Classification

Based on the **normalised bounding-box height** (taller = closer):

| Condition | Proximity |
|---|---|
| bbox_h ≥ 0.55 | `CLOSE` ("very close") |
| bbox_h ≥ 0.30 | `MEDIUM` ("ahead") |
| bbox_h < 0.30 | `FAR` ("in the distance") |

### Urgency

A detection is `is_urgent = True` when **all three** conditions are met:
- Priority level = 1 (person, vehicle, stairs)
- Zone = CENTRE (blocking the path ahead)
- Proximity = CLOSE (≥ 55 % of frame height)

---

## 6. Topological Map Explained

The topological map is how a visually impaired person mentally structures a
familiar environment — not GPS coordinates, but a sequence of recognisable
*places* linked by *transitions*.

```
[Location 0] ── [Location 1] ── [Location 3]
                     │
               [Location 2]
```

Each **node** stores:
- `feature`: 80-dim histogram of YOLO class confidences seen here
- `label`: auto-generated `"Location N"` (can be renamed)
- `visit_count`: how many times the user has passed through
- `neighbours`: connected node IDs (traversal graph)

The feature vector acts as a **visual fingerprint** — the same corridor will
consistently produce similar YOLO detections (same furniture, same layout),
allowing recognition without GPS.

---

## 7. Voice Commands

| Utterance (examples) | Response |
|---|---|
| `"describe"`, `"what do you see"`, `"surroundings"` | Full scene count + path status |
| `"where am I"`, `"location"`, `"route"` | Topological location description |
| `"stop"`, `"quit"`, `"silence"` | Stops navigation |
| `"help"`, `"commands"` | Lists available commands |
| Anything else | `"I heard: <text>. Say help for commands."` |

Commands are matched by **keyword intersection** — partial matches work, e.g.
`"what can you see around me"` correctly triggers the describe flow.

---

## 8. Running the Project

### Install dependencies

```
pip install -r requirements.txt
```

> `picamera2` is pre-installed on Raspberry Pi OS Bookworm. Uncomment it in
> `requirements.txt` only if installing into a clean virtualenv on the RPi.

### Desktop simulation (webcam)

```
SIMULATE=1 python simulate.py
```

### Desktop simulation (video file)

```
SIMULATE=1 python simulate.py --video path/to/video.mp4
```

### Desktop simulation (no camera — synthetic frames)

```
SIMULATE=1 python simulate.py --synthetic
```

### Hardware mode (Raspberry Pi 5)

```
python src/main.py
```

### Debug logging

```
DEBUG=1 SIMULATE=1 python simulate.py
```

---

## 9. Key Configuration Values

All values live in `src/config.py`.

| Parameter | Default | Effect |
|---|---|---|
| `DETECT_FPS` | `10` | Inference rate — lower = less CPU |
| `YOLO_CONF_THRESH` | `0.45` | Raise to reduce false positives |
| `DIST_CLOSE_THRESH` | `0.55` | Raise to trigger warnings later |
| `FEEDBACK_MIN_INTERVAL_S` | `2.5` | Raise to reduce speech frequency |
| `WHISPER_MODEL` | `"tiny"` | `"base"` improves accuracy on faster hardware |
| `TOPO_SIMILARITY_THRESH` | `0.72` | Lower = more sensitive place-matching |
| `TTS_RATE` | `165` | Lower = slower, clearer speech |

---

## 10. Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8n model inference |
| `opencv-python` | Camera capture, image processing, simulation window |
| `numpy` | Array operations, feature vectors |
| `openai-whisper` | Local speech-to-text (no internet) |
| `sounddevice` | Cross-platform microphone capture (ALSA / CoreAudio) |
| `pyttsx3` | Offline text-to-speech (espeak on Linux, SAPI5 on Windows) |
| `picamera2` | RPi camera module (hardware only, pre-installed on RPi OS) |
