#!/usr/bin/env bash
# deploy/install.sh — one-shot install on a fresh Raspberry Pi 5 (Bookworm)
set -euo pipefail

echo "=== Navigation Assistant — RPi 5 Setup ==="

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/4] System packages …"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-venv \
    espeak-ng \
    libasound2-dev \
    libportaudio2 \
    libcamera-dev \
    python3-picamera2 \
    v4l-utils

# ── 2. Python virtualenv ──────────────────────────────────────────────────────
echo "[2/4] Creating virtualenv …"
python3 -m venv --system-site-packages venv
source venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

echo ""
echo "[2/4] Preparing YOLO11n ONNX model …"
python - <<'EOF'
import sys
sys.path.insert(0, "src")
from detector import Detector

Detector().load()
print("YOLO11n ONNX ready")
EOF

echo "[2/4] Downloading Whisper tiny weights …"
python - <<'EOF'
from faster_whisper import WhisperModel
WhisperModel("tiny", device="cpu", compute_type="int8")
print("faster-whisper tiny ready")
EOF

# ── 3. Systemd service ────────────────────────────────────────────────────────
echo "[3/4] Installing systemd service …"
sudo cp deploy/nav-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nav-assistant

# ── 4. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Install complete ==="
echo ""
echo "Test simulation (no RPi hardware needed):"
echo "  SIMULATE=1 python simulate.py"
echo ""
echo "Run on RPi hardware:"
echo "  source venv/bin/activate && python src/main.py"
echo ""
echo "Start as service (boots automatically):"
echo "  sudo systemctl start nav-assistant"
echo "  journalctl -u nav-assistant -f"
