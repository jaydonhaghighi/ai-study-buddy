### AI Study Buddy — Raspberry Pi Agent (Python)

This folder contains the **Raspberry Pi focus-tracking agent**, written in Python.

### What it does (high level)
- Pairs the device (claim code → device token).
- Polls the backend for an **active focus session** assigned to this device.
- When active, captures frames (optional), runs inference (optional), applies temporal focus logic, and logs state transitions **locally**.
- When the focus session ends, computes an **aggregated summary** and **POSTs** it to the backend with retries.

### Design goals
- Outbound HTTPS only (no inbound ports on the Pi)
- Privacy-preserving (no raw video uploaded; only aggregated summary JSON)
- Reliable under intermittent connectivity (local queue + retries)

### Quick start (development / simulation mode)
1) Create a venv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Set config (example):

```bash
export STUDYBUDDY_BASE_URL="https://<your-cloud-functions-host>"
export STUDYBUDDY_CLAIM_CODE="ABCD-1234"
export STUDYBUDDY_SIMULATE=1
```

3) Pair and run:

```bash
python -m studybuddy_pi pair
python -m studybuddy_pi run
```

### Running on Raspberry Pi (production-like)
- Prefer `tflite-runtime` rather than full TensorFlow (later, when the model is added).
- Install OpenCV for encoding/face detection (`opencv-python`).
- Use the provided systemd unit in `systemd/ai-study-buddy.service`.

### Training a simple "looking at screen" model (fine-tune)
If you need a **deep-learning model** (fine-tuned) instead of heuristics, use the scripts in `train/`:

- `train/train_tf.py` fine-tunes MobileNetV2 and exports TFLite.

See `train/README.md` for step-by-step instructions.

### If OpenCV can't read frames (common on PiCam)
This agent uses **Picamera2/libcamera** as the camera capture stack (recommended on Raspberry Pi).

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

### Calibration preview (local demo)
For the demo, the Pi agent can expose a **LAN-only live camera preview** (MJPEG) to help the student align the camera before eye tracking.

- This preview is **not saved**.
- The web UI will auto-stop the preview after alignment is “good” for ~3 seconds.
- Only use this on a trusted local network during demos.

Enable (defaults to on):

```bash
export STUDYBUDDY_ENABLE_PREVIEW_SERVER=1
export STUDYBUDDY_PREVIEW_PORT=8080
export STUDYBUDDY_CAMERA_FORMAT=RGB888
```

Pi endpoints:
- `POST /start` (begin preview capture)
- `POST /stop` (stop preview capture)
- `GET /status` (returns `{ faceDetected, aligned, ... }`)
- `GET /stream.mjpg` (MJPEG stream)

### Backend endpoints expected (PRD-aligned)
Firebase Cloud Functions expose endpoints using the **exported function name**:
- `POST /deviceRegister` (claim code registration)
- `GET /devicePairingStatus` (poll until paired, returns device token once)
- `GET /deviceCurrentFocusSession` (poll/long-poll assignment)
- `POST /deviceSessionSummary` (upload aggregated summary)

Focus session control (web -> backend):
- `POST /deviceClaim` (user claims device by claim code)
- `POST /focusStart` (start focus tracking; assigns active focusSessionId to device)
- `POST /focusStop` (stop focus tracking; clears device assignment)


