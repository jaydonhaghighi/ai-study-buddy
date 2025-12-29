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
- Prefer `tflite-runtime` rather than full TensorFlow.
- Install OpenCV for camera capture if needed.
- Use the provided systemd unit in `systemd/ai-study-buddy.service`.

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


