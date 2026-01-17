## Training: "Looking at screen" classifier (fine-tune)

This folder provides a **simple fine-tuning pipeline** for a binary classifier:

- `looking` vs `not_looking`
- MobileNetV2 backbone (pretrained) + small classifier head
- Export to **TFLite** for Raspberry Pi inference

### 1) Collect data (on the Pi)

```bash
cd pi-agent
python train/collect_data.py --out-dir data --participant p01 --session s01 --placement monitor_top --cycles 6 --looking-seconds 10 --away-seconds 10 --save-face --require-face
```

This will save face crops to:

```
data/run_<timestamp>/face/<participant>/<session>/<placement>/looking/*
data/run_<timestamp>/face/<participant>/<session>/<placement>/not_looking/*
```

### 2) Prepare train/val/test splits (recommended)

This avoids leakage by splitting **by participant** (or by session).

```bash
cd pi-agent
python train/prepare_dataset.py --runs-dir data --out-dir data/splits --split-by participant
```

Outputs:

```
data/splits/train/{looking,not_looking}/*
data/splits/val/{looking,not_looking}/*
data/splits/test/{looking,not_looking}/*
```

### 3) Train + export TFLite (on your laptop/desktop)

```bash
cd pi-agent
python -m venv .venv-train
source .venv-train/bin/activate
pip install tensorflow
python train/train_tf.py --train-dir data/splits/train --val-dir data/splits/val --test-dir data/splits/test --quantize
```

Output:

```
models/focus_model.tflite
```

### 3) Run on the Pi

Copy the TFLite model to the Pi and run the agent with:

```bash
export STUDYBUDDY_MODEL_PATH="/home/pi/models/focus_model.tflite"
export STUDYBUDDY_MODEL_THRESHOLD="0.5"
python -m studybuddy_pi run
```

### Notes
- The model expects **face crops** (from the same collection script).
- If no face is detected at runtime, it counts as **not looking**.
- You can tune `STUDYBUDDY_MODEL_THRESHOLD` to trade off false positives vs false negatives.

### Collecting from participants on their own laptops (recommended for scaling)
For a guided webcam session with preview + face box + beeps:

```bash
cd pi-agent
pip install -r requirements-collect.txt
python -m studybuddy_pi collect-data --participant p02 --session s01 --placement laptop_webcam
```

See `pi-agent/PARTICIPANT_GUIDE.md` for the “send this to anyone” instructions.
