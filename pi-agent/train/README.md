## Training: attention direction classifier (5-way, fine-tune)

This folder provides a **simple fine-tuning pipeline** for a 5-way attention direction classifier:

- `screen`
- `away_left`
- `away_right`
- `away_up`
- `away_down`

It uses a **MobileNetV2 backbone** (ImageNet pretrained) + a small softmax head, and exports to **TFLite** for Raspberry Pi inference.

Important: label order is fixed across training + inference and must match:

```python
["screen", "away_left", "away_right", "away_up", "away_down"]
```

### 1) Collect data (on the Pi)

Data collection is now done via the **web-based JS collector** (more reliable on macOS/Windows).

1) Run the web app and open the collector page at:
- `http://localhost:5173/collect` (dev), or
- your deployed site at `/collect`

2) Fill in `participant` / `session` / `placement` and run the guided flow.

3) It downloads a zip like `run_<timestamp>.zip`. Unzip it and copy the `run_<timestamp>/` folder into:

```
pi-agent/data/
```

So you end up with:

```
data/run_<timestamp>/face/<participant>/<session>/<placement_condition>/screen/*
data/run_<timestamp>/face/<participant>/<session>/<placement_condition>/away_left/*
data/run_<timestamp>/face/<participant>/<session>/<placement_condition>/away_right/*
data/run_<timestamp>/face/<participant>/<session>/<placement_condition>/away_up/*
data/run_<timestamp>/face/<participant>/<session>/<placement_condition>/away_down/*
```

### 2) Prepare train/val/test splits (recommended)

This avoids leakage by splitting **by participant** (or by session).

```bash
cd pi-agent
python train/prepare_dataset.py --runs-dir data --out-dir data/splits --split-by participant
```

Outputs:

```
data/splits/train/{screen,away_left,away_right,away_up,away_down}/*
data/splits/val/{screen,away_left,away_right,away_up,away_down}/*
data/splits/test/{screen,away_left,away_right,away_up,away_down}/*
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
- The model expects **face crops** (from the same collection pipeline). At runtime, the Pi agent detects a face and classifies the face crop into one of the 5 labels.
- If no face is detected at runtime, the agent treats it as **not focused**.
- You can tune `STUDYBUDDY_MODEL_THRESHOLD` to trade off false positives vs false negatives for the `screen` class.
- Augmentations are designed to be label-safe (notably: **no horizontal flips**, since that would swap left/right).

### Optional: clean bad samples before splitting

You can automatically flag (and optionally quarantine) low-quality or suspiciously labeled images:

- **Blur check**: Laplacian variance threshold
- **Label check**: head-pose-derived label mismatch

Dry-run (report only):

```bash
cd pi-agent
python train/clean_dataset.py \
  --runs-dir data_aligned_m025_v2 \
  --merge-screen-down-focus \
  --blur-laplacian-var-min 80 \
  --out-report artifacts/data_clean/report.json \
  --out-csv artifacts/data_clean/report.csv
```

Apply quarantine (move flagged images):

```bash
cd pi-agent
python train/clean_dataset.py \
  --runs-dir data_aligned_m025_v2 \
  --merge-screen-down-focus \
  --quarantine-dir artifacts/data_clean/quarantine \
  --apply
```

Tip: start with dry-run, inspect `report.csv`, then run with `--apply`.
