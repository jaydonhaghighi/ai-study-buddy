# StudyBuddy ML Pipeline (Lean + Capstone-Ready)

This directory contains a Dockerized TensorFlow/Keras pipeline for:

1. Data ingest + validation
2. LOSO (participant-held-out) split generation
3. Baseline training with MLflow tracking
4. Core metrics evaluation (macro-F1, precision/recall, confusion matrix)
5. Best checkpoint export
6. Local GPU inference API with temporal logic

## Dataset expectations

The pipeline expects:

- dataset root with `run_*/meta.jsonl`
- each `meta.jsonl` line includes at least:
  - `label`
  - `participant`
  - `facePath`
- labels in:
  - `screen`
  - `away_left`
  - `away_right`
  - `away_up`
  - `away_down`

**Data location:** By default the Makefile uses `ml/artifacts/data` as the dataset root (so your `run_*/face/...` data lives next to the manifest). To use a different folder, set `PI_AGENT_DATA_ROOT` when running make (e.g. `PI_AGENT_DATA_ROOT=/path/to/data make validate`).

## Quick start (from repo root)

Default dataset root is `ml/artifacts/data`. Override if needed:

```bash
# optional: use a different dataset root
export PI_AGENT_DATA_ROOT=/path/to/your/run_*_data
```

Then run:

```bash
make ml-run
```

`make ml-run` performs build, validation, LOSO split, training, evaluation, best-model export,
starts inference in detached mode, and runs a tiny inference smoke test.

If you prefer step-by-step execution:

```bash
make ml-build
make mlflow-up
make validate
make split-loso
make train-loso
make eval-loso
make export-best
make serve-gpu
```

If your machine uses `docker-compose` (instead of `docker compose`), run:

```bash
DOCKER_COMPOSE=docker-compose make ml-build
```

MLflow UI: `http://localhost:5001`
Inference API: `http://localhost:8001`

Run smoke test manually at any time:

```bash
make smoke-inference
```

Generate capstone-ready Markdown + plots (LOSO report):

```bash
make capstone-report
```

Clean artifacts before a fresh run:

```bash
make ml-clean
make ml-fresh-pipeline
```

Notes:
- `ml-clean` removes split/training/export outputs and non-versioned report files.
- It preserves `ml/artifacts/reports/runs/` (timestamped report history) and `ml/artifacts/mlflow/`.
- Use `make ml-clean-all` only when you want to delete all ML artifacts, including history.
- If cleanup fails with `Permission denied`, run `make fix-artifact-perms` once.

Run recommended next experiment presets (fresh pipeline + timestamped report):

```bash
make ml-exp-stability-v1
make ml-exp-regularized-v2
make ml-exp-hires-v3
make ml-exp-v4
```

`ml-exp-v4` adds two-stage fine-tuning, label smoothing, and stronger label-safe augmentation.

Use a custom config without editing `baseline.yaml`:

```bash
make ml-fresh-pipeline ML_CONFIG=/app/configs/exp_stability_v1.yaml
make capstone-report ML_CONFIG=/app/configs/exp_stability_v1.yaml
```

## App integration (Milestone 6)

The web app can stream webcam frames to the inference API during focus sessions.

Frontend env (`.env` in repo root):

```bash
VITE_INFERENCE_API_BASE_URL=http://localhost:8001
```

Inference API CORS is controlled with:

```bash
STUDYBUDDY_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

This is already wired in `ml/docker-compose.yml` with sensible defaults for local Vite dev.

## CLI commands (inside container or local venv)

```bash
studybuddy-ml validate
studybuddy-ml split-loso
studybuddy-ml train-loso
studybuddy-ml eval-loso
studybuddy-ml export-best
studybuddy-ml serve
```

Use `studybuddy-ml --help` for full options.

## Artifacts produced

- `ml/artifacts/data/manifest.csv`
- `ml/artifacts/reports/data_validation.json`
- `ml/artifacts/splits/fold_*.json`
- `ml/artifacts/training/folds/fold_*/best_model.keras`
- `ml/artifacts/training/loso_summary.csv`
- `ml/artifacts/reports/loso_aggregate.json`
- `ml/artifacts/export/best_model.keras`
- `ml/artifacts/export/saved_model/`
- `ml/artifacts/export/model_meta.json`

## Temporal logic defaults (in inference API)

- EMA smoothing alpha: `0.35`
- Distract transition threshold: `2.5s`
- Refocus threshold: `1.0s`
- Confidence threshold: `0.55`
- Min dwell between transitions: `0.5s`

These can be tuned after first LOSO baseline.


Best way is to build it in **milestones with strict acceptance checks**, not all at once.

## Recommended Development Approach

- Split work into **2 systems + 1 contract layer**:
  - `training pipeline` (TensorFlow/Keras + LOSO + MLflow)
  - `inference service` (GPU API + temporal logic)
  - `shared schemas` (session/summary payloads so frontend/backend stay aligned)
- Deliver one vertical slice at a time, with runnable commands after each slice.
- Keep it lean: one strong baseline model first, then only a few high-impact improvements.

## Milestone Plan I’d Execute

1. **Scaffold + standards (day 1)**
- Create `ml/` package structure, configs, CLI entrypoints, MLflow setup.
- Define canonical labels and summary schema.

2. **Data ingest + validate + LOSO splits (day 1-2)**
- Build manifest generator and validator.
- Generate participant-held-out folds.
- Output validation + split artifacts.

3. **Baseline training + MLflow logging (day 2-4)**
- Train one Keras baseline per LOSO fold.
- Log fold metrics/artifacts in MLflow.
- Compute aggregate mean/std macro-F1.

4. **Evaluation + export (day 4-5)**
- Save best checkpoints and exported model.
- Produce confusion matrices + per-class metrics report.

5. **GPU inference API + temporal logic (day 5-8)**
- FastAPI service (WebSocket at 5 FPS).
- Add smoothing + hysteresis state machine.
- Emit aggregated session summaries only (no raw frame storage).

6. **App integration + dashboard path (day 8-10)**
- Wire webcam stream to inference service.
- Keep `focusStart`/`focusStop` flow.
- Persist summaries to Firestore and verify dashboard.

7. **Capstone polish (day 10-12)**
- Reproducible commands, docs, demo script, progression charts from MLflow.

## Why this is the best path

- Minimizes risk by validating each layer early.
- Gives you professor-friendly evidence: reproducibility, progression, and metrics.
- Keeps scope lean while still looking like real engineering.

## MLflow vs alternatives

Use **MLflow** (local) for this project.
- It’s perfect for capstone documentation of training progression.
- Lower friction than WandB for your setup.
- Easy to defend in report: params, metrics, artifacts, fold comparisons, best-model lineage.

## What I need from you to start implementation

- Dataset root path
- Annotation format sample (5-10 lines)
- Participant ID field naming
- Python version on your 3070 Ti server
- Whether you want Docker now or later

If you want, I’ll start with Milestone 1 + 2 immediately and set up the exact command surface (`validate`, `split-loso`, `train-loso`, `eval-loso`, `export`, `serve`).