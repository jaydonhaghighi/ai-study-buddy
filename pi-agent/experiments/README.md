## Experiments automation (local loop)

This folder helps you iterate on the attention model in a repeatable loop:

**split → train → test/evaluate → document → decide next experiment → repeat**

All runs are tracked in **MLflow** using:
- **SQLite tracking store**: `pi-agent/mlflow.db`
- **File artifact store**: `pi-agent/mlflow_artifacts/`

### Quickstart

From the repo root:

```bash
cd pi-agent
python experiments/run_experiment.py --dry-run
```

### One command (runs everything)

```bash
cd pi-agent
python experiments/all.py --dry-run
```

### Run one real experiment (requires TensorFlow installed)
```bash
cd pi-agent
python experiments/run_experiment.py \
  --splits-dir data/splits \
  --backbone mobilenetv3small \
  --input-size 224 \
  --quantize \
  --tag baseline
```

### Prepare splits (optional helper)

If you have collected runs under `pi-agent/data/run_*`, you can generate `data/splits/`:

```bash
cd pi-agent
python experiments/run_experiment.py --prepare-splits --runs-dir data --split-by participant --dry-run
```

### Run a sweep (small grid)

```bash
cd pi-agent
python experiments/sweep.py --dry-run
```

### Where to look
- **MLflow UI**: run:
  - `mlflow ui --backend-store-uri sqlite:///mlflow.db --artifacts-destination mlflow_artifacts --host 127.0.0.1 --port 5000`
  - then open the browser at `http://127.0.0.1:5000`
- **Batch summaries**: sweep/LOPO/auto write a `brief.md` artifact on the parent run, plus `summary.json`.

### LOPO evaluation (best accuracy estimate for new users)

Run **leave-one-participant-out** for a single config:

```bash
cd pi-agent
python experiments/all.py --lopo --mode one --prepare-splits --runs-dir data --tag lopo_baseline --backbone mobilenetv3small --input-size 224 --quantize
```

### Head-pose baseline (often better LOPO generalization)

This is a geometry-first baseline: **face landmarks → head pose (yaw/pitch) → 5-way bins**.
It’s usually much more stable across new participants than a CNN classifier when data is small.

Run LOPO with head-pose:

```bash
cd pi-agent
python experiments/lopo_headpose.py --runs-dir data --tag lopo_headpose_v1 --download-task
```

Or evaluate once on an existing `splits/` folder:

```bash
cd pi-agent
python experiments/headpose_eval.py --splits-dir /path/to/splits --tag headpose_eval --download-task
```

### Auto strategy: sweep → pick top K → LOPO

This runs a sweep, picks the top K configs by mean macro-F1, then runs LOPO on those configs. Everything is tracked in MLflow with nested runs.

```bash
cd pi-agent
python experiments/auto.py --sweep-tag sweep_auto --top-k 2 --quantize
```

### Cleanup

Preview what would be deleted (safe dry-run):

```bash
cd pi-agent
python experiments/cleanup.py --prune-artifacts --prune-runs --keep 5 --pycache
```

Actually delete:

```bash
cd pi-agent
python experiments/cleanup.py --prune-artifacts --prune-runs --keep 5 --pycache --apply
```

### Start fresh (reset MLflow)

Dry-run (shows what would be deleted):

```bash
cd pi-agent
python experiments/reset_mlflow.py
```

Actually delete `mlflow.db` + `mlflow_artifacts/`:

```bash
cd pi-agent
python experiments/reset_mlflow.py --apply
```

