## ENTIRE ML WORKFLOW END-TO-END

#### **0) (Optional) Prep for a fresh run**
From repo root:

```bash
make ml-clean
```

`ml-clean` removes split/training/export outputs and non-versioned report files, while preserving:
- `ml/artifacts/reports/runs/` (timestamped report history)
- `ml/artifacts/mlflow` (MLflow history)

If you ever hit `Permission denied` on cleanup, run once:

```bash
make fix-artifact-perms
```

Useful one-command variants:

```bash
make ml-fresh-pipeline   # ml-clean + ml-pipeline
make ml-fresh-run        # ml-clean + ml-run (includes serve + smoke test)
```

If you really want to delete all ML artifacts/history:

```bash
make ml-clean-all
```

#### **Preset experiment runs (recommended next configs)**
Each command runs a fresh pipeline and generates a timestamped capstone report
with the matching config snapshot:

```bash
make ml-exp-stability-v1
make ml-exp-regularized-v2
make ml-exp-hires-v3
make ml-exp-v4
```

`ml-exp-v4` adds two-stage fine-tuning, label smoothing, and stronger label-safe augmentation.

#### **1) Train + evaluate + export your best model**
From repo root:

```bash
make mlflow-up
make validate
make split-loso
make train-loso
make eval-loso
make export-best
```

(Or if you want it all in one go)

```bash
make ml-pipeline
```

Full one-command run (pipeline + inference startup + smoke test):

```bash
make ml-run
```

If you want to run any command with a specific config file:

```bash
make ml-fresh-pipeline ML_CONFIG=/app/configs/exp_stability_v1.yaml
make capstone-report ML_CONFIG=/app/configs/exp_stability_v1.yaml
```

#### **2) Generate a saved copy of the report (timestamped folder)**
```bash
make capstone-report
```

This creates a new folder every time under:

- `ml/artifacts/reports/runs/<YYYYMMDD-HHMMSS>/capstone_report.md`
- `ml/artifacts/reports/runs/<YYYYMMDD-HHMMSS>/plots/`

#### **3) Run the model locally and use it in the web app (live)**
Start inference:

```bash
make serve-gpu-detached
make smoke-inference
```

Run the web app:

```bash
npm run dev
```

Then in the app:
- Start Focus → you should see **live pose label + confidence**, duration, and the distracted/focused toasts/sounds.

#### **4) Repeat for documentation**
Each time you retrain or tweak configs:
- rerun `make ml-pipeline` (or `make ml-fresh-pipeline` if you want a clean slate first)
- rerun `make capstone-report` to keep an immutable record of that run’s results.