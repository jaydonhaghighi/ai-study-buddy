# AI Study Buddy

AI Study Buddy is a web app that combines AI chat help with webcam-based focus tracking for study sessions.

## Problem Statement

Students often lose focus while studying and do not have a simple way to measure attention and get contextual academic help in one place.

## Solution Statement

This app provides:
- AI chat support for study questions
- Real-time focus tracking from webcam head-pose signals
- Focus session summaries and dashboard analytics
- Course/session-based organization of study chats

## Technology Used

- Frontend: React, TypeScript, Vite
- Backend: Firebase Cloud Functions (Node.js, TypeScript), Genkit, OpenAI
- Platform: Firebase Auth, Firestore, Storage
- Vision + Focus: MediaPipe (browser) + local FastAPI inference service
- ML Pipeline: Python, scikit-learn/pandas/numpy, MLflow, Docker Compose, Makefile

## Start Frontend Web App

From the repo root:

```bash
cp .env.example .env
npm install
npm run dev
```

Open the local URL shown by Vite (usually `http://localhost:5173`).

## Entire ML Pipeline (End-to-End)

Prerequisites: Docker, Docker Compose, GNU Make.

### 0) Optional fresh start

```bash
make ml-clean
```

If cleanup permissions fail:

```bash
make fix-artifact-perms
```

Convenience commands:

```bash
make ml-fresh-pipeline   # ml-clean + ml-pipeline
make ml-fresh-run        # ml-clean + ml-run
make ml-clean-all        # remove all ML artifacts/history
```

### 1) Train + evaluate + export best model

```bash
make mlflow-up
make validate
make split-loso
make train-loso
make eval-loso
make export-best
```

One-command alternatives:

```bash
make ml-pipeline
make ml-run
```

### 2) Preset experiment runs

```bash
make ml-exp-stability-v1
make ml-exp-regularized-v2
make ml-exp-hires-v3
make ml-exp-v4
```

### 3) Generate timestamped report

```bash
make capstone-report
```

Artifacts are saved under:
- `ml/artifacts/reports/runs/<YYYYMMDD-HHMMSS>/capstone_report.md`
- `ml/artifacts/reports/runs/<YYYYMMDD-HHMMSS>/plots/`

### 4) Run local inference and use it in the web app

```bash
make serve-gpu-detached
make smoke-inference
npm run dev
```

In the app, start a focus session to see live pose label/confidence and focus events.

### 5) Run pipeline with a specific config

```bash
make ml-fresh-pipeline ML_CONFIG=/app/configs/exp_stability_v1.yaml
make capstone-report ML_CONFIG=/app/configs/exp_stability_v1.yaml
```