# =============================================================================
# AI Study Buddy - ML Pipeline Makefile
# =============================================================================
# Override dataset root: PI_AGENT_DATA_ROOT=/path/to/run_*_data make <target>
# Override config:       ML_CONFIG=/app/configs/exp_v4.yaml make <target>
# =============================================================================

# Dataset root: in-repo ml/artifacts/data (run_*/face/...) or override with PI_AGENT_DATA_ROOT
PI_AGENT_DATA_ROOT ?= $(CURDIR)/ml/artifacts/data
DOCKER_COMPOSE ?= docker compose
LOCAL_UID ?= $(shell id -u)
LOCAL_GID ?= $(shell id -g)
ML_COMPOSE = PI_AGENT_DATA_ROOT="$(PI_AGENT_DATA_ROOT)" LOCAL_UID="$(LOCAL_UID)" LOCAL_GID="$(LOCAL_GID)" $(DOCKER_COMPOSE) -f ml/docker-compose.yml
TRAINER_CLI = $(ML_COMPOSE) run --rm trainer studybuddy-ml
INFERENCE_URL ?= http://localhost:8001
REPORT_TAG ?= $(shell date +%Y%m%d-%H%M%S)
ML_CONFIG ?= /app/configs/baseline.yaml
ML_CONFIG_STABILITY_V1 = /app/configs/exp_stability_v1.yaml
ML_CONFIG_REGULARIZED_V2 = /app/configs/exp_regularized_v2.yaml
ML_CONFIG_HIRES_V3 = /app/configs/exp_hires_v3.yaml
ML_CONFIG_V4 = /app/configs/exp_v4.yaml

.PHONY: compose-check ml-build mlflow-up mlflow-down validate split-loso train-loso eval-loso train-production export-best serve-gpu serve-gpu-detached smoke-inference ml-pipeline ml-run capstone-report ml-clean ml-clean-all fix-artifact-perms ml-fresh-pipeline ml-fresh-run ml-exp-stability-v1 ml-exp-regularized-v2 ml-exp-hires-v3 ml-exp-v4

# -----------------------------------------------------------------------------
# Prerequisites & infrastructure
# -----------------------------------------------------------------------------

# Check that docker compose is available; fail with a hint if not
compose-check:
	@$(DOCKER_COMPOSE) version >/dev/null 2>&1 || ( \
		echo "Docker compose command not available."; \
		echo "Try: DOCKER_COMPOSE='docker-compose' make <target>"; \
		echo "or install Docker Compose plugin so 'docker compose' works."; \
		exit 1; \
	)

# Build Docker images for trainer and inference services
ml-build: compose-check
	$(ML_COMPOSE) build trainer inference

# Start MLflow server in background (UI at http://localhost:5001)
mlflow-up: compose-check
	$(ML_COMPOSE) up -d mlflow

# Stop MLflow server
mlflow-down: compose-check
	$(ML_COMPOSE) stop mlflow

# -----------------------------------------------------------------------------
# Data & training pipeline (run in order, or use ml-pipeline)
# -----------------------------------------------------------------------------

# Scan dataset, validate labels/images, write manifest.csv and data_validation.json
validate: compose-check
	$(TRAINER_CLI) validate \
		--dataset-root /datasets \
		--manifest-out /app/artifacts/data/manifest.csv \
		--report-out /app/artifacts/reports/data_validation.json

# Generate leave-one-subject-out folds (fold_00.json, fold_01.json, ...) for evaluation
split-loso: compose-check
	$(TRAINER_CLI) split-loso \
		--manifest-csv /app/artifacts/data/manifest.csv \
		--participant-column participant_id \
		--split-out-dir /app/artifacts/splits

# Train one model per LOSO fold; log to MLflow; write loso_summary.csv and fold checkpoints
train-loso: compose-check
	$(TRAINER_CLI) train-loso \
		--manifest-csv /app/artifacts/data/manifest.csv \
		--split-dir /app/artifacts/splits \
		--config-path $(ML_CONFIG) \
		--output-dir /app/artifacts/training \
		--tracking-uri http://mlflow:5000

# Aggregate fold metrics (mean/std) into loso_aggregate.json
eval-loso: compose-check
	$(TRAINER_CLI) eval-loso \
		--summary-csv /app/artifacts/training/loso_summary.csv \
		--aggregate-out /app/artifacts/reports/loso_aggregate.json

# Train one production model on full dataset; writes to artifacts/training/production/
train-production: compose-check
	$(TRAINER_CLI) train-production \
		--manifest-csv /app/artifacts/data/manifest.csv \
		--config-path $(ML_CONFIG) \
		--output-dir /app/artifacts/training \
		--tracking-uri http://mlflow:5000

# Copy best model to artifacts/export/ (prefers production model; fallback: best LOSO fold by val metric)
export-best: compose-check
	$(TRAINER_CLI) export-best \
		--summary-csv /app/artifacts/training/loso_summary.csv \
		--export-dir /app/artifacts/export \
		--criterion val_macro_f1 \
		--production-summary-json /app/artifacts/training/production/production_summary.json \
		--prefer-production

# -----------------------------------------------------------------------------
# Inference & one-shot runs
# -----------------------------------------------------------------------------

# Start inference API in foreground (Ctrl+C to stop); serves at INFERENCE_URL (default 8001)
serve-gpu: compose-check
	$(ML_COMPOSE) up inference

# Start inference API in background (detached)
serve-gpu-detached: compose-check
	$(ML_COMPOSE) up -d inference

# Hit /health and /predict to verify inference API is working
smoke-inference:
	bash ml/scripts/smoke_inference.sh "$(INFERENCE_URL)"

# -----------------------------------------------------------------------------
# Full pipeline (build → MLflow → validate → split → train LOSO → eval → train production → export)
ml-pipeline: compose-check
	$(MAKE) ml-build
	$(MAKE) mlflow-up
	$(MAKE) validate
	$(MAKE) split-loso
	$(MAKE) train-loso
	$(MAKE) eval-loso
	$(MAKE) train-production
	$(MAKE) export-best

# ml-pipeline + start inference in background + run smoke test
ml-run: compose-check
	$(MAKE) ml-pipeline
	$(MAKE) serve-gpu-detached
	$(MAKE) smoke-inference
	@echo "Done. MLflow: http://localhost:5001 | Inference: $(INFERENCE_URL)"

# -----------------------------------------------------------------------------
# Cleanup & report
# -----------------------------------------------------------------------------

# Remove splits, training, export, manifest, and report files; keep reports/runs and mlflow
ml-clean:
	-$(MAKE) fix-artifact-perms
	rm -rf ml/artifacts/splits ml/artifacts/training ml/artifacts/export ml/artifacts/reports/plots
	rm -f ml/artifacts/data/manifest.csv ml/artifacts/reports/data_validation.json ml/artifacts/reports/loso_aggregate.json ml/artifacts/reports/capstone_report.md
	@echo "Cleaned training/eval/export artifacts."
	@echo "Preserved: ml/artifacts/reports/runs and ml/artifacts/mlflow"

# Remove entire ml/artifacts (including MLflow DB and timestamped reports)
ml-clean-all:
	-$(MAKE) fix-artifact-perms
	rm -rf ml/artifacts
	@echo "Removed all ML artifacts, including MLflow history and timestamped reports."

# Fix ownership of ml/artifacts so host user can delete files (run if clean fails with permission errors)
fix-artifact-perms: compose-check
	$(ML_COMPOSE) run --rm --user root trainer bash -lc "mkdir -p /app/artifacts && chown -R $(LOCAL_UID):$(LOCAL_GID) /app/artifacts"
	@echo "Normalized ml/artifacts ownership to UID=$(LOCAL_UID) GID=$(LOCAL_GID)."

# Clean then run full pipeline (no serve/smoke)
ml-fresh-pipeline: compose-check
	$(MAKE) ml-clean
	$(MAKE) ml-pipeline

# Clean then run full pipeline + serve + smoke test
ml-fresh-run: compose-check
	$(MAKE) ml-clean
	$(MAKE) ml-run

# -----------------------------------------------------------------------------
# Preset experiments (clean, run pipeline with specific config, generate capstone report)
# -----------------------------------------------------------------------------

# Experiment: stability-focused config
ml-exp-stability-v1: compose-check
	$(MAKE) ml-fresh-pipeline ML_CONFIG=$(ML_CONFIG_STABILITY_V1)
	$(MAKE) capstone-report ML_CONFIG=$(ML_CONFIG_STABILITY_V1)

# Experiment: regularized / label-safe augmentation
ml-exp-regularized-v2: compose-check
	$(MAKE) ml-fresh-pipeline ML_CONFIG=$(ML_CONFIG_REGULARIZED_V2)
	$(MAKE) capstone-report ML_CONFIG=$(ML_CONFIG_REGULARIZED_V2)

# Experiment: higher-resolution input
ml-exp-hires-v3: compose-check
	$(MAKE) ml-fresh-pipeline ML_CONFIG=$(ML_CONFIG_HIRES_V3)
	$(MAKE) capstone-report ML_CONFIG=$(ML_CONFIG_HIRES_V3)

# Experiment: two-stage fine-tuning, label smoothing, stronger augmentation (recommended)
ml-exp-v4: compose-check
	$(MAKE) ml-fresh-pipeline ML_CONFIG=$(ML_CONFIG_V4)
	$(MAKE) capstone-report ML_CONFIG=$(ML_CONFIG_V4)

# Generate capstone_report.md and plots in artifacts/reports/runs/$(REPORT_TAG)/
capstone-report: compose-check
	$(TRAINER_CLI) capstone-report \
		--summary-csv /app/artifacts/training/loso_summary.csv \
		--folds-dir /app/artifacts/training/folds \
		--run-dir /app/artifacts/reports/runs/$(REPORT_TAG) \
		--criterion test_macro_f1 \
		--config-path $(ML_CONFIG) \
		--data-validation-json /app/artifacts/reports/data_validation.json
