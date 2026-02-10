PI_AGENT_DATA_ROOT ?= /home/haghighi/Documents/pi-agent-data
DOCKER_COMPOSE ?= docker compose
ML_COMPOSE = PI_AGENT_DATA_ROOT="$(PI_AGENT_DATA_ROOT)" $(DOCKER_COMPOSE) -f ml/docker-compose.yml
TRAINER_CLI = $(ML_COMPOSE) run --rm trainer studybuddy-ml
INFERENCE_URL ?= http://localhost:8001

.PHONY: compose-check ml-build mlflow-up mlflow-down validate split-loso train-loso eval-loso export-best serve-gpu serve-gpu-detached smoke-inference ml-pipeline ml-run

compose-check:
	@$(DOCKER_COMPOSE) version >/dev/null 2>&1 || ( \
		echo "Docker compose command not available."; \
		echo "Try: DOCKER_COMPOSE='docker-compose' make <target>"; \
		echo "or install Docker Compose plugin so 'docker compose' works."; \
		exit 1; \
	)

ml-build: compose-check
	$(ML_COMPOSE) build trainer inference

mlflow-up: compose-check
	$(ML_COMPOSE) up -d mlflow

mlflow-down: compose-check
	$(ML_COMPOSE) stop mlflow

validate: compose-check
	$(TRAINER_CLI) validate \
		--dataset-root /datasets \
		--manifest-out /app/artifacts/data/manifest.csv \
		--report-out /app/artifacts/reports/data_validation.json

split-loso: compose-check
	$(TRAINER_CLI) split-loso \
		--manifest-csv /app/artifacts/data/manifest.csv \
		--participant-column participant_id \
		--split-out-dir /app/artifacts/splits

train-loso: compose-check
	$(TRAINER_CLI) train-loso \
		--manifest-csv /app/artifacts/data/manifest.csv \
		--split-dir /app/artifacts/splits \
		--config-path /app/configs/baseline.yaml \
		--output-dir /app/artifacts/training \
		--tracking-uri http://mlflow:5000

eval-loso: compose-check
	$(TRAINER_CLI) eval-loso \
		--summary-csv /app/artifacts/training/loso_summary.csv \
		--aggregate-out /app/artifacts/reports/loso_aggregate.json

export-best: compose-check
	$(TRAINER_CLI) export-best \
		--summary-csv /app/artifacts/training/loso_summary.csv \
		--export-dir /app/artifacts/export \
		--criterion test_macro_f1

serve-gpu: compose-check
	$(ML_COMPOSE) up inference

serve-gpu-detached: compose-check
	$(ML_COMPOSE) up -d inference

smoke-inference:
	bash ml/scripts/smoke_inference.sh "$(INFERENCE_URL)"

ml-pipeline: compose-check
	$(MAKE) ml-build
	$(MAKE) mlflow-up
	$(MAKE) validate
	$(MAKE) split-loso
	$(MAKE) train-loso
	$(MAKE) eval-loso
	$(MAKE) export-best

ml-run: compose-check
	$(MAKE) ml-pipeline
	$(MAKE) serve-gpu-detached
	$(MAKE) smoke-inference
	@echo "Done. MLflow: http://localhost:5001 | Inference: $(INFERENCE_URL)"
