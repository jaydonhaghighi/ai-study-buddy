# StudyBuddy ML Training Rundown (Complete Guide)

This document is a full beginner-to-advanced explanation of the ML system in this repository.
It is project-specific and maps directly to the code in:

- `ml/studybuddy_ml/pipeline.py`
- `ml/studybuddy_ml/cli.py`
- `ml/studybuddy_ml/serve_api.py`
- `ml/studybuddy_ml/temporal.py`
- `Makefile`
- `ml/configs/*.yaml`

---

## Part 1: Full In-Depth Rundown

## 1) What this ML project does

The model predicts head-pose attention labels from face images.

Labels (from `ml/studybuddy_ml/__init__.py`):

- `screen`
- `away_left`
- `away_right`
- `away_up`
- `away_down`

So this is a 5-class image classification problem.

High-level pipeline:

1. Validate and clean raw data
2. Build a canonical manifest CSV
3. Generate LOSO splits by participant
4. Train one model per fold
5. Evaluate each fold on held-out participant
6. Aggregate fold metrics
7. Train one production model on full data
8. Export deployable model
9. Serve model for real-time inference with temporal smoothing

---

## 2) End-to-end command flow (what actually runs)

From repo root (via `Makefile`):

```bash
make validate
make split-loso
make train-loso
make eval-loso
make train-production
make export-best
```

Or one shot:

```bash
make ml-pipeline
```

Where each step maps:

- `validate` -> `studybuddy-ml validate` -> `build_manifest()`
- `split-loso` -> `studybuddy-ml split-loso` -> `generate_loso_splits()`
- `train-loso` -> `studybuddy-ml train-loso` -> `train_loso()`
- `eval-loso` -> `studybuddy-ml eval-loso` -> `aggregate_loso()`
- `train-production` -> `studybuddy-ml train-production` -> `train_production()`
- `export-best` -> `studybuddy-ml export-best` -> `export_production_model()` or `export_best_model()`

---

## 3) Data layer: where data comes from and how it is validated

Expected dataset structure (from `ml/README.md` and pipeline code):

- Root contains `run_*/meta.jsonl`
- Each JSONL line should include:
  - `label`
  - `participant`
  - `facePath`

In Docker trainer service:

- host dataset is mounted as `/datasets`

Validation and manifest building (`build_manifest`):

- discovers all `run_*/meta.jsonl`
- parses each line
- rejects malformed rows
- rejects rows with missing required fields
- rejects rows with invalid labels
- rejects rows where image file does not exist
- normalizes participant IDs
- stores records in a dataframe
- drops duplicate `image_path` rows
- computes summary statistics report

Outputs:

- `ml/artifacts/data/manifest.csv`
- `ml/artifacts/reports/data_validation.json`

Current dataset snapshot (from your artifacts):

- `rows_seen`: 10640
- `rows_kept`: 10260
- `rows_missing_image`: 380
- `participants`: 8
- `classes`: 5

---

## 4) Splitting strategy: LOSO (Leave-One-Subject-Out)

Implemented in `generate_loso_splits()` and validated in `_validate_loso_splits()`.

Why LOSO:

- Webcam frames from the same person are highly correlated.
- Random image split can cause leakage (same person appears in train and test).
- LOSO forces evaluation on unseen participants, which is methodologically stronger.

What one fold contains:

- train participants: all except val/test
- val participants: exactly one
- test participants: exactly one

Current split stats (from `ml/artifacts/splits/index.json`):

- `num_folds`: 8
- train per fold: 7395 to 8145 (mean 7695)
- val per fold: 1030 to 1537 (mean 1282.5)
- test per fold: 1030 to 1537 (mean 1282.5)

Integrity checks include:

- no overlap between train/val/test participant sets
- exactly one val and one test participant per fold
- all participants are known and covered
- split counts match current manifest
- each participant appears exactly once as test

---

## 5) Input pipeline: preprocessing and augmentation

Implemented in `_prepare_dataset()`.

Base preprocessing (all splits):

1. load image from `image_path`
2. decode JPEG, force 3 channels (RGB)
3. resize to `[image_size, image_size]`
4. cast to `float32`
5. batch by `batch_size`
6. prefetch for throughput

Training-only augmentation:

- random brightness (`aug_brightness_delta`)
- random contrast (`aug_contrast_lower`, `aug_contrast_upper`)
- random saturation (`aug_saturation_lower`, `aug_saturation_upper`)
- optional Gaussian noise (`aug_gaussian_noise_stddev`)

Important:

- validation and test pipelines do not apply augmentation
- this avoids evaluation contamination

---

## 6) Model architecture and transfer learning

Implemented in `_build_model()`.

Backbone options:

- `efficientnetv2b0` (default)
- `mobilenetv2`

Model structure:

1. Input: `(image_size, image_size, 3)`
2. Pretrained backbone (ImageNet weights)
3. GlobalAveragePooling2D
4. Dropout
5. Dense(5, softmax) head

Output:

- one probability per class
- predicted label = argmax probability

Transfer learning behavior:

- backbone starts frozen
- optional fine-tuning unfreezes top layers later

---

## 7) Training loop mechanics (what "training" means here)

Main functions:

- `train_loso()` for cross-subject evaluation
- `train_production()` for final full-dataset model

Compilation (`_compile_model()`):

- optimizer: Adam
- loss:
  - `SparseCategoricalCrossentropy` if no label smoothing
  - custom smoothed categorical loss if `label_smoothing > 0`
- metric: accuracy

Class imbalance handling:

- class weights computed from training labels using `compute_class_weight(class_weight="balanced", ...)`
- class weights passed into `model.fit()`

Callbacks:

- `ModelCheckpoint(save_best_only=True)`
- `EarlyStopping(restore_best_weights=True)`

---

## 8) Crucial definitions (epoch, batch size, freeze epochs, etc.)

### Batch
Small group of samples processed together in one forward/backward pass.

### Batch size (`batch_size`)
How many samples per batch.

Effects:

- higher batch size: fewer steps per epoch, more VRAM use
- lower batch size: more steps per epoch, less VRAM use, noisier gradient updates

### Step / Iteration
One optimizer update, usually one batch.

### Epoch
One full pass through the training split.

Approximate formula:

`steps_per_epoch ~= ceil(train_samples / batch_size)`

### Epochs (`epochs`)
General epoch count field in config.

Code nuance:

- if `freeze_epochs` is not explicitly set, code sets `freeze_epochs = epochs`
- actual two-stage training length is `freeze_epochs + fine_tune_epochs`

### Freeze phase (`freeze_epochs`)
Backbone frozen; only classification head trains.

Why:

- stable adaptation to new task
- lower overfitting risk at start

### Fine-tune phase (`fine_tune_epochs`)
Top backbone layers are unfrozen and trained with smaller LR.

Why:

- adapt pretrained features to project-specific data

### Fine-tune trainable layers (`fine_tune_trainable_layers`)
How many top layers in backbone are made trainable in fine-tuning.

### Learning rate (`learning_rate`, `fine_tune_learning_rate`)
Step size for weight updates.

- larger LR: faster but can overshoot/diverge
- smaller LR: slower but often more stable
- fine-tune LR is usually smaller than initial LR

### Dropout (`dropout`)
Regularization layer that randomly drops activations during training.

Why:

- reduces overfitting

### Label smoothing (`label_smoothing`)
Softens one-hot targets slightly.

Why:

- reduces overconfident predictions
- can improve generalization/calibration

### Early stopping (`early_stopping_patience`)
Stops training when monitored metric stops improving for N epochs.

In LOSO training:

- monitor is `val_accuracy`

In production training:

- monitor is `accuracy` (train-only scope)

---

## 9) Metrics: what they mean and why they matter

Computed in `_classification_metrics()`:

- `accuracy`
- `macro_f1`
- `macro_precision`
- `macro_recall`
- `balanced_accuracy`

Definitions:

- Accuracy: fraction of all predictions that are correct.
- Precision (per class): when model predicts class C, how often it is correct.
- Recall (per class): of all true class C samples, how many were found.
- F1 (per class): harmonic mean of precision and recall.
- Macro version: simple average across classes (equal class importance).
- Balanced accuracy: average recall across classes.

Why macro metrics are important here:

- class distributions are not perfectly uniform
- macro metrics prevent majority classes from dominating the score

Confusion matrix:

- visual count table of true vs predicted classes
- helps diagnose specific confusions (for example, `away_left` vs `away_right`)

---

## 10) LOSO fold outputs and aggregate outputs

Per fold (`ml/artifacts/training/folds/fold_XX/`):

- `best_model.keras`
- `history.json`
- `test_classification_report.json`
- `test_confusion_matrix.png`
- `fold_summary.json`

Global LOSO outputs:

- `ml/artifacts/training/loso_summary.csv`
- `ml/artifacts/reports/loso_aggregate.json` (when `eval-loso` is run)

Aggregation (`aggregate_loso()`):

- computes mean/std/min/max over all numeric `val_*` and `test_*` columns

---

## 11) Production training and deployment selection

`train_production()`:

- trains on full manifest (all rows)
- logs train metrics
- writes:
  - `ml/artifacts/training/production/best_model.keras`
  - `ml/artifacts/training/production/production_summary.json`

`export-best` behavior:

1. if production summary exists and prefer-production is enabled, export production model
2. otherwise choose best LOSO fold by validation criterion (default `val_macro_f1`)

Guardrail in `export_best_model()`:

- refuses `test_*` criterion for deployment model selection
- avoids selecting deployment model directly on test performance

Export outputs:

- `ml/artifacts/export/best_model.keras`
- `ml/artifacts/export/saved_model/`
- `ml/artifacts/export/model_meta.json`

---

## 12) Reproducibility and lineage

The pipeline records:

- `seed`
- `git_sha`
- `manifest_sha256`
- `config_sha256`
- split index SHA (for LOSO)
- TensorFlow deterministic ops requested flag

This appears in run params/tags and reproducibility JSONs.

Why this matters:

- you can trace exactly what code + config + data produced a result
- essential for scientific validity and capstone defense

---

## 13) Full config glossary (all training config fields)

From `TrainConfig` in `pipeline.py`:

- `experiment_name`: MLflow experiment name
- `seed`: random seed for reproducibility
- `image_size`: square resize target
- `backbone`: pretrained architecture choice
- `batch_size`: samples per batch
- `epochs`: base epoch count (used to default `freeze_epochs`)
- `freeze_epochs`: stage-1 epochs with frozen backbone
- `fine_tune_epochs`: stage-2 epochs with partially unfrozen backbone
- `learning_rate`: LR for freeze stage
- `fine_tune_learning_rate`: LR for fine-tune stage
- `fine_tune_trainable_layers`: number of top backbone layers to unfreeze
- `dropout`: classifier head dropout rate
- `label_smoothing`: label smoothing factor [0,1)
- `early_stopping_patience`: no-improvement epoch tolerance
- `aug_brightness_delta`: train-time brightness jitter amount
- `aug_contrast_lower`: lower contrast bound
- `aug_contrast_upper`: upper contrast bound
- `aug_saturation_lower`: lower saturation bound
- `aug_saturation_upper`: upper saturation bound
- `aug_gaussian_noise_stddev`: optional gaussian noise intensity
- `participant_column`: participant ID column in manifest
- `label_column`: label column in manifest
- `path_column`: image path column in manifest

---

## 14) Baseline and experiment presets

Default Makefile config:

- `ML_CONFIG=/app/configs/baseline.yaml`

Config presets available:

- `exp_stability_v1.yaml`
- `exp_regularized_v2.yaml`
- `exp_hires_v3.yaml`
- `exp_v4.yaml`

Examples:

- `exp_hires_v3` raises `image_size` to 256 and lowers `batch_size` to 12
- `exp_v4` enables two-stage fine-tuning, label smoothing, stronger augmentation

---

## 15) Inference API and temporal behavior (post-training)

Inference service (`serve_api.py`):

- loads `best_model.keras`
- decodes incoming image
- resizes to inferred model input size
- returns label + confidence + class probabilities

Session endpoints:

- start session
- send frames
- stop session and get summary metrics

Temporal smoothing (`temporal.py`):

- EMA smoothing on class probabilities (`ema_alpha`)
- confidence threshold for considering state changes
- hysteresis timing:
  - requires longer time to transition to distracted (`distract_seconds`)
  - shorter time to refocus (`refocus_seconds`)
- minimum dwell time between transitions (`min_dwell_seconds`)

This improves UX by avoiding rapid flicker from noisy frame-by-frame predictions.

---

## 16) Important artifacts and where to look first

- Data checks: `ml/artifacts/reports/data_validation.json`
- Split definitions: `ml/artifacts/splits/index.json` + `fold_*.json`
- Fold training outputs: `ml/artifacts/training/folds/`
- Fold summary table: `ml/artifacts/training/loso_summary.csv`
- Aggregate report: `ml/artifacts/reports/loso_aggregate.json`
- Production summary: `ml/artifacts/training/production/production_summary.json`
- Deployment model: `ml/artifacts/export/best_model.keras`
- Experiment tracking: `ml/artifacts/mlflow/`

---

## 17) How to evaluate quality correctly

For scientific claims:

- prioritize LOSO metrics (cross-subject generalization)
- report mean and std across folds, not only best fold
- include confusion matrix and per-class behavior

For deployment:

- export production model after LOSO methodology is validated
- use temporal logic for stable session-level behavior

---

## 18) Common pitfalls and how this repo handles them

Potential pitfall -> Current handling:

- participant leakage -> LOSO by participant + validation checks
- stale split definitions -> split count consistency validation
- overfitting during long training -> early stopping
- class imbalance -> balanced class weights
- accidental test-based model selection -> export guard against `test_*` criterion
- weak experiment traceability -> git/config/manifest hashes + MLflow logs

---

## 19) What is not implemented (so you know current boundaries)

- no LR scheduler callback (`ReduceLROnPlateau`, cosine, etc.)
- no mixed precision training policy in pipeline
- no optimizer gradient clipping config
- no automated hyperparameter search loop
- production stage uses train-only evaluation (by design)

These are possible future improvements, not necessarily bugs.

---

## 20) One-page recap (mental model)

Think of this project in two layers:

1) **Training/evaluation layer**
- clean and validate data
- split by participant (LOSO)
- train/evaluate fold models
- aggregate metrics
- train full-data production model
- export deployable artifact

2) **Inference/behavior layer**
- run model on incoming frames
- smooth predictions over time
- apply focused/distracted transition logic
- output session summary metrics

This is why the project is strong: it combines good ML methodology with an end-to-end deployable application workflow.

---

## Part 2: How to Explain This to a Professor / Interviewer

## A) 30-second version

"We built a 5-class head-pose classifier for focus tracking. The pipeline validates raw data, builds a manifest, and evaluates with participant-level LOSO to avoid identity leakage. We train transfer-learning models with EfficientNetV2B0, class weighting, early stopping, and fold-level reporting. Then we aggregate cross-fold metrics for scientific validity, train a production model on full data for deployment, and serve it through a FastAPI inference API with temporal smoothing and hysteresis to produce stable focus session analytics."

---

## B) 2-minute technical version

"Our dataset is organized as `run_*/meta.jsonl` plus face images. We parse that into a canonical `manifest.csv` after filtering malformed rows, missing files, and invalid labels. We currently have 10,260 valid images across 8 participants and 5 classes.

For evaluation, we use participant-level LOSO: each fold has one test subject and one validation subject, with all remaining subjects for training. This avoids train/test contamination from subject identity, which is a major risk in webcam datasets.

Model-wise, we use transfer learning with EfficientNetV2B0 (or MobileNetV2), then a GAP + dropout + softmax head. Training is staged: frozen-backbone phase and optional fine-tuning phase with a lower learning rate. We apply train-only augmentation and balanced class weights. We checkpoint on validation accuracy and use early stopping.

Per fold, we compute accuracy, macro precision/recall/F1, and balanced accuracy, and generate classification reports plus confusion matrices. Then we aggregate metrics across folds (mean/std/min/max) for robust reporting.

After LOSO validation, we train one production model on the full dataset and export it as Keras + SavedModel. In serving, predictions are smoothed with EMA and state transitions require dwell thresholds to avoid noisy focus/distracted flicker."

---

## C) Why your methodology is defensible

If asked "why should we trust this?", answer:

1. **Leakage-aware split design**: LOSO by participant, not random image split.
2. **Reproducibility**: seed + git SHA + config/data hashes are logged.
3. **Class-aware metrics**: macro metrics and balanced accuracy, not only raw accuracy.
4. **Artifacts and traceability**: per-fold reports, confusion matrices, MLflow runs.
5. **Deployment discipline**: evaluation pipeline separated from production export policy.

---

## D) Likely questions + strong answers

### "Why LOSO instead of random split?"
Because random split can leak person-specific patterns across train/test. LOSO measures true cross-person generalization.

### "Why macro-F1?"
Macro-F1 gives equal weight to each class and is more informative under class imbalance than accuracy alone.

### "What does freeze vs fine-tune do?"
Freeze trains only the new classification head first for stability; fine-tuning then adapts selected pretrained layers with smaller LR.

### "How do you prevent overfitting?"
LOSO split design, train-only augmentation, class weighting, dropout, early stopping, and validation-based checkpointing.

### "How do you ensure reproducibility?"
Fixed seed, deterministic-op request, and hashes for manifest/config/splits plus git SHA in tracking metadata.

### "How is inference made stable?"
Temporal EMA smoothing plus hysteresis thresholds for distract/refocus transitions and minimum dwell time.

---

## E) What to say you would improve next

If asked for future work:

- add LR scheduler (for example `ReduceLROnPlateau`)
- add mixed precision for faster GPU training
- run systematic hyperparameter search
- add calibration analysis and threshold tuning
- evaluate domain shift robustness (lighting/device variations)

---

## F) Presentation cheat sheet (memorize this)

- Problem: 5-class head-pose attention classification.
- Data: 10,260 valid images, 8 participants, 5 labels.
- Method: participant-level LOSO cross-validation.
- Model: EfficientNetV2B0 transfer learning + dropout head.
- Training: class weights, augmentation, early stopping, checkpointing.
- Metrics: accuracy + macro precision/recall/F1 + balanced accuracy.
- Evidence: fold-level artifacts + aggregate stats.
- Deploy: production model export + FastAPI + temporal smoothing.

---

End of guide.

