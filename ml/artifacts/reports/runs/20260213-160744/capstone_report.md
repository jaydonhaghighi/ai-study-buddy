# StudyBuddy Head-Pose Model (LOSO) — Capstone Report

- **Generated at (UTC)**: 2026-02-13 21:07:47Z
- **Summary CSV**: `/app/artifacts/training/loso_summary.csv`
- **Folds dir**: `/app/artifacts/training/folds`
- **Selection criterion**: `test_macro_f1`
- **Best fold**: `05` (test_macro_f1=0.8196)

## Key plots

- `test_macro_f1_by_fold.png`
- `val_accuracy_by_epoch.png`
- `val_loss_by_epoch.png`
- `best_fold_confusion_matrix.png`

![test_macro_f1 by fold](plots/test_macro_f1_by_fold.png)

## Aggregate metrics (mean ± std across folds)

- **test_macro_f1**: mean=0.7227 std=0.1178 (min=0.4393, max=0.8196)
- **test_balanced_accuracy**: mean=0.7485 std=0.1118 (min=0.4785, max=0.8309)
- **test_accuracy**: mean=0.7510 std=0.1105 (min=0.4810, max=0.8476)
- **val_macro_f1**: mean=0.7636 std=0.1160 (min=0.4980, max=0.8938)
- **val_accuracy**: mean=0.7723 std=0.1115 (min=0.5098, max=0.8986)

## Per-fold results

| fold_id | val_participants | test_participants | test_macro_f1 | test_balanced_accuracy | test_accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | daniel_esenwa | can_meto | 0.7204 | 0.7949 | 0.7760 |
| 1 | isaiah_hunte | daniel_esenwa | 0.8131 | 0.8309 | 0.8476 |
| 2 | jaydon | isaiah_hunte | 0.8129 | 0.8300 | 0.8258 |
| 3 | johnny | jaydon | 0.7148 | 0.7409 | 0.7430 |
| 4 | mais | johnny | 0.6820 | 0.6902 | 0.7116 |
| 5 | phoenix_bastien | mais | 0.8196 | 0.8216 | 0.8224 |
| 6 | yara | phoenix_bastien | 0.7798 | 0.8008 | 0.8008 |
| 7 | can_meto | yara | 0.4393 | 0.4785 | 0.4810 |

## Learning curves (validation)

![val_accuracy_by_epoch](plots/val_accuracy_by_epoch.png)

![val_loss_by_epoch](plots/val_loss_by_epoch.png)

## Best fold confusion matrix

![best_fold_confusion_matrix](plots/best_fold_confusion_matrix.png)

## Training configuration snapshot

```yaml
experiment_name: studybuddy-headpose-loso-v5-no-kareem
seed: 42
image_size: 256
backbone: efficientnetv2b0
batch_size: 12
epochs: 28
freeze_epochs: 10
fine_tune_epochs: 18
learning_rate: 0.00025
fine_tune_learning_rate: 0.00003
fine_tune_trainable_layers: 30
dropout: 0.4
label_smoothing: 0.08
early_stopping_patience: 8
aug_brightness_delta: 0.16
aug_contrast_lower: 0.85
aug_contrast_upper: 1.20
aug_saturation_lower: 0.85
aug_saturation_upper: 1.20
aug_gaussian_noise_stddev: 3.0
label_column: label
participant_column: participant_id
path_column: image_path
```

## Data validation snapshot

```json
{
  "dataset_root": "/datasets",
  "meta_files_found": 8,
  "rows_seen": 10640,
  "rows_kept": 10260,
  "rows_malformed": 0,
  "rows_invalid_label": 0,
  "rows_missing_image": 380,
  "participants": [
    "can_meto",
    "daniel_esenwa",
    "isaiah_hunte",
    "jaydon",
    "johnny",
    "mais",
    "phoenix_bastien",
    "yara"
  ],
  "class_counts": {
    "away_down": 2007,
    "away_left": 2121,
    "away_right": 2099,
    "away_up": 1988,
    "screen": 2045
  },
  "participant_counts": {
    "can_meto": 1085,
    "daniel_esenwa": 1030,
    "isaiah_hunte": 1148,
    "jaydon": 1537,
    "johnny": 1328,
    "mais": 1458,
    "phoenix_bastien": 1200,
    "yara": 1474
  },
  "per_participant_class_counts": [
    {
      "participant_id": "can_meto",
      "label": "away_down",
      "count": 187
    },
    {
      "participant_id": "can_meto",
      "label": "away_left",
      "count": 237
    },
    {
      "participant_id": "can_meto",
      "label": "away_right",
      "count": 221
    },
    {
      "participant_id": "can_meto",
      "label": "away_up",
      "count": 205
    },
    {
      "participant_id": "can_meto",
      "label": "screen",
      "count": 235
    },
    {
      "participant_id": "daniel_esenwa",
      "label": "away_down",
      "count": 182
    },
    {
      "participant_id": "daniel_esenwa",
      "label": "away_left",
      "count": 232
    },
    {
      "participant_id": "daniel_esenwa",
      "label": "away_right",
      "count": 215
    },
    {
      "participant_id": "daniel_esenwa",
      "label": "away_up",
      "count": 223
    },
    {
      "participant_id": "daniel_esenwa",
      "label": "screen",
      "count": 178
    },
    {
      "participant_id": "isaiah_hunte",
      "label": "away_down",
      "count": 211
    },
    {
      "participant_id": "isaiah_hunte",
      "label": "away_left",
      "count": 237
    },
    {
      "participant_id": "isaiah_hunte",
      "label": "away_right",
      "count": 236
    },
    {
      "participant_id": "isaiah_hunte",
      "label": "away_up",
      "count": 231
    },
    {
      "participant_id": "isaiah_hunte",
      "label": "screen",
      "count": 233
    },
    {
      "participant_id": "jaydon",
      "label": "away_down",
      "count": 300
    },
    {
      "participant_id": "jaydon",
      "label": "away_left",
      "count": 320
    },
    {
      "participant_id": "jaydon",
      "label": "away_right",
      "count": 294
    },
    {
      "participant_id": "jaydon",
      "label": "away_up",
      "count": 308
    },
    {
      "participant_id": "jaydon",
      "label": "screen",
      "count": 315
    },
    {
      "participant_id": "johnny",
      "label": "away_down",
      "count": 298
    },
    {
      "participant_id": "johnny",
      "label": "away_left",
      "count": 244
    },
    {
      "participant_id": "johnny",
      "label": "away_right",
      "count": 301
    },
    {
      "participant_id": "johnny",
      "label": "away_up",
      "count": 242
    },
    {
      "participant_id": "johnny",
      "label": "screen",
      "count": 243
    },
    {
      "participant_id": "mais",
      "label": "away_down",
      "count": 305
    },
    {
      "participant_id": "mais",
      "label": "away_left",
      "count": 299
    },
    {
      "participant_id": "mais",
      "label": "away_right",
      "count": 304
    },
    {
      "participant_id": "mais",
      "label": "away_up",
      "count": 253
    },
    {
      "participant_id": "mais",
      "label": "screen",
      "count": 297
    },
    {
      "participant_id": "phoenix_bastien",
      "label": "away_down",
      "count": 240
    },
    {
      "participant_id": "phoenix_bastien",
      "label": "away_left",
      "count": 240
    },
    {
      "participant_id": "phoenix_bastien",
      "label": "away_right",
      "count": 240
    },
    {
      "participant_id": "phoenix_bastien",
      "label": "away_up",
      "count": 240
    },
    {
      "participant_id": "phoenix_bastien",
      "label": "screen",
      "count": 240
    },
    {
      "participant_id": "yara",
      "label": "away_down",
      "count": 284
    },
    {
      "participant_id": "yara",
      "label": "away_left",
      "count": 312
    },
    {
      "participant_id": "yara",
      "label": "away_right",
      "count": 288
    },
    {
      "participant_id": "yara",
      "label": "away_up",
      "count": 286
    },
    {
      "participant_id": "yara",
      "label": "screen",
      "count": 304
    }
  ]
}
```
