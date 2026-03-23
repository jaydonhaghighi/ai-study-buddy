# StudyBuddy Head-Pose Model (LOSO) — Capstone Report

- **Generated at (UTC)**: 2026-02-13 08:04:02Z
- **Summary CSV**: `/app/artifacts/training/loso_summary.csv`
- **Folds dir**: `/app/artifacts/training/folds`
- **Selection criterion**: `test_macro_f1`
- **Best fold**: `03` (test_macro_f1=0.8573)

## Key plots

- `test_macro_f1_by_fold.png`
- `val_accuracy_by_epoch.png`
- `val_loss_by_epoch.png`
- `best_fold_confusion_matrix.png`

![test_macro_f1 by fold](plots/test_macro_f1_by_fold.png)

## Aggregate metrics (mean ± std across folds)

- **test_macro_f1**: mean=0.6830 std=0.1670 (min=0.2932, max=0.8573)
- **test_balanced_accuracy**: mean=0.7103 std=0.1498 (min=0.3799, max=0.8606)
- **test_accuracy**: mean=0.7078 std=0.1534 (min=0.3621, max=0.8595)
- **val_macro_f1**: mean=0.6699 std=0.1916 (min=0.3098, max=0.8710)
- **val_accuracy**: mean=0.7098 std=0.1619 (min=0.3806, max=0.8711)

## Per-fold results

| fold_id | val_participants | test_participants | test_macro_f1 | test_balanced_accuracy | test_accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | daniel_esenwa | can_meto | 0.7167 | 0.7856 | 0.7668 |
| 1 | isaiah_hunte | daniel_esenwa | 0.7754 | 0.7929 | 0.8087 |
| 2 | jaydon | isaiah_hunte | 0.8250 | 0.8437 | 0.8397 |
| 3 | johnny | jaydon | 0.8573 | 0.8606 | 0.8595 |
| 4 | kareem_mourad | johnny | 0.6258 | 0.6226 | 0.6370 |
| 5 | mais | kareem_mourad | 0.2932 | 0.3799 | 0.3621 |
| 6 | yara | mais | 0.6227 | 0.6296 | 0.6262 |
| 7 | can_meto | yara | 0.7479 | 0.7678 | 0.7626 |

## Learning curves (validation)

![val_accuracy_by_epoch](plots/val_accuracy_by_epoch.png)

![val_loss_by_epoch](plots/val_loss_by_epoch.png)

## Best fold confusion matrix

![best_fold_confusion_matrix](plots/best_fold_confusion_matrix.png)

## Training configuration snapshot

```yaml
experiment_name: studybuddy-headpose-loso-v4-finetune
seed: 42
image_size: 256
backbone: efficientnetv2b0
batch_size: 12
epochs: 24
freeze_epochs: 8
fine_tune_epochs: 16
learning_rate: 0.0003
fine_tune_learning_rate: 0.00005
fine_tune_trainable_layers: 40
dropout: 0.3
label_smoothing: 0.05
early_stopping_patience: 6
aug_brightness_delta: 0.18
aug_contrast_lower: 0.8
aug_contrast_upper: 1.25
aug_saturation_lower: 0.8
aug_saturation_upper: 1.25
aug_gaussian_noise_stddev: 4.0
label_column: label
participant_column: participant_id
path_column: image_path
```

## Data validation snapshot

```json
{
  "dataset_root": "/datasets",
  "meta_files_found": 8,
  "rows_seen": 11187,
  "rows_kept": 10731,
  "rows_malformed": 0,
  "rows_invalid_label": 0,
  "rows_missing_image": 456,
  "participants": [
    "can_meto",
    "daniel_esenwa",
    "isaiah_hunte",
    "jaydon",
    "johnny",
    "kareem_mourad",
    "mais",
    "yara"
  ],
  "class_counts": {
    "away_down": 2111,
    "away_left": 2229,
    "away_right": 2206,
    "away_up": 2044,
    "screen": 2141
  },
  "participant_counts": {
    "can_meto": 1085,
    "daniel_esenwa": 1030,
    "isaiah_hunte": 1148,
    "jaydon": 1537,
    "johnny": 1328,
    "kareem_mourad": 1671,
    "mais": 1458,
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
      "participant_id": "kareem_mourad",
      "label": "away_down",
      "count": 344
    },
    {
      "participant_id": "kareem_mourad",
      "label": "away_left",
      "count": 348
    },
    {
      "participant_id": "kareem_mourad",
      "label": "away_right",
      "count": 347
    },
    {
      "participant_id": "kareem_mourad",
      "label": "away_up",
      "count": 296
    },
    {
      "participant_id": "kareem_mourad",
      "label": "screen",
      "count": 336
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
