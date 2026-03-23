# StudyBuddy Head-Pose Model (LOSO) — Capstone Report

- **Generated at (UTC)**: 2026-02-11 06:43:49Z
- **Summary CSV**: `/app/artifacts/training/loso_summary.csv`
- **Folds dir**: `/app/artifacts/training/folds`
- **Selection criterion**: `test_macro_f1`
- **Best fold**: `05` (test_macro_f1=0.8177)

## Key plots

- `test_macro_f1_by_fold.png`
- `val_accuracy_by_epoch.png`
- `val_loss_by_epoch.png`
- `best_fold_confusion_matrix.png`

![test_macro_f1 by fold](plots/test_macro_f1_by_fold.png)

## Aggregate metrics (mean ± std across folds)

- **test_macro_f1**: mean=0.5815 std=0.1856 (min=0.3309, max=0.8177)
- **test_balanced_accuracy**: mean=0.6289 std=0.1484 (min=0.4086, max=0.8139)
- **test_accuracy**: mean=0.6307 std=0.1462 (min=0.4169, max=0.8138)
- **val_macro_f1**: mean=0.6412 std=0.1589 (min=0.3841, max=0.8750)
- **val_accuracy**: mean=0.6802 std=0.1413 (min=0.4432, max=0.8746)

## Per-fold results

| fold_id | val_participants | test_participants | test_macro_f1 | test_balanced_accuracy | test_accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | jaydon | can_meto | 0.6478 | 0.6823 | 0.6830 |
| 1 | johnny | jaydon | 0.7926 | 0.7958 | 0.7959 |
| 2 | kareem_mourad | johnny | 0.3309 | 0.4086 | 0.4169 |
| 3 | mais | kareem_mourad | 0.4019 | 0.5086 | 0.5100 |
| 4 | yara | mais | 0.4981 | 0.5643 | 0.5649 |
| 5 | can_meto | yara | 0.8177 | 0.8139 | 0.8138 |

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
  "meta_files_found": 6,
  "rows_seen": 9899,
  "rows_kept": 9847,
  "rows_malformed": 0,
  "rows_invalid_label": 0,
  "rows_missing_image": 52,
  "participants": [
    "can_meto",
    "jaydon",
    "johnny",
    "kareem_mourad",
    "mais",
    "yara"
  ],
  "class_counts": {
    "away_down": 1978,
    "away_left": 1991,
    "away_right": 1933,
    "away_up": 1964,
    "screen": 1981
  },
  "participant_counts": {
    "can_meto": 1227,
    "jaydon": 1754,
    "johnny": 1708,
    "kareem_mourad": 1747,
    "mais": 1671,
    "yara": 1740
  },
  "per_participant_class_counts": [
    {
      "participant_id": "can_meto",
      "label": "away_down",
      "count": 239
    },
    {
      "participant_id": "can_meto",
      "label": "away_left",
      "count": 257
    },
    {
      "participant_id": "can_meto",
      "label": "away_right",
      "count": 234
    },
    {
      "participant_id": "can_meto",
      "label": "away_up",
      "count": 240
    },
    {
      "participant_id": "can_meto",
      "label": "screen",
      "count": 257
    },
    {
      "participant_id": "jaydon",
      "label": "away_down",
      "count": 351
    },
    {
      "participant_id": "jaydon",
      "label": "away_left",
      "count": 351
    },
    {
      "participant_id": "jaydon",
      "label": "away_right",
      "count": 351
    },
    {
      "participant_id": "jaydon",
      "label": "away_up",
      "count": 351
    },
    {
      "participant_id": "jaydon",
      "label": "screen",
      "count": 350
    },
    {
      "participant_id": "johnny",
      "label": "away_down",
      "count": 350
    },
    {
      "participant_id": "johnny",
      "label": "away_left",
      "count": 351
    },
    {
      "participant_id": "johnny",
      "label": "away_right",
      "count": 333
    },
    {
      "participant_id": "johnny",
      "label": "away_up",
      "count": 335
    },
    {
      "participant_id": "johnny",
      "label": "screen",
      "count": 339
    },
    {
      "participant_id": "kareem_mourad",
      "label": "away_down",
      "count": 351
    },
    {
      "participant_id": "kareem_mourad",
      "label": "away_left",
      "count": 348
    },
    {
      "participant_id": "kareem_mourad",
      "label": "away_right",
      "count": 348
    },
    {
      "participant_id": "kareem_mourad",
      "label": "away_up",
      "count": 351
    },
    {
      "participant_id": "kareem_mourad",
      "label": "screen",
      "count": 349
    },
    {
      "participant_id": "mais",
      "label": "away_down",
      "count": 335
    },
    {
      "participant_id": "mais",
      "label": "away_left",
      "count": 333
    },
    {
      "participant_id": "mais",
      "label": "away_right",
      "count": 334
    },
    {
      "participant_id": "mais",
      "label": "away_up",
      "count": 335
    },
    {
      "participant_id": "mais",
      "label": "screen",
      "count": 334
    },
    {
      "participant_id": "yara",
      "label": "away_down",
      "count": 352
    },
    {
      "participant_id": "yara",
      "label": "away_left",
      "count": 351
    },
    {
      "participant_id": "yara",
      "label": "away_right",
      "count": 333
    },
    {
      "participant_id": "yara",
      "label": "away_up",
      "count": 352
    },
    {
      "participant_id": "yara",
      "label": "screen",
      "count": 352
    }
  ]
}
```
