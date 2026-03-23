# StudyBuddy Head-Pose Model (LOSO) — Capstone Report

- **Generated at (UTC)**: 2026-02-11 03:09:30Z
- **Summary CSV**: `/app/artifacts/training/loso_summary.csv`
- **Folds dir**: `/app/artifacts/training/folds`
- **Selection criterion**: `test_macro_f1`
- **Best fold**: `05` (test_macro_f1=0.7161)

## Key plots

- `test_macro_f1_by_fold.png`
- `val_accuracy_by_epoch.png`
- `val_loss_by_epoch.png`
- `best_fold_confusion_matrix.png`

![test_macro_f1 by fold](plots/test_macro_f1_by_fold.png)

## Aggregate metrics (mean ± std across folds)

- **test_macro_f1**: mean=0.3996 std=0.2040 (min=0.1413, max=0.7161)
- **test_balanced_accuracy**: mean=0.4456 std=0.1699 (min=0.2196, max=0.7142)
- **test_accuracy**: mean=0.4474 std=0.1691 (min=0.2248, max=0.7149)
- **val_macro_f1**: mean=0.5215 std=0.2128 (min=0.1447, max=0.7769)
- **val_accuracy**: mean=0.5750 std=0.1824 (min=0.2529, max=0.7765)

## Per-fold results

| fold_id | val_participants | test_participants | test_macro_f1 | test_balanced_accuracy | test_accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | jaydon | can_meto | 0.4164 | 0.4803 | 0.4841 |
| 1 | johnny | jaydon | 0.5739 | 0.5745 | 0.5747 |
| 2 | kareem_mourad | johnny | 0.1413 | 0.2196 | 0.2248 |
| 3 | mais | kareem_mourad | 0.1726 | 0.2684 | 0.2685 |
| 4 | yara | mais | 0.3771 | 0.4167 | 0.4171 |
| 5 | can_meto | yara | 0.7161 | 0.7142 | 0.7149 |

## Learning curves (validation)

![val_accuracy_by_epoch](plots/val_accuracy_by_epoch.png)

![val_loss_by_epoch](plots/val_loss_by_epoch.png)

## Best fold confusion matrix

![best_fold_confusion_matrix](plots/best_fold_confusion_matrix.png)

## Training configuration snapshot

```yaml
experiment_name: studybuddy-headpose-loso
seed: 42
image_size: 224
backbone: efficientnetv2b0
batch_size: 32
epochs: 12
learning_rate: 0.001
dropout: 0.2
early_stopping_patience: 3
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
