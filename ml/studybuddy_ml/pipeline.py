from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

from . import LABELS


@dataclass
class TrainConfig:
    experiment_name: str = "studybuddy-headpose-loso"
    seed: int = 42
    image_size: int = 224
    backbone: str = "efficientnetv2b0"
    batch_size: int = 32
    epochs: int = 12
    freeze_epochs: int | None = None
    fine_tune_epochs: int = 0
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-4
    fine_tune_trainable_layers: int = 30
    dropout: float = 0.2
    label_smoothing: float = 0.0
    early_stopping_patience: int = 3
    aug_brightness_delta: float = 0.12
    aug_contrast_lower: float = 0.9
    aug_contrast_upper: float = 1.1
    aug_saturation_lower: float = 0.9
    aug_saturation_upper: float = 1.1
    aug_gaussian_noise_stddev: float = 0.0
    participant_column: str = "participant_id"
    label_column: str = "label"
    path_column: str = "image_path"


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML: {path}")
    return payload


def load_train_config(config_path: Path) -> TrainConfig:
    payload = load_yaml(config_path)
    config = TrainConfig()
    for key, value in payload.items():
        if hasattr(config, key):
            setattr(config, key, value)
    _validate_train_config(config)
    return config


def _validate_train_config(config: TrainConfig) -> None:
    if config.freeze_epochs is None:
        config.freeze_epochs = int(config.epochs)
    config.freeze_epochs = int(config.freeze_epochs)
    config.fine_tune_epochs = int(config.fine_tune_epochs)
    config.fine_tune_trainable_layers = int(config.fine_tune_trainable_layers)
    config.batch_size = int(config.batch_size)
    if config.freeze_epochs < 0:
        raise ValueError(f"freeze_epochs must be >= 0, got {config.freeze_epochs}")
    if config.fine_tune_epochs < 0:
        raise ValueError(f"fine_tune_epochs must be >= 0, got {config.fine_tune_epochs}")
    if config.freeze_epochs == 0 and config.fine_tune_epochs == 0:
        raise ValueError("At least one of freeze_epochs or fine_tune_epochs must be > 0")

    if not 0.0 <= config.label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {config.label_smoothing}")

    if config.fine_tune_learning_rate <= 0:
        raise ValueError(
            f"fine_tune_learning_rate must be positive, got {config.fine_tune_learning_rate}"
        )
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")

    if config.aug_contrast_lower <= 0 or config.aug_contrast_upper <= 0:
        raise ValueError("aug_contrast_lower/upper must be positive")
    if config.aug_contrast_lower > config.aug_contrast_upper:
        raise ValueError("aug_contrast_lower must be <= aug_contrast_upper")

    if config.aug_saturation_lower <= 0 or config.aug_saturation_upper <= 0:
        raise ValueError("aug_saturation_lower/upper must be positive")
    if config.aug_saturation_lower > config.aug_saturation_upper:
        raise ValueError("aug_saturation_lower must be <= aug_saturation_upper")
    if config.aug_brightness_delta < 0:
        raise ValueError("aug_brightness_delta must be >= 0")
    if config.aug_gaussian_noise_stddev < 0:
        raise ValueError("aug_gaussian_noise_stddev must be >= 0")


def load_participant_map(map_path: Path | None) -> dict[str, str]:
    if map_path is None:
        return {}
    payload = load_yaml(map_path)
    mapped: dict[str, str] = {}
    for raw_name, participant_id in payload.items():
        mapped[str(raw_name).strip()] = str(participant_id).strip()
    return mapped


def _discover_meta_files(dataset_root: Path) -> list[Path]:
    meta_files = sorted(dataset_root.glob("run_*/meta.jsonl"))
    if not meta_files:
        raise FileNotFoundError(
            f"No run_*/meta.jsonl files found under {dataset_root}"
        )
    return meta_files


def build_manifest(
    dataset_root: Path,
    participant_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    participant_map = participant_map or {}
    dataset_root = dataset_root.resolve()
    meta_files = _discover_meta_files(dataset_root)

    missing_images = 0
    invalid_labels = 0
    malformed_rows = 0
    total_rows = 0

    rows: list[dict[str, Any]] = []
    for meta_file in meta_files:
        run_id = meta_file.parent.name
        with meta_file.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                payload_raw = line.strip()
                if not payload_raw:
                    continue
                total_rows += 1
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    malformed_rows += 1
                    continue

                label = str(payload.get("label", "")).strip()
                face_path = str(payload.get("facePath", "")).strip()
                participant_raw = str(payload.get("participant", "")).strip()
                if not label or not face_path or not participant_raw:
                    malformed_rows += 1
                    continue
                if label not in LABELS:
                    invalid_labels += 1
                    continue

                image_path = (dataset_root / face_path).resolve()
                if not image_path.exists():
                    missing_images += 1
                    continue

                participant_id = participant_map.get(participant_raw, _slugify(participant_raw))
                rows.append(
                    {
                        "run_id": run_id,
                        "line_index": line_idx,
                        "image_path": str(image_path),
                        "face_path": face_path,
                        "label": label,
                        "participant_name_raw": participant_raw,
                        "participant_id": participant_id,
                        "session": str(payload.get("session", "")).strip() or None,
                        "condition": str(payload.get("condition", "")).strip() or None,
                        "timestamp": payload.get("timestamp"),
                        "away_direction": payload.get("awayDirection"),
                    }
                )

    if not rows:
        raise ValueError("Manifest is empty after filtering. Check dataset path and labels.")

    manifest = pd.DataFrame(rows)
    manifest = manifest.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    class_counts = (
        manifest["label"].value_counts().sort_index().to_dict()
    )
    participant_counts = (
        manifest["participant_id"].value_counts().sort_index().to_dict()
    )
    per_participant_class_counts = (
        manifest.groupby(["participant_id", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["participant_id", "label"])
        .to_dict(orient="records")
    )

    report = {
        "dataset_root": str(dataset_root),
        "meta_files_found": len(meta_files),
        "rows_seen": total_rows,
        "rows_kept": int(len(manifest)),
        "rows_malformed": malformed_rows,
        "rows_invalid_label": invalid_labels,
        "rows_missing_image": missing_images,
        "participants": sorted(manifest["participant_id"].unique().tolist()),
        "class_counts": class_counts,
        "participant_counts": participant_counts,
        "per_participant_class_counts": per_participant_class_counts,
    }
    return manifest, report


def write_manifest_and_report(
    manifest: pd.DataFrame,
    report: dict[str, Any],
    manifest_out: Path,
    report_out: Path,
) -> None:
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_out, index=False)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def generate_loso_splits(
    manifest: pd.DataFrame,
    participant_column: str,
    split_out_dir: Path,
) -> list[dict[str, Any]]:
    participants = sorted(manifest[participant_column].dropna().unique().tolist())
    if len(participants) < 3:
        raise ValueError(
            f"LOSO requires at least 3 participants, found {len(participants)}"
        )

    split_out_dir.mkdir(parents=True, exist_ok=True)
    split_records: list[dict[str, Any]] = []
    for fold_idx, test_pid in enumerate(participants):
        remaining = [p for p in participants if p != test_pid]
        val_pid = remaining[fold_idx % len(remaining)]
        train_pids = [p for p in remaining if p != val_pid]

        train_count = int(manifest[manifest[participant_column].isin(train_pids)].shape[0])
        val_count = int(manifest[manifest[participant_column] == val_pid].shape[0])
        test_count = int(manifest[manifest[participant_column] == test_pid].shape[0])

        payload = {
            "fold_id": fold_idx,
            "train_participants": train_pids,
            "val_participants": [val_pid],
            "test_participants": [test_pid],
            "counts": {
                "train": train_count,
                "val": val_count,
                "test": test_count,
            },
        }
        split_records.append(payload)
        (split_out_dir / f"fold_{fold_idx:02d}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    (split_out_dir / "index.json").write_text(
        json.dumps(
            {
                "strategy": "LOSO",
                "participant_column": participant_column,
                "num_folds": len(split_records),
                "folds": split_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return split_records


def _load_split_files(split_dir: Path) -> list[Path]:
    split_files = sorted(split_dir.glob("fold_*.json"))
    if not split_files:
        raise FileNotFoundError(f"No fold_*.json files found in {split_dir}")
    return split_files


def _prepare_dataset(
    frame: pd.DataFrame,
    config: TrainConfig,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = frame["image_path"].astype(str).tolist()
    labels = frame["label_id"].astype(np.int32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000), seed=seed, reshuffle_each_iteration=True)

    def _map(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, [config.image_size, config.image_size])
        image = tf.cast(image, tf.float32)
        if training:
            image = tf.image.random_brightness(image, max_delta=config.aug_brightness_delta)
            image = tf.image.random_contrast(
                image,
                lower=config.aug_contrast_lower,
                upper=config.aug_contrast_upper,
            )
            image = tf.image.random_saturation(
                image,
                lower=config.aug_saturation_lower,
                upper=config.aug_saturation_upper,
            )
            if config.aug_gaussian_noise_stddev > 0:
                noise = tf.random.normal(
                    tf.shape(image),
                    mean=0.0,
                    stddev=config.aug_gaussian_noise_stddev,
                    dtype=tf.float32,
                )
                image = tf.clip_by_value(image + noise, 0.0, 255.0)
        return image, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _build_model(
    image_size: int,
    num_classes: int,
    dropout: float,
    backbone: str = "efficientnetv2b0",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    bb = _slugify(backbone)
    if bb in {"efficientnetv2b0", "efficientnet_v2_b0", "efficientnetv2_b0"}:
        # EfficientNetV2 models include their own preprocessing when
        # include_preprocessing=True. Our dataset produces float32 images in
        # the 0..255 range, which matches the expected input for built-in preprocessing.
        x = inputs
        base = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
            include_preprocessing=True,
        )
    elif bb in {"mobilenetv2", "mobilenet_v2"}:
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    else:
        raise ValueError(
            "Unsupported backbone. Use 'efficientnetv2b0' or 'mobilenetv2'. "
            f"Got: {backbone!r}"
        )

    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="studybuddy_headpose")
    return model, base


def _make_classification_loss(
    label_smoothing: float,
    num_classes: int,
) -> tf.keras.losses.Loss | Any:
    if label_smoothing <= 0:
        return tf.keras.losses.SparseCategoricalCrossentropy()

    categorical_loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )

    def _smoothed_sparse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
        return categorical_loss(y_true_one_hot, y_pred)

    return _smoothed_sparse_loss


def _compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    label_smoothing: float,
    num_classes: int,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=_make_classification_loss(
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        ),
        metrics=["accuracy"],
    )


def _set_backbone_trainable_layers(
    base: tf.keras.Model,
    trainable_layers: int,
) -> int:
    base.trainable = True
    total_layers = len(base.layers)
    if trainable_layers <= 0 or trainable_layers >= total_layers:
        first_trainable_idx = 0
    else:
        first_trainable_idx = total_layers - trainable_layers

    trainable_count = 0
    for idx, layer in enumerate(base.layers):
        is_trainable_region = idx >= first_trainable_idx
        # Keep batch norm frozen for stability on small participant datasets.
        layer.trainable = is_trainable_region and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
        if layer.trainable:
            trainable_count += 1
    return trainable_count


def _merge_history_dicts(histories: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, [])
            merged[key].extend(float(v) for v in values)
    return merged


def _dataset_labels(dataset: tf.data.Dataset) -> np.ndarray:
    labels: list[np.ndarray] = []
    for _batch_x, batch_y in dataset:
        labels.append(batch_y.numpy())
    if not labels:
        return np.array([], dtype=np.int32)
    return np.concatenate(labels, axis=0)


def _evaluate_dataset(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    probs = model.predict(dataset, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = _dataset_labels(dataset)
    return y_true, y_pred


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "balanced_accuracy": 0.0,
        }
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(LABELS)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test Fold)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def train_loso(
    manifest_csv: Path,
    split_dir: Path,
    config_path: Path,
    output_dir: Path,
    tracking_uri: str | None = None,
) -> Path:
    config = load_train_config(config_path)
    set_global_seed(config.seed)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(manifest_csv)
    if config.label_column not in frame.columns:
        raise KeyError(f"Label column '{config.label_column}' not found in manifest")
    if config.path_column not in frame.columns:
        raise KeyError(f"Path column '{config.path_column}' not found in manifest")
    if config.participant_column not in frame.columns:
        raise KeyError(f"Participant column '{config.participant_column}' not found in manifest")

    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    frame["label_id"] = frame[config.label_column].map(label_to_idx)
    if frame["label_id"].isna().any():
        unknown = sorted(frame[frame["label_id"].isna()][config.label_column].unique().tolist())
        raise ValueError(f"Unknown labels in manifest: {unknown}")
    frame["label_id"] = frame["label_id"].astype(int)

    fold_summaries: list[dict[str, Any]] = []
    for split_file in _load_split_files(split_dir):
        split = json.loads(split_file.read_text(encoding="utf-8"))
        fold_id = int(split["fold_id"])
        train_p = split["train_participants"]
        val_p = split["val_participants"]
        test_p = split["test_participants"]

        train_df = frame[frame[config.participant_column].isin(train_p)].copy()
        val_df = frame[frame[config.participant_column].isin(val_p)].copy()
        test_df = frame[frame[config.participant_column].isin(test_p)].copy()
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(f"Fold {fold_id} has empty split. Check LOSO generation.")

        train_ds = _prepare_dataset(train_df, config, True, config.seed)
        val_ds = _prepare_dataset(val_df, config, False, config.seed)
        test_ds = _prepare_dataset(test_df, config, False, config.seed)

        class_values = np.unique(train_df["label_id"].to_numpy())
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=class_values,
            y=train_df["label_id"].to_numpy(),
        )
        class_weight = {int(label): float(weight) for label, weight in zip(class_values, class_weights)}

        fold_dir = output_dir / "folds" / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_dir / "best_model.keras"

        with mlflow.start_run(run_name=f"loso_fold_{fold_id:02d}") as run:
            mlflow.set_tags(
                {
                    "fold_id": fold_id,
                    "train_participants": ",".join(train_p),
                    "val_participants": ",".join(val_p),
                    "test_participants": ",".join(test_p),
                    "strategy": "LOSO",
                }
            )
            mlflow.log_params(
                {
                    "image_size": config.image_size,
                    "backbone": config.backbone,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "freeze_epochs": config.freeze_epochs,
                    "fine_tune_epochs": config.fine_tune_epochs,
                    "learning_rate": config.learning_rate,
                    "fine_tune_learning_rate": config.fine_tune_learning_rate,
                    "fine_tune_trainable_layers": config.fine_tune_trainable_layers,
                    "dropout": config.dropout,
                    "label_smoothing": config.label_smoothing,
                    "early_stopping_patience": config.early_stopping_patience,
                    "aug_brightness_delta": config.aug_brightness_delta,
                    "aug_contrast_lower": config.aug_contrast_lower,
                    "aug_contrast_upper": config.aug_contrast_upper,
                    "aug_saturation_lower": config.aug_saturation_lower,
                    "aug_saturation_upper": config.aug_saturation_upper,
                    "aug_gaussian_noise_stddev": config.aug_gaussian_noise_stddev,
                }
            )

            model, base = _build_model(
                image_size=config.image_size,
                num_classes=len(LABELS),
                dropout=config.dropout,
                backbone=config.backbone,
            )
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
            )
            history_chunks: list[dict[str, list[float]]] = []
            completed_epochs = 0

            if config.freeze_epochs and config.freeze_epochs > 0:
                base.trainable = False
                _compile_model(
                    model,
                    config.learning_rate,
                    config.label_smoothing,
                    num_classes=len(LABELS),
                )
                freeze_history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    initial_epoch=completed_epochs,
                    epochs=completed_epochs + config.freeze_epochs,
                    class_weight=class_weight,
                    callbacks=[
                        checkpoint_callback,
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_accuracy",
                            mode="max",
                            patience=config.early_stopping_patience,
                            restore_best_weights=True,
                        ),
                    ],
                    verbose=2,
                )
                history_chunks.append(freeze_history.history)
                completed_epochs += len(freeze_history.history.get("loss", []))

            if config.fine_tune_epochs > 0:
                trainable_backbone_layers = _set_backbone_trainable_layers(
                    base,
                    config.fine_tune_trainable_layers,
                )
                mlflow.log_param("fine_tune_trainable_layers_effective", trainable_backbone_layers)
                _compile_model(
                    model,
                    config.fine_tune_learning_rate,
                    config.label_smoothing,
                    num_classes=len(LABELS),
                )
                fine_tune_history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    initial_epoch=completed_epochs,
                    epochs=completed_epochs + config.fine_tune_epochs,
                    class_weight=class_weight,
                    callbacks=[
                        checkpoint_callback,
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_accuracy",
                            mode="max",
                            patience=config.early_stopping_patience,
                            restore_best_weights=True,
                        ),
                    ],
                    verbose=2,
                )
                history_chunks.append(fine_tune_history.history)

            history = _merge_history_dicts(history_chunks)
            history_out = fold_dir / "history.json"
            history_out.write_text(json.dumps(history, indent=2), encoding="utf-8")

            best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
            val_true, val_pred = _evaluate_dataset(best_model, val_ds)
            test_true, test_pred = _evaluate_dataset(best_model, test_ds)
            val_metrics = _classification_metrics(val_true, val_pred)
            test_metrics = _classification_metrics(test_true, test_pred)

            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            test_report = classification_report(
                test_true,
                test_pred,
                target_names=LABELS,
                labels=np.arange(len(LABELS)),
                output_dict=True,
                zero_division=0,
            )
            report_path = fold_dir / "test_classification_report.json"
            report_path.write_text(json.dumps(test_report, indent=2), encoding="utf-8")

            cm_path = fold_dir / "test_confusion_matrix.png"
            _plot_confusion_matrix(test_true, test_pred, cm_path)
            mlflow.log_artifacts(str(fold_dir), artifact_path=f"fold_{fold_id:02d}")

            fold_summary = {
                "run_id": run.info.run_id,
                "fold_id": fold_id,
                "train_participants": ",".join(train_p),
                "val_participants": ",".join(val_p),
                "test_participants": ",".join(test_p),
                "model_path": str(checkpoint_path),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            fold_summaries.append(fold_summary)

            (fold_dir / "fold_summary.json").write_text(
                json.dumps(fold_summary, indent=2), encoding="utf-8"
            )

    summary_df = pd.DataFrame(fold_summaries).sort_values("fold_id")
    summary_csv = output_dir / "loso_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    return summary_csv


def aggregate_loso(summary_csv: Path, aggregate_out: Path) -> dict[str, Any]:
    frame = pd.read_csv(summary_csv)
    metric_columns = [c for c in frame.columns if c.startswith("val_") or c.startswith("test_")]
    aggregate: dict[str, Any] = {
        "num_folds": int(frame.shape[0]),
        "metrics": {},
        "skipped_columns": [],
    }
    for column in metric_columns:
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        values = numeric_values.to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            aggregate["skipped_columns"].append(column)
            continue
        aggregate["metrics"][column] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    aggregate_out.parent.mkdir(parents=True, exist_ok=True)
    aggregate_out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def export_best_model(
    summary_csv: Path,
    export_dir: Path,
    criterion: str = "test_macro_f1",
) -> dict[str, Any]:
    frame = pd.read_csv(summary_csv)
    if criterion not in frame.columns:
        raise KeyError(f"Criterion column '{criterion}' not found in {summary_csv}")
    best_idx = frame[criterion].astype(float).idxmax()
    best_row = frame.loc[best_idx]

    src_model = Path(str(best_row["model_path"]))
    if not src_model.exists():
        raise FileNotFoundError(f"Best model path not found: {src_model}")

    export_dir.mkdir(parents=True, exist_ok=True)
    dst_model = export_dir / "best_model.keras"
    shutil.copy2(src_model, dst_model)

    model = tf.keras.models.load_model(dst_model, compile=False)
    saved_model_dir = export_dir / "saved_model"
    model.export(str(saved_model_dir))

    meta = {
        "criterion": criterion,
        "best_fold_id": int(best_row["fold_id"]),
        "criterion_value": float(best_row[criterion]),
        "labels": LABELS,
        "keras_model_path": str(dst_model),
        "saved_model_path": str(saved_model_dir),
        "source_summary_csv": str(summary_csv),
        "train_participants": str(best_row["train_participants"]),
        "val_participants": str(best_row["val_participants"]),
        "test_participants": str(best_row["test_participants"]),
    }
    (export_dir / "model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta
