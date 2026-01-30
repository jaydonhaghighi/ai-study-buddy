from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import tensorflow as tf

# Keep label order fixed across training + TFLite inference.
LABELS = ["screen", "away_left", "away_right", "away_up", "away_down"]


def build_augmenter(seed: int) -> tf.keras.Model:
    """
    Label-safe augmentations to improve generalization to new people/cameras.
    IMPORTANT: Do NOT horizontally flip (it would swap away_left/away_right).
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.06,
                width_factor=0.06,
                fill_mode="reflect",
                seed=seed,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.08, 0.10),
                width_factor=(-0.08, 0.10),
                fill_mode="reflect",
                seed=seed + 1,
            ),
            tf.keras.layers.RandomRotation(
                factor=0.02,
                fill_mode="reflect",
                seed=seed + 2,
            ),
        ],
        name="augmenter",
    )


def compute_class_weights(train_dir: Path) -> tuple[dict[int, float], dict[str, int]]:
    counts: dict[str, int] = {}
    for lab in LABELS:
        d = train_dir / lab
        if not d.exists():
            counts[lab] = 0
            continue
        counts[lab] = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.jpeg"))) + len(list(d.glob("*.png")))

    total = sum(counts.values())
    n = len(LABELS)
    weights: dict[int, float] = {}
    for i, lab in enumerate(LABELS):
        c = counts[lab]
        # If a class is missing, set weight to 0 (training will likely fail anyway due to missing class).
        weights[i] = float(total / (n * c)) if c > 0 else 0.0
    return weights, counts


def build_model(input_size: int, num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
        name="backbone",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def load_dir_dataset(data_dir: Path, input_size: int, batch_size: int, shuffle: bool, seed: int):
    return tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        class_names=LABELS,
    )

def find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """
    Keras layer ordering can vary across versions (especially Keras 3).
    Don't rely on model.layers[index]. Instead, grab the backbone by name,
    falling back to a heuristic if needed.
    """
    try:
        layer = model.get_layer("backbone")
        if isinstance(layer, tf.keras.Model):
            return layer
    except Exception:
        pass

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and hasattr(layer, "layers") and len(layer.layers) > 10:
            return layer

    raise RuntimeError("Could not locate MobileNetV2 backbone layer to fine-tune.")

def evaluate_detailed(model: tf.keras.Model, ds) -> dict:
    """
    Returns confusion matrix + per-class metrics and macro-F1.
    Keeps dependencies minimal (pure TF/py).
    """
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_true.extend(tf.argmax(y, axis=1).numpy().tolist())
        y_pred.extend(tf.argmax(probs, axis=1).numpy().tolist())

    n = len(LABELS)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=n).numpy().tolist()

    per_class: dict[str, dict] = {}
    for i, lab in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        support = sum(cm[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        per_class[lab] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    macro_f1 = sum(per_class[lab]["f1"] for lab in LABELS) / max(1, n)
    overall_acc = sum(cm[i][i] for i in range(n)) / max(1, sum(sum(r) for r in cm))

    return {
        "accuracy_from_cm": float(overall_acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a MobileNetV2 attention direction classifier (5-way softmax).")
    parser.add_argument("--data-dir", help="Directory with class subfolders (quick experiments)")
    parser.add_argument("--train-dir", help="Directory with class subfolders for training (preferred)")
    parser.add_argument("--val-dir", help="Directory with class subfolders for validation (preferred)")
    parser.add_argument("--test-dir", help="Directory with class subfolders for test (recommended)")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation (not recommended for LOPO)")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights (not recommended if imbalanced)")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--tflite-name", default="focus_model.tflite")
    parser.add_argument("--quantize", action="store_true", help="Enable dynamic range quantization")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    train_dir = Path(args.train_dir).expanduser().resolve() if args.train_dir else None
    val_dir = Path(args.val_dir).expanduser().resolve() if args.val_dir else None
    test_dir = Path(args.test_dir).expanduser().resolve() if args.test_dir else None
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if train_dir and val_dir:
        train_ds = load_dir_dataset(train_dir, args.input_size, args.batch_size, shuffle=True, seed=args.seed)
        val_ds = load_dir_dataset(val_dir, args.input_size, args.batch_size, shuffle=False, seed=args.seed)
        test_ds = load_dir_dataset(test_dir, args.input_size, args.batch_size, shuffle=False, seed=args.seed) if test_dir else None
    elif data_dir:
        # Fallback: simple random split (not leakage-safe across people/sessions)
        full = load_dir_dataset(data_dir, args.input_size, args.batch_size, shuffle=True, seed=args.seed)
        n = tf.data.experimental.cardinality(full).numpy()
        n_val = max(1, int(round(n * 0.2)))
        val_ds = full.take(n_val)
        train_ds = full.skip(n_val)
        test_ds = None
    else:
        raise SystemExit("Provide either --train-dir/--val-dir (preferred) or --data-dir (quick).")

    autotune = tf.data.AUTOTUNE
    # Augment only the training set (label-safe: no flips).
    if not args.no_augment:
        augmenter = build_augmenter(args.seed)

        def _augment(x, y):
            x = augmenter(x, training=True)
            # Light photometric jitter for lighting/camera variance.
            x = tf.image.random_brightness(x, max_delta=0.08)
            x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
            x = tf.clip_by_value(x, 0.0, 255.0)
            return x, y

        train_ds = train_ds.map(_augment, num_parallel_calls=autotune)

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.prefetch(autotune)

    class_weight = None
    if train_dir and (not args.no_class_weights):
        class_weight, counts = compute_class_weights(train_dir)
        print("Train class counts:", counts)
        print("Using class_weight:", class_weight)

    model = build_model(args.input_size, num_classes=len(LABELS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("Training head...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Fine-tune last layers
    base = find_backbone(model)
    base.trainable = True
    for layer in base.layers[: args.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"],
    )
    print("Fine-tuning...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    if test_ds is not None:
        print("Evaluating on test...")
        results = model.evaluate(test_ds, return_dict=True)
        print("Test metrics:", results)
        detailed = evaluate_detailed(model, test_ds)
        merged = dict(results)
        merged.update(detailed)
        # Save metrics for automation (e.g., LOPO runs)
        (out_dir / "metrics_test.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"Test macro_f1: {merged.get('macro_f1'):.4f}")

    saved_model_dir = out_dir / "focus_model_saved"
    # Keras 3: model.save() requires .keras/.h5. For SavedModel (needed for TFLite),
    # use model.export(<dir>).
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    model.export(str(saved_model_dir))
    # Also save a .keras file for convenience.
    model.save(str(out_dir / "focus_model.keras"))

    # Persist label order for inference.
    (out_dir / "focus_model_labels.json").write_text(json.dumps(LABELS, indent=2) + "\n", encoding="utf-8")

    print("Exporting TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = out_dir / args.tflite_name
    tflite_path.write_bytes(tflite_model)

    print(f"Saved TFLite model: {tflite_path}")
    print("Use STUDYBUDDY_MODEL_PATH to point the Pi agent at this file.")


if __name__ == "__main__":
    main()
