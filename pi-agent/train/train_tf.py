from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

# Keep label order fixed across training + TFLite inference.
LABELS = ["screen", "away_left", "away_right", "away_up", "away_down"]


def build_model(input_size: int, num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
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
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.prefetch(autotune)

    model = build_model(args.input_size, num_classes=len(LABELS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("Training head...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head)

    # Fine-tune last layers
    base = model.layers[2]
    base.trainable = True
    for layer in base.layers[: args.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    print("Fine-tuning...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune)

    if test_ds is not None:
        print("Evaluating on test...")
        results = model.evaluate(test_ds, return_dict=True)
        print("Test metrics:", results)
        # Save metrics for automation (e.g., LOPO runs)
        (out_dir / "metrics_test.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    saved_model_dir = out_dir / "focus_model_saved"
    model.save(saved_model_dir)

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
