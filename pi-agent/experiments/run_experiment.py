from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory

try:
    # When executed as a module: python -m experiments.run_experiment
    from .mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
except Exception:  # pragma: no cover
    # When executed as a script: python experiments/run_experiment.py
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from experiments.mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _short_id(n: int = 4) -> str:
    # 4 hex chars is enough to avoid collisions for local runs.
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout_s: int | None = None) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    return p.returncode, p.stdout, p.stderr


def _git_info(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"repoRoot": str(repo_root)}
    rc, out, _ = _run_cmd(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    info["commit"] = out.strip() if rc == 0 else None
    rc, out, _ = _run_cmd(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    info["branch"] = out.strip() if rc == 0 else None
    rc, out, _ = _run_cmd(["git", "-C", str(repo_root), "status", "--porcelain"])
    info["dirty"] = bool(out.strip()) if rc == 0 else None
    info["statusPorcelain"] = out.strip().splitlines() if rc == 0 else None
    return info


def _count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    n = 0
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            n += 1
    return n


def _class_counts(split_dir: Path, labels: list[str]) -> dict[str, int]:
    return {lab: _count_images(split_dir / lab) for lab in labels}


def _load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _top_confusions(cm: list[list[int]], labels: list[str], k: int = 8) -> list[dict[str, Any]]:
    pairs: list[tuple[int, str, str]] = []
    n = min(len(labels), len(cm))
    for i in range(n):
        row = cm[i]
        for j in range(min(n, len(row))):
            if i == j:
                continue
            c = int(row[j])
            if c > 0:
                pairs.append((c, labels[i], labels[j]))
    pairs.sort(reverse=True, key=lambda t: t[0])
    return [{"count": c, "true": t, "pred": p} for c, t, p in pairs[:k]]


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    # train_tf.py writes accuracy_from_cm, macro_f1, per_class, confusion_matrix
    labels = []
    cm = None
    if isinstance(metrics.get("confusion_matrix"), dict):
        labels = metrics["confusion_matrix"].get("labels") or []
        cm = metrics["confusion_matrix"].get("matrix")
    top_conf = []
    if isinstance(labels, list) and isinstance(cm, list):
        top_conf = _top_confusions(cm, labels, k=8)
    return {
        "macro_f1": metrics.get("macro_f1"),
        "accuracy_from_cm": metrics.get("accuracy_from_cm"),
        "top_confusions": top_conf,
    }


def _default_labels() -> list[str]:
    # Must match train/train_tf.py and studybuddy_pi/inference.py
    return ["screen", "away_left", "away_right", "away_up", "away_down"]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    mlflow_run_id: str
    ok: bool
    metrics: dict[str, Any]
    config: dict[str, Any]


def _tee_subprocess(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return int(p.wait())


def run_one(args: argparse.Namespace) -> RunResult:
    # Resolve important roots
    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]  # pi-agent/
    repo_root = pi_agent_dir.parent

    run_id = args.run_id or f"{_utc_now_compact()}_{_short_id()}"

    labels = _default_labels()
    mlflow = configure_mlflow(pi_agent_dir=pi_agent_dir, experiment_name="ai-study-buddy-pi-agent")

    # Dataset split dirs (defaults match prepare_dataset.py output)
    splits_dir = Path(args.splits_dir).expanduser().resolve() if args.splits_dir else (pi_agent_dir / "data" / "splits")
    train_dir = splits_dir / "train"
    val_dir = splits_dir / "val"
    test_dir = splits_dir / "test"

    prep_log = ""
    if args.prepare_splits:
        prep = [
            sys.executable,
            str(pi_agent_dir / "train" / "prepare_dataset.py"),
            "--runs-dir",
            str(args.runs_dir),
            "--out-dir",
            str(splits_dir),
            "--split-by",
            str(args.split_by),
        ]
        if args.holdout_participant:
            prep += ["--holdout-participant", str(args.holdout_participant)]
        if args.copy:
            prep += ["--copy"]
        rc, out, err = _run_cmd(prep, cwd=pi_agent_dir)
        prep_log = out + "\n" + err
        if rc != 0:
            raise SystemExit(f"prepare_dataset.py failed (exit={rc}).")

    counts = {
        "train": _class_counts(train_dir, labels),
        "val": _class_counts(val_dir, labels),
        "test": _class_counts(test_dir, labels),
    }

    cfg: dict[str, Any] = {
        "runId": run_id,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
        "git": _git_info(repo_root),
        "data": {
            "splitsDir": str(splits_dir),
            "counts": counts,
            "splitBy": args.split_by,
            "holdoutParticipant": args.holdout_participant,
        },
        "train": {
            "backbone": args.backbone,
            "inputSize": args.input_size,
            "batchSize": args.batch_size,
            "epochsHead": args.epochs_head,
            "epochsFinetune": args.epochs_finetune,
            "fineTuneAt": args.fine_tune_at,
            "labelSmoothing": args.label_smoothing,
            "noAugment": args.no_augment,
            "noClassWeights": args.no_class_weights,
            "quantize": args.quantize,
            "seed": args.seed,
        },
        "artifacts": {},
        "status": {"ok": None, "exitCode": None, "dryRun": bool(args.dry_run)},
    }

    run_name = str(args.tag or run_id)
    with mlflow.start_run(run_name=run_name, nested=bool(getattr(args, "mlflow_nested", False))):
        active = mlflow.active_run()
        assert active is not None
        mlflow_run_id = active.info.run_id

        mlflow.set_tag("run_id", run_id)
        if args.tag:
            mlflow.set_tag("tag", str(args.tag))
        mlflow.set_tag("dry_run", bool(args.dry_run))
        if args.holdout_participant:
            mlflow.set_tag("holdout_participant", str(args.holdout_participant))

        # Log core params
        for k, v in cfg.get("train", {}).items():
            if k == "command":
                continue
            mlflow.log_param(f"train.{k}", v)
        mlflow.log_param("data.splitBy", cfg["data"].get("splitBy"))
        mlflow.log_param("data.holdoutParticipant", cfg["data"].get("holdoutParticipant"))

        # Small artifacts
        log_dict_as_json(mlflow, cfg["git"], "meta/git.json")
        log_dict_as_json(mlflow, cfg["data"].get("counts") or {}, "data/counts.json")
        if prep_log.strip():
            log_text(mlflow, prep_log, "data/prepare_splits.log")

        # Prepare a temp artifact dir for training outputs
        with TemporaryDirectory(prefix="ai-study-buddy-artifacts-") as td:
            artifacts_dir = Path(td)
            metrics_json = artifacts_dir / "metrics_test.json"
            log_txt = artifacts_dir / "train.log"
            cfg["artifacts"] = {
                "tmpArtifactsDir": str(artifacts_dir),
                "metricsTestJson": str(metrics_json),
            }

            cmd = [
                sys.executable,
                str(pi_agent_dir / "train" / "train_tf.py"),
                "--train-dir",
                str(train_dir),
                "--val-dir",
                str(val_dir),
                "--test-dir",
                str(test_dir),
                "--backbone",
                str(args.backbone),
                "--input-size",
                str(args.input_size),
                "--batch-size",
                str(args.batch_size),
                "--epochs-head",
                str(args.epochs_head),
                "--epochs-finetune",
                str(args.epochs_finetune),
                "--fine-tune-at",
                str(args.fine_tune_at),
                "--label-smoothing",
                str(args.label_smoothing),
                "--out-dir",
                str(artifacts_dir),
                "--tflite-name",
                str(args.tflite_name),
                "--seed",
                str(args.seed),
            ]
            if args.no_augment:
                cmd.append("--no-augment")
            if args.no_class_weights:
                cmd.append("--no-class-weights")
            if args.quantize:
                cmd.append("--quantize")

            cfg["train"]["command"] = " ".join(cmd)
            log_dict_as_json(mlflow, cfg, "meta/config.json")

            if args.dry_run:
                dummy = {
                    "accuracy": 0.0,
                    "accuracy_from_cm": 0.0,
                    "macro_f1": 0.0,
                    "per_class": {lab: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for lab in labels},
                    "confusion_matrix": {"labels": labels, "matrix": [[0 for _ in labels] for _ in labels]},
                    "_dry_run": True,
                }
                _write_json(metrics_json, dummy)
                mlflow.log_metric("macro_f1", 0.0)
                mlflow.log_metric("accuracy_from_cm", 0.0)
                mlflow.log_artifact(str(metrics_json), artifact_path="artifacts")
                return RunResult(run_id=run_id, mlflow_run_id=mlflow_run_id, ok=True, metrics=dummy, config=cfg)

            start = time.time()
            exit_code = _tee_subprocess(cmd, cwd=pi_agent_dir, log_path=log_txt)
            elapsed_s = time.time() - start

            mlflow.log_metric("elapsed_seconds", float(elapsed_s))
            mlflow.log_param("status.exit_code", int(exit_code))
            mlflow.log_artifact(str(log_txt), artifact_path="logs")

            if exit_code != 0:
                cfg["status"] = {"ok": False, "exitCode": int(exit_code), "dryRun": False, "elapsedSeconds": elapsed_s}
                log_dict_as_json(mlflow, cfg, "meta/config_failed.json")
                raise SystemExit(f"Training failed (exit={exit_code}).")

            metrics = _load_json(metrics_json) or {}

            # Core metrics
            if isinstance(metrics.get("macro_f1"), (int, float)):
                mlflow.log_metric("macro_f1", float(metrics["macro_f1"]))
            if isinstance(metrics.get("accuracy_from_cm"), (int, float)):
                mlflow.log_metric("accuracy_from_cm", float(metrics["accuracy_from_cm"]))

            per = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
            if isinstance(per, dict):
                for lab, d in per.items():
                    if isinstance(d, dict) and isinstance(d.get("recall"), (int, float)):
                        mlflow.log_metric(f"recall.{lab}", float(d["recall"]))

            # Artifacts
            mlflow.log_artifact(str(metrics_json), artifact_path="artifacts")
            for name in [args.tflite_name, "focus_model_labels.json", "focus_model.keras", "focus_model_saved"]:
                p = artifacts_dir / name
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="artifacts")

            cfg["status"] = {"ok": True, "exitCode": 0, "dryRun": False, "elapsedSeconds": elapsed_s}
            log_dict_as_json(mlflow, cfg["status"], "meta/status.json")
            return RunResult(run_id=run_id, mlflow_run_id=mlflow_run_id, ok=True, metrics=metrics, config=cfg)
    cmd = [
        sys.executable,
        str(pi_agent_dir / "train" / "train_tf.py"),
        "--train-dir",
        str(train_dir),
        "--val-dir",
        str(val_dir),
        "--test-dir",
        str(test_dir),
        "--backbone",
        str(args.backbone),
        "--input-size",
        str(args.input_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs-head",
        str(args.epochs_head),
        "--epochs-finetune",
        str(args.epochs_finetune),
        "--fine-tune-at",
        str(args.fine_tune_at),
        "--label-smoothing",
        str(args.label_smoothing),
        "--out-dir",
        str(paths.artifacts_dir),
        "--tflite-name",
        str(args.tflite_name),
        "--seed",
        str(args.seed),
    ]
    if args.no_augment:
        cmd.append("--no-augment")
    if args.no_class_weights:
        cmd.append("--no-class-weights")
    if args.quantize:
        cmd.append("--quantize")

    cfg["train"]["command"] = " ".join(cmd)

    # Write config early (so even failed runs are inspectable)
    _write_json(paths.config_json, cfg)

    # Start MLflow run immediately (so failures are recorded)
    run_name = str(args.tag or run_id)
    with mlflow.start_run(run_name=run_name, nested=bool(getattr(args, "mlflow_nested", False))):
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("tag", args.tag or "")
        mlflow.set_tag("dry_run", bool(args.dry_run))
        if args.holdout_participant:
            mlflow.set_tag("holdout_participant", str(args.holdout_participant))

        # Log core params
        for k, v in cfg.get("train", {}).items():
            if k in {"command"}:
                continue
            mlflow.log_param(f"train.{k}", v)
        mlflow.log_param("data.splitBy", cfg["data"].get("splitBy"))
        mlflow.log_param("data.holdoutParticipant", cfg["data"].get("holdoutParticipant"))

        # Log counts as artifact (small + useful)
        log_dict_as_json(mlflow, cfg["data"].get("counts") or {}, "data/counts.json")

        if args.dry_run:
            # Produce a tiny placeholder metrics file so reporting can be tested without TF installed.
            dummy = {
                "accuracy": 0.0,
                "accuracy_from_cm": 0.0,
                "macro_f1": 0.0,
                "per_class": {lab: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for lab in labels},
                "confusion_matrix": {"labels": labels, "matrix": [[0 for _ in labels] for _ in labels]},
                "_dry_run": True,
            }
            _write_json(paths.metrics_json, dummy)
            cfg["status"] = {"ok": True, "exitCode": 0, "dryRun": True}
            _write_json(paths.config_json, cfg)
            paths.notes_md.write_text(
                "# Notes\n\nDry-run completed. TensorFlow not required; this run exists to validate automation/reporting.\n",
                encoding="utf-8",
            )

            mlflow.log_metric("macro_f1", 0.0)
            mlflow.log_metric("accuracy_from_cm", 0.0)
            mlflow.log_artifact(str(paths.config_json))
            mlflow.log_artifact(str(paths.metrics_json))
            return paths

        start = time.time()
        exit_code = _tee_subprocess(cmd, cwd=pi_agent_dir, log_path=paths.log_txt)
        elapsed_s = time.time() - start

        cfg["status"] = {"ok": exit_code == 0, "exitCode": int(exit_code), "dryRun": False, "elapsedSeconds": elapsed_s}
        _write_json(paths.config_json, cfg)
        mlflow.log_metric("elapsed_seconds", float(elapsed_s))
        mlflow.log_param("status.exit_code", int(exit_code))

        # Always log logs/config even on failure.
        mlflow.log_artifact(str(paths.config_json))
        if paths.log_txt.exists():
            mlflow.log_artifact(str(paths.log_txt))
        if exit_code != 0:
            raise SystemExit(f"Training failed (exit={exit_code}). See {paths.log_txt}")

        metrics = _load_json(paths.metrics_json) or {}
        summary = _summarize_metrics(metrics)

        # MLflow metrics
        if isinstance(metrics.get("macro_f1"), (int, float)):
            mlflow.log_metric("macro_f1", float(metrics["macro_f1"]))
        if isinstance(metrics.get("accuracy_from_cm"), (int, float)):
            mlflow.log_metric("accuracy_from_cm", float(metrics["accuracy_from_cm"]))
        # Per-class recall metrics
        per = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
        if isinstance(per, dict):
            for lab, d in per.items():
                if isinstance(d, dict) and isinstance(d.get("recall"), (int, float)):
                    mlflow.log_metric(f"recall.{lab}", float(d["recall"]))

        # Artifacts
        mlflow.log_artifact(str(paths.metrics_json))
        mlflow.log_artifact(str(paths.config_json))
        if paths.log_txt.exists():
            mlflow.log_artifact(str(paths.log_txt))
        # Model artifacts (if present)
        for name in ["focus_model.tflite", "focus_model_labels.json", "metrics_test.json"]:
            p = paths.artifacts_dir / name
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="artifacts")

        notes = [
            "# Notes",
            "",
            f"- run_id: `{run_id}`",
            f"- macro_f1: `{summary.get('macro_f1')}`",
            f"- accuracy_from_cm: `{summary.get('accuracy_from_cm')}`",
            "",
            "## Top confusions",
        ]
        for item in summary.get("top_confusions") or []:
            notes.append(f"- **{item['true']} → {item['pred']}**: {item['count']}")
        notes.append("")
        paths.notes_md.write_text("\n".join(notes) + "\n", encoding="utf-8")
        return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single training/eval experiment and log to MLflow.")
    parser.add_argument("--run-id", default=None, help="Optional run_id (default: timestamp + random suffix)")
    parser.add_argument("--tag", default=None, help="Optional tag to attach to the run (e.g. 'baseline')")

    parser.add_argument("--splits-dir", default=None, help="Directory containing train/val/test split folders")

    # Optional: prepare splits from collected data runs.
    parser.add_argument("--prepare-splits", action="store_true", help="Run prepare_dataset.py before training")
    parser.add_argument("--runs-dir", default="data", help="Directory containing run_* folders (for prepare_dataset.py)")
    parser.add_argument("--split-by", choices=["participant", "session"], default="participant")
    parser.add_argument("--holdout-participant", default=None, help="Force one participant into test split (LOPO)")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink when preparing splits")

    # Training knobs (mirrors train_tf.py)
    parser.add_argument(
        "--backbone",
        default="mobilenetv3small",
        choices=["mobilenetv3small", "mobilenetv3large", "mobilenetv2", "efficientnetlite0"],
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tflite-name", default="focus_model.tflite")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute TensorFlow training; writes placeholder metrics for validating automation/reporting.",
    )

    args = parser.parse_args(argv)
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

