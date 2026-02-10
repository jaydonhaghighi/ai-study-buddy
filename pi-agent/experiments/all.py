from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    # When executed as a module: python -m experiments.all
    from .mlflow_utils import configure_mlflow  # type: ignore
    from .run_experiment import run_one  # type: ignore
    from .sweep import run_sweep  # type: ignore
    from .lopo import run_lopo  # type: ignore
except Exception:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from experiments.mlflow_utils import configure_mlflow  # type: ignore
    from experiments.run_experiment import run_one  # type: ignore
    from experiments.sweep import run_sweep  # type: ignore
    from experiments.lopo import run_lopo  # type: ignore


def _run(cmd: list[str], *, cwd: Path) -> int:
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(cwd))
    return int(p.returncode)


def main(argv: list[str] | None = None) -> int:
    """
    One-command orchestrator:
    - (optional) prepare_dataset.py
    - run one / sweep / LOPO
    - everything logged to MLflow (SQLite)
    """
    parser = argparse.ArgumentParser(description="Run the full local loop (MLflow-only).")
    parser.add_argument(
        "--mode",
        choices=["one", "sweep"],
        default="one",
        help="Run a single experiment or a small sweep.",
    )
    parser.add_argument(
        "--lopo",
        action="store_true",
        help="Run LOPO evaluation (one run per participant holdout). Only applies to --mode one.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip TensorFlow training; still logs a run to MLflow.")

    # Optional split-prep
    parser.add_argument("--prepare-splits", action="store_true", help="Generate data/splits from data/run_* before training.")
    parser.add_argument("--runs-dir", default="data", help="Directory containing run_* folders (for prepare_dataset.py)")
    parser.add_argument("--splits-dir", default="data/splits", help="Output splits directory (train/val/test)")
    parser.add_argument("--split-by", choices=["participant", "session"], default="participant")
    parser.add_argument("--holdout-participant", default=None)
    parser.add_argument("--copy", action="store_true")

    # Training knobs (forwarded)
    parser.add_argument("--tag", default="baseline")
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

    # Sweep knobs (when mode=sweep)
    parser.add_argument("--backbones", default=None, help="Comma-separated list (sweep only)")
    parser.add_argument("--input-sizes", default=None, help="Comma-separated list (sweep only)")
    parser.add_argument("--seeds", default=None, help="Comma-separated list (sweep only)")

    args = parser.parse_args(argv)

    pi_agent_dir = Path(__file__).resolve().parents[1]

    # Start a parent run so this command is a single "demo-able" unit in MLflow UI.
    mlflow = configure_mlflow(pi_agent_dir=pi_agent_dir, experiment_name="ai-study-buddy-pi-agent")
    parent_name = f"all:{args.mode}:{args.tag}" + (":lopo" if args.lopo else "")
    with mlflow.start_run(run_name=parent_name):
        mlflow.set_tag("kind", "all")
        mlflow.set_tag("created_at_utc", datetime.now(timezone.utc).isoformat())

        # 1) Prepare splits (optional; mainly useful for mode=one and mode=sweep)
        if args.prepare_splits and not args.lopo:
            cmd = [
                sys.executable,
                str(pi_agent_dir / "train" / "prepare_dataset.py"),
                "--runs-dir",
                str(args.runs_dir),
                "--out-dir",
                str(args.splits_dir),
                "--split-by",
                str(args.split_by),
            ]
            if args.holdout_participant:
                cmd += ["--holdout-participant", str(args.holdout_participant)]
            if args.copy:
                cmd += ["--copy"]

            rc = _run(cmd, cwd=pi_agent_dir)
            if rc != 0:
                return rc

        # 2) Run experiment / LOPO / sweep (nested under the parent run)
        if args.mode == "one" and args.lopo:
            ns = argparse.Namespace(**vars(args))
            ns.mlflow_nested = True
            # lopo.py expects these names
            ns.tflite_name = "focus_model.tflite"
            run_lopo(ns)

        elif args.mode == "one":
            ns = argparse.Namespace(**vars(args))
            ns.run_id = None
            ns.prepare_splits = bool(args.prepare_splits)
            ns.mlflow_nested = True
            ns.tflite_name = "focus_model.tflite"
            run_one(ns)

        else:
            ns = argparse.Namespace(**vars(args))
            ns.mlflow_nested = True
            ns.tflite_name = "focus_model.tflite"
            # sweep.py expects CSV strings for lists
            ns.backbones = args.backbones or "mobilenetv3small,mobilenetv2"
            ns.input_sizes = args.input_sizes or "160,192,224"
            ns.seeds = args.seeds or "1337,1338"
            run_sweep(ns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

