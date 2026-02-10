from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    # When executed as a module: python -m experiments.lopo
    from .mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
    from .run_experiment import RunResult, run_one  # type: ignore
except Exception:  # pragma: no cover
    # When executed as a script: python experiments/lopo.py
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from experiments.mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
    from experiments.run_experiment import RunResult, run_one  # type: ignore


def _discover_participants(runs_dir: Path) -> list[str]:
    """
    Discover participants from collected run_* folders:
      run_*/face/<participant>/...
    """
    participants: set[str] = set()
    for run_dir in sorted([p for p in runs_dir.glob("run_*") if p.is_dir()]):
        face_root = run_dir / "face"
        if not face_root.exists():
            continue
        for p in face_root.iterdir():
            if p.is_dir():
                participants.add(p.name)
    return sorted(participants)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric(r: RunResult, key: str) -> float | None:
    v = r.metrics.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def run_lopo(args: argparse.Namespace) -> list[RunResult]:
    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]
    mlflow = configure_mlflow(pi_agent_dir=pi_agent_dir, experiment_name="ai-study-buddy-pi-agent")

    runs_dir = (
        (pi_agent_dir / args.runs_dir).expanduser().resolve()
        if not Path(args.runs_dir).is_absolute()
        else Path(args.runs_dir).expanduser().resolve()
    )

    participants = _discover_participants(runs_dir)
    if not participants:
        raise SystemExit(f"[lopo] No participants found under {runs_dir}/run_*/face/<participant>/")

    results: list[RunResult] = []
    parent_name = f"lopo:{args.tag}"
    with mlflow.start_run(run_name=parent_name, nested=bool(getattr(args, "mlflow_nested", False))):
        mlflow.set_tag("kind", "lopo")
        mlflow.set_tag("tag_base", str(args.tag))
        mlflow.set_tag("created_at_utc", _utc_now_iso())
        mlflow.log_param("lopo.n_participants", int(len(participants)))
        mlflow.log_param("lopo.participants", ",".join(participants))

        total = len(participants)
        for i, participant in enumerate(participants, start=1):
            ns: Any = argparse.Namespace()
            ns.run_id = None
            ns.tag = f"{args.tag}:{participant} ({i}/{total})"

            ns.splits_dir = args.splits_dir
            ns.prepare_splits = True
            ns.runs_dir = str(runs_dir)
            ns.split_by = "participant"
            ns.holdout_participant = participant
            ns.copy = bool(args.copy)
            ns.mlflow_nested = True

            ns.backbone = args.backbone
            ns.input_size = args.input_size
            ns.batch_size = args.batch_size
            ns.epochs_head = args.epochs_head
            ns.epochs_finetune = args.epochs_finetune
            ns.fine_tune_at = args.fine_tune_at
            ns.label_smoothing = args.label_smoothing
            ns.no_augment = bool(args.no_augment)
            ns.no_class_weights = bool(args.no_class_weights)
            ns.quantize = bool(args.quantize)
            ns.seed = args.seed
            ns.tflite_name = args.tflite_name
            ns.dry_run = bool(args.dry_run)

            print(f"\n[lopo] Holdout={participant} ({i}/{total}) backbone={args.backbone} input={args.input_size}\n")
            res = run_one(ns)
            results.append(res)

        macro_f1s = [m for m in (_metric(r, "macro_f1") for r in results) if m is not None]
        if macro_f1s:
            mlflow.log_metric("lopo.macro_f1_mean", float(mean(macro_f1s)))
            mlflow.log_metric("lopo.macro_f1_std", float(pstdev(macro_f1s)) if len(macro_f1s) > 1 else 0.0)

        accs = [a for a in (_metric(r, "accuracy_from_cm") for r in results) if a is not None]
        if accs:
            mlflow.log_metric("lopo.acc_mean", float(mean(accs)))
            mlflow.log_metric("lopo.acc_std", float(pstdev(accs)) if len(accs) > 1 else 0.0)

        # Per-class recall means (if present)
        recalls: dict[str, list[float]] = {}
        for r in results:
            for k, v in r.metrics.items():
                if not isinstance(k, str) or not k.startswith("recall."):
                    continue
                if isinstance(v, (int, float)):
                    recalls.setdefault(k, []).append(float(v))
        for k, vals in recalls.items():
            mlflow.log_metric(f"lopo.{k}_mean", float(mean(vals)))

        # Brief artifact
        top = sorted(
            [r for r in results if _metric(r, "macro_f1") is not None],
            key=lambda r: float(_metric(r, "macro_f1") or -1.0),
            reverse=True,
        )[: min(10, len(results))]
        lines = [
            "# LOPO brief",
            "",
            f"- tag_base: `{args.tag}`",
            f"- participants: `{len(participants)}`",
            f"- created_at_utc: `{_utc_now_iso()}`",
            "",
            "## Aggregate",
            "",
            f"- macro_f1_mean: `{mean(macro_f1s):.4f}`" if macro_f1s else "- macro_f1_mean: `N/A`",
            f"- macro_f1_std: `{(pstdev(macro_f1s) if len(macro_f1s)>1 else 0.0):.4f}`" if macro_f1s else "- macro_f1_std: `N/A`",
            f"- acc_mean: `{mean(accs):.4f}`" if accs else "- acc_mean: `N/A`",
            "",
            "## Best holdouts",
            "",
            "| rank | holdout | macro_f1 | acc | mlflow_run_id |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
        for rank, r in enumerate(top, start=1):
            hold = r.config.get("data", {}).get("holdoutParticipant") if isinstance(r.config.get("data"), dict) else None
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        str(hold or ""),
                        f"{_metric(r,'macro_f1'):.4f}" if _metric(r, "macro_f1") is not None else "",
                        f"{_metric(r,'accuracy_from_cm'):.4f}" if _metric(r, "accuracy_from_cm") is not None else "",
                        str(r.mlflow_run_id),
                    ]
                )
                + " |"
            )
        lines.append("")
        log_text(mlflow, "\n".join(lines) + "\n", "brief.md")

        summary = {
            "tag_base": args.tag,
            "participants": participants,
            "children": [
                {
                    "run_id": r.run_id,
                    "mlflow_run_id": r.mlflow_run_id,
                    "tag": r.config.get("tag"),
                    "holdout_participant": (r.config.get("data") or {}).get("holdoutParticipant") if isinstance(r.config.get("data"), dict) else None,
                    "macro_f1": r.metrics.get("macro_f1"),
                    "accuracy_from_cm": r.metrics.get("accuracy_from_cm"),
                }
                for r in results
            ],
        }
        log_dict_as_json(mlflow, summary, "summary.json")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LOPO (leave-one-participant-out) experiments (MLflow-only).")
    parser.add_argument("--runs-dir", default="data", help="Directory containing run_* folders")
    parser.add_argument("--splits-dir", default="data/splits", help="Split output dir (overwritten per holdout)")
    parser.add_argument("--split-by", choices=["participant"], default="participant")
    parser.add_argument("--tag", default="lopo", help="Base tag applied to all LOPO runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--copy", action="store_true")

    # Training knobs (forwarded to run_experiment)
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

    args = parser.parse_args(argv)
    run_lopo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

