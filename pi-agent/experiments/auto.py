from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from statistics import mean

try:
    # When executed as a module: python -m experiments.auto
    from .mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
    from .sweep import run_sweep  # type: ignore
    from .lopo import run_lopo  # type: ignore
    from .run_experiment import RunResult  # type: ignore
except Exception:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from experiments.mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
    from experiments.sweep import run_sweep  # type: ignore
    from experiments.lopo import run_lopo  # type: ignore
    from experiments.run_experiment import RunResult  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Fingerprint a training config for ranking.
    Excludes seed so we can average across seeds.
    """
    tr = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    if not isinstance(tr, dict):
        tr = {}
    keys = [
        "backbone",
        "inputSize",
        "batchSize",
        "epochsHead",
        "epochsFinetune",
        "fineTuneAt",
        "labelSmoothing",
        "noAugment",
        "noClassWeights",
        "quantize",
    ]
    return {k: tr.get(k) for k in keys}


@dataclass(frozen=True)
class RankedConfig:
    fingerprint: dict[str, Any]
    mean_macro_f1: float
    n: int


def _metric(r: RunResult, key: str) -> float | None:
    v = r.metrics.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _rank_from_sweep_results(results: list[RunResult]) -> list[RankedConfig]:
    buckets: dict[str, list[float]] = {}
    fp_by_key: dict[str, dict[str, Any]] = {}
    for r in results:
        mf1 = _metric(r, "macro_f1")
        if mf1 is None:
            continue
        fp = _fingerprint(r.config)
        key = json.dumps(fp, sort_keys=True)
        buckets.setdefault(key, []).append(mf1)
        fp_by_key[key] = fp

    ranked: list[RankedConfig] = []
    for key, vals in buckets.items():
        ranked.append(RankedConfig(fingerprint=fp_by_key[key], mean_macro_f1=float(mean(vals)), n=len(vals)))
    ranked.sort(key=lambda r: (-r.mean_macro_f1, -r.n))
    return ranked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto loop: sweep → pick top K → LOPO (MLflow-only).")
    parser.add_argument("--runs-dir", default="data", help="Directory containing run_* folders")
    parser.add_argument("--splits-dir", default="data/splits", help="Directory for generated splits")

    # Sweep settings
    parser.add_argument("--sweep-tag", default="sweep_auto", help="Base tag for sweep runs")
    parser.add_argument("--backbones", default="mobilenetv3small,mobilenetv2")
    parser.add_argument("--input-sizes", default="160,192,224")
    parser.add_argument("--seeds", default="1337,1338")

    # Training knobs (applied to both sweep + LOPO)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--quantize", action="store_true")

    # Selection + LOPO
    parser.add_argument("--top-k", type=int, default=2, help="How many configs from sweep to LOPO-evaluate")
    parser.add_argument("--lopo-seed", type=int, default=1337, help="Seed used for LOPO runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pi_agent_dir = Path(__file__).resolve().parents[1]
    mlflow = configure_mlflow(pi_agent_dir=pi_agent_dir, experiment_name="ai-study-buddy-pi-agent")

    parent_name = f"auto:{args.sweep_tag}"
    with mlflow.start_run(run_name=parent_name):
        mlflow.set_tag("kind", "auto")
        mlflow.set_tag("created_at_utc", _utc_now_iso())
        mlflow.log_param("auto.sweep_tag", str(args.sweep_tag))
        mlflow.log_param("auto.top_k", int(args.top_k))
        mlflow.log_param("auto.lopo_seed", int(args.lopo_seed))

        # 1) Sweep (nested parent run inside auto)
        sweep_args = argparse.Namespace(
            backbones=args.backbones,
            input_sizes=args.input_sizes,
            seeds=args.seeds,
            splits_dir=args.splits_dir,
            dry_run=bool(args.dry_run),
            tag=str(args.sweep_tag),
            batch_size=args.batch_size,
            epochs_head=args.epochs_head,
            epochs_finetune=args.epochs_finetune,
            fine_tune_at=args.fine_tune_at,
            label_smoothing=args.label_smoothing,
            no_augment=bool(args.no_augment),
            no_class_weights=bool(args.no_class_weights),
            quantize=bool(args.quantize),
            tflite_name="focus_model.tflite",
            mlflow_nested=True,
        )
        sweep_results = run_sweep(sweep_args)

        ranked = _rank_from_sweep_results(sweep_results)
        if not ranked:
            raise SystemExit(f"[auto] Sweep produced no usable results for tag_base={args.sweep_tag!r}.")

        top = ranked[: max(1, int(args.top_k))]
        print("\n[auto] Top configs from sweep (by mean macro_f1):")
        for i, rcfg in enumerate(top, start=1):
            print(f"  {i}) mean_macro_f1={rcfg.mean_macro_f1:.4f} n={rcfg.n} fp={rcfg.fingerprint}")

        log_dict_as_json(
            mlflow,
            {
                "sweep_tag": args.sweep_tag,
                "ranked": [
                    {"rank": i + 1, "mean_macro_f1": r.mean_macro_f1, "n": r.n, "fingerprint": r.fingerprint}
                    for i, r in enumerate(ranked)
                ],
            },
            "auto/sweep_ranking.json",
        )

        # 2) LOPO on top configs (each is its own nested LOPO parent run)
        lopo_runs: list[dict[str, Any]] = []
        for i, rcfg in enumerate(top, start=1):
            fp = rcfg.fingerprint
            lopo_tag = f"lopo_from_{args.sweep_tag}_top{i}"

            lopo_args = argparse.Namespace(
                runs_dir=str(args.runs_dir),
                splits_dir=str(args.splits_dir),
                split_by="participant",
                tag=lopo_tag,
                dry_run=bool(args.dry_run),
                copy=False,
                backbone=str(fp.get("backbone") or "mobilenetv3small"),
                input_size=int(fp.get("inputSize") or 224),
                batch_size=int(fp.get("batchSize") or args.batch_size),
                epochs_head=int(fp.get("epochsHead") or args.epochs_head),
                epochs_finetune=int(fp.get("epochsFinetune") or args.epochs_finetune),
                fine_tune_at=int(fp.get("fineTuneAt") or args.fine_tune_at),
                label_smoothing=float(fp.get("labelSmoothing") or args.label_smoothing),
                no_augment=bool(fp.get("noAugment") or args.no_augment),
                no_class_weights=bool(fp.get("noClassWeights") or args.no_class_weights),
                quantize=bool(fp.get("quantize") or args.quantize),
                seed=int(args.lopo_seed),
                tflite_name="focus_model.tflite",
                mlflow_nested=True,
            )
            lopo_results = run_lopo(lopo_args)
            lopo_runs.append(
                {
                    "rank": i,
                    "lopo_tag": lopo_tag,
                    "fingerprint": fp,
                    "n_participants": len(lopo_results),
                }
            )

        log_dict_as_json(mlflow, {"lopo_runs": lopo_runs}, "auto/lopo_runs.json")

        # 3) Brief artifact at top-level auto run
        brief = [
            "# Auto brief",
            "",
            f"- sweep_tag: `{args.sweep_tag}`",
            f"- top_k: `{args.top_k}`",
            f"- created_at_utc: `{_utc_now_iso()}`",
            "",
            "## Top configs chosen for LOPO",
            "",
        ]
        for i, rcfg in enumerate(top, start=1):
            brief.append(f"- {i}) mean_macro_f1={rcfg.mean_macro_f1:.4f} n={rcfg.n} fp={rcfg.fingerprint}")
        brief.append("")
        log_text(mlflow, "\n".join(brief) + "\n", "brief.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

