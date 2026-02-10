from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from statistics import mean, pstdev

try:
    # When executed as a module: python -m experiments.sweep
    from .mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
    from .run_experiment import RunResult, run_one  # type: ignore
except Exception:  # pragma: no cover
    # When executed as a script: python experiments/sweep.py
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from experiments.mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
    from experiments.run_experiment import RunResult, run_one  # type: ignore


@dataclass(frozen=True)
class SweepSpec:
    backbones: list[str]
    input_sizes: list[int]
    seeds: list[int]


def _parse_csv_list(raw: str, cast) -> list:
    items = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric(r: RunResult, key: str) -> float | None:
    v = r.metrics.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def run_sweep(args: argparse.Namespace) -> list[RunResult]:
    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]
    mlflow = configure_mlflow(pi_agent_dir=pi_agent_dir, experiment_name="ai-study-buddy-pi-agent")

    spec = SweepSpec(
        backbones=_parse_csv_list(args.backbones, str) or ["mobilenetv3small"],
        input_sizes=_parse_csv_list(args.input_sizes, int) or [224],
        seeds=_parse_csv_list(args.seeds, int) or [1337],
    )

    total = len(spec.backbones) * len(spec.input_sizes) * len(spec.seeds)
    results: list[RunResult] = []

    parent_name = f"sweep:{args.tag}"
    with mlflow.start_run(run_name=parent_name, nested=bool(getattr(args, "mlflow_nested", False))):
        mlflow.set_tag("kind", "sweep")
        mlflow.set_tag("tag_base", str(args.tag))
        mlflow.set_tag("created_at_utc", _utc_now_iso())
        mlflow.log_param("sweep.backbones", ",".join(spec.backbones))
        mlflow.log_param("sweep.input_sizes", ",".join(str(x) for x in spec.input_sizes))
        mlflow.log_param("sweep.seeds", ",".join(str(x) for x in spec.seeds))
        mlflow.log_param("sweep.total", int(total))

        i = 0
        for backbone, input_size, seed in itertools.product(spec.backbones, spec.input_sizes, spec.seeds):
            i += 1
            ns: Any = argparse.Namespace(**vars(args))
            ns.run_id = None
            ns.prepare_splits = False
            ns.runs_dir = "data"
            ns.split_by = "participant"
            ns.holdout_participant = None
            ns.copy = False
            ns.mlflow_nested = True

            ns.backbone = backbone
            ns.input_size = int(input_size)
            ns.seed = int(seed)
            ns.tag = f"{args.tag}:{i}/{total}"

            print(f"\n[sweep] {i}/{total} backbone={backbone} input_size={input_size} seed={seed}\n")
            res = run_one(ns)
            results.append(res)

        macro_f1s = [m for m in (_metric(r, "macro_f1") for r in results) if m is not None]
        if macro_f1s:
            mlflow.log_metric("macro_f1_mean", float(mean(macro_f1s)))
            mlflow.log_metric("macro_f1_std", float(pstdev(macro_f1s)) if len(macro_f1s) > 1 else 0.0)

        best = None
        for r in results:
            mf1 = _metric(r, "macro_f1")
            if mf1 is None:
                continue
            if best is None or mf1 > best[0]:
                best = (mf1, r)
        if best is not None:
            mlflow.log_metric("best.macro_f1", float(best[0]))
            mlflow.set_tag("best.mlflow_run_id", best[1].mlflow_run_id)
            mlflow.set_tag("best.run_id", best[1].run_id)

        # Brief artifact
        top = sorted(
            [r for r in results if _metric(r, "macro_f1") is not None],
            key=lambda r: float(_metric(r, "macro_f1") or -1.0),
            reverse=True,
        )[: min(10, len(results))]
        lines = [
            "# Sweep brief",
            "",
            f"- tag_base: `{args.tag}`",
            f"- total runs: `{total}`",
            f"- created_at_utc: `{_utc_now_iso()}`",
            "",
            "## Top runs",
            "",
            "| rank | macro_f1 | acc | backbone | input | seed | child_run_name | mlflow_run_id |",
            "| ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
        for rank, r in enumerate(top, start=1):
            tr = r.config.get("train") if isinstance(r.config.get("train"), dict) else {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        f"{_metric(r,'macro_f1'):.4f}" if _metric(r, "macro_f1") is not None else "",
                        f"{_metric(r,'accuracy_from_cm'):.4f}" if _metric(r, "accuracy_from_cm") is not None else "",
                        str(tr.get("backbone") or ""),
                        str(tr.get("inputSize") or ""),
                        str(tr.get("seed") or ""),
                        str(r.config.get("tag") or ""),
                        str(r.mlflow_run_id),
                    ]
                )
                + " |"
            )
        lines.append("")
        log_text(mlflow, "\n".join(lines) + "\n", "brief.md")

        # Machine-readable summary
        summary = {
            "tag_base": args.tag,
            "total": total,
            "children": [
                {
                    "run_id": r.run_id,
                    "mlflow_run_id": r.mlflow_run_id,
                    "tag": r.config.get("tag"),
                    "train": r.config.get("train"),
                    "macro_f1": r.metrics.get("macro_f1"),
                    "accuracy_from_cm": r.metrics.get("accuracy_from_cm"),
                }
                for r in results
            ],
        }
        log_dict_as_json(mlflow, summary, "summary.json")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small local sweep over training configs (MLflow-only).")
    parser.add_argument("--backbones", default="mobilenetv3small,mobilenetv2", help="Comma-separated list")
    parser.add_argument("--input-sizes", default="160,192,224", help="Comma-separated list")
    parser.add_argument("--seeds", default="1337,1338", help="Comma-separated list")

    # Forwarded to run_experiment
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", default="sweep")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--tflite-name", default="focus_model.tflite")

    args = parser.parse_args(argv)
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

