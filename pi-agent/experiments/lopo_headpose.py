from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    from .mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
    from .headpose_eval import evaluate_headpose  # type: ignore
except Exception:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from experiments.mlflow_utils import configure_mlflow, log_dict_as_json, log_text  # type: ignore
    from experiments.headpose_eval import evaluate_headpose  # type: ignore


LABELS = ["screen", "away_left", "away_right", "away_up", "away_down"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _discover_participants(runs_dir: Path) -> list[str]:
    participants: set[str] = set()
    for run_dir in sorted([p for p in runs_dir.glob("run_*") if p.is_dir()]):
        face_root = run_dir / "face"
        if not face_root.exists():
            continue
        for p in face_root.iterdir():
            if p.is_dir():
                participants.add(p.name)
    return sorted(participants)


def _run_cmd(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return int(p.returncode), str(p.stdout)


def run_lopo_headpose(args: argparse.Namespace) -> int:
    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]
    mlflow = configure_mlflow(pi_agent_dir=pi_agent_dir, experiment_name="ai-study-buddy-pi-agent")

    runs_dir = (pi_agent_dir / args.runs_dir).expanduser().resolve() if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir).expanduser().resolve()
    participants = _discover_participants(runs_dir)
    if not participants:
        raise SystemExit(f"[lopo_headpose] No participants found under {runs_dir}/run_*/face/<participant>/")

    task_path = (pi_agent_dir / args.task_path).resolve() if not Path(args.task_path).is_absolute() else Path(args.task_path).resolve()

    parent_name = f"lopo_headpose:{args.tag}"
    with mlflow.start_run(run_name=parent_name, nested=bool(getattr(args, "mlflow_nested", False))):
        mlflow.set_tag("kind", "lopo_headpose")
        mlflow.set_tag("tag_base", str(args.tag))
        mlflow.set_tag("created_at_utc", _utc_now_iso())
        mlflow.log_param("method", "headpose")
        mlflow.log_param("runs_dir", str(runs_dir))
        mlflow.log_param("splits_dir", str(args.splits_dir))
        mlflow.log_param("yaw_deg", float(args.yaw_deg))
        mlflow.log_param("pitch_deg", float(args.pitch_deg))
        mlflow.log_param("deadzone_yaw_deg", float(args.deadzone_yaw_deg))
        mlflow.log_param("deadzone_pitch_deg", float(args.deadzone_pitch_deg))
        mlflow.log_param("dominance_ratio", float(args.dominance_ratio))
        mlflow.log_param("use_dominant_axis", not bool(args.no_dominant_axis))
        mlflow.log_param("invert_yaw", bool(args.invert_yaw))
        mlflow.log_param("invert_pitch", bool(args.invert_pitch))
        mlflow.log_param("merge_screen_down_focus", bool(args.merge_screen_down_focus))
        mlflow.log_param("calibration_screens", int(args.calibration_screens))
        mlflow.log_param("task_path", str(task_path))
        mlflow.log_param("lopo.n_participants", int(len(participants)))
        mlflow.log_param("lopo.participants", ",".join(participants))

        child_metrics: list[dict[str, Any]] = []
        macro_f1s: list[float] = []
        accs: list[float] = []

        total = len(participants)
        for i, participant in enumerate(participants, start=1):
            tag = f"{args.tag}:{participant} ({i}/{total})"
            print(f"\n[lopo_headpose] Holdout={participant} ({i}/{total})\n")

            # Prepare splits for this holdout.
            prep = [
                sys.executable,
                str(pi_agent_dir / "train" / "prepare_dataset.py"),
                "--runs-dir",
                str(runs_dir),
                "--out-dir",
                str(args.splits_dir),
                "--split-by",
                "participant",
                "--holdout-participant",
                str(participant),
            ]
            rc, out = _run_cmd(prep, cwd=pi_agent_dir)
            if rc != 0:
                log_text(mlflow, out, f"prepare_splits/{participant}.log")
                raise SystemExit(f"prepare_dataset.py failed for holdout={participant} (exit={rc})")
            log_text(mlflow, out, f"prepare_splits/{participant}.log")

            m = evaluate_headpose(
                splits_dir=Path(args.splits_dir).expanduser().resolve(),
                tag=tag,
                holdout_participant=participant,
                yaw_deg=float(args.yaw_deg),
                pitch_deg=float(args.pitch_deg),
                deadzone_yaw_deg=float(args.deadzone_yaw_deg),
                deadzone_pitch_deg=float(args.deadzone_pitch_deg),
                dominance_ratio=float(args.dominance_ratio),
                use_dominant_axis=not bool(args.no_dominant_axis),
                invert_yaw=bool(args.invert_yaw),
                invert_pitch=bool(args.invert_pitch),
                merge_screen_down_focus=bool(args.merge_screen_down_focus),
                calibration_screens=int(args.calibration_screens),
                task_path=task_path,
                mlflow_nested=True,
            )

            child_metrics.append({"holdout": participant, "macro_f1": m.get("macro_f1"), "accuracy_from_cm": m.get("accuracy_from_cm")})
            if isinstance(m.get("macro_f1"), (int, float)):
                macro_f1s.append(float(m["macro_f1"]))
            if isinstance(m.get("accuracy_from_cm"), (int, float)):
                accs.append(float(m["accuracy_from_cm"]))

        if macro_f1s:
            mlflow.log_metric("lopo.macro_f1_mean", float(mean(macro_f1s)))
            mlflow.log_metric("lopo.macro_f1_std", float(pstdev(macro_f1s)) if len(macro_f1s) > 1 else 0.0)
        if accs:
            mlflow.log_metric("lopo.acc_mean", float(mean(accs)))
            mlflow.log_metric("lopo.acc_std", float(pstdev(accs)) if len(accs) > 1 else 0.0)

        # Brief
        best = sorted(
            [d for d in child_metrics if isinstance(d.get("macro_f1"), (int, float))],
            key=lambda d: float(d.get("macro_f1") or -1.0),
            reverse=True,
        )
        lines = [
            "# LOPO head-pose brief",
            "",
            f"- tag_base: `{args.tag}`",
            f"- participants: `{len(participants)}`",
            f"- created_at_utc: `{_utc_now_iso()}`",
            "",
            "## Aggregate",
            "",
            f"- macro_f1_mean: `{mean(macro_f1s):.4f}`" if macro_f1s else "- macro_f1_mean: `N/A`",
            f"- acc_mean: `{mean(accs):.4f}`" if accs else "- acc_mean: `N/A`",
            "",
            "## Holdouts",
            "",
            "| rank | holdout | macro_f1 | acc |",
            "| ---: | --- | ---: | ---: |",
        ]
        for rank, d in enumerate(best, start=1):
            lines.append(f"| {rank} | {d['holdout']} | {float(d['macro_f1']):.4f} | {float(d['accuracy_from_cm']):.4f} |")
        lines.append("")
        log_text(mlflow, "\n".join(lines) + "\n", "brief.md")
        log_dict_as_json(mlflow, {"tag_base": args.tag, "participants": participants, "children": child_metrics}, "summary.json")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LOPO evaluation for head-pose baseline (MLflow-only).")
    parser.add_argument("--runs-dir", default="data", help="Directory containing run_* folders")
    parser.add_argument("--splits-dir", default="data/splits_headpose", help="Split output dir (overwritten per holdout)")
    parser.add_argument("--tag", default="lopo_headpose")
    parser.add_argument("--yaw-deg", type=float, default=15.0)
    parser.add_argument("--pitch-deg", type=float, default=12.0)
    parser.add_argument("--deadzone-yaw-deg", type=float, default=10.0)
    parser.add_argument("--deadzone-pitch-deg", type=float, default=8.0)
    parser.add_argument("--dominance-ratio", type=float, default=1.15)
    parser.add_argument("--no-dominant-axis", action="store_true")
    parser.add_argument("--invert-yaw", action="store_true")
    parser.add_argument("--invert-pitch", action="store_true")
    parser.add_argument("--merge-screen-down-focus", action="store_true", help="Map screen+away_down into focused.")
    parser.add_argument("--calibration-screens", type=int, default=40, help="Use first N screen frames for neutral calibration.")
    parser.add_argument("--task-path", default="models/mediapipe/face_landmarker.task")
    parser.add_argument("--download-task", action="store_true")
    args = parser.parse_args(argv)

    # Optional download (local convenience)
    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]
    task_path = (pi_agent_dir / args.task_path).resolve() if not Path(args.task_path).is_absolute() else Path(args.task_path).resolve()
    if args.download_task:
        from experiments.headpose_eval import _ensure_face_landmarker_task  # type: ignore

        _ensure_face_landmarker_task(task_path)

    return run_lopo_headpose(args)


if __name__ == "__main__":
    raise SystemExit(main())

