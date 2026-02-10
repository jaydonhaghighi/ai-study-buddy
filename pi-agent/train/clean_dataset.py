from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    # Running from pi-agent root (python train/clean_dataset.py ...)
    from experiments.headpose_eval import (  # type: ignore
        HeadPoseEstimator,
        _decide_label,
        _ensure_face_landmarker_task,
        _map_label,
    )
except Exception:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.headpose_eval import (  # type: ignore
        HeadPoseEstimator,
        _decide_label,
        _ensure_face_landmarker_task,
        _map_label,
    )


RAW_LABELS = {"screen", "away_left", "away_right", "away_up", "away_down"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class CleanerConfig:
    runs_dir: Path
    out_report: Path
    out_csv: Path
    quarantine_dir: Path | None
    apply: bool
    copy_instead_of_move: bool
    blur_laplacian_var_min: float
    require_pose_match: bool
    allow_no_landmarks: bool
    yaw_deg: float
    pitch_deg: float
    deadzone_yaw_deg: float
    deadzone_pitch_deg: float
    dominance_ratio: float
    no_dominant_axis: bool
    invert_yaw: bool
    invert_pitch: bool
    merge_screen_down_focus: bool
    calibration_screens: int
    task_path: Path
    download_task: bool
    max_files: int | None


def _find_label(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part in RAW_LABELS:
            return part
    return None


def _extract_participant(path: Path) -> str:
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part == "face" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def _iter_items(runs_dir: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for p in sorted(runs_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lab = _find_label(p)
        if lab is None:
            continue
        items.append((p, lab))
    return items


def _laplacian_variance(cv2: Any, img: Any) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _calc_neutral_offsets(
    *,
    cv2: Any,
    est: HeadPoseEstimator,
    items: list[tuple[Path, str]],
    calibration_screens: int,
    invert_yaw: bool,
    invert_pitch: bool,
) -> dict[str, tuple[float, float]]:
    by_person_screen: dict[str, list[Path]] = defaultdict(list)
    for p, lab in items:
        if lab != "screen":
            continue
        person = _extract_participant(p)
        by_person_screen[person].append(p)

    out: dict[str, tuple[float, float]] = {}
    for person, paths in by_person_screen.items():
        yaws: list[float] = []
        pitches: list[float] = []
        for p in paths[: max(0, int(calibration_screens))]:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            yp = est.estimate_yaw_pitch(img)
            if yp is None:
                continue
            yaw, pitch = yp
            if invert_yaw:
                yaw = -yaw
            if invert_pitch:
                pitch = -pitch
            yaws.append(float(yaw))
            pitches.append(float(pitch))
        if yaws and pitches:
            out[person] = (float(sum(yaws) / len(yaws)), float(sum(pitches) / len(pitches)))
    return out


def clean_dataset(cfg: CleanerConfig) -> dict[str, Any]:
    import cv2  # type: ignore

    if cfg.download_task:
        _ensure_face_landmarker_task(cfg.task_path)

    items = _iter_items(cfg.runs_dir)
    if cfg.max_files is not None:
        items = items[: max(0, int(cfg.max_files))]
    if not items:
        raise SystemExit(f"[clean_dataset] No labeled images found under: {cfg.runs_dir}")

    est = HeadPoseEstimator(task_path=cfg.task_path)
    neutral_offsets = _calc_neutral_offsets(
        cv2=cv2,
        est=est,
        items=items,
        calibration_screens=cfg.calibration_screens,
        invert_yaw=cfg.invert_yaw,
        invert_pitch=cfg.invert_pitch,
    )

    rows: list[dict[str, Any]] = []
    counts = defaultdict(int)
    per_label = defaultdict(lambda: defaultdict(int))

    for img_path, raw_label in items:
        label = _map_label(raw_label, merge_screen_down_focus=cfg.merge_screen_down_focus)
        participant = _extract_participant(img_path)
        row: dict[str, Any] = {
            "path": str(img_path),
            "participant": participant,
            "raw_label": raw_label,
            "effective_label": label,
            "blur_score": None,
            "yaw_deg": None,
            "pitch_deg": None,
            "predicted_label_from_pose": None,
            "status": "ok",
            "reasons": [],
        }
        counts["total"] += 1
        per_label[label]["total"] += 1

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            row["status"] = "flagged"
            row["reasons"].append("unreadable")
            counts["unreadable"] += 1
            rows.append(row)
            per_label[label]["flagged"] += 1
            continue

        blur_score = _laplacian_variance(cv2, img)
        row["blur_score"] = float(blur_score)
        if blur_score < cfg.blur_laplacian_var_min:
            row["status"] = "flagged"
            row["reasons"].append("blurry")
            counts["blurry"] += 1

        yp = est.estimate_yaw_pitch(img)
        if yp is None:
            row["reasons"].append("no_landmarks")
            counts["no_landmarks"] += 1
            if not cfg.allow_no_landmarks:
                row["status"] = "flagged"
        else:
            yaw, pitch = yp
            if cfg.invert_yaw:
                yaw = -yaw
            if cfg.invert_pitch:
                pitch = -pitch
            neutral_yaw, neutral_pitch = neutral_offsets.get(participant, (0.0, 0.0))
            dyaw = float(yaw - neutral_yaw)
            dpitch = float(pitch - neutral_pitch)
            row["yaw_deg"] = dyaw
            row["pitch_deg"] = dpitch
            pred = _decide_label(
                yaw_deg=dyaw,
                pitch_deg=dpitch,
                yaw_th=cfg.yaw_deg,
                pitch_th=cfg.pitch_deg,
                deadzone_yaw_deg=cfg.deadzone_yaw_deg,
                deadzone_pitch_deg=cfg.deadzone_pitch_deg,
                dominance_ratio=cfg.dominance_ratio,
                use_dominant_axis=not cfg.no_dominant_axis,
                merge_screen_down_focus=cfg.merge_screen_down_focus,
            )
            row["predicted_label_from_pose"] = pred
            if cfg.require_pose_match and pred != label:
                row["status"] = "flagged"
                row["reasons"].append("label_pose_mismatch")
                counts["label_pose_mismatch"] += 1

        if row["status"] == "flagged":
            counts["flagged"] += 1
            per_label[label]["flagged"] += 1
        rows.append(row)

    # Optional quarantine stage.
    moved = 0
    if cfg.quarantine_dir is not None and cfg.apply:
        qroot = cfg.quarantine_dir.resolve()
        qroot.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r["status"] != "flagged":
                continue
            src = Path(r["path"]).resolve()
            try:
                rel = src.relative_to(cfg.runs_dir.resolve())
            except Exception:
                # Safety: never move unknown external paths.
                continue
            dst = qroot / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if cfg.copy_instead_of_move:
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))
            moved += 1

    summary = {
        "runs_dir": str(cfg.runs_dir),
        "total": int(counts["total"]),
        "flagged": int(counts["flagged"]),
        "flagged_rate": float(counts["flagged"] / max(1, counts["total"])),
        "blurry": int(counts["blurry"]),
        "unreadable": int(counts["unreadable"]),
        "no_landmarks": int(counts["no_landmarks"]),
        "label_pose_mismatch": int(counts["label_pose_mismatch"]),
        "quarantine_applied": bool(cfg.apply and cfg.quarantine_dir is not None),
        "quarantined_count": int(moved),
        "neutral_offsets_found_for_participants": int(len(neutral_offsets)),
        "config": {
            "blur_laplacian_var_min": float(cfg.blur_laplacian_var_min),
            "require_pose_match": bool(cfg.require_pose_match),
            "allow_no_landmarks": bool(cfg.allow_no_landmarks),
            "yaw_deg": float(cfg.yaw_deg),
            "pitch_deg": float(cfg.pitch_deg),
            "deadzone_yaw_deg": float(cfg.deadzone_yaw_deg),
            "deadzone_pitch_deg": float(cfg.deadzone_pitch_deg),
            "dominance_ratio": float(cfg.dominance_ratio),
            "no_dominant_axis": bool(cfg.no_dominant_axis),
            "invert_yaw": bool(cfg.invert_yaw),
            "invert_pitch": bool(cfg.invert_pitch),
            "merge_screen_down_focus": bool(cfg.merge_screen_down_focus),
            "calibration_screens": int(cfg.calibration_screens),
        },
        "per_label": {
            lab: {
                "total": int(stats["total"]),
                "flagged": int(stats["flagged"]),
                "flagged_rate": float(stats["flagged"] / max(1, stats["total"])),
            }
            for lab, stats in sorted(per_label.items())
        },
    }

    cfg.out_report.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_report.write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")

    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "participant",
                "raw_label",
                "effective_label",
                "status",
                "reasons",
                "blur_score",
                "yaw_deg",
                "pitch_deg",
                "predicted_label_from_pose",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "path": r["path"],
                    "participant": r["participant"],
                    "raw_label": r["raw_label"],
                    "effective_label": r["effective_label"],
                    "status": r["status"],
                    "reasons": "|".join(r["reasons"]),
                    "blur_score": r["blur_score"],
                    "yaw_deg": r["yaw_deg"],
                    "pitch_deg": r["pitch_deg"],
                    "predicted_label_from_pose": r["predicted_label_from_pose"],
                }
            )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flag low-quality or label-inconsistent face images (blur + head-pose checks)."
    )
    parser.add_argument("--runs-dir", required=True, help="Dataset root (e.g. data/, data_aligned_m025_v2/)")
    parser.add_argument("--out-report", default="artifacts/data_clean/report.json")
    parser.add_argument("--out-csv", default="artifacts/data_clean/report.csv")
    parser.add_argument("--quarantine-dir", default=None, help="If set with --apply, move/copy flagged files here.")
    parser.add_argument("--apply", action="store_true", help="Actually move/copy flagged files to quarantine.")
    parser.add_argument("--copy-instead-of-move", action="store_true", help="With --apply, copy flagged files instead of moving.")
    parser.add_argument("--blur-laplacian-var-min", type=float, default=80.0, help="Blur threshold. Lower is blurrier.")
    parser.add_argument("--no-require-pose-match", action="store_true", help="Disable label-vs-pose mismatch flagging.")
    parser.add_argument("--allow-no-landmarks", action="store_true", help="Do not flag no-landmark images.")
    parser.add_argument("--yaw-deg", type=float, default=15.0)
    parser.add_argument("--pitch-deg", type=float, default=12.0)
    parser.add_argument("--deadzone-yaw-deg", type=float, default=10.0)
    parser.add_argument("--deadzone-pitch-deg", type=float, default=8.0)
    parser.add_argument("--dominance-ratio", type=float, default=1.15)
    parser.add_argument("--no-dominant-axis", action="store_true")
    parser.add_argument("--invert-yaw", action="store_true")
    parser.add_argument("--invert-pitch", action="store_true")
    parser.add_argument("--merge-screen-down-focus", action="store_true")
    parser.add_argument("--calibration-screens", type=int, default=40)
    parser.add_argument("--task-path", default="models/mediapipe/face_landmarker.task")
    parser.add_argument("--download-task", action="store_true")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick trial runs.")
    args = parser.parse_args(argv)

    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_report = (pi_agent_dir / args.out_report).resolve() if not Path(args.out_report).is_absolute() else Path(args.out_report).resolve()
    out_csv = (pi_agent_dir / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv).resolve()
    quarantine_dir = None
    if args.quarantine_dir:
        quarantine_dir = (pi_agent_dir / args.quarantine_dir).resolve() if not Path(args.quarantine_dir).is_absolute() else Path(args.quarantine_dir).resolve()

    task_path = (pi_agent_dir / args.task_path).resolve() if not Path(args.task_path).is_absolute() else Path(args.task_path).resolve()

    cfg = CleanerConfig(
        runs_dir=runs_dir,
        out_report=out_report,
        out_csv=out_csv,
        quarantine_dir=quarantine_dir,
        apply=bool(args.apply),
        copy_instead_of_move=bool(args.copy_instead_of_move),
        blur_laplacian_var_min=float(args.blur_laplacian_var_min),
        require_pose_match=not bool(args.no_require_pose_match),
        allow_no_landmarks=bool(args.allow_no_landmarks),
        yaw_deg=float(args.yaw_deg),
        pitch_deg=float(args.pitch_deg),
        deadzone_yaw_deg=float(args.deadzone_yaw_deg),
        deadzone_pitch_deg=float(args.deadzone_pitch_deg),
        dominance_ratio=float(args.dominance_ratio),
        no_dominant_axis=bool(args.no_dominant_axis),
        invert_yaw=bool(args.invert_yaw),
        invert_pitch=bool(args.invert_pitch),
        merge_screen_down_focus=bool(args.merge_screen_down_focus),
        calibration_screens=int(args.calibration_screens),
        task_path=task_path,
        download_task=bool(args.download_task),
        max_files=int(args.max_files) if args.max_files is not None else None,
    )

    summary = clean_dataset(cfg)
    print(json.dumps(summary, indent=2))
    if cfg.quarantine_dir is not None and not cfg.apply:
        print("[clean_dataset] Dry-run only. Re-run with --apply to move/copy flagged files to quarantine.")
    print(f"[clean_dataset] report: {cfg.out_report}")
    print(f"[clean_dataset] csv: {cfg.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

