from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    # Module mode: python -m experiments.headpose_eval
    from .mlflow_utils import configure_mlflow, log_dict_as_json  # type: ignore
except Exception:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from experiments.mlflow_utils import configure_mlflow, log_dict_as_json  # type: ignore


RAW_LABELS = ["screen", "away_left", "away_right", "away_up", "away_down"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:  # nosec - intended local download
        data = r.read()
    dst.write_bytes(data)


def _ensure_face_landmarker_task(task_path: Path) -> None:
    if task_path.exists() and task_path.stat().st_size > 1_000_000:
        return
    url = "https://storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker.task"
    print(f"[headpose] downloading face_landmarker.task to {task_path}")
    _download(url, task_path)


@dataclass(frozen=True)
class HeadPoseConfig:
    yaw_deg: float
    pitch_deg: float
    deadzone_yaw_deg: float
    deadzone_pitch_deg: float
    dominance_ratio: float
    use_dominant_axis: bool
    invert_yaw: bool
    invert_pitch: bool
    merge_screen_down_focus: bool
    calibration_screens: int
    model_task_path: str


class HeadPoseEstimator:
    """
    Geometry-first baseline:
      image -> face landmarks -> solvePnP -> yaw/pitch -> bin into 5 labels
    """

    def __init__(self, *, task_path: Path):
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
            from mediapipe.tasks.python.core import base_options as _base_options  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Head-pose baseline requires `mediapipe` + OpenCV. "
                "Install: `pip install -U mediapipe opencv-python`"
            ) from e

        self._cv2 = cv2
        self._np = np
        self._mp = mp
        self._vision = vision
        self._base_options = _base_options

        base = _base_options.BaseOptions(model_asset_path=str(task_path))
        opts = vision.FaceLandmarkerOptions(base_options=base, output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=1)
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)

    def estimate_yaw_pitch(self, bgr_image) -> tuple[float, float] | None:
        cv2 = self._cv2
        np = self._np
        mp = self._mp

        h, w = bgr_image.shape[:2]
        if h < 20 or w < 20:
            return None

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_image)
        if not res.face_landmarks:
            return None

        lms = res.face_landmarks[0]
        # FaceMesh landmark indices (common subset).
        idxs = {
            "nose": 1,
            "chin": 152,
            "leye": 33,
            "reye": 263,
            "lmouth": 61,
            "rmouth": 291,
        }

        def pt(i: int) -> tuple[float, float]:
            p = lms[i]
            return float(p.x) * w, float(p.y) * h

        image_points = np.array([pt(idxs[k]) for k in ["nose", "chin", "leye", "reye", "lmouth", "rmouth"]], dtype=np.float64)

        # Generic 3D face model points (mm-ish). Scale doesn’t matter for angles.
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # nose tip
                (0.0, -63.6, -12.5),  # chin
                (-43.3, 32.7, -26.0),  # left eye outer corner
                (43.3, 32.7, -26.0),  # right eye outer corner
                (-28.9, -28.9, -24.1),  # left mouth corner
                (28.9, -28.9, -24.1),  # right mouth corner
            ],
            dtype=np.float64,
        )

        focal = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        # Euler angles from rotation matrix (x=pitch, y=yaw, z=roll)
        sy = math.sqrt(float(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0]))
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(float(rmat[2, 1]), float(rmat[2, 2]))
            yaw = math.atan2(float(-rmat[2, 0]), float(sy))
        else:
            pitch = math.atan2(float(-rmat[1, 2]), float(rmat[1, 1]))
            yaw = math.atan2(float(-rmat[2, 0]), float(sy))

        pitch_deg = float(pitch * 180.0 / math.pi)
        yaw_deg = float(yaw * 180.0 / math.pi)
        return yaw_deg, pitch_deg


def _effective_labels(*, merge_screen_down_focus: bool) -> list[str]:
    if merge_screen_down_focus:
        return ["focused", "away_left", "away_right", "away_up"]
    return list(RAW_LABELS)


def _map_label(raw_label: str, *, merge_screen_down_focus: bool) -> str:
    if merge_screen_down_focus and raw_label in {"screen", "away_down"}:
        return "focused"
    return raw_label


def _decide_label(
    *,
    yaw_deg: float,
    pitch_deg: float,
    yaw_th: float,
    pitch_th: float,
    deadzone_yaw_deg: float,
    deadzone_pitch_deg: float,
    dominance_ratio: float,
    use_dominant_axis: bool,
    merge_screen_down_focus: bool,
) -> str:
    # Deadzone around neutral.
    if abs(yaw_deg) <= deadzone_yaw_deg and abs(pitch_deg) <= deadzone_pitch_deg:
        return "focused" if merge_screen_down_focus else "screen"

    # Dominant-axis logic helps avoid frequent diagonal misclassification.
    ny = abs(yaw_deg) / max(1e-6, yaw_th)
    np = abs(pitch_deg) / max(1e-6, pitch_th)
    if use_dominant_axis:
        if ny >= np * dominance_ratio and abs(yaw_deg) >= yaw_th:
            return "away_left" if yaw_deg < 0 else "away_right"
        if np >= ny * dominance_ratio and abs(pitch_deg) >= pitch_th:
            if pitch_deg < 0:
                return "away_up"
            return "focused" if merge_screen_down_focus else "away_down"

    # Fallback if no strong dominant axis.
    if abs(yaw_deg) >= yaw_th and abs(yaw_deg) >= abs(pitch_deg):
        return "away_left" if yaw_deg < 0 else "away_right"
    if abs(pitch_deg) >= pitch_th:
        if pitch_deg < 0:
            return "away_up"
        return "focused" if merge_screen_down_focus else "away_down"
    return "focused" if merge_screen_down_focus else "screen"


def _iter_labeled_images(split_dir: Path, *, labels: list[str]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for lab in labels:
        d = split_dir / lab
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpg")):
            out.append((p, lab))
        for p in sorted(d.glob("*.jpeg")):
            out.append((p, lab))
        for p in sorted(d.glob("*.png")):
            out.append((p, lab))
    return out


def _compute_metrics(y_true: list[str], y_pred: list[str], *, labels: list[str]) -> dict[str, Any]:
    # Confusion matrix
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        cm[idx[t]][idx[p]] += 1

    per_class: dict[str, dict[str, Any]] = {}
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        support = sum(cm[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        per_class[lab] = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "support": int(support)}

    macro_f1 = sum(per_class[lab]["f1"] for lab in labels) / max(1, n)
    total = sum(sum(r) for r in cm)
    acc = sum(cm[i][i] for i in range(n)) / max(1, total)

    return {
        "accuracy_from_cm": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


def evaluate_headpose(
    *,
    splits_dir: Path,
    tag: str,
    holdout_participant: str | None,
    yaw_deg: float,
    pitch_deg: float,
    deadzone_yaw_deg: float,
    deadzone_pitch_deg: float,
    dominance_ratio: float,
    use_dominant_axis: bool,
    invert_yaw: bool,
    invert_pitch: bool,
    merge_screen_down_focus: bool,
    calibration_screens: int,
    task_path: Path,
    mlflow_nested: bool,
) -> dict[str, Any]:
    import cv2  # type: ignore

    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]
    mlflow = configure_mlflow(pi_agent_dir=pi_agent_dir, experiment_name="ai-study-buddy-pi-agent")

    cfg = HeadPoseConfig(
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        deadzone_yaw_deg=float(deadzone_yaw_deg),
        deadzone_pitch_deg=float(deadzone_pitch_deg),
        dominance_ratio=float(dominance_ratio),
        use_dominant_axis=bool(use_dominant_axis),
        invert_yaw=bool(invert_yaw),
        invert_pitch=bool(invert_pitch),
        merge_screen_down_focus=bool(merge_screen_down_focus),
        calibration_screens=int(calibration_screens),
        model_task_path=str(task_path),
    )

    labels = _effective_labels(merge_screen_down_focus=cfg.merge_screen_down_focus)
    est = HeadPoseEstimator(task_path=task_path)
    test_items = _iter_labeled_images(splits_dir / "test", labels=list(RAW_LABELS))
    if not test_items:
        raise SystemExit(f"[headpose] No test images under {splits_dir / 'test'}")

    # Per-participant neutral calibration from screen frames (simulates a short setup phase).
    calib_candidates = [p for p, raw_lab in test_items if raw_lab == "screen"]
    calib_paths = set(calib_candidates[: max(0, int(cfg.calibration_screens))])
    neutral_yaws: list[float] = []
    neutral_pitches: list[float] = []
    n_calib_fail = 0
    for p in calib_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            n_calib_fail += 1
            continue
        yp = est.estimate_yaw_pitch(img)
        if yp is None:
            n_calib_fail += 1
            continue
        yaw0, pitch0 = yp
        if cfg.invert_yaw:
            yaw0 = -yaw0
        if cfg.invert_pitch:
            pitch0 = -pitch0
        neutral_yaws.append(float(yaw0))
        neutral_pitches.append(float(pitch0))

    neutral_yaw = float(sum(neutral_yaws) / len(neutral_yaws)) if neutral_yaws else 0.0
    neutral_pitch = float(sum(neutral_pitches) / len(neutral_pitches)) if neutral_pitches else 0.0

    y_true: list[str] = []
    y_pred: list[str] = []
    n_fail = 0

    for p, raw_lab in test_items:
        # Exclude calibration frames from evaluation to avoid leakage.
        if p in calib_paths:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            n_fail += 1
            pred = "focused" if cfg.merge_screen_down_focus else "screen"
        else:
            yp = est.estimate_yaw_pitch(img)
            if yp is None:
                n_fail += 1
                pred = "focused" if cfg.merge_screen_down_focus else "screen"
            else:
                yaw, pitch = yp
                if cfg.invert_yaw:
                    yaw = -yaw
                if cfg.invert_pitch:
                    pitch = -pitch
                dyaw = float(yaw - neutral_yaw)
                dpitch = float(pitch - neutral_pitch)
                pred = _decide_label(
                    yaw_deg=dyaw,
                    pitch_deg=dpitch,
                    yaw_th=cfg.yaw_deg,
                    pitch_th=cfg.pitch_deg,
                    deadzone_yaw_deg=cfg.deadzone_yaw_deg,
                    deadzone_pitch_deg=cfg.deadzone_pitch_deg,
                    dominance_ratio=cfg.dominance_ratio,
                    use_dominant_axis=cfg.use_dominant_axis,
                    merge_screen_down_focus=cfg.merge_screen_down_focus,
                )

        y_true.append(_map_label(raw_lab, merge_screen_down_focus=cfg.merge_screen_down_focus))
        y_pred.append(pred)

    metrics = _compute_metrics(y_true, y_pred, labels=labels)
    metrics["face_landmark_fail_rate"] = float(n_fail / max(1, len(test_items)))
    metrics["n_test"] = int(len(y_true))
    metrics["calibration_used"] = int(len(calib_paths))
    metrics["calibration_success"] = int(len(neutral_yaws))
    metrics["calibration_fail"] = int(n_calib_fail)
    metrics["neutral_yaw_deg"] = float(neutral_yaw)
    metrics["neutral_pitch_deg"] = float(neutral_pitch)

    run_name = f"headpose:{tag}"
    with mlflow.start_run(run_name=run_name, nested=bool(mlflow_nested)):
        mlflow.set_tag("kind", "headpose_eval")
        mlflow.set_tag("created_at_utc", _utc_now_iso())
        mlflow.set_tag("tag", str(tag))
        if holdout_participant:
            mlflow.set_tag("holdout_participant", str(holdout_participant))

        mlflow.log_param("method", "headpose")
        mlflow.log_param("splits_dir", str(splits_dir))
        mlflow.log_param("yaw_deg", float(cfg.yaw_deg))
        mlflow.log_param("pitch_deg", float(cfg.pitch_deg))
        mlflow.log_param("deadzone_yaw_deg", float(cfg.deadzone_yaw_deg))
        mlflow.log_param("deadzone_pitch_deg", float(cfg.deadzone_pitch_deg))
        mlflow.log_param("dominance_ratio", float(cfg.dominance_ratio))
        mlflow.log_param("use_dominant_axis", bool(cfg.use_dominant_axis))
        mlflow.log_param("invert_yaw", bool(cfg.invert_yaw))
        mlflow.log_param("invert_pitch", bool(cfg.invert_pitch))
        mlflow.log_param("merge_screen_down_focus", bool(cfg.merge_screen_down_focus))
        mlflow.log_param("calibration_screens", int(cfg.calibration_screens))
        mlflow.log_param("task_path", str(task_path))

        mlflow.log_metric("macro_f1", float(metrics["macro_f1"]))
        mlflow.log_metric("accuracy_from_cm", float(metrics["accuracy_from_cm"]))
        mlflow.log_metric("face_landmark_fail_rate", float(metrics["face_landmark_fail_rate"]))

        per = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
        for lab in labels:
            d = per.get(lab) if isinstance(per, dict) else None
            if isinstance(d, dict) and isinstance(d.get("recall"), (int, float)):
                mlflow.log_metric(f"recall.{lab}", float(d["recall"]))

        # Artifacts
        log_dict_as_json(
            mlflow,
            {
                "tag": tag,
                "holdout_participant": holdout_participant,
                "splits_dir": str(splits_dir),
                "config": cfg.__dict__,
            },
            "meta/config.json",
        )
        log_dict_as_json(mlflow, metrics, "artifacts/metrics_test.json")

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a head-pose baseline on a prepared splits dir and log to MLflow.")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--tag", default="headpose")
    parser.add_argument("--holdout-participant", default=None)
    parser.add_argument("--yaw-deg", type=float, default=15.0)
    parser.add_argument("--pitch-deg", type=float, default=12.0)
    parser.add_argument("--deadzone-yaw-deg", type=float, default=10.0)
    parser.add_argument("--deadzone-pitch-deg", type=float, default=8.0)
    parser.add_argument("--dominance-ratio", type=float, default=1.15)
    parser.add_argument("--no-dominant-axis", action="store_true")
    parser.add_argument("--invert-yaw", action="store_true")
    parser.add_argument("--invert-pitch", action="store_true")
    parser.add_argument("--merge-screen-down-focus", action="store_true", help="Map screen+away_down to focused (4-class).")
    parser.add_argument("--calibration-screens", type=int, default=40, help="Use first N screen frames from test for neutral calibration.")
    parser.add_argument("--task-path", default="models/mediapipe/face_landmarker.task")
    parser.add_argument("--download-task", action="store_true", help="Download the MediaPipe task model if missing.")
    parser.add_argument("--mlflow-nested", action="store_true")
    args = parser.parse_args(argv)

    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]
    task_path = (pi_agent_dir / args.task_path).resolve() if not Path(args.task_path).is_absolute() else Path(args.task_path).resolve()
    if args.download_task:
        _ensure_face_landmarker_task(task_path)

    evaluate_headpose(
        splits_dir=Path(args.splits_dir).expanduser().resolve(),
        tag=str(args.tag),
        holdout_participant=str(args.holdout_participant) if args.holdout_participant else None,
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
        mlflow_nested=bool(args.mlflow_nested),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

