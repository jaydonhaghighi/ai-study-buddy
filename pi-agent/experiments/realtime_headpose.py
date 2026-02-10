from __future__ import annotations

import argparse
import statistics
import time
from collections import deque
from pathlib import Path

try:
    # Module mode: python -m experiments.realtime_headpose
    from .headpose_eval import HeadPoseEstimator, _decide_label, _ensure_face_landmarker_task  # type: ignore
except Exception:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from experiments.headpose_eval import HeadPoseEstimator, _decide_label, _ensure_face_landmarker_task  # type: ignore


def _put_line(cv2, frame, text: str, line_idx: int, *, color=(0, 255, 0)) -> None:
    cv2.putText(
        frame,
        text,
        (12, 28 + line_idx * 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Realtime webcam head-pose viewer (yaw/pitch/roll + direction label)."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--yaw-deg", type=float, default=15.0)
    parser.add_argument("--pitch-deg", type=float, default=12.0)
    parser.add_argument("--deadzone-yaw-deg", type=float, default=10.0)
    parser.add_argument("--deadzone-pitch-deg", type=float, default=8.0)
    parser.add_argument("--dominance-ratio", type=float, default=1.15)
    parser.add_argument("--no-dominant-axis", action="store_true")
    parser.add_argument("--invert-yaw", action="store_true")
    parser.add_argument("--invert-pitch", action="store_true")
    parser.add_argument(
        "--task-path",
        default="models/mediapipe/face_landmarker.task",
        help="Path to MediaPipe face_landmarker.task",
    )
    parser.add_argument("--download-task", action="store_true")
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=40,
        help="Number of frames used when you press 'c' to calibrate neutral screen pose.",
    )
    parser.add_argument("--mirror", action="store_true", help="Mirror preview for selfie-style interaction.")
    parser.add_argument(
        "--median-window",
        type=int,
        default=5,
        help="Median smoothing window size for angles (odd numbers work best).",
    )
    parser.add_argument(
        "--angle-ema-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor for angles in range (0,1]. Lower = smoother.",
    )
    parser.add_argument(
        "--label-stable-frames",
        type=int,
        default=4,
        help="Require N consecutive frames before changing displayed label.",
    )
    args = parser.parse_args(argv)

    import cv2  # type: ignore

    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]
    task_path = (pi_agent_dir / args.task_path).resolve() if not Path(args.task_path).is_absolute() else Path(args.task_path).resolve()
    if args.download_task:
        _ensure_face_landmarker_task(task_path)

    est = HeadPoseEstimator(task_path=task_path)

    cap = cv2.VideoCapture(int(args.camera_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    cap.set(cv2.CAP_PROP_FPS, int(args.fps))
    if not cap.isOpened():
        raise SystemExit(f"[realtime_headpose] Could not open camera index {args.camera_index}")

    neutral_yaw = 0.0
    neutral_pitch = 0.0
    calib_left = 0
    calib_sum_yaw = 0.0
    calib_sum_pitch = 0.0
    median_window = max(1, int(args.median_window))
    alpha = float(args.angle_ema_alpha)
    if alpha <= 0.0 or alpha > 1.0:
        raise SystemExit("[realtime_headpose] --angle-ema-alpha must be in (0,1].")

    yaw_hist: deque[float] = deque(maxlen=median_window)
    pitch_hist: deque[float] = deque(maxlen=median_window)
    roll_hist: deque[float] = deque(maxlen=median_window)
    ema_yaw: float | None = None
    ema_pitch: float | None = None
    ema_roll: float | None = None

    stable_frames = max(1, int(args.label_stable_frames))
    current_label = "no_face"
    candidate_label = "no_face"
    candidate_count = 0

    print("[realtime_headpose] Running. Keys: q=quit, c=calibrate-neutral, r=reset-neutral")

    prev_t = time.time()
    ema_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            ypr = est.estimate_yaw_pitch_roll(frame)
            direction = "no_face"
            yaw_text = pitch_text = roll_text = "--"
            color = (0, 200, 255)

            if ypr is not None:
                yaw, pitch, roll = ypr
                if args.invert_yaw:
                    yaw = -yaw
                if args.invert_pitch:
                    pitch = -pitch

                # First pass: median smoothing to remove single-frame spikes.
                yaw_hist.append(float(yaw))
                pitch_hist.append(float(pitch))
                roll_hist.append(float(roll))
                yaw_med = float(statistics.median(yaw_hist))
                pitch_med = float(statistics.median(pitch_hist))
                roll_med = float(statistics.median(roll_hist))

                # Second pass: EMA smoothing for continuous stability.
                if ema_yaw is None:
                    ema_yaw = yaw_med
                    ema_pitch = pitch_med
                    ema_roll = roll_med
                else:
                    ema_yaw = (1.0 - alpha) * ema_yaw + alpha * yaw_med
                    ema_pitch = (1.0 - alpha) * ema_pitch + alpha * pitch_med
                    ema_roll = (1.0 - alpha) * ema_roll + alpha * roll_med

                if calib_left > 0:
                    calib_sum_yaw += float(ema_yaw)
                    calib_sum_pitch += float(ema_pitch)
                    calib_left -= 1
                    if calib_left == 0:
                        n = max(1, int(args.calibration_frames))
                        neutral_yaw = float(calib_sum_yaw / n)
                        neutral_pitch = float(calib_sum_pitch / n)

                dyaw = float(ema_yaw - neutral_yaw)
                dpitch = float(ema_pitch - neutral_pitch)
                raw_direction = _decide_label(
                    yaw_deg=dyaw,
                    pitch_deg=dpitch,
                    yaw_th=float(args.yaw_deg),
                    pitch_th=float(args.pitch_deg),
                    deadzone_yaw_deg=float(args.deadzone_yaw_deg),
                    deadzone_pitch_deg=float(args.deadzone_pitch_deg),
                    dominance_ratio=float(args.dominance_ratio),
                    use_dominant_axis=not bool(args.no_dominant_axis),
                    merge_screen_down_focus=False,  # user asked for up/down/left/right/screen.
                )
                # Hysteresis: only commit label changes after stable_frames.
                if raw_direction == current_label:
                    candidate_label = raw_direction
                    candidate_count = 0
                else:
                    if raw_direction == candidate_label:
                        candidate_count += 1
                    else:
                        candidate_label = raw_direction
                        candidate_count = 1
                    if candidate_count >= stable_frames:
                        current_label = candidate_label
                        candidate_count = 0
                direction = current_label

                yaw_text = f"{dyaw:+.1f}"
                pitch_text = f"{dpitch:+.1f}"
                roll_text = f"{float(ema_roll):+.1f}"
                color = (0, 255, 0) if direction == "screen" else (0, 180, 255)
            else:
                # Lose face => reset temporal filters to avoid stale values.
                yaw_hist.clear()
                pitch_hist.clear()
                roll_hist.clear()
                ema_yaw = None
                ema_pitch = None
                ema_roll = None
                current_label = "no_face"
                candidate_label = "no_face"
                candidate_count = 0

            now = time.time()
            dt = max(1e-6, now - prev_t)
            prev_t = now
            fps = 1.0 / dt
            ema_fps = fps if ema_fps == 0.0 else (0.9 * ema_fps + 0.1 * fps)

            cv2.rectangle(frame, (6, 6), (610, 210), (0, 0, 0), thickness=-1)
            cv2.addWeighted(frame, 0.75, frame, 0.25, 0, frame)
            _put_line(cv2, frame, f"Direction: {direction}", 0, color=color)
            _put_line(cv2, frame, f"Yaw: {yaw_text} deg", 1)
            _put_line(cv2, frame, f"Pitch: {pitch_text} deg", 2)
            _put_line(cv2, frame, f"Roll: {roll_text} deg", 3)
            _put_line(cv2, frame, f"Neutral(y,p): ({neutral_yaw:+.1f}, {neutral_pitch:+.1f})", 4, color=(255, 220, 0))
            if calib_left > 0:
                _put_line(cv2, frame, f"Calibrating... {calib_left} frames left (look at screen)", 5, color=(255, 220, 0))
            else:
                _put_line(cv2, frame, "Keys: c=calibrate  r=reset  q=quit", 5, color=(220, 220, 220))
            _put_line(cv2, frame, f"FPS: {ema_fps:.1f} | smooth(win={median_window}, alpha={alpha:.2f}, hold={stable_frames})", 6, color=(200, 200, 200))

            cv2.imshow("AI Study Buddy - Realtime Head Pose", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calib_left = max(1, int(args.calibration_frames))
                calib_sum_yaw = 0.0
                calib_sum_pitch = 0.0
            if key == ord("r"):
                neutral_yaw = 0.0
                neutral_pitch = 0.0
                calib_left = 0
                calib_sum_yaw = 0.0
                calib_sum_pitch = 0.0
                yaw_hist.clear()
                pitch_hist.clear()
                roll_hist.clear()
                ema_yaw = None
                ema_pitch = None
                ema_roll = None
                current_label = "no_face"
                candidate_label = "no_face"
                candidate_count = 0
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

