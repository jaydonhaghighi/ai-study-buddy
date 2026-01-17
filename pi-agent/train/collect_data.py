from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV is required for data collection. Install opencv-python.") from e
    return cv2


def _haarcascade_path(cv2, filename: str) -> str:
    try:
        base = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if base:
            return str(base) + filename
    except Exception:
        pass

    env = os.getenv("OPENCV_HAAR_PATH")
    if env:
        p = Path(env) / filename
        if p.exists():
            return str(p)

    candidates = [
        Path("/usr/share/opencv4/haarcascades") / filename,
        Path("/usr/share/opencv/haarcascades") / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise RuntimeError(
        f"Could not locate OpenCV haarcascade file: {filename}. "
        "If you're using system OpenCV, install haarcascades or set OPENCV_HAAR_PATH."
    )


def _load_camera(width: int, height: int, fmt: str):
    from studybuddy_pi.camera_manager import CameraManager

    cam = CameraManager(width=width, height=height, format=fmt, target_fps=15.0)
    cam.acquire("collect")
    return cam


def _detect_face(gray, face_cascade) -> tuple[int, int, int, int] | None:
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def _crop_face(frame, box, pad_ratio: float = 0.2):
    x, y, w, h = box
    fh, fw = frame.shape[:2]
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(fw, x + w + px)
    y2 = min(fh, y + h + py)
    face = frame[y1:y2, x1:x2]
    return face if face.size else None


def _save_image(path: Path, image, cv2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def collect_label(
    *,
    participant: str,
    session: str,
    placement: str,
    label: str,
    duration_seconds: float,
    cam: Any,
    face_cascade,
    out_dir: Path,
    fps: float,
    save_full: bool,
    save_face: bool,
    require_face: bool,
    meta_file: Path,
    show_preview: bool,
):
    cv2 = _load_cv2()
    start = time.time()
    frame_idx = 0
    saved = 0
    skipped_no_face = 0
    while time.time() - start < duration_seconds:
        latest = cam.get_latest()
        if latest is None or latest.frame is None:
            time.sleep(0.01)
            continue

        frame = latest.frame
        ts = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_box = _detect_face(gray, face_cascade)

        if require_face and not face_box:
            skipped_no_face += 1
            if show_preview:
                try:
                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        f"{label} (no face)  saved={saved}  skipped={skipped_no_face}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("studybuddy-collect", preview)
                    cv2.waitKey(1)
                except Exception:
                    pass
            time.sleep(max(0.0, (1.0 / max(fps, 1.0)) - 0.001))
            continue

        sample = {
            "label": label,
            "timestamp": ts,
            "face_box": face_box,
            "participant": participant,
            "session": session,
            "placement": placement,
        }

        file_stem = f"{int(ts * 1000)}_{frame_idx:05d}"
        if save_full:
            full_path = out_dir / "raw" / participant / session / placement / label / f"{file_stem}.jpg"
            _save_image(full_path, frame, cv2)
            sample["full_path"] = str(full_path)

        if save_face and face_box:
            face = _crop_face(frame, face_box)
            if face is not None:
                face_path = out_dir / "face" / participant / session / placement / label / f"{file_stem}.jpg"
                _save_image(face_path, face, cv2)
                sample["face_path"] = str(face_path)
                saved += 1

        meta_file.parent.mkdir(parents=True, exist_ok=True)
        with meta_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")

        frame_idx += 1
        if show_preview:
            try:
                preview = frame.copy()
                if face_box:
                    x, y, w, h = face_box
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"{label}  saved={saved}  skipped={skipped_no_face}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    preview,
                    f"{participant}/{session}/{placement}",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )
                cv2.imshow("studybuddy-collect", preview)
                cv2.waitKey(1)
            except Exception:
                pass
        time.sleep(max(0.0, (1.0 / max(fps, 1.0)) - 0.001))


def main():
    parser = argparse.ArgumentParser(description="Collect labeled eye-contact data for fine-tuning.")
    parser.add_argument("--out-dir", default="data", help="Output directory for the dataset")
    parser.add_argument("--participant", default=os.getenv("STUDYBUDDY_PARTICIPANT", "p01"))
    parser.add_argument("--session", default=os.getenv("STUDYBUDDY_SESSION", f"s{int(time.time())}"))
    parser.add_argument("--placement", default=os.getenv("STUDYBUDDY_PLACEMENT", "monitor_top"))
    parser.add_argument("--looking-seconds", type=float, default=10.0)
    parser.add_argument("--away-seconds", type=float, default=10.0)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--format", default="RGB888")
    parser.add_argument("--save-full", action="store_true", help="Save full frames in raw/")
    parser.add_argument("--save-face", action="store_true", help="Save face crops in face/ (recommended)")
    parser.add_argument("--require-face", action="store_true", help="Only save samples when a face is detected")
    parser.add_argument("--preview", action="store_true", help="Show a live preview window (needs GUI)")
    args = parser.parse_args()

    if not args.save_full and not args.save_face:
        args.save_face = True

    out_dir = Path(args.out_dir).expanduser().resolve()
    run_dir = out_dir / f"run_{int(time.time())}"
    meta_file = run_dir / "meta.jsonl"

    cv2 = _load_cv2()
    face_path = _haarcascade_path(cv2, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(face_path)
    if face_cascade is None or face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {face_path}")

    cam = _load_camera(args.width, args.height, args.format)

    print("\n=== Data Collection ===")
    print("Follow the prompts. Keep your face visible.")
    print(f"Saving to: {run_dir}")
    print(f"participant={args.participant} session={args.session} placement={args.placement}")
    print()
    time.sleep(1.0)

    try:
        for i in range(args.cycles):
            print(f"[{i+1}/{args.cycles}] LOOK at the screen for {args.looking_seconds}s...")
            collect_label(
                participant=args.participant,
                session=args.session,
                placement=args.placement,
                label="looking",
                duration_seconds=args.looking_seconds,
                cam=cam,
                face_cascade=face_cascade,
                out_dir=run_dir,
                fps=args.fps,
                save_full=args.save_full,
                save_face=args.save_face,
                require_face=args.require_face,
                meta_file=meta_file,
                show_preview=args.preview,
            )

            print(f"[{i+1}/{args.cycles}] LOOK AWAY for {args.away_seconds}s...")
            collect_label(
                participant=args.participant,
                session=args.session,
                placement=args.placement,
                label="not_looking",
                duration_seconds=args.away_seconds,
                cam=cam,
                face_cascade=face_cascade,
                out_dir=run_dir,
                fps=args.fps,
                save_full=args.save_full,
                save_face=args.save_face,
                require_face=args.require_face,
                meta_file=meta_file,
                show_preview=args.preview,
            )

        print("\nDone. You can combine multiple runs under a single data folder.")
        print("Saved face crops under:")
        print("  face/<participant>/<session>/<placement>/{looking,not_looking}/*.jpg")
    finally:
        try:
            if args.preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            cam.release("collect")
        except Exception:
            pass


if __name__ == "__main__":
    main()
