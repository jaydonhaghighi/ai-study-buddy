from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _beep():
    # Cross-platform "best effort" beep.
    try:
        import winsound  # type: ignore

        winsound.Beep(880, 140)
        return
    except Exception:
        pass
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass


def _has_display() -> bool:
    # Linux/Unix GUI presence check
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def _slug(s: str) -> str:
    return "".join([c.lower() if c.isalnum() else "_" for c in s]).strip("_")


def _load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenCV is required for laptop data collection. "
            "Run: pip install -r pi-agent/requirements-collect.txt"
        ) from e
    return cv2


def _load_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("NumPy is required. Run: pip install -r pi-agent/requirements-collect.txt") from e
    return np


def _haarcascade_path(cv2, filename: str) -> str:
    # Prefer cv2.data if available (opencv-python), fallback to common system locations.
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

    # Common linux paths
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


@dataclass(frozen=True)
class Condition:
    name: str
    instructions: str


DEFAULT_CONDITIONS: list[Condition] = [
    Condition(
        name="normal",
        instructions="Normal posture, normal lighting. Sit as you would during study.",
    ),
    Condition(
        name="lean_back",
        instructions="Lean back slightly (like relaxing in your chair), keep laptop in same place.",
    ),
    Condition(
        name="lean_forward",
        instructions="Lean forward slightly (like concentrating), keep laptop in same place.",
    ),
    Condition(
        name="glasses",
        instructions="If you have glasses: put them on now. If not, press 'S' to skip this condition.",
    ),
    Condition(
        name="dim_light",
        instructions="Dim the room a bit (if possible). If not possible, press 'S' to skip.",
    ),
]


def _draw_overlay(
    frame,
    cv2,
    *,
    title: str,
    subtitle: str,
    face_box: tuple[int, int, int, int] | None,
    face_ok: bool,
    progress: str,
):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # top bar
    cv2.rectangle(overlay, (0, 0), (w, 110), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, title, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    cv2.putText(frame, subtitle, (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2)
    cv2.putText(frame, progress, (18, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)

    if face_box:
        x, y, fw, fh = face_box
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 200, 0) if face_ok else (0, 0, 255), 2)


def _wait_for_key(cv2, win: str, allowed: set[str]) -> str:
    # returns lowercase key char
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == 255:
            continue
        try:
            ch = chr(k).lower()
        except Exception:
            continue
        if ch in allowed:
            return ch


def _probe_webcams(cv2, max_index: int = 6) -> list[int]:
    ok: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if cap is not None and cap.isOpened():
                ok.append(i)
        finally:
            try:
                cap.release()
            except Exception:
                pass
    return ok


def run_guided_collection(
    *,
    out_dir: Path,
    participant: str,
    session: str,
    placement: str,
    webcam_index: int,
    width: int,
    height: int,
    fps: float,
    cycles: int,
    look_seconds: float,
    away_seconds: float,
    require_face: bool,
    save_full: bool,
    save_face: bool,
    preview: bool,
):
    cv2 = _load_cv2()
    np = _load_numpy()
    _ = np  # silence unused for type-checkers

    run_dir = out_dir / f"run_{int(time.time())}"
    meta_file = run_dir / "meta.jsonl"

    face_path = _haarcascade_path(cv2, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(face_path)
    if face_cascade is None or face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {face_path}")

    # Auto-disable preview if there's no display (common on headless Linux).
    if preview and not _has_display() and os.name != "nt":
        print("[collect-data] No DISPLAY found; running without preview window.")
        preview = False

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        # Provide actionable hints.
        try:
            avail = _probe_webcams(cv2, max_index=8)
        except Exception:
            avail = []
        hint = ""
        if avail:
            hint = f" Available indices: {avail}. Try: --webcam {avail[0]}"
        raise RuntimeError(
            f"Could not open webcam index {webcam_index}.{hint}\n"
            "If you're on macOS, also ensure your Terminal/Python has Camera permission "
            "(System Settings → Privacy & Security → Camera)."
        )

    # Best-effort set size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    win = "AI Study Buddy - Data Collection"
    if preview:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, width, height)
        except Exception as e:
            print(f"[collect-data] Preview window failed to initialize ({e}); running without preview.")
            preview = False

    def read_frame():
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame

    def show_screen(title: str, subtitle: str, progress: str, face_box=None, face_ok=False):
        if not preview:
            return
        frame = read_frame()
        if frame is None:
            return
        _draw_overlay(frame, cv2, title=title, subtitle=subtitle, face_box=face_box, face_ok=face_ok, progress=progress)
        cv2.imshow(win, frame)

    def countdown(seconds: int, title: str, subtitle: str, progress: str):
        for i in range(seconds, 0, -1):
            _beep()
            t0 = time.time()
            while time.time() - t0 < 1.0:
                show_screen(title, f"{subtitle}  Starting in {i}…", progress)
                if preview:
                    k = cv2.waitKey(20) & 0xFF
                    if k != 255:
                        ch = chr(k).lower()
                        if ch == "q":
                            raise KeyboardInterrupt()
                        if ch == "s":
                            return "skip"
            # next second
        _beep()
        return "ok"

    def capture_segment(label: str, duration: float, condition_tag: str, cycle_idx: int):
        start = time.time()
        saved = 0
        skipped = 0
        frame_idx = 0
        while time.time() - start < duration:
            frame = read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            ts = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_box = _detect_face(gray, face_cascade)
            face_ok = bool(face_box)

            if require_face and not face_ok:
                skipped += 1
                if preview:
                    _draw_overlay(
                        frame,
                        cv2,
                        title=("LOOK AT SCREEN" if label == "looking" else "LOOK AWAY"),
                        subtitle="No face detected — adjust position/lighting",
                        face_box=None,
                        face_ok=False,
                        progress=f"{condition_tag}  cycle {cycle_idx+1}/{cycles}  saved={saved}  skipped={skipped}  (Q quit / S skip)",
                    )
                    cv2.imshow(win, frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k != 255:
                        ch = chr(k).lower()
                        if ch == "q":
                            raise KeyboardInterrupt()
                        if ch == "s":
                            return "skip"
                time.sleep(max(0.0, (1.0 / max(fps, 1.0)) - 0.001))
                continue

            sample: dict[str, Any] = {
                "label": label,
                "timestamp": ts,
                "face_box": face_box,
                "participant": participant,
                "session": session,
                "placement": condition_tag,  # keeps compatibility with prepare_dataset.py
                "basePlacement": placement,
                "condition": condition_tag,
            }

            file_stem = f"{int(ts * 1000)}_{frame_idx:05d}"
            if save_full:
                full_path = run_dir / "raw" / participant / session / condition_tag / label / f"{file_stem}.jpg"
                _save_image(full_path, frame, cv2)
                sample["full_path"] = str(full_path)

            if save_face and face_box:
                face = _crop_face(frame, face_box)
                if face is not None:
                    face_path = run_dir / "face" / participant / session / condition_tag / label / f"{file_stem}.jpg"
                    _save_image(face_path, face, cv2)
                    sample["face_path"] = str(face_path)
                    saved += 1

            meta_file.parent.mkdir(parents=True, exist_ok=True)
            with meta_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            if preview:
                _draw_overlay(
                    frame,
                    cv2,
                    title=("LOOK AT SCREEN" if label == "looking" else "LOOK AWAY"),
                    subtitle=f"Keep still. ({label})",
                    face_box=face_box,
                    face_ok=face_ok,
                    progress=f"{condition_tag}  cycle {cycle_idx+1}/{cycles}  saved={saved}  skipped={skipped}  (Q quit / S skip)",
                )
                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF
                if k != 255:
                    ch = chr(k).lower()
                    if ch == "q":
                        raise KeyboardInterrupt()
                    if ch == "s":
                        return "skip"

            frame_idx += 1
            time.sleep(max(0.0, (1.0 / max(fps, 1.0)) - 0.001))

        _beep()
        return "ok"

    # Intro screen
    print("\n=== Guided Data Collection (Laptop Webcam) ===")
    print(f"participant={participant} session={session} placement={placement}")
    print("Keys: [Enter] start, Q quit, S skip condition/segment (preview mode)")
    print(f"Saving to: {run_dir}\n")

    # Wait for Enter in window
    if preview:
        while True:
            show_screen(
                "Ready to collect data",
                "Position laptop so your face is clearly visible. Press ENTER to begin.",
                "Q quit",
            )
            k = cv2.waitKey(50) & 0xFF
            if k == 13 or k == 10:  # Enter
                break
            if k != 255 and chr(k).lower() == "q":
                cap.release()
                cv2.destroyAllWindows()
                return
    else:
        print("Preview disabled. Starting in 3 seconds...")
        time.sleep(3.0)

    try:
        for cond in DEFAULT_CONDITIONS:
            condition_tag = f"{placement}_{_slug(cond.name)}"

            # Condition instruction screen
            while True:
                show_screen(
                    f"Condition: {cond.name}",
                    f"{cond.instructions}  (ENTER start / S skip / Q quit)",
                    condition_tag,
                )
                k = cv2.waitKey(50) & 0xFF
                if k == 13 or k == 10:
                    break
                if k != 255:
                    ch = chr(k).lower()
                    if ch == "q":
                        raise KeyboardInterrupt()
                    if ch == "s":
                        condition_tag = None
                        break

            if condition_tag is None:
                continue

            for cycle_idx in range(cycles):
                r = countdown(3, "LOOK AT SCREEN", "Get ready to look at the screen.", f"{condition_tag}  cycle {cycle_idx+1}/{cycles}")
                if r == "skip":
                    break
                r = capture_segment("looking", look_seconds, condition_tag, cycle_idx)
                if r == "skip":
                    break

                r = countdown(3, "LOOK AWAY", "Get ready to look away (off-screen).", f"{condition_tag}  cycle {cycle_idx+1}/{cycles}")
                if r == "skip":
                    break
                r = capture_segment("not_looking", away_seconds, condition_tag, cycle_idx)
                if r == "skip":
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()

    print("\nDone.")
    print("Send this folder back to the project owner:")
    print(f"  {run_dir}")
    print("It contains face crops under:")
    print("  face/<participant>/<session>/<placement_condition>/{looking,not_looking}/*.jpg")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="studybuddy collect-data", description="Guided labeled webcam capture (looking vs not looking).")
    parser.add_argument("--out-dir", default="data", help="Output directory (will create run_<timestamp>/)")
    parser.add_argument("--participant", default=os.getenv("STUDYBUDDY_PARTICIPANT", "p01"))
    parser.add_argument("--session", default=os.getenv("STUDYBUDDY_SESSION", f"s{int(time.time())}"))
    parser.add_argument("--placement", default=os.getenv("STUDYBUDDY_PLACEMENT", "laptop_webcam"))
    parser.add_argument("--webcam", type=int, default=int(os.getenv("STUDYBUDDY_WEBCAM", "0")))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--look-seconds", type=float, default=6.0)
    parser.add_argument("--away-seconds", type=float, default=6.0)
    parser.add_argument("--require-face", action="store_true", default=True)
    parser.add_argument("--save-full", action="store_true", help="Also save full frames (bigger dataset)")
    parser.add_argument("--no-save-face", action="store_true", help="Do not save face crops")
    parser.add_argument("--no-preview", action="store_true", help="Run without a GUI preview window (headless)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    run_guided_collection(
        out_dir=out_dir,
        participant=str(args.participant),
        session=str(args.session),
        placement=str(args.placement),
        webcam_index=int(args.webcam),
        width=int(args.width),
        height=int(args.height),
        fps=float(args.fps),
        cycles=int(args.cycles),
        look_seconds=float(args.look_seconds),
        away_seconds=float(args.away_seconds),
        require_face=bool(args.require_face),
        save_full=bool(args.save_full),
        save_face=not bool(args.no_save_face),
        preview=not bool(args.no_preview),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

