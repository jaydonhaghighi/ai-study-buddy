from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .opencv_tuning import apply_opencv_videoio_env


def _device_path_to_index(device: str) -> int | None:
    # "/dev/video0" -> 0
    try:
        if device.startswith("/dev/video"):
            return int(device.replace("/dev/video", ""))
    except Exception:
        return None
    return None


@contextmanager
def open_camera(index: int = 0, device: str | None = None) -> Iterator[Any]:
    """
    Opens an OpenCV camera device if available.
    In simulation mode you can skip calling this entirely.
    """
    try:
        apply_opencv_videoio_env()
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is not installed. Install opencv-python on the Pi.") from e

    # Prefer V4L2 backend on Raspberry Pi. If a device path is provided, convert to an index.
    cam_index = index
    if device:
        idx = _device_path_to_index(device)
        if idx is not None:
            cam_index = idx
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback to default backend
        cap.release()
        cap = cv2.VideoCapture(cam_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera ({device or f'index={cam_index}'})")

        # Try common formats/resolutions that tend to work on Pi
        candidates = [
            (640, 480, "MJPG"),
            (640, 480, "YUYV"),
            (1280, 720, "MJPG"),
            (1280, 720, "YUYV"),
        ]
        for w, h, fourcc in candidates:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            except Exception:
                continue
            # Warm-up reads; if frames start coming in, keep this config
            ok_any = False
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    ok_any = True
                    break
            if ok_any:
                break

        yield cap
    finally:
        cap.release()


def read_frame(cap: Any) -> Any | None:
    ok, frame = cap.read()
    return frame if ok else None


