from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def open_camera(index: int = 0) -> Iterator[Any]:
    """
    Opens an OpenCV camera device if available.
    In simulation mode you can skip calling this entirely.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is not installed. Install opencv-python on the Pi.") from e

    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera index={index}")
        yield cap
    finally:
        cap.release()


def read_frame(cap: Any) -> Any | None:
    ok, frame = cap.read()
    return frame if ok else None


