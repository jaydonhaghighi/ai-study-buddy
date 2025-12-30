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


class FrameSource:
    """
    Minimal interface used by the agent + preview server.
    Must expose:
    - read() -> (ok: bool, frame: ndarray | None)
    - release() -> None
    """

    def read(self) -> tuple[bool, Any | None]:  # pragma: no cover
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover
        raise NotImplementedError


class Cv2FrameSource(FrameSource):
    def __init__(self, cap: Any):
        self._cap = cap

    def read(self) -> tuple[bool, Any | None]:
        ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        self._cap.release()


class Picamera2FrameSource(FrameSource):
    def __init__(self, picam2: Any, *, swap_rgb_to_bgr: bool):
        self._picam2 = picam2
        self._swap_rgb_to_bgr = swap_rgb_to_bgr

    def read(self) -> tuple[bool, Any | None]:
        # Picamera2 returns whatever we configured (prefer BGR888).
        frame = self._picam2.capture_array()
        if frame is None:
            return False, None
        if self._swap_rgb_to_bgr:
            # Avoid relying on cv2 color conversion; just swap channels.
            try:
                import numpy as np  # type: ignore
                frame = np.ascontiguousarray(frame[:, :, ::-1])
            except Exception:
                # Best-effort fallback (may still work if frame is already BGR)
                pass
        return True, frame

    def release(self) -> None:
        try:
            self._picam2.stop()
        except Exception:
            pass
        try:
            self._picam2.close()
        except Exception:
            pass


def open_frame_source(index: int = 0, device: str | None = None) -> FrameSource:
    """
    Try OpenCV VideoCapture first (fast), then fallback to Picamera2/libcamera (reliable on PiCam).
    """
    apply_opencv_videoio_env()

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is not installed. Install opencv-python on the Pi.") from e

    cam_index = index
    if device:
        idx = _device_path_to_index(device)
        if idx is not None:
            cam_index = idx

    # OpenCV path
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(cam_index)

    if cap.isOpened():
        # Try common formats/resolutions; if we can get a frame, we accept it.
        candidates = [
            (640, 480, "YUYV"),
            (640, 480, "MJPG"),
            (1280, 720, "YUYV"),
            (1280, 720, "MJPG"),
        ]
        got_frame = False
        for w, h, fourcc in candidates:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            except Exception:
                pass
            for _ in range(20):
                ok, frame = cap.read()
                if ok and frame is not None:
                    got_frame = True
                    break
            if got_frame:
                break

        if got_frame:
            return Cv2FrameSource(cap)

    # If OpenCV couldn't produce frames, fallback to Picamera2
    try:
        from picamera2 import Picamera2  # type: ignore
    except Exception as e:
        # Provide a precise error so you know what to install
        try:
            cap.release()
        except Exception:
            pass
        raise RuntimeError(
            "OpenCV could not read frames from the camera. Install Picamera2 for a reliable fallback "
            "(e.g., `sudo apt install -y python3-picamera2`)."
        ) from e

    picam2 = Picamera2()
    # Keep it simple: preview-sized frames for calibration and stub inference
    swap_rgb_to_bgr = False
    try:
        # Prefer BGR888 so OpenCV/JPEG encoding has correct channel order.
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
    except Exception:
        # If BGR888 isn't available, fall back to RGB888 and swap manually.
        try:
            config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
            picam2.configure(config)
            swap_rgb_to_bgr = True
        except Exception:
            # last-resort fallback to defaults
            swap_rgb_to_bgr = False
    picam2.start()
    # Warm-up
    timeouts = 0
    for _ in range(10):
        try:
            _ = picam2.capture_array()
            break
        except Exception:
            timeouts += 1
            if timeouts >= 3:
                break
    return Picamera2FrameSource(picam2, swap_rgb_to_bgr=swap_rgb_to_bgr)


@contextmanager
def open_camera(index: int = 0, device: str | None = None) -> Iterator[FrameSource]:
    """
    Opens an OpenCV camera device if available.
    In simulation mode you can skip calling this entirely.
    """
    src = open_frame_source(index=index, device=device)
    try:
        yield src
    finally:
        src.release()


def read_frame(cap: FrameSource) -> Any | None:
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


