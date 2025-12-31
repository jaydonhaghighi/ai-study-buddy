from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CameraFrame:
    ts: float
    frame: Any  # numpy ndarray (RGB/BGR depending on configuration)


class CameraManager:
    """
    Single camera owner using Picamera2/libcamera.

    - Supports reference-counted "acquire/release" by feature (preview/tracking)
    - Runs a background capture loop and stores the latest frame
    - Never uses cv2.VideoCapture (unreliable on PiCam for many images)
    """

    def __init__(self, *, width: int = 640, height: int = 480, format: str = "RGB888", target_fps: float = 15.0):
        self.width = width
        self.height = height
        self.format = format  # Picamera2 format string (e.g. "RGB888")
        self.target_fps = target_fps

        self._lock = threading.Lock()
        self._refcount = 0
        self._owners: set[str] = set()

        self._picam2: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._latest: CameraFrame | None = None
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def acquire(self, owner: str) -> None:
        with self._lock:
            if owner in self._owners:
                return
            self._owners.add(owner)
            self._refcount += 1
            if self._refcount == 1:
                self._start_locked()

    def release(self, owner: str) -> None:
        with self._lock:
            if owner not in self._owners:
                return
            self._owners.remove(owner)
            self._refcount = max(0, self._refcount - 1)
            if self._refcount == 0:
                self._stop_locked()

    def get_latest(self) -> CameraFrame | None:
        with self._lock:
            return self._latest

    def _start_locked(self) -> None:
        # Start Picamera2 + capture thread
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as e:
            raise RuntimeError("Picamera2 is required for camera capture on Raspberry Pi. Install python3-picamera2.") from e

        self._stop_event.clear()
        self._picam2 = Picamera2()
        try:
            config = self._picam2.create_preview_configuration(main={"size": (self.width, self.height), "format": self.format})
            self._picam2.configure(config)
        except Exception:
            # fallback to defaults if configuration fails
            pass
        self._picam2.start()

        # Warm up a few frames
        for _ in range(5):
            try:
                _ = self._picam2.capture_array()
                break
            except Exception:
                time.sleep(0.05)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _stop_locked(self) -> None:
        self._stop_event.set()
        picam2 = self._picam2
        self._picam2 = None

        try:
            if picam2 is not None:
                picam2.stop()
        except Exception:
            pass
        try:
            if picam2 is not None:
                picam2.close()
        except Exception:
            pass

        self._thread = None
        self._latest = None
        self._last_error = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            picam2 = None
            with self._lock:
                picam2 = self._picam2
            if picam2 is None:
                time.sleep(0.05)
                continue

            try:
                frame = picam2.capture_array()
                with self._lock:
                    self._latest = CameraFrame(ts=time.time(), frame=frame)
                    self._last_error = None
            except Exception as e:
                with self._lock:
                    self._last_error = str(e)

            # Target FPS pacing
            elapsed = time.time() - start
            sleep_for = max(0.0, (1.0 / max(self.target_fps, 0.1)) - elapsed)
            time.sleep(sleep_for)


