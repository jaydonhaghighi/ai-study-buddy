from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceResult:
    is_focused: bool
    confidence: float | None = None


class FocusInference:
    """
    Abstraction over the focus model.

    - In simulation mode: generates plausible focus/distract patterns without camera/model deps.
    - In real mode (current): uses a simple OpenCV heuristic to estimate if the student is
      "looking at the screen" (face present + roughly centered + eyes detected).
      This is intentionally lightweight and is meant as a practical baseline, not a robust model.
    """

    def __init__(
        self,
        simulate: bool = False,
        *,
        frame_format: str = "RGB888",
        inference_interval_seconds: float = 0.25,
        center_tolerance_ratio: float = 0.22,
        require_eyes: bool = False,
    ):
        self.simulate = simulate
        self._sim_counter = 0
        self.frame_format = frame_format
        self.inference_interval_seconds = max(0.05, float(inference_interval_seconds))
        self.center_tolerance_ratio = max(0.05, float(center_tolerance_ratio))
        self.require_eyes = require_eyes

        # Cached outputs (to avoid running detection every frame).
        self._last_ts: float = 0.0
        self._last_result: InferenceResult = InferenceResult(is_focused=False, confidence=None)

        # Lazy-loaded OpenCV detectors
        self._cv2 = None
        self._face_cascade = None
        self._eye_cascade = None

    def _ensure_detectors(self) -> None:
        if self._cv2 is not None:
            return
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenCV is required for real focus inference. Install `opencv-python` (or system OpenCV) "
                "or set STUDYBUDDY_SIMULATE=1."
            ) from e

        self._cv2 = cv2
        # Built-in Haar cascades shipped with OpenCV-python
        face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self._face_cascade = cv2.CascadeClassifier(face_path)
        self._eye_cascade = cv2.CascadeClassifier(eye_path)
        if self._face_cascade is None or self._face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar face cascade at {face_path}")
        if self._eye_cascade is None or self._eye_cascade.empty():
            # Eye cascade isn't strictly required; keep running without it.
            self._eye_cascade = None

    def _to_gray(self, frame: Any):
        cv2 = self._cv2
        # Picamera2 usually provides RGB when configured with "RGB888".
        fmt = (self.frame_format or "").upper()
        if "RGB" in fmt and "BGR" not in fmt:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _heuristic_predict(self, frame: Any | None) -> InferenceResult:
        self._ensure_detectors()
        cv2 = self._cv2
        assert cv2 is not None
        assert self._face_cascade is not None

        if frame is None:
            return InferenceResult(is_focused=False, confidence=0.0)

        try:
            h, w = frame.shape[:2]
        except Exception:
            return InferenceResult(is_focused=False, confidence=0.0)

        gray = self._to_gray(frame)
        # Downsample for speed; detection doesn't need full-res
        scale = 0.5
        try:
            small = cv2.resize(gray, (int(w * scale), int(h * scale)))
        except Exception:
            small = gray
            scale = 1.0

        faces = self._face_cascade.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(60 * scale), int(60 * scale)),
        )
        if len(faces) == 0:
            return InferenceResult(is_focused=False, confidence=0.1)

        # Choose largest face (by area)
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        # Rescale back to full-res coordinates
        x = int(x / scale)
        y = int(y / scale)
        fw = int(fw / scale)
        fh = int(fh / scale)

        # Face centering: assume "looking at screen" when face center is near frame center.
        cx = x + fw / 2.0
        cy = y + fh / 2.0
        dx = abs(cx - w / 2.0) / max(w, 1)
        dy = abs(cy - h / 2.0) / max(h, 1)
        centered = (dx <= self.center_tolerance_ratio) and (dy <= self.center_tolerance_ratio)

        # Eye detection within the upper half of the face box (best-effort).
        eyes_found = False
        if self._eye_cascade is not None:
            try:
                x2 = max(0, x)
                y2 = max(0, y)
                x3 = min(w, x + fw)
                y3 = min(h, y + int(0.6 * fh))
                roi = gray[y2:y3, x2:x3]
                eyes = self._eye_cascade.detectMultiScale(
                    roi,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(24, 24),
                )
                eyes_found = len(eyes) >= 1
            except Exception:
                eyes_found = False

        # Confidence scoring (0..1). Keep it simple and explainable.
        score = 0.0
        score += 0.65 if centered else 0.35
        score += 0.35 if eyes_found else 0.0
        score = max(0.0, min(1.0, score))

        if self.require_eyes:
            is_focused = centered and eyes_found
        else:
            is_focused = score >= 0.6
        return InferenceResult(is_focused=bool(is_focused), confidence=score)

    def predict(self, frame: Any | None = None) -> InferenceResult:
        if self.simulate:
            # Simple, deterministic-ish pattern:
            # focused for ~45 ticks, distracted for ~10 ticks, repeat.
            self._sim_counter += 1
            phase = self._sim_counter % 55
            is_focused = phase < 45
            # Add tiny noise
            if random.random() < 0.02:
                is_focused = not is_focused
            return InferenceResult(is_focused=is_focused, confidence=None)

        # Real (heuristic) inference: rate-limit detection and reuse last output for stability/perf.
        now = time.time()
        if (now - self._last_ts) < self.inference_interval_seconds:
            return self._last_result

        res = self._heuristic_predict(frame)
        self._last_ts = now
        self._last_result = res
        return res