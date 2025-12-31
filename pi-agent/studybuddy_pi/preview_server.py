from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .camera_manager import CameraManager

class _SharedState:
    def __init__(self, camera: CameraManager):
        self.camera = camera
        self.enabled = False
        self.lock = threading.Lock()

        self.last_jpeg: bytes | None = None
        self.last_frame_ts: float | None = None

        self.face_detected: bool = False
        self.aligned: bool = False
        self.face_box: list[int] | None = None  # [x, y, w, h]
        self.last_error: str | None = None
        # Deterministic preview color handling:
        # On this Pi setup, swapping R/B was producing incorrect colors.
        # We keep a debug overlay (red/blue squares) to verify channel order visually.
        self.last_swap_applied: bool = False
        self.frame_count: int = 0

        self._stop = False
        self._thread: threading.Thread | None = None

        self._face_cascade = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_all(self) -> None:
        self._stop = True
        try:
            self.camera.release("preview")
        except Exception:
            pass

    def enable(self) -> None:
        # Acquire the shared camera (refcounted)
        self.camera.acquire("preview")
        with self.lock:
            self.enabled = True

    def disable(self) -> None:
        with self.lock:
            self.enabled = False
            self.last_jpeg = None
            self.last_frame_ts = None
            self.face_detected = False
            self.aligned = False
            self.face_box = None
            self.last_error = None
        try:
            self.camera.release("preview")
        except Exception:
            pass

    def _loop(self) -> None:
        # Background capture loop when enabled
        while not self._stop:
            time.sleep(0.05)  # ~20Hz cap; browser will pull what it needs
            with self.lock:
                if not self.enabled:
                    continue

            try:
                import cv2  # type: ignore
            except Exception:
                with self.lock:
                    self.last_error = "cv2 import failed in capture thread"
                continue

            latest = self.camera.get_latest()
            if latest is None:
                with self.lock:
                    self.last_error = self.camera.last_error or "No frames yet"
                continue

            frame = latest.frame
            if frame is None:
                with self.lock:
                    self.last_error = "Camera returned null frame"
                continue

            h, w = frame.shape[:2]

            # Face detection (simple alignment heuristic)
            face_detected = False
            aligned = False
            face_box = None

            # Do NOT swap channels by default. (If colors look swapped, we can flip this later,
            # but the debug overlay below will tell us definitively.)

            # Ensure frame is contiguous for OpenCV ops (rectangle/imencode)
            try:
                import numpy as np  # type: ignore
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
            except Exception:
                pass

            if self._face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) > 0:
                    # pick largest face
                    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                    face_detected = True
                    face_box = [int(x), int(y), int(fw), int(fh)]

                    # alignment: face center near frame center (within 15% of width/height)
                    cx = x + fw / 2.0
                    cy = y + fh / 2.0
                    aligned = (abs(cx - w / 2.0) <= 0.15 * w) and (abs(cy - h / 2.0) <= 0.15 * h)

                    # draw rectangle (visual feedback only, not saved)
                    try:
                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + fw, y + fh),
                            (0, 255, 0) if aligned else (0, 165, 255),
                            2,
                        )
                    except Exception:
                        # Don't crash the capture thread if drawing fails
                        pass

            # Debug color reference overlay (small, bottom-left):
            # - Red square (BGR: 0,0,255)
            # - Blue square (BGR: 255,0,0)
            # If these appear swapped, the pipeline is interpreting channels differently.
            try:
                size = 20
                pad = 6
                # Red
                cv2.rectangle(frame, (pad, h - pad - size), (pad + size, h - pad), (0, 0, 255), -1)
                # Blue
                cv2.rectangle(frame, (pad + size + 6, h - pad - size), (pad + size + 6 + size, h - pad), (255, 0, 0), -1)
            except Exception:
                pass

            # Encode JPEG
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok2:
                with self.lock:
                    self.last_error = "cv2.imencode(.jpg) failed"
                continue

            jpeg = buf.tobytes()
            ts = latest.ts
            with self.lock:
                self.last_jpeg = jpeg
                self.last_frame_ts = ts
                self.frame_count += 1
                self.face_detected = face_detected
                self.aligned = aligned
                self.face_box = face_box
                self.last_error = None


class PreviewServer:
    """
    LAN-only calibration preview server.

    Endpoints:
    - GET /health
    - POST /start  (enables capture)
    - POST /stop   (disables capture)
    - GET /status  (alignment + timestamp)
    - GET /stream.mjpg (multipart MJPEG stream)
    """

    def __init__(self, *, host: str = "0.0.0.0", port: int = 8080, camera: CameraManager):
        self.host = host
        self.port = port
        self.state = _SharedState(camera=camera)
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.state.start()

        server = self

        class Handler(BaseHTTPRequestHandler):
            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path

                if path == "/health":
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
                    return

                if path == "/status":
                    with server.state.lock:
                        body = {
                            # Cast everything to plain JSON-serializable primitives
                            "enabled": bool(server.state.enabled),
                            "lastFrameTs": float(server.state.last_frame_ts) if server.state.last_frame_ts is not None else None,
                            "faceDetected": bool(server.state.face_detected),
                            "aligned": bool(server.state.aligned),
                            "faceBox": [int(x) for x in server.state.face_box] if server.state.face_box else None,
                            "lastError": str(server.state.last_error) if server.state.last_error is not None else None,
                            "swapApplied": bool(server.state.last_swap_applied),
                            "frameCount": int(server.state.frame_count),
                            "cameraRequestedFormat": str(server.state.camera.requested_format),
                            "cameraActualFormat": str(server.state.camera.actual_format) if server.state.camera.actual_format is not None else None,
                        }
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(body).encode("utf-8"))
                    return

                if path == "/stream.mjpg":
                    # Stream MJPEG
                    self.send_response(200)
                    self._cors()
                    boundary = "frame"
                    self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                    self.end_headers()

                    while True:
                        with server.state.lock:
                            jpeg = server.state.last_jpeg
                            enabled = server.state.enabled

                        if not enabled or jpeg is None:
                            time.sleep(0.1)
                            continue

                        try:
                            self.wfile.write(f"--{boundary}\r\n".encode())
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            time.sleep(0.05)
                        except BrokenPipeError:
                            return
                        except ConnectionResetError:
                            return
                        except Exception:
                            return

                self.send_response(404)
                self._cors()
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)

                if path == "/start":
                    try:
                        server.state.enable()
                        self.send_response(200)
                        self._cors()
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"ok": True, "enabled": True}).encode("utf-8"))
                    except Exception as e:
                        try:
                            server.state.disable()
                        except Exception:
                            pass
                        self.send_response(500)
                        self._cors()
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
                    return

                if path == "/stop":
                    server.state.disable()
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": True, "enabled": False}).encode("utf-8"))
                    return

                self.send_response(404)
                self._cors()
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                # Keep stdout clean
                return

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.state.stop_all()


