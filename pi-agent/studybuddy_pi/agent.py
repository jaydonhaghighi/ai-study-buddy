from __future__ import annotations

import json
import os
import time
from typing import Any

from .api import StudyBuddyApi
from .config import Config
from .camera_manager import CameraManager
from .focus_state import FocusStateMachine
from .inference import FocusInference
from .preview_server import PreviewServer
from .storage import StoredAuth, enqueue_summary, list_queued_summaries, load_auth, save_auth
from .summary import compute_focus_summary, summary_to_payload


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self._preview_server: PreviewServer | None = None
        self._camera = CameraManager(
            width=config.camera_width,
            height=config.camera_height,
            format=config.camera_format,
            target_fps=max(config.target_fps, 10.0),
        )

    def pair(self) -> StoredAuth:
        if not self.config.claim_code:
            raise RuntimeError("Missing STUDYBUDDY_CLAIM_CODE")

        api = StudyBuddyApi(self.config.base_url, device_token=None)

        # Best-effort registration (backend may return OK even if already registered)
        try:
            api.register_device(self.config.claim_code, device_id=self.config.device_id)
        except Exception:
            # Registration may be optional depending on backend design; pairingStatus may still work
            pass

        # Poll until paired
        backoff = 1.0
        while True:
            status = api.pairing_status(self.config.claim_code, device_id=self.config.device_id)
            if status.paired and status.deviceToken:
                auth = StoredAuth(device_token=status.deviceToken, device_id=status.deviceId, paired_at=time.time())
                save_auth(self.config.state_dir, auth)
                return auth

            time.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)

    def _load_auth_or_error(self) -> StoredAuth:
        auth = load_auth(self.config.state_dir)
        if not auth:
            raise RuntimeError(
                "Device is not paired. Run `python -m studybuddy_pi pair` or set STUDYBUDDY_CLAIM_CODE."
            )
        return auth

    def flush_outbox(self, api: StudyBuddyApi) -> None:
        for p in list_queued_summaries(self.config.state_dir):
            try:
                payload = json.loads(p.read_text())
                api.post_session_summary(payload)
                p.unlink()
            except Exception:
                # Stop on first failure to avoid hot-looping; retry next cycle
                break

    def run_forever(self) -> None:
        auth = self._load_auth_or_error()
        api = StudyBuddyApi(self.config.base_url, device_token=auth.device_token)

        inference = FocusInference(simulate=self.config.simulate)

        if self.config.enable_preview_server:
            try:
                self._preview_server = PreviewServer(
                    host=self.config.preview_host,
                    port=self.config.preview_port,
                    camera=self._camera,
                )
                self._preview_server.start()
                print(
                    f"[ai-study-buddy] Calibration preview server listening on http://{self.config.preview_host}:{self.config.preview_port}"
                )
                print("[ai-study-buddy] Endpoints: POST /start, POST /stop, GET /status, GET /stream.mjpg")
            except Exception as e:
                print(f"[ai-study-buddy] Failed to start preview server: {e}")

        device_id = auth.device_id or self.config.device_id
        current_focus_session_id: str | None = None
        current_meta: dict[str, Any] = {}

        poll_sleep = self.config.poll_interval_seconds
        last_control_check = 0.0

        # Active session state
        fsm: FocusStateMachine | None = None
        session_start_ts: float | None = None
        frames_captured = 0
        last_heartbeat = 0.0
        stopping_focus_session = False

        while True:
            # Always try to flush any queued summaries
            self.flush_outbox(api)

        # now = time.time()  # reserved for future use (e.g., heartbeat)

            # If not currently running a focus session, use polling (or long-poll) to wait for one.
            if current_focus_session_id is None:
                try:
                    resp = api.current_focus_session(
                        device_id=device_id,
                        wait_seconds=self.config.long_poll_wait_seconds,
                        since_epoch_ms=None,
                    )
                    if resp.focusSessionId:
                        current_focus_session_id = resp.focusSessionId
                        current_meta = {"courseId": resp.courseId, "courseSessionId": resp.courseSessionId}
                        fsm = FocusStateMachine(
                            distract_threshold_seconds=self.config.distract_threshold_seconds,
                            refocus_threshold_seconds=self.config.refocus_threshold_seconds,
                        )
                        session_start_ts = time.time()
                        frames_captured = 0
                        last_heartbeat = 0.0
                        print(f"[ai-study-buddy] Focus session START: {current_focus_session_id}")
                        # Acquire the shared camera for tracking
                        try:
                            self._camera.acquire("tracking")
                        except Exception as e:
                            print(f"[ai-study-buddy] Camera acquire failed: {e}")
                        last_control_check = 0.0
                        poll_sleep = self.config.poll_interval_seconds
                    else:
                        # idle: backoff polling if not using long-poll
                        if self.config.long_poll_wait_seconds is None:
                            time.sleep(poll_sleep)
                            poll_sleep = min(poll_sleep * 1.2, self.config.poll_max_interval_seconds)
                except Exception:
                    time.sleep(poll_sleep)
                    poll_sleep = min(poll_sleep * 1.5, self.config.poll_max_interval_seconds)
                continue

            # Active focus session: run capture/inference loop at target fps, and periodically check control plane.
            assert fsm is not None and session_start_ts is not None

            tick_start = time.time()
            try:
                latest = self._camera.get_latest()
                frame = latest.frame if latest is not None else None
                if frame is not None:
                    frames_captured += 1

                res = inference.predict(frame=frame)
                fsm.update(res.is_focused, now=tick_start)
            except NotImplementedError:
                # If real inference isn't wired, require simulate
                raise
            except Exception:
                # Ignore transient capture/model issues; keep loop alive
                pass

            # Heartbeat while active (helps confirm camera is running)
            if (tick_start - last_heartbeat) >= 5.0:
                last_heartbeat = tick_start
                cam_status = "ON" if self._camera.get_latest() is not None else "OFF"
                print(f"[ai-study-buddy] Active {current_focus_session_id} camera={cam_status} frames={frames_captured}")

            # Check session assignment (control) ~1Hz
            if (tick_start - last_control_check) >= 1.0:
                last_control_check = tick_start
                try:
                    resp = api.current_focus_session(
                        device_id=device_id,
                        wait_seconds=None,
                        since_epoch_ms=None,
                    )
                    if (not resp.focusSessionId or resp.focusSessionId != current_focus_session_id) and not stopping_focus_session:
                        # Session ended or switched
                        stopping_focus_session = True
                        ended_focus_session_id = current_focus_session_id
                        end_ts = time.time()
                        print(f"[ai-study-buddy] Focus session STOP: {ended_focus_session_id}")

                        try:
                            computed = compute_focus_summary(
                                transitions=fsm.transitions,
                                distractions=fsm.distractions,
                                start_ts=session_start_ts,
                                end_ts=end_ts,
                            )
                            payload = summary_to_payload(
                                focus_session_id=ended_focus_session_id,
                                device_id=device_id,
                                course_id=current_meta.get("courseId"),
                                course_session_id=current_meta.get("courseSessionId"),
                                start_ts=session_start_ts,
                                end_ts=end_ts,
                                distractions=fsm.distractions,
                                computed=computed,
                            )

                            # Always enqueue first (reliability), then try upload
                            enqueue_summary(self.config.state_dir, payload)
                            self.flush_outbox(api)
                        except Exception as e:
                            # Never let summary/upload failures prevent cleanup/reset
                            print(f"[ai-study-buddy] Stop handling error (will retry outbox later): {e}")
                        finally:
                            # Release camera ownership for tracking
                            try:
                                self._camera.release("tracking")
                            except Exception:
                                pass
                            print("[ai-study-buddy] Camera RELEASED")

                            # Reset local state so frames stop counting up for the ended session
                            current_focus_session_id = None
                            current_meta = {}
                            fsm = None
                            session_start_ts = None
                            poll_sleep = self.config.poll_interval_seconds
                            stopping_focus_session = False
                except Exception:
                    # Keep running; control checks will retry
                    pass

            # Sleep to hit target fps
            elapsed = time.time() - tick_start
            sleep_for = max(0.0, (1.0 / max(self.config.target_fps, 0.1)) - elapsed)
            time.sleep(sleep_for)