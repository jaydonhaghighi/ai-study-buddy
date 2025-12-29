from __future__ import annotations

import json
import os
import time
from typing import Any

from .api import StudyBuddyApi
from .config import Config
from .focus_state import FocusStateMachine
from .inference import FocusInference
from .camera import open_camera, read_frame
from .preview_server import PreviewServer
from .storage import StoredAuth, enqueue_summary, list_queued_summaries, load_auth, save_auth
from .summary import compute_focus_summary, summary_to_payload


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self._preview_server: PreviewServer | None = None

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
                    camera_index=0,
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
        cap: Any | None = None
        camera_ctx: Any | None = None
        video_writer: Any | None = None
        video_path: str | None = None
        frames_captured = 0
        last_heartbeat = 0.0

        while True:
            # Always try to flush any queued summaries
            self.flush_outbox(api)

            now = time.time()

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
                        # Open camera on session start (even if inference is simulated)
                        if self.config.enable_camera:
                            camera_ctx = open_camera(index=0)
                            cap = camera_ctx.__enter__()
                            print("[ai-study-buddy] Camera OPENED (cv2.VideoCapture)")
                            # Optional local recording to a file for verification/debug
                            if self.config.record_dir:
                                try:
                                    import cv2  # type: ignore
                                    os.makedirs(self.config.record_dir, exist_ok=True)
                                    video_path = os.path.join(
                                        self.config.record_dir,
                                        f"{current_focus_session_id}.avi",
                                    )
                                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                                    video_writer = cv2.VideoWriter(video_path, fourcc, self.config.target_fps, (w, h))
                                    print(f"[ai-study-buddy] Recording ENABLED: {video_path} ({w}x{h} @ {self.config.target_fps}fps)")
                                except Exception:
                                    video_writer = None
                                    video_path = None
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
                frame = None
                if cap is not None:
                    frame = read_frame(cap)
                    if frame is not None and video_writer is not None:
                        try:
                            video_writer.write(frame)
                        except Exception:
                            pass
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
                cam_status = "ON" if cap is not None else "OFF"
                rec_status = f"REC={video_path}" if video_path else "REC=off"
                print(f"[ai-study-buddy] Active {current_focus_session_id} camera={cam_status} frames={frames_captured} {rec_status}")

            # Check session assignment (control) ~1Hz
            if (tick_start - last_control_check) >= 1.0:
                last_control_check = tick_start
                try:
                    resp = api.current_focus_session(
                        device_id=device_id,
                        wait_seconds=None,
                        since_epoch_ms=None,
                    )
                    if not resp.focusSessionId or resp.focusSessionId != current_focus_session_id:
                        # Session ended or switched
                        end_ts = time.time()
                        print(f"[ai-study-buddy] Focus session STOP: {current_focus_session_id}")
                        computed = compute_focus_summary(
                            transitions=fsm.transitions,
                            distractions=fsm.distractions,
                            start_ts=session_start_ts,
                            end_ts=end_ts,
                        )
                        payload = summary_to_payload(
                            focus_session_id=current_focus_session_id,
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

                        # Close camera/recording
                        try:
                            if video_writer is not None:
                                video_writer.release()
                        except Exception:
                            pass
                        video_writer = None
                        if video_path:
                            print(f"[ai-study-buddy] Recording CLOSED: {video_path}")
                        video_path = None
                        try:
                            if camera_ctx is not None:
                                camera_ctx.__exit__(None, None, None)
                        except Exception:
                            pass
                        cap = None
                        camera_ctx = None
                        print("[ai-study-buddy] Camera CLOSED")

                        # Reset local state
                        current_focus_session_id = None
                        current_meta = {}
                        fsm = None
                        session_start_ts = None
                        poll_sleep = self.config.poll_interval_seconds
                except Exception:
                    # Keep running; control checks will retry
                    pass

            # Sleep to hit target fps
            elapsed = time.time() - tick_start
            sleep_for = max(0.0, (1.0 / max(self.config.target_fps, 0.1)) - elapsed)
            time.sleep(sleep_for)


