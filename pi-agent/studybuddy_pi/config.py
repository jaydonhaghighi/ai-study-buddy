from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    # Backend base URL, e.g. "https://us-central1-<project>.cloudfunctions.net"
    base_url: str

    # Pairing (claim code shown/entered by user)
    claim_code: str | None

    # Local storage
    state_dir: str

    # Polling
    poll_interval_seconds: float
    poll_max_interval_seconds: float
    long_poll_wait_seconds: int | None

    # Focus temporal logic thresholds
    distract_threshold_seconds: float
    refocus_threshold_seconds: float

    # Capture/inference loop (when active)
    target_fps: float
    simulate: bool
    record_dir: str | None
    enable_preview_server: bool
    preview_host: str
    preview_port: int
    # Preview color handling:
    # - None: auto (based on camera output format)
    # - True: force swap RB (treat input as RGB -> convert to BGR)
    # - False: force no swap (treat input as already BGR)
    preview_swap_rb: bool | None

    # Camera (Picamera2)
    camera_width: int
    camera_height: int
    camera_format: str

    # Optional device identity hints (depending on backend design)
    device_id: str | None


def load_config() -> Config:
    base_url = os.getenv("STUDYBUDDY_BASE_URL", "").rstrip("/")
    if not base_url:
        raise RuntimeError("Missing STUDYBUDDY_BASE_URL")

    # Use a stable per-user dir by default
    state_dir = os.getenv("STUDYBUDDY_STATE_DIR", os.path.expanduser("~/.ai-study-buddy"))

    return Config(
        base_url=base_url,
        claim_code=os.getenv("STUDYBUDDY_CLAIM_CODE"),
        state_dir=state_dir,
        poll_interval_seconds=float(os.getenv("STUDYBUDDY_POLL_INTERVAL_SECONDS", "2.0")),
        poll_max_interval_seconds=float(os.getenv("STUDYBUDDY_POLL_MAX_INTERVAL_SECONDS", "30.0")),
        long_poll_wait_seconds=(
            int(os.getenv("STUDYBUDDY_LONG_POLL_WAIT_SECONDS", "25"))
            if os.getenv("STUDYBUDDY_LONG_POLL_WAIT_SECONDS") is not None
            else None
        ),
        distract_threshold_seconds=float(os.getenv("STUDYBUDDY_DISTRACT_THRESHOLD_SECONDS", "30.0")),
        refocus_threshold_seconds=float(os.getenv("STUDYBUDDY_REFOCUS_THRESHOLD_SECONDS", "3.0")),
        target_fps=float(os.getenv("STUDYBUDDY_TARGET_FPS", "10.0")),
        simulate=_env_bool("STUDYBUDDY_SIMULATE", False),
        record_dir=os.getenv("STUDYBUDDY_RECORD_DIR"),
        enable_preview_server=_env_bool("STUDYBUDDY_ENABLE_PREVIEW_SERVER", True),
        preview_host=os.getenv("STUDYBUDDY_PREVIEW_HOST", "0.0.0.0"),
        preview_port=int(os.getenv("STUDYBUDDY_PREVIEW_PORT", "8080")),
        preview_swap_rb=(
            None
            if os.getenv("STUDYBUDDY_PREVIEW_SWAP_RB") is None
            else _env_bool("STUDYBUDDY_PREVIEW_SWAP_RB", False)
        ),
        camera_width=int(os.getenv("STUDYBUDDY_CAMERA_WIDTH", "640")),
        camera_height=int(os.getenv("STUDYBUDDY_CAMERA_HEIGHT", "480")),
        # Prefer BGR888 to match OpenCV's expected channel order.
        camera_format=os.getenv("STUDYBUDDY_CAMERA_FORMAT", "BGR888"),
        device_id=os.getenv("STUDYBUDDY_DEVICE_ID"),
    )