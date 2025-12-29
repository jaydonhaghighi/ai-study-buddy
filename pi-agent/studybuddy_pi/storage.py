from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StoredAuth:
    device_token: str
    device_id: str | None = None
    paired_at: float | None = None  # epoch seconds


def ensure_dir(path: str) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def auth_file(state_dir: str) -> Path:
    return ensure_dir(state_dir) / "device_auth.json"


def queue_dir(state_dir: str) -> Path:
    return ensure_dir(state_dir) / "outbox"


def load_auth(state_dir: str) -> StoredAuth | None:
    p = auth_file(state_dir)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    token = data.get("deviceToken")
    if not token:
        return None
    return StoredAuth(
        device_token=token,
        device_id=data.get("deviceId"),
        paired_at=data.get("pairedAt"),
    )


def save_auth(state_dir: str, auth: StoredAuth) -> None:
    p = auth_file(state_dir)
    payload = {
        "deviceToken": auth.device_token,
        "deviceId": auth.device_id,
        "pairedAt": auth.paired_at or time.time(),
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))


def enqueue_summary(state_dir: str, summary: dict[str, Any]) -> Path:
    out = queue_dir(state_dir)
    ts = int(time.time() * 1000)
    focus_session_id = summary.get("focusSessionId") or "unknown"
    name = f"{ts}-{focus_session_id}.json"
    p = out / name
    p.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return p


def list_queued_summaries(state_dir: str) -> list[Path]:
    out = queue_dir(state_dir)
    if not out.exists():
        return []
    return sorted([p for p in out.glob("*.json") if p.is_file()])


def delete_queued(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)  # py3.8+: supported; on older, ignore
    except TypeError:
        if path.exists():
            path.unlink()


