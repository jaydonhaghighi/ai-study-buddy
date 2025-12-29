from __future__ import annotations

import time
from typing import Any

import requests

from .models import CurrentFocusSessionResponse, PairingStatusResponse


class StudyBuddyApi:
    def __init__(self, base_url: str, device_token: str | None = None, timeout_seconds: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.device_token = device_token
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.device_token:
            headers["Authorization"] = f"Bearer {self.device_token}"
        return headers

    def register_device(self, claim_code: str, device_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"claimCode": claim_code}
        if device_id:
            payload["deviceId"] = device_id
        r = requests.post(
            f"{self.base_url}/device/register",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        r.raise_for_status()
        return r.json()

    def pairing_status(self, claim_code: str, device_id: str | None = None) -> PairingStatusResponse:
        params: dict[str, Any] = {"claimCode": claim_code}
        if device_id:
            params["deviceId"] = device_id
        r = requests.get(
            f"{self.base_url}/device/pairingStatus",
            params=params,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        r.raise_for_status()
        return PairingStatusResponse.model_validate(r.json())

    def current_focus_session(
        self,
        device_id: str | None = None,
        wait_seconds: int | None = None,
        since_epoch_ms: int | None = None,
    ) -> CurrentFocusSessionResponse:
        params: dict[str, Any] = {}
        if device_id:
            params["deviceId"] = device_id
        if wait_seconds is not None:
            params["waitSeconds"] = wait_seconds
        if since_epoch_ms is not None:
            params["since"] = since_epoch_ms

        r = requests.get(
            f"{self.base_url}/device/currentFocusSession",
            params=params,
            headers=self._headers(),
            timeout=(wait_seconds + 5) if wait_seconds else self.timeout_seconds,
        )
        r.raise_for_status()
        return CurrentFocusSessionResponse.model_validate(r.json())

    def post_session_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/device/sessionSummary",
            json=summary,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        r.raise_for_status()
        return r.json() if r.content else {"ok": True, "uploadedAt": time.time()}


