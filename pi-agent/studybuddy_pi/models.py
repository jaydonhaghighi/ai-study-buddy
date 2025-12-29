from __future__ import annotations

from pydantic import BaseModel, Field


class PairingStatusResponse(BaseModel):
    paired: bool = False
    deviceToken: str | None = None
    deviceId: str | None = None


class CurrentFocusSessionResponse(BaseModel):
    # If no active focus session, focusSessionId is null/None
    focusSessionId: str | None = None
    # Optional metadata the backend may return
    courseId: str | None = None
    courseSessionId: str | None = None


class FocusSummary(BaseModel):
    focusSessionId: str
    deviceId: str | None = None

    # Optional linking for analytics
    courseId: str | None = None
    courseSessionId: str | None = None

    startTs: float = Field(..., description="Epoch seconds")
    endTs: float = Field(..., description="Epoch seconds")

    focusedMs: int
    distractedMs: int
    longestFocusedMs: int
    longestDistractedMs: int
    distractions: int
    avgFocusBeforeDistractMs: int | None = None
    focusPercent: float


