from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .focus_state import Transition


@dataclass
class ComputedSummary:
    focused_ms: int
    distracted_ms: int
    longest_focused_ms: int
    longest_distracted_ms: int
    focus_percent: float
    avg_focus_before_distract_ms: int | None


def _durations_ms(transitions: list[Transition], start_ts: float, end_ts: float) -> tuple[list[tuple[str, int]], int]:
    """
    Returns a list of (state, durationMs) segments and total duration ms.
    """
    if end_ts <= start_ts:
        return [], 0

    segs: list[tuple[str, int]] = []

    # Ensure transitions sorted
    transitions = sorted(transitions, key=lambda t: t.ts)

    # Clamp first transition to start
    current_state = transitions[0].state if transitions else "FOCUSED"
    current_ts = start_ts

    for t in transitions[1:]:
        ts = min(max(t.ts, start_ts), end_ts)
        if ts <= current_ts:
            current_state = t.state
            continue
        segs.append((current_state, int(round((ts - current_ts) * 1000))))
        current_state = t.state
        current_ts = ts

    if current_ts < end_ts:
        segs.append((current_state, int(round((end_ts - current_ts) * 1000))))

    total = sum(d for _, d in segs)
    return segs, total


def compute_focus_summary(
    *,
    transitions: list[Transition],
    distractions: int,
    start_ts: float,
    end_ts: float,
) -> ComputedSummary:
    segs, total_ms = _durations_ms(transitions, start_ts, end_ts)

    focused_ms = sum(d for s, d in segs if s == "FOCUSED")
    distracted_ms = sum(d for s, d in segs if s == "DISTRACTED")

    longest_focused_ms = max([d for s, d in segs if s == "FOCUSED"] or [0])
    longest_distracted_ms = max([d for s, d in segs if s == "DISTRACTED"] or [0])

    focus_percent = (focused_ms / total_ms * 100.0) if total_ms > 0 else 0.0

    # Avg focus streak before each distraction event:
    # Interpret as the focused segment immediately preceding a distracted segment.
    focus_before: list[int] = []
    for i in range(1, len(segs)):
        prev_state, prev_d = segs[i - 1]
        state, _d = segs[i]
        if state == "DISTRACTED" and prev_state == "FOCUSED":
            focus_before.append(prev_d)
    avg_focus_before_distract_ms = int(round(sum(focus_before) / len(focus_before))) if focus_before else None

    # Use provided distractions count (state machine count), but if missing, fallback to observed
    _ = distractions

    return ComputedSummary(
        focused_ms=focused_ms,
        distracted_ms=distracted_ms,
        longest_focused_ms=longest_focused_ms,
        longest_distracted_ms=longest_distracted_ms,
        focus_percent=round(focus_percent, 2),
        avg_focus_before_distract_ms=avg_focus_before_distract_ms,
    )


def summary_to_payload(
    *,
    focus_session_id: str,
    device_id: str | None,
    start_ts: float,
    end_ts: float,
    distractions: int,
    computed: ComputedSummary,
    attention_label_counts: dict[str, int] | None = None,
    course_id: str | None = None,
    course_session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "focusSessionId": focus_session_id,
        "deviceId": device_id,
        "courseId": course_id,
        "courseSessionId": course_session_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "focusedMs": computed.focused_ms,
        "distractedMs": computed.distracted_ms,
        "longestFocusedMs": computed.longest_focused_ms,
        "longestDistractedMs": computed.longest_distracted_ms,
        "distractions": distractions,
        "avgFocusBeforeDistractMs": computed.avg_focus_before_distract_ms,
        "focusPercent": computed.focus_percent,
        "attentionLabelCounts": attention_label_counts or {},
    }