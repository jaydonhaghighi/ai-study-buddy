from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import LABELS, SCREEN_LABEL


@dataclass
class TemporalConfig:
    ema_alpha: float = 0.35
    distract_seconds: float = 2.5
    refocus_seconds: float = 1.0
    confidence_threshold: float = 0.55
    min_dwell_seconds: float = 0.5


@dataclass
class TemporalResult:
    timestamp_ms: int
    raw_label: str
    raw_confidence: float
    smoothed_label: str
    smoothed_confidence: float
    state: str
    transitioned: bool
    previous_state: str


class TemporalFocusEngine:
    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()
        self.state = "focused"
        self.last_transition_ms: int | None = None
        self.pending_state: str | None = None
        self.pending_since_ms: int | None = None
        self.smoothed_probs: np.ndarray | None = None

    def update(self, probs: np.ndarray, timestamp_ms: int) -> TemporalResult:
        probs = np.asarray(probs, dtype=np.float32)
        raw_idx = int(np.argmax(probs))
        raw_label = LABELS[raw_idx]
        raw_conf = float(probs[raw_idx])

        if self.smoothed_probs is None:
            self.smoothed_probs = probs.copy()
        else:
            alpha = self.config.ema_alpha
            self.smoothed_probs = alpha * probs + (1.0 - alpha) * self.smoothed_probs

        smooth_idx = int(np.argmax(self.smoothed_probs))
        smooth_label = LABELS[smooth_idx]
        smooth_conf = float(self.smoothed_probs[smooth_idx])

        candidate_state = self.state
        if smooth_conf >= self.config.confidence_threshold:
            candidate_state = "focused" if smooth_label == SCREEN_LABEL else "distracted"

        transitioned = False
        previous_state = self.state

        if candidate_state != self.state:
            if self.pending_state != candidate_state:
                self.pending_state = candidate_state
                self.pending_since_ms = timestamp_ms
            else:
                assert self.pending_since_ms is not None
                elapsed = (timestamp_ms - self.pending_since_ms) / 1000.0
                required = (
                    self.config.distract_seconds
                    if candidate_state == "distracted"
                    else self.config.refocus_seconds
                )
                last_transition_ok = True
                if self.last_transition_ms is not None:
                    dwell = (timestamp_ms - self.last_transition_ms) / 1000.0
                    last_transition_ok = dwell >= self.config.min_dwell_seconds
                if elapsed >= required and last_transition_ok:
                    self.state = candidate_state
                    self.last_transition_ms = timestamp_ms
                    transitioned = True
                    self.pending_state = None
                    self.pending_since_ms = None
        else:
            self.pending_state = None
            self.pending_since_ms = None

        return TemporalResult(
            timestamp_ms=timestamp_ms,
            raw_label=raw_label,
            raw_confidence=raw_conf,
            smoothed_label=smooth_label,
            smoothed_confidence=smooth_conf,
            state=self.state,
            transitioned=transitioned,
            previous_state=previous_state,
        )


@dataclass
class SessionSummaryAccumulator:
    start_ms: int
    focused_ms: int = 0
    distracted_ms: int = 0
    distractions: int = 0
    longest_focused_ms: int = 0
    longest_distracted_ms: int = 0
    avg_focus_before_distract_ms: float = 0.0
    label_counts: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in LABELS}
    )
    _active_state: str | None = None
    _active_state_started_ms: int | None = None
    _last_frame_ms: int | None = None
    _focus_streaks_before_distract: list[int] = field(default_factory=list)

    def update(self, result: TemporalResult) -> None:
        now_ms = int(result.timestamp_ms)
        self.label_counts[result.smoothed_label] = self.label_counts.get(result.smoothed_label, 0) + 1

        if self._active_state is None:
            self._active_state = result.state
            self._active_state_started_ms = now_ms
            self._last_frame_ms = now_ms
            return

        assert self._last_frame_ms is not None
        delta_ms = max(0, now_ms - self._last_frame_ms)
        if self._active_state == "focused":
            self.focused_ms += delta_ms
        else:
            self.distracted_ms += delta_ms
        self._last_frame_ms = now_ms

        if result.transitioned and self._active_state_started_ms is not None:
            streak_ms = max(0, now_ms - self._active_state_started_ms)
            if self._active_state == "focused":
                self.longest_focused_ms = max(self.longest_focused_ms, streak_ms)
                if result.state == "distracted":
                    self.distractions += 1
                    self._focus_streaks_before_distract.append(streak_ms)
            else:
                self.longest_distracted_ms = max(self.longest_distracted_ms, streak_ms)

            self._active_state = result.state
            self._active_state_started_ms = now_ms

    def finalize(self, end_ms: int) -> dict[str, Any]:
        if self._active_state is not None and self._active_state_started_ms is not None:
            streak_ms = max(0, end_ms - self._active_state_started_ms)
            if self._active_state == "focused":
                self.longest_focused_ms = max(self.longest_focused_ms, streak_ms)
            else:
                self.longest_distracted_ms = max(self.longest_distracted_ms, streak_ms)

        if self._focus_streaks_before_distract:
            self.avg_focus_before_distract_ms = float(
                np.mean(self._focus_streaks_before_distract)
            )

        total_ms = self.focused_ms + self.distracted_ms
        focus_percent = (100.0 * self.focused_ms / total_ms) if total_ms > 0 else 0.0

        attention_label_counts = {
            label: int(count)
            for label, count in self.label_counts.items()
            if count > 0
        }
        return {
            "startTs": int(self.start_ms / 1000),
            "endTs": int(end_ms / 1000),
            "focusedMs": int(self.focused_ms),
            "distractedMs": int(self.distracted_ms),
            "longestFocusedMs": int(self.longest_focused_ms),
            "longestDistractedMs": int(self.longest_distracted_ms),
            "distractions": int(self.distractions),
            "avgFocusBeforeDistractMs": float(round(self.avg_focus_before_distract_ms, 2)),
            "focusPercent": float(round(focus_percent, 2)),
            "attentionLabelCounts": attention_label_counts,
        }
