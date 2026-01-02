from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    ts: float  # epoch seconds
    state: str  # "FOCUSED" | "DISTRACTED"


class FocusStateMachine:
    """
    Temporal smoothing over noisy frame-level predictions.

    - Transition to DISTRACTED only after continuous not-focused for distract_threshold_seconds.
    - Transition back to FOCUSED only after continuous focused for refocus_threshold_seconds.
    """

    def __init__(self, distract_threshold_seconds: float = 30.0, refocus_threshold_seconds: float = 3.0):
        self.distract_threshold_seconds = distract_threshold_seconds
        self.refocus_threshold_seconds = refocus_threshold_seconds

        self.state: str = "FOCUSED"
        self._unfocused_since: float | None = None
        self._focused_since: float | None = None

        now = time.time()
        self.transitions: list[Transition] = [Transition(ts=now, state=self.state)]
        self.distractions: int = 0

    def update(self, is_focused: bool, now: float | None = None) -> str:
        ts = now if now is not None else time.time()

        if self.state == "FOCUSED":
            if is_focused:
                self._unfocused_since = None
                return self.state

            # not focused
            if self._unfocused_since is None:
                self._unfocused_since = ts
            if (ts - self._unfocused_since) >= self.distract_threshold_seconds:
                self.state = "DISTRACTED"
                self._focused_since = None
                self.transitions.append(Transition(ts=ts, state=self.state))
                self.distractions += 1
            return self.state

        # DISTRACTED
        if not is_focused:
            self._focused_since = None
            return self.state

        # focused while distracted
        if self._focused_since is None:
            self._focused_since = ts
        if (ts - self._focused_since) >= self.refocus_threshold_seconds:
            self.state = "FOCUSED"
            self._unfocused_since = None
            self.transitions.append(Transition(ts=ts, state=self.state))
        return self.state