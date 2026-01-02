from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceResult:
    is_focused: bool
    confidence: float | None = None


class FocusInference:
    """
    Abstraction over the focus model.

    - In simulation mode: generates plausible focus/distract patterns without camera/model deps.
    - In real mode: expected to be backed by TFLite + preprocessing (to be implemented when model is ready).
    """

    def __init__(self, simulate: bool = False):
        self.simulate = simulate
        self._sim_counter = 0

    def predict(self, frame: Any | None = None) -> InferenceResult:
        if self.simulate:
            # Simple, deterministic-ish pattern:
            # focused for ~45 ticks, distracted for ~10 ticks, repeat.
            self._sim_counter += 1
            phase = self._sim_counter % 55
            is_focused = phase < 45
            # Add tiny noise
            if random.random() < 0.02:
                is_focused = not is_focused
            return InferenceResult(is_focused=is_focused, confidence=None)

        # Placeholder for real TFLite inference.
        # When implementing for real:
        # - load tflite model (tflite_runtime.interpreter)
        # - preprocess frame to input tensor
        # - run invoke()
        # - map output to is_focused + confidence
        raise NotImplementedError("Real inference not wired yet. Set STUDYBUDDY_SIMULATE=1 for now.")