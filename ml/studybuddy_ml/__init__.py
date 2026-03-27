"""StudyBuddy ML pipeline package."""

LABELS = ["screen", "away_left", "away_right", "away_up", "away_down"]
SCREEN_LABEL = "screen"
FOCUSED_LABELS = frozenset({SCREEN_LABEL, "away_down"})


def is_focused_label(label: str) -> bool:
    return label in FOCUSED_LABELS
