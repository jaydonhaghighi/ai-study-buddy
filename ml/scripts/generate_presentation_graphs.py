#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready ML charts from existing capstone artifacts."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("ml/artifacts/reports/runs/20260213-160744"),
        help="Run directory containing capstone_report.md and plots/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ml/artifacts/reports/presentation"),
        help="Output directory for generated charts.",
    )
    parser.add_argument(
        "--timeline-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with frame-level timeline data. "
            "Columns: timestamp_s, raw_confidence, smoothed_confidence, state"
        ),
    )
    return parser.parse_args()


def _parse_markdown_table_rows(markdown: str) -> list[dict[str, str]]:
    lines = markdown.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("| fold_id |"):
            header_idx = i
            break
    if header_idx < 0:
        raise ValueError("Could not find per-fold results table in capstone_report.md")

    headers = [h.strip() for h in lines[header_idx].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[header_idx + 2 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    if not rows:
        raise ValueError("Per-fold table exists but has no rows")
    return rows


def _extract_best_fold(markdown: str) -> str:
    match = re.search(r"Best fold\*\*: `(\d+)`", markdown)
    if not match:
        return "unknown"
    return match.group(1).zfill(2)


def plot_participant_performance(table_rows: list[dict[str, str]], out_path: Path) -> None:
    frame = pd.DataFrame(table_rows)
    needed = {"test_participants", "test_macro_f1", "test_balanced_accuracy"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in table: {sorted(missing)}")

    frame["test_macro_f1"] = pd.to_numeric(frame["test_macro_f1"], errors="coerce")
    frame["test_balanced_accuracy"] = pd.to_numeric(
        frame["test_balanced_accuracy"], errors="coerce"
    )
    frame = frame.dropna(subset=["test_macro_f1", "test_balanced_accuracy"])
    frame = frame.rename(columns={"test_participants": "participant"})
    frame = frame.sort_values("test_macro_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(
        frame["participant"],
        frame["test_macro_f1"],
        color="#0f172a",
        alpha=0.92,
        label="Test Macro F1",
    )
    ax.scatter(
        frame["test_balanced_accuracy"],
        frame["participant"],
        color="#16a34a",
        marker="D",
        s=52,
        label="Test Balanced Accuracy",
        zorder=3,
    )

    for bar, score in zip(bars, frame["test_macro_f1"]):
        ax.text(
            score + 0.007,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0.0, min(1.0, float(frame[["test_macro_f1", "test_balanced_accuracy"]].max().max() + 0.12)))
    ax.set_xlabel("Score")
    ax.set_ylabel("Held-out participant")
    ax.set_title("LOSO Held-out Participant Performance")
    ax.grid(axis="x", alpha=0.22, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_confusion_matrix_slide(source_png: Path, out_path: Path, best_fold: str) -> None:
    if not source_png.exists():
        raise FileNotFoundError(f"Missing confusion matrix image: {source_png}")
    image = plt.imread(source_png)
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        f"Best Fold Confusion Matrix (Fold {best_fold})",
        fontsize=16,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _timeline_from_csv(timeline_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(timeline_csv)
    required = {"timestamp_s", "raw_confidence", "smoothed_confidence", "state"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Timeline CSV missing required columns: {sorted(missing)}"
        )
    frame = frame.copy()
    frame["timestamp_s"] = pd.to_numeric(frame["timestamp_s"], errors="coerce")
    frame["raw_confidence"] = pd.to_numeric(frame["raw_confidence"], errors="coerce")
    frame["smoothed_confidence"] = pd.to_numeric(
        frame["smoothed_confidence"], errors="coerce"
    )
    frame = frame.dropna(subset=["timestamp_s", "raw_confidence", "smoothed_confidence"])
    return frame.sort_values("timestamp_s")


def _illustrative_timeline() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t = np.arange(0, 301, 2)
    segments = [
        (0, 56, "focused", 0.80),
        (56, 92, "distracted", 0.69),
        (92, 184, "focused", 0.77),
        (184, 226, "distracted", 0.66),
        (226, 302, "focused", 0.83),
    ]

    states: list[str] = []
    smoothed: list[float] = []
    for ts in t:
        for start, end, state, base_conf in segments:
            if start <= ts < end:
                states.append(state)
                smoothed.append(float(np.clip(base_conf + rng.normal(0, 0.035), 0.35, 0.98)))
                break

    raw = np.clip(np.array(smoothed) + rng.normal(0, 0.08, size=len(smoothed)), 0.2, 0.99)
    return pd.DataFrame(
        {
            "timestamp_s": t,
            "raw_confidence": raw,
            "smoothed_confidence": smoothed,
            "state": states,
        }
    )


def plot_session_timeline(frame: pd.DataFrame, out_path: Path, from_real_data: bool) -> None:
    x = frame["timestamp_s"].to_numpy()
    raw = frame["raw_confidence"].to_numpy()
    smooth = frame["smoothed_confidence"].to_numpy()
    states = frame["state"].astype(str).to_numpy()
    is_focused = np.array([1 if s == "focused" else 0 for s in states], dtype=int)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top, ax_bottom = axes

    ax_top.plot(x, raw, color="#94a3b8", linewidth=1.4, alpha=0.8, label="Raw confidence")
    ax_top.plot(x, smooth, color="#0f172a", linewidth=2.4, label="Smoothed confidence")
    ax_top.axhline(0.55, color="#dc2626", linestyle="--", linewidth=1.3, label="Confidence threshold (0.55)")
    ax_top.set_ylim(0.0, 1.0)
    ax_top.set_ylabel("Confidence")
    ax_top.grid(alpha=0.24, linestyle="--")
    ax_top.legend(loc="lower right")
    ax_top.set_title("Session Timeline: Confidence and Focus State Transitions")

    ax_bottom.step(x, is_focused, where="post", color="#0f766e", linewidth=2.2)
    ax_bottom.fill_between(x, 0, is_focused, step="post", color="#22c55e", alpha=0.28)
    ax_bottom.fill_between(x, 0, 1 - is_focused, step="post", color="#f97316", alpha=0.22)
    ax_bottom.set_yticks([0, 1], labels=["Distracted", "Focused"])
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("State")
    ax_bottom.grid(alpha=0.18, linestyle="--")

    if not from_real_data:
        fig.text(
            0.01,
            0.01,
            "Note: illustrative timeline generated because frame-level session logs are not persisted in current artifacts.",
            fontsize=9,
            color="#374151",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    report_md = run_dir / "capstone_report.md"
    cm_png = run_dir / "plots" / "best_fold_confusion_matrix.png"
    if not report_md.exists():
        raise FileNotFoundError(f"Missing report markdown: {report_md}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = report_md.read_text(encoding="utf-8")
    table_rows = _parse_markdown_table_rows(markdown)
    best_fold = _extract_best_fold(markdown)

    participant_chart = out_dir / "loso_heldout_participant_performance.png"
    confusion_slide = out_dir / "best_fold_confusion_matrix_presentation.png"
    timeline_chart = out_dir / "session_timeline_confidence_state.png"

    plot_participant_performance(table_rows, participant_chart)
    plot_confusion_matrix_slide(cm_png, confusion_slide, best_fold=best_fold)

    if args.timeline_csv and args.timeline_csv.exists():
        timeline_df = _timeline_from_csv(args.timeline_csv.resolve())
        from_real_data = True
    else:
        timeline_df = _illustrative_timeline()
        from_real_data = False
    plot_session_timeline(timeline_df, timeline_chart, from_real_data=from_real_data)

    print("Generated charts:")
    print(f"- {participant_chart}")
    print(f"- {confusion_slide}")
    print(f"- {timeline_chart}")
    if not from_real_data:
        print(
            "Timeline note: generated illustrative timeline because no --timeline-csv was provided."
        )


if __name__ == "__main__":
    main()
