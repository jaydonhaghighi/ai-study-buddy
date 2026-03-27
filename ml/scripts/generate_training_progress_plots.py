#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AGG_LINE_RE = re.compile(
    r"- \*\*(?P<metric>[a-z0-9_]+)\*\*: "
    r"mean=(?P<mean>[0-9.]+) std=(?P<std>[0-9.]+) "
    r"\(min=(?P<min>[0-9.]+), max=(?P<max>[0-9.]+)\)"
)
BEST_FOLD_RE = re.compile(
    r"- \*\*Best fold\*\*: `(?P<fold_id>\d+)` "
    r"\((?P<criterion>[a-z0-9_]+)=(?P<score>[0-9.]+)\)"
)


@dataclass
class RunSummary:
    run_id: str
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    best_fold_id: int | None
    best_fold_criterion: str | None
    best_fold_score: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ML training-progress charts from archived capstone report markdown."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("ml/artifacts/reports/runs"),
        help="Directory containing run folders with capstone_report.md",
    )
    parser.add_argument(
        "--selected-run",
        type=str,
        default="20260213-160744",
        help="Run id to use for fold-level charts",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ml/artifacts/reports/presentation/training_progress"),
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--exclude-run-ids",
        type=str,
        default="",
        help="Comma-separated run ids to exclude (e.g., 20260213-030358)",
    )
    parser.add_argument(
        "--exclude-experiment-labels",
        type=str,
        default="",
        help="Comma-separated experiment labels to exclude (e.g., experiment-11)",
    )
    return parser.parse_args()


def parse_fold_table(markdown: str) -> pd.DataFrame:
    lines = markdown.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("| fold_id |"):
            header_idx = i
            break
    if header_idx < 0:
        raise ValueError("Could not find per-fold table in report markdown")

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
        raise ValueError("Per-fold table was found but contains no rows")

    frame = pd.DataFrame(rows)
    numeric_cols = [
        "fold_id",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_accuracy",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["fold_id", "test_macro_f1", "test_balanced_accuracy", "test_accuracy"])
    frame["fold_id"] = frame["fold_id"].astype(int)
    return frame.sort_values("fold_id").reset_index(drop=True)


def parse_run_summary(run_id: str, markdown: str) -> RunSummary:
    metrics_mean: dict[str, float] = {}
    metrics_std: dict[str, float] = {}
    for match in AGG_LINE_RE.finditer(markdown):
        metric = match.group("metric")
        metrics_mean[metric] = float(match.group("mean"))
        metrics_std[metric] = float(match.group("std"))

    best_match = BEST_FOLD_RE.search(markdown)
    if best_match:
        best_fold_id = int(best_match.group("fold_id"))
        best_criterion = best_match.group("criterion")
        best_score = float(best_match.group("score"))
    else:
        best_fold_id = None
        best_criterion = None
        best_score = None

    return RunSummary(
        run_id=run_id,
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
        best_fold_id=best_fold_id,
        best_fold_criterion=best_criterion,
        best_fold_score=best_score,
    )


def load_all_run_summaries(runs_root: Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        report_path = run_dir / "capstone_report.md"
        if not report_path.exists():
            continue
        markdown = report_path.read_text(encoding="utf-8")
        summaries.append(parse_run_summary(run_dir.name, markdown))
    if not summaries:
        raise ValueError(f"No run summaries found under {runs_root}")
    return summaries


def load_selected_fold_frame(runs_root: Path, run_id: str) -> pd.DataFrame:
    report_path = runs_root / run_id / "capstone_report.md"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing selected run report: {report_path}")
    return parse_fold_table(report_path.read_text(encoding="utf-8"))


def build_experiment_label_map(run_ids: list[str]) -> dict[str, str]:
    unique_sorted = sorted(set(run_ids))
    return {
        run_id: f"experiment-{index:02d}"
        for index, run_id in enumerate(unique_sorted, start=1)
    }


def _parse_csv_arg(value: str) -> set[str]:
    return {token.strip() for token in value.split(",") if token.strip()}


def plot_test_metrics_by_fold(frame: pd.DataFrame, out_path: Path, run_label: str) -> None:
    x = np.arange(len(frame))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics = [
        ("test_macro_f1", "Macro F1", "#0f172a"),
        ("test_balanced_accuracy", "Balanced Accuracy", "#2563eb"),
        ("test_accuracy", "Accuracy", "#16a34a"),
    ]
    for idx, (col, label, color) in enumerate(metrics):
        offset = (idx - 1) * width
        ax.bar(x + offset, frame[col], width=width, label=label, color=color, alpha=0.92)

    ax.set_xticks(x, [str(fid) for fid in frame["fold_id"]])
    ax.set_xlabel("LOSO fold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Test Metrics by Fold ({run_label})")
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fold_metric_distribution(frame: pd.DataFrame, out_path: Path, run_label: str) -> None:
    cols = ["test_macro_f1", "test_balanced_accuracy", "test_accuracy"]
    labels = ["Macro F1", "Balanced Accuracy", "Accuracy"]
    data = [frame[c].to_numpy() for c in cols]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.35)
        body.set_facecolor("#0f172a")
    parts["cmeans"].set_color("#dc2626")
    parts["cmedians"].set_color("#16a34a")

    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Fold-to-Fold Metric Distribution ({run_label})")
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def summaries_to_frame(summaries: list[RunSummary]) -> pd.DataFrame:
    rows: list[dict[str, float | str | None]] = []
    for summary in summaries:
        row: dict[str, float | str | None] = {
            "run_id": summary.run_id,
            "best_fold_id": summary.best_fold_id,
            "best_fold_score": summary.best_fold_score,
        }
        for metric, value in summary.metrics_mean.items():
            row[f"{metric}_mean"] = value
        for metric, value in summary.metrics_std.items():
            row[f"{metric}_std"] = value
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    return frame


def plot_run_metric_trends(run_frame: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(run_frame))
    labels = run_frame["experiment_label"].tolist()
    fig, ax = plt.subplots(figsize=(13, 6))

    metric_specs = [
        ("test_macro_f1_mean", "Test Macro F1", "#0f172a"),
        ("val_macro_f1_mean", "Val Macro F1", "#2563eb"),
        ("test_accuracy_mean", "Test Accuracy", "#16a34a"),
        ("val_accuracy_mean", "Val Accuracy", "#7c3aed"),
    ]
    for col, label, color in metric_specs:
        if col not in run_frame.columns:
            continue
        ax.plot(x, run_frame[col], marker="o", linewidth=2, label=label, color=color)

    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean score")
    ax.set_title("Run-to-Run Mean Validation/Test Metric Trends")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_generalization_gap_trend(run_frame: pd.DataFrame, out_path: Path) -> None:
    required_cols = ["val_macro_f1_mean", "test_macro_f1_mean", "val_accuracy_mean", "test_accuracy_mean"]
    missing = [c for c in required_cols if c not in run_frame.columns]
    if missing:
        raise ValueError(f"Missing columns for generalization-gap plot: {missing}")

    x = np.arange(len(run_frame))
    # Use absolute gap magnitude so the chart focuses on mismatch size
    # rather than sign direction.
    gap_macro = (run_frame["val_macro_f1_mean"] - run_frame["test_macro_f1_mean"]).abs()
    gap_acc = (run_frame["val_accuracy_mean"] - run_frame["test_accuracy_mean"]).abs()

    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.38
    ax.bar(x - width / 2, gap_macro, width=width, label="|Val-Test| Macro F1 gap", color="#0f172a", alpha=0.9)
    ax.bar(x + width / 2, gap_acc, width=width, label="|Val-Test| Accuracy gap", color="#2563eb", alpha=0.9)
    ax.axhline(0.0, color="#374151", linewidth=1.1)
    ax.set_xticks(x, run_frame["experiment_label"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Absolute gap")
    ax.set_title("Generalization Gap Magnitude by Run")
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_best_vs_mean_test_macro_f1(run_frame: pd.DataFrame, out_path: Path) -> None:
    if "test_macro_f1_mean" not in run_frame.columns:
        raise ValueError("Missing test_macro_f1_mean for best-vs-mean plot")

    x = np.arange(len(run_frame))
    labels = run_frame["experiment_label"].tolist()
    mean_vals = run_frame["test_macro_f1_mean"].to_numpy(dtype=float)
    best_vals = pd.to_numeric(run_frame["best_fold_score"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, mean_vals, marker="o", linewidth=2.4, color="#0f172a", label="Mean test macro F1")
    ax.plot(x, best_vals, marker="D", linewidth=2.0, color="#dc2626", label="Best-fold test macro F1")
    ax.fill_between(x, mean_vals, best_vals, color="#fca5a5", alpha=0.23)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Best-Fold vs Mean Test Macro F1 by Run")
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_all_run_summaries(runs_root)
    full_run_ids = [s.run_id for s in summaries]
    experiment_label_map = build_experiment_label_map(full_run_ids)

    exclude_run_ids = _parse_csv_arg(args.exclude_run_ids)
    exclude_labels = _parse_csv_arg(args.exclude_experiment_labels)
    if exclude_labels:
        for run_id, label in experiment_label_map.items():
            if label in exclude_labels:
                exclude_run_ids.add(run_id)

    if args.selected_run in exclude_run_ids:
        selected_label = experiment_label_map.get(args.selected_run, args.selected_run)
        raise ValueError(
            f"Selected run {args.selected_run} ({selected_label}) is excluded. "
            "Choose a different selected run or remove it from exclusions."
        )

    included_summaries = [summary for summary in summaries if summary.run_id not in exclude_run_ids]
    if not included_summaries:
        raise ValueError("All runs were excluded; nothing to plot.")

    selected_fold_frame = load_selected_fold_frame(runs_root, args.selected_run)
    run_frame = summaries_to_frame(included_summaries)
    run_frame["experiment_label"] = run_frame["run_id"].map(experiment_label_map)
    selected_run_label = experiment_label_map.get(args.selected_run, args.selected_run)

    outputs = [
        out_dir / "test_metrics_by_fold_grouped.png",
        out_dir / "fold_metric_distribution.png",
        out_dir / "run_metric_trends_val_vs_test.png",
        out_dir / "run_generalization_gap_trend.png",
        out_dir / "run_best_vs_mean_test_macro_f1.png",
    ]
    mapping_csv = out_dir / "experiment_label_mapping.csv"

    plot_test_metrics_by_fold(selected_fold_frame, outputs[0], run_label=selected_run_label)
    plot_fold_metric_distribution(selected_fold_frame, outputs[1], run_label=selected_run_label)
    plot_run_metric_trends(run_frame, outputs[2])
    plot_generalization_gap_trend(run_frame, outputs[3])
    plot_best_vs_mean_test_macro_f1(run_frame, outputs[4])
    mapping_rows = []
    for run_id, label in sorted(experiment_label_map.items(), key=lambda row: row[1]):
        mapping_rows.append(
            {
                "experiment_label": label,
                "run_id": run_id,
                "included_in_charts": run_id not in exclude_run_ids,
            }
        )
    pd.DataFrame(mapping_rows).to_csv(mapping_csv, index=False)

    print("Generated training-progress plots:")
    for path in outputs:
        print(f"- {path}")
    print(f"- {mapping_csv}")


if __name__ == "__main__":
    main()
