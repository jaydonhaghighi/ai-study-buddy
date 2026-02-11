from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _read_text_if_exists(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _to_float_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    return values


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def generate_capstone_report(
    *,
    summary_csv: Path,
    folds_dir: Path,
    out_md: Path,
    plots_dir: Path,
    criterion: str = "test_macro_f1",
    config_path: Path | None = None,
    data_validation_json: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    summary_csv = summary_csv.resolve()
    folds_dir = folds_dir.resolve()
    if run_dir is not None:
        run_dir = run_dir.resolve()
        out_md = (run_dir / "capstone_report.md").resolve()
        plots_dir = (run_dir / "plots").resolve()
    else:
        out_md = out_md.resolve()
        plots_dir = plots_dir.resolve()

    frame = pd.read_csv(summary_csv)
    if frame.empty:
        raise ValueError(f"Summary CSV is empty: {summary_csv}")
    if criterion not in frame.columns:
        raise KeyError(f"Criterion column '{criterion}' not found in {summary_csv}")

    criterion_values = _to_float_series(frame, criterion)
    if criterion_values.size == 0:
        raise ValueError(f"Criterion column '{criterion}' has no numeric values in {summary_csv}")

    best_idx = pd.to_numeric(frame[criterion], errors="coerce").idxmax()
    best_row = frame.loc[best_idx]
    best_fold_id = int(best_row["fold_id"])
    best_value = float(best_row[criterion])

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # Plot 1: bar chart of test_macro_f1 by fold (or the selected criterion).
    bar_png = plots_dir / f"{criterion}_by_fold.png"
    fold_ids = frame["fold_id"].astype(int).to_list()
    fold_scores = pd.to_numeric(frame[criterion], errors="coerce").fillna(0.0).to_list()

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar([str(x) for x in fold_ids], fold_scores, color="#111827")
    for i, b in enumerate(bars):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.005,
            f"{fold_scores[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylim(0.0, max(1.0, float(max(fold_scores) + 0.05)))
    plt.title(f"{criterion} by LOSO fold")
    plt.xlabel("Fold ID")
    plt.ylabel(criterion)
    plt.tight_layout()
    plt.savefig(bar_png, dpi=160)
    plt.close()

    # Plot 2: validation accuracy curves across folds (from fold history.json).
    val_acc_png = plots_dir / "val_accuracy_by_epoch.png"
    val_loss_png = plots_dir / "val_loss_by_epoch.png"

    histories: dict[int, dict[str, list[float]]] = {}
    for fold_id in fold_ids:
        fold_path = folds_dir / f"fold_{int(fold_id):02d}" / "history.json"
        if not fold_path.exists():
            continue
        payload = json.loads(fold_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        histories[int(fold_id)] = {
            k: [float(x) for x in v] for k, v in payload.items() if isinstance(v, list)
        }

    if histories:
        plt.figure(figsize=(9, 4.5))
        for fold_id, hist in histories.items():
            y = hist.get("val_accuracy")
            if not y:
                continue
            x = list(range(1, len(y) + 1))
            plt.plot(x, y, linewidth=2, label=f"fold {fold_id:02d}")
        plt.title("Validation accuracy over epochs (all folds)")
        plt.xlabel("Epoch")
        plt.ylabel("val_accuracy")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(val_acc_png, dpi=160)
        plt.close()

        plt.figure(figsize=(9, 4.5))
        for fold_id, hist in histories.items():
            y = hist.get("val_loss")
            if not y:
                continue
            x = list(range(1, len(y) + 1))
            plt.plot(x, y, linewidth=2, label=f"fold {fold_id:02d}")
        plt.title("Validation loss over epochs (all folds)")
        plt.xlabel("Epoch")
        plt.ylabel("val_loss")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(val_loss_png, dpi=160)
        plt.close()

    # Plot 3: copy best fold confusion matrix (already generated during training).
    best_cm_src = folds_dir / f"fold_{best_fold_id:02d}" / "test_confusion_matrix.png"
    best_cm_dst = plots_dir / "best_fold_confusion_matrix.png"
    if best_cm_src.exists():
        shutil.copy2(best_cm_src, best_cm_dst)

    # Aggregate metrics (numeric only) from summary CSV for quick reporting.
    numeric_metric_cols = [
        c for c in frame.columns if c.startswith("val_") or c.startswith("test_")
    ]
    numeric_metrics: dict[str, dict[str, float]] = {}
    for col in numeric_metric_cols:
        values = _to_float_series(frame, col)
        if values.size == 0:
            continue
        numeric_metrics[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    # Markdown table with core per-fold metrics
    table_cols = [
        "fold_id",
        "val_participants",
        "test_participants",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_accuracy",
    ]
    present_cols = [c for c in table_cols if c in frame.columns]

    header = "| " + " | ".join(present_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(present_cols)) + " |"
    rows: list[str] = []
    for _, r in frame.sort_values("fold_id").iterrows():
        cells: list[str] = []
        for c in present_cols:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                cells.append(f"{v:.4f}")
            else:
                cells.append(_md_escape(str(v)))
        rows.append("| " + " | ".join(cells) + " |")

    config_text = _read_text_if_exists(config_path)
    validation_text = _read_text_if_exists(data_validation_json)

    def metric_line(key: str) -> str | None:
        m = numeric_metrics.get(key)
        if not m:
            return None
        return f"- **{key}**: mean={m['mean']:.4f} std={m['std']:.4f} (min={m['min']:.4f}, max={m['max']:.4f})"

    metric_lines = [
        metric_line("test_macro_f1"),
        metric_line("test_balanced_accuracy"),
        metric_line("test_accuracy"),
        metric_line("val_macro_f1"),
        metric_line("val_accuracy"),
    ]
    metric_lines = [x for x in metric_lines if x]

    md = "\n".join(
        [
            "# StudyBuddy Head-Pose Model (LOSO) — Capstone Report",
            "",
            f"- **Generated at (UTC)**: {generated_at}",
            f"- **Summary CSV**: `{summary_csv}`",
            f"- **Folds dir**: `{folds_dir}`",
            f"- **Selection criterion**: `{criterion}`",
            f"- **Best fold**: `{best_fold_id:02d}` ({criterion}={best_value:.4f})",
            "",
            "## Key plots",
            "",
            f"- `{bar_png.name}`",
            f"- `{val_acc_png.name}`" if histories else "- (no history plots found)",
            f"- `{val_loss_png.name}`" if histories else "",
            f"- `{best_cm_dst.name}`" if best_cm_dst.exists() else "- (best confusion matrix not found)",
            "",
            f"![{criterion} by fold](plots/{bar_png.name})",
            "",
            "## Aggregate metrics (mean ± std across folds)",
            "",
            *metric_lines,
            "",
            "## Per-fold results",
            "",
            header,
            sep,
            *rows,
            "",
            "## Learning curves (validation)",
            "",
            (
                f"![val_accuracy_by_epoch](plots/{val_acc_png.name})\n\n"
                f"![val_loss_by_epoch](plots/{val_loss_png.name})"
            )
            if histories
            else "_No `history.json` files found under folds; skipping learning-curve plots._",
            "",
            "## Best fold confusion matrix",
            "",
            f"![best_fold_confusion_matrix](plots/{best_cm_dst.name})"
            if best_cm_dst.exists()
            else "_Confusion matrix image not found._",
            "",
            "## Training configuration snapshot",
            "",
            "```yaml\n" + config_text.strip() + "\n```" if config_text else "_No config file provided._",
            "",
            "## Data validation snapshot",
            "",
            "```json\n" + validation_text.strip() + "\n```"
            if validation_text
            else "_No data validation JSON provided._",
            "",
        ]
    ).strip() + "\n"

    out_md.write_text(md, encoding="utf-8")

    return {
        "out_md": str(out_md),
        "plots_dir": str(plots_dir),
        "best_fold_id": best_fold_id,
        "best_value": best_value,
        "criterion": criterion,
    }

