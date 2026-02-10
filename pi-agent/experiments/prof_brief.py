from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRow:
    run_id: str
    created_at: str
    tag: str
    backbone: str
    input_size: int
    seed: int
    macro_f1: float
    acc: float


def _load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_index(index_csv: Path) -> list[RunRow]:
    out: list[RunRow] = []
    with index_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.append(
                    RunRow(
                        run_id=r["run_id"],
                        created_at=r.get("createdAtUtc", ""),
                        tag=r.get("tag", ""),
                        backbone=r.get("backbone", ""),
                        input_size=int(r.get("inputSize") or 0),
                        seed=int(r.get("seed") or 0),
                        macro_f1=float(r.get("macro_f1") or 0.0),
                        acc=float(r.get("accuracy_from_cm") or 0.0),
                    )
                )
            except Exception:
                continue
    return out


def _tag_base(tag: str) -> str:
    if ":" in tag:
        return tag.split(":", 1)[0].strip()
    return tag.strip()


def _lopo_participant(tag: str) -> str | None:
    # lopo_from_xxx:Name (5/8)
    m = re.match(r"^[^:]+:([^(]+)\(\d+/\d+\)\s*$", tag.replace(" (", "("))
    return m.group(1).strip() if m else None


def _worst_participants(rows: list[RunRow]) -> list[tuple[str, float]]:
    vals: dict[str, list[float]] = {}
    for rr in rows:
        if not rr.tag.startswith("lopo_from_"):
            continue
        p = _lopo_participant(rr.tag)
        if not p:
            continue
        vals.setdefault(p, []).append(rr.macro_f1)
    agg = [(p, sum(v) / len(v)) for p, v in vals.items() if v]
    agg.sort(key=lambda t: t[1])
    return agg


def main() -> int:
    pi_agent_dir = Path(__file__).resolve().parents[1]
    runs_dir = pi_agent_dir / "runs"
    index_csv = runs_dir / "index.csv"
    report_md = runs_dir / "REPORT.md"

    rows = _read_index(index_csv)

    sweep = [r for r in rows if _tag_base(r.tag) == "sweep_auto"]
    sweep_best = max(sweep, key=lambda r: r.macro_f1, default=None)

    # Best single run overall (highest macro_f1)
    best_run = max(rows, key=lambda r: r.macro_f1, default=None)
    best_metrics = None
    best_cfg = None
    if best_run:
        best_cfg = _load_json(runs_dir / best_run.run_id / "config.json")
        best_metrics = _load_json(runs_dir / best_run.run_id / "artifacts" / "metrics_test.json")

    worst = _worst_participants(rows)[:4]

    now = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines += [
        "# Experiment brief (professor-friendly)",
        "",
        f"Generated at: `{now}`",
        "",
        "This file summarizes what I did, what results I got, and what I plan to do next.",
        "",
        "## Where the raw artifacts live",
        "",
        f"- Index (all runs): `{index_csv}`",
        f"- Report (LOPO aggregates + best run + AI-ready block): `{report_md}`",
        "",
    ]

    lines += [
        "## Summary of experiments so far",
        "",
        "### Experiment 1: sweep over model choices (random split test set)",
        "",
        "- Sweep grid: backbones × input sizes × seeds.",
        "- Purpose: quickly find promising configurations before expensive LOPO.",
        "",
    ]
    if sweep_best:
        lines += [
            f"- Best sweep run: `{sweep_best.run_id}`",
            f"  - cfg: `{sweep_best.backbone}` input `{sweep_best.input_size}` seed `{sweep_best.seed}`",
            f"  - macro_f1: `{sweep_best.macro_f1:.4f}`  acc: `{sweep_best.acc:.4f}`",
            "",
        ]
    else:
        lines += ["- No sweep runs found in `index.csv` (expected tag base: `sweep_auto`).", ""]

    lines += [
        "### Experiment 2: LOPO (leave-one-participant-out) evaluation",
        "",
        "- Purpose: estimate how well the model generalizes to **new people** (the real deployment scenario).",
        "- Method: train on 7 participants, test on 1 held-out participant, repeat for all participants; report mean ± std.",
        "",
        "The LOPO aggregate table is at the top of `runs/REPORT.md`.",
        "",
    ]

    lines += [
        "## Key findings",
        "",
        "- The dominant error mode is **left/right confusion**, specifically `away_right → away_left` in the best single run.",
        "- Participant difficulty varies a lot; the hardest holdouts (lowest mean macro-F1) are:",
    ]
    for p, m in worst:
        lines.append(f"  - `{p}`: mean macro_f1 ≈ `{m:.4f}`")
    lines += ["", "## Best single run deep-dive", ""]

    if best_run and best_metrics:
        lines += [
            f"- Best run_id: `{best_run.run_id}`",
            f"- tag: `{best_run.tag}`",
            f"- macro_f1: `{best_run.macro_f1:.4f}`  acc_from_cm: `{best_run.acc:.4f}`",
        ]
        if isinstance(best_cfg, dict):
            tr = best_cfg.get("train") if isinstance(best_cfg.get("train"), dict) else {}
            if isinstance(tr, dict):
                lines += [
                    f"- train cmd: `{tr.get('command','')}`",
                ]
        # Per-class recall summary
        per = best_metrics.get("per_class") if isinstance(best_metrics.get("per_class"), dict) else {}
        if isinstance(per, dict):
            lines += ["", "Per-class recall (best run):"]
            for lab in ["screen", "away_left", "away_right", "away_up", "away_down"]:
                d = per.get(lab)
                if isinstance(d, dict) and isinstance(d.get("recall"), (int, float)):
                    lines.append(f"- `{lab}` recall: `{float(d['recall']):.3f}`")
        lines += ["", "Top confusions (from confusion matrix):"]
        cm = best_metrics.get("confusion_matrix") if isinstance(best_metrics.get("confusion_matrix"), dict) else {}
        if isinstance(cm, dict):
            labels = cm.get("labels")
            mat = cm.get("matrix")
            if isinstance(labels, list) and isinstance(mat, list):
                # Compute top off-diagonal counts
                pairs = []
                for i, row in enumerate(mat):
                    if not isinstance(row, list):
                        continue
                    for j, c in enumerate(row):
                        if i == j:
                            continue
                        try:
                            c = int(c)
                        except Exception:
                            continue
                        if c > 0 and i < len(labels) and j < len(labels):
                            pairs.append((c, labels[i], labels[j]))
                pairs.sort(reverse=True)
                for c, t, p in pairs[:5]:
                    lines.append(f"- **{t} → {p}**: {c}")
    else:
        lines += ["No best run metrics found. (Expected: `runs/<run_id>/artifacts/metrics_test.json`)", ""]

    lines += [
        "",
        "## Plan for the next experiment (what I will do next)",
        "",
        "Goal: reduce `away_right ↔ away_left` confusion and improve the worst LOPO holdouts without deleting them.",
        "",
        "1) **Label audit for left/right** (high impact):",
        "   - Spot-check a small sample of `away_left` and `away_right` images for the hardest participants.",
        "   - Confirm the collector UI is not mirrored in a way that swaps semantic left/right.",
        "",
        "2) **Targeted data collection** (if labels are correct):",
        "   - Collect additional `away_right` examples across lighting and posture, especially for the hardest participants.",
        "",
        "3) **One controlled training change + LOPO**:",
        "   - Try lower label smoothing (often helps under-represented/confused classes):",
        "",
        "```bash",
        "cd pi-agent",
        "python experiments/all.py --lopo --mode one --prepare-splits --runs-dir data \\",
        "  --tag lopo_next_ls002 --backbone mobilenetv2 --input-size 224 --quantize --label-smoothing 0.02",
        "```",
        "",
        "Then review `runs/REPORT.md` (LOPO aggregates) and decide whether to keep the change.",
        "",
    ]

    out_path = runs_dir / "PROFESSOR_BRIEF.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[prof_brief] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

