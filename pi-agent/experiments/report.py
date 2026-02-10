from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _top_confusions(cm: list[list[int]], labels: list[str], k: int = 8) -> list[dict[str, Any]]:
    pairs: list[tuple[int, str, str]] = []
    n = min(len(labels), len(cm))
    for i in range(n):
        row = cm[i]
        for j in range(min(n, len(row))):
            if i == j:
                continue
            c = int(row[j])
            if c > 0:
                pairs.append((c, labels[i], labels[j]))
    pairs.sort(reverse=True, key=lambda t: t[0])
    return [{"count": c, "true": t, "pred": p} for c, t, p in pairs[:k]]


def _extract_metrics(metrics: dict[str, Any]) -> tuple[float | None, float | None, list[dict[str, Any]]]:
    macro_f1 = metrics.get("macro_f1")
    acc = metrics.get("accuracy_from_cm") or metrics.get("accuracy")
    top_conf: list[dict[str, Any]] = []
    cm = None
    labels = None
    if isinstance(metrics.get("confusion_matrix"), dict):
        labels = metrics["confusion_matrix"].get("labels")
        cm = metrics["confusion_matrix"].get("matrix")
    if isinstance(labels, list) and isinstance(cm, list):
        top_conf = _top_confusions(cm, labels, k=8)
    return (float(macro_f1) if isinstance(macro_f1, (int, float)) else None, float(acc) if isinstance(acc, (int, float)) else None, top_conf)


@dataclass
class RunRecord:
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    metrics: dict[str, Any] | None
    macro_f1: float | None
    acc: float | None
    top_confusions: list[dict[str, Any]]


def _tag_base(tag: Any) -> str:
    """
    For LOPO runs we tag like: "lopo_baseline:Joshua (5/8)".
    Group by the prefix before the first colon.
    """
    if not isinstance(tag, str):
        return ""
    s = tag.strip()
    if ":" in s:
        return s.split(":", 1)[0].strip()
    return s


def _get_holdout_participant(cfg: dict[str, Any]) -> str | None:
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    if isinstance(data, dict):
        hp = data.get("holdoutParticipant")
        if isinstance(hp, str) and hp.strip():
            return hp.strip()
    return None


def _train_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Stable subset of training knobs that define an experiment identity.
    """
    tr = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    if not isinstance(tr, dict):
        tr = {}
    keys = [
        "backbone",
        "inputSize",
        "batchSize",
        "epochsHead",
        "epochsFinetune",
        "fineTuneAt",
        "labelSmoothing",
        "noAugment",
        "noClassWeights",
        "quantize",
        "seed",
    ]
    return {k: tr.get(k) for k in keys}


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(max(0.0, var))


def _per_class_recall(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
    if not isinstance(per_class, dict):
        return out
    for lab, d in per_class.items():
        if not isinstance(d, dict):
            continue
        rec = d.get("recall")
        if isinstance(rec, (int, float)):
            out[str(lab)] = float(rec)
    return out


@dataclass
class LopoAggregate:
    tag_base: str
    train_fingerprint: dict[str, Any]
    n: int
    macro_f1_mean: float
    macro_f1_std: float
    acc_mean: float
    acc_std: float
    recall_mean: dict[str, float]


def _iter_runs(pi_agent_dir: Path) -> list[RunRecord]:
    runs_dir = pi_agent_dir / "runs"
    if not runs_dir.exists():
        return []

    out: list[RunRecord] = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        run_id = run_dir.name
        cfg = _load_json(run_dir / "config.json")
        if not cfg:
            continue
        metrics_path = None
        artifacts = cfg.get("artifacts") if isinstance(cfg.get("artifacts"), dict) else {}
        if isinstance(artifacts, dict) and artifacts.get("metricsTestJson"):
            metrics_path = Path(str(artifacts["metricsTestJson"]))
        if metrics_path is None:
            metrics_path = run_dir / "artifacts" / "metrics_test.json"
        metrics = _load_json(metrics_path) if metrics_path.exists() else None
        macro_f1, acc, top_conf = (None, None, [])
        if metrics:
            macro_f1, acc, top_conf = _extract_metrics(metrics)
        out.append(
            RunRecord(
                run_id=run_id,
                run_dir=run_dir,
                config=cfg,
                metrics=metrics,
                macro_f1=macro_f1,
                acc=acc,
                top_confusions=top_conf,
            )
        )
    return out


def _pick_baseline(runs: list[RunRecord]) -> RunRecord | None:
    # Prefer an explicit tag containing "baseline"
    for r in runs:
        tag = r.config.get("tag")
        if isinstance(tag, str) and "baseline" in tag.lower():
            return r
    # Otherwise, prefer a common old default config (mobilenetv2 @ 224) if present.
    for r in runs:
        tr = r.config.get("train") if isinstance(r.config.get("train"), dict) else {}
        if isinstance(tr, dict) and tr.get("backbone") == "mobilenetv2" and int(tr.get("inputSize") or 0) == 224:
            return r
    # Otherwise, earliest run by createdAtUtc.
    def _ts(rr: RunRecord) -> str:
        return str(rr.config.get("createdAtUtc") or "")

    return sorted(runs, key=_ts)[0] if runs else None


def _compute_lopo_aggregates(runs: list[RunRecord]) -> list[LopoAggregate]:
    """
    Aggregate LOPO runs (runs with a holdoutParticipant) into groups with mean/std metrics.
    Groups are defined by: tag_base + training fingerprint.
    """
    buckets: dict[str, list[RunRecord]] = {}

    for r in runs:
        hp = _get_holdout_participant(r.config)
        if not hp:
            continue
        if r.macro_f1 is None or r.metrics is None:
            continue
        tag_base = _tag_base(r.config.get("tag"))
        fp = _train_fingerprint(r.config)
        key = json.dumps({"tagBase": tag_base, "train": fp}, sort_keys=True)
        buckets.setdefault(key, []).append(r)

    aggs: list[LopoAggregate] = []
    for _key, rs in buckets.items():
        # If the same holdout participant was run multiple times, keep the best macro_f1.
        best_by_holdout: dict[str, RunRecord] = {}
        for r in rs:
            hp = _get_holdout_participant(r.config) or r.run_id
            if hp not in best_by_holdout or (r.macro_f1 or 0.0) > (best_by_holdout[hp].macro_f1 or 0.0):
                best_by_holdout[hp] = r
        rs = list(best_by_holdout.values())
        if not rs:
            continue

        tag_base = _tag_base(rs[0].config.get("tag"))
        fp = _train_fingerprint(rs[0].config)

        macro_vals = [float(r.macro_f1 or 0.0) for r in rs if r.macro_f1 is not None]
        acc_vals = [float(r.acc or 0.0) for r in rs if r.acc is not None]
        macro_mean, macro_std = _mean_std(macro_vals)
        acc_mean, acc_std = _mean_std(acc_vals) if acc_vals else (0.0, 0.0)

        recall_lists: dict[str, list[float]] = {}
        for r in rs:
            rec = _per_class_recall(r.metrics or {})
            for lab, v in rec.items():
                recall_lists.setdefault(lab, []).append(float(v))
        recall_mean = {lab: _mean_std(vals)[0] for lab, vals in recall_lists.items() if vals}

        aggs.append(
            LopoAggregate(
                tag_base=tag_base,
                train_fingerprint=fp,
                n=len(rs),
                macro_f1_mean=macro_mean,
                macro_f1_std=macro_std,
                acc_mean=acc_mean,
                acc_std=acc_std,
                recall_mean=recall_mean,
            )
        )

    aggs.sort(key=lambda a: (-a.macro_f1_mean, a.macro_f1_std))
    return aggs


def _heuristic_suggestions(best: RunRecord, baseline: RunRecord | None) -> list[str]:
    sugg: list[str] = []
    m = best.metrics or {}
    per_class = m.get("per_class") if isinstance(m.get("per_class"), dict) else {}
    cm_obj = m.get("confusion_matrix") if isinstance(m.get("confusion_matrix"), dict) else {}
    labels = cm_obj.get("labels") if isinstance(cm_obj.get("labels"), list) else []
    cm = cm_obj.get("matrix") if isinstance(cm_obj.get("matrix"), list) else None

    # 1) Target worst recall class
    worst = None
    if isinstance(per_class, dict):
        for lab, d in per_class.items():
            if not isinstance(d, dict):
                continue
            rec = d.get("recall")
            if not isinstance(rec, (int, float)):
                continue
            if worst is None or rec < worst[1]:
                worst = (lab, float(rec))
    if worst:
        sugg.append(
            f"Worst recall is **{worst[0]}** (recall={worst[1]:.3f}). Collect more varied data for that class (lighting, glasses, camera distance) and/or increase `--input-size`."
        )

    # 2) Confusion-specific suggestion
    if isinstance(labels, list) and isinstance(cm, list) and len(labels) >= 5:
        top = _top_confusions(cm, labels, k=3)
        for item in top:
            t = item["true"]
            p = item["pred"]
            c = item["count"]
            if {t, p} == {"away_left", "away_right"}:
                sugg.append(
                    f"High left/right confusion (**{t}→{p}** count={c}). Avoid any mirroring, consider larger input size, and collect extra left/right examples with subtle head turns."
                )
            if {t, p} == {"away_up", "away_down"}:
                sugg.append(
                    f"High up/down confusion (**{t}→{p}** count={c}). Collect more up/down samples with consistent camera height and reduce face crop padding variability."
                )

    # 3) Config-oriented suggestions
    train_cfg = best.config.get("train") if isinstance(best.config.get("train"), dict) else {}
    if isinstance(train_cfg, dict):
        if float(train_cfg.get("labelSmoothing") or 0.0) > 0.06:
            sugg.append("Try reducing `--label-smoothing` (e.g. 0.02–0.05) if the model is under-confident.")
        sugg.append("Run 2–3 seeds for the top configuration and pick the most stable (macro-F1 variance).")

    # 4) Baseline delta hint
    if baseline and baseline.macro_f1 is not None and best.macro_f1 is not None:
        delta = best.macro_f1 - baseline.macro_f1
        sugg.append(f"Best vs baseline macro-F1 delta: `{delta:+.4f}`. Keep the winning change and sweep only one new knob next.")

    # Keep suggestions concise
    dedup = []
    seen = set()
    for s in sugg:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup[:8]


def _write_index_csv(pi_agent_dir: Path, runs: list[RunRecord]) -> Path:
    out_path = pi_agent_dir / "runs" / "index.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "createdAtUtc",
        "tag",
        "backbone",
        "inputSize",
        "seed",
        "macro_f1",
        "accuracy_from_cm",
        "dirty",
        "commit",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            cfg = r.config
            tr = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
            git = cfg.get("git") if isinstance(cfg.get("git"), dict) else {}
            row = {
                "run_id": r.run_id,
                "createdAtUtc": cfg.get("createdAtUtc"),
                "tag": cfg.get("tag"),
                "backbone": tr.get("backbone") if isinstance(tr, dict) else None,
                "inputSize": tr.get("inputSize") if isinstance(tr, dict) else None,
                "seed": tr.get("seed") if isinstance(tr, dict) else None,
                "macro_f1": _fmt(r.macro_f1),
                "accuracy_from_cm": _fmt(r.acc),
                "dirty": git.get("dirty") if isinstance(git, dict) else None,
                "commit": (git.get("commit") if isinstance(git, dict) else None),
            }
            w.writerow(row)
    return out_path


def _ai_ready_block(best: RunRecord, runners_up: list[RunRecord], baseline: RunRecord | None) -> str:
    # Build a compact JSON payload for copy/paste.
    def pack(r: RunRecord) -> dict[str, Any]:
        cfg = r.config
        train = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
        data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
        git = cfg.get("git") if isinstance(cfg.get("git"), dict) else {}
        packed = {
            "runId": r.run_id,
            "createdAtUtc": cfg.get("createdAtUtc"),
            "tag": cfg.get("tag"),
            "git": {"commit": git.get("commit"), "dirty": git.get("dirty"), "branch": git.get("branch")},
            "data": {"counts": data.get("counts"), "splitBy": data.get("splitBy"), "holdoutParticipant": data.get("holdoutParticipant")},
            "train": {
                "backbone": train.get("backbone"),
                "inputSize": train.get("inputSize"),
                "batchSize": train.get("batchSize"),
                "epochsHead": train.get("epochsHead"),
                "epochsFinetune": train.get("epochsFinetune"),
                "fineTuneAt": train.get("fineTuneAt"),
                "labelSmoothing": train.get("labelSmoothing"),
                "noAugment": train.get("noAugment"),
                "noClassWeights": train.get("noClassWeights"),
                "quantize": train.get("quantize"),
                "seed": train.get("seed"),
            },
            "metrics": r.metrics,
            "topConfusions": r.top_confusions,
        }
        return packed

    payload = {
        "best": pack(best),
        "runnersUp": [pack(r) for r in runners_up],
        "baseline": pack(baseline) if baseline else None,
    }

    prompt = (
        "You are helping improve a 5-way attention direction classifier "
        "(screen / away_left / away_right / away_up / away_down) for Raspberry Pi.\n"
        "Task:\n"
        "1) Identify the top 2 failure modes from per-class recall + confusion matrix.\n"
        "2) Propose the next 5 experiments as a ranked list. Each experiment must change only 1–2 knobs and include exact parameter values.\n"
        "3) Propose targeted data collection improvements tied to the worst confusions.\n"
        "4) If you propose code changes, specify which file(s) and what to change.\n"
    )

    return "\n".join(
        [
            "## AI-ready block (copy/paste)",
            "",
            "Paste the block below into your AI assistant:",
            "",
            "```",
            prompt.strip(),
            "",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


def _write_report_md(pi_agent_dir: Path, runs: list[RunRecord]) -> Path:
    out_path = pi_agent_dir / "runs" / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by macro_f1 desc, put missing metrics at bottom.
    runs_sorted = sorted(runs, key=lambda r: (r.macro_f1 is None, -(r.macro_f1 or 0.0)))
    best = next((r for r in runs_sorted if r.macro_f1 is not None), None)
    baseline = _pick_baseline(runs_sorted)
    runners_up = [r for r in runs_sorted if r.macro_f1 is not None and best and r.run_id != best.run_id][:2]

    lines = [
        "# Runs report",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}Z`",
        "",
    ]

    # LOPO aggregate section (if present)
    lopo_aggs = _compute_lopo_aggregates(runs)
    if lopo_aggs:
        lines += [
            "## LOPO aggregates (mean ± std across held-out participants)",
            "",
            "This is the best accuracy estimate for how the model generalizes to **new people**.",
            "",
            "| rank | tag_base | n | macro_f1 | acc | screen_recall | left | right | up | down | backbone | input | seed |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]

        def rmean(a: LopoAggregate, lab: str) -> str:
            v = a.recall_mean.get(lab)
            return f"{v:.3f}" if isinstance(v, (int, float)) else ""

        for i, a in enumerate(lopo_aggs[:10], start=1):
            fp = a.train_fingerprint
            lines.append(
                f"| {i} | `{a.tag_base}` | {a.n} | {a.macro_f1_mean:.4f} ± {a.macro_f1_std:.4f} | {a.acc_mean:.4f} ± {a.acc_std:.4f} | "
                + " | ".join(
                    [
                        rmean(a, "screen"),
                        rmean(a, "away_left"),
                        rmean(a, "away_right"),
                        rmean(a, "away_up"),
                        rmean(a, "away_down"),
                        str(fp.get("backbone") or ""),
                        str(fp.get("inputSize") or ""),
                        str(fp.get("seed") or ""),
                    ]
                )
                + " |"
            )
        lines += ["", "## Leaderboard (top 10)", ""]
    else:
        lines += ["## Leaderboard (top 10)", ""]

    lines += [
        "| rank | run_id | macro_f1 | acc | backbone | input | tag |",
        "| ---: | --- | ---: | ---: | --- | ---: | --- |",
    ]

    rank = 0
    for r in runs_sorted:
        if r.macro_f1 is None:
            continue
        rank += 1
        tr = r.config.get("train") if isinstance(r.config.get("train"), dict) else {}
        lines.append(
            f"| {rank} | `{r.run_id}` | {r.macro_f1:.4f} | {(_fmt(r.acc) or '')} | {tr.get('backbone')} | {tr.get('inputSize')} | {r.config.get('tag') or ''} |"
        )
        if rank >= 10:
            break

    lines += ["", "## Notes", ""]
    if not best:
        lines.append("No completed runs with metrics found yet. Run an experiment, then re-generate this report.")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    # Best run deep-dive
    lines += [f"### Best run: `{best.run_id}`", ""]
    lines.append(f"- macro_f1: `{_fmt(best.macro_f1)}`")
    lines.append(f"- acc_from_cm: `{_fmt(best.acc)}`")
    if baseline and baseline.run_id != best.run_id:
        if baseline.macro_f1 is not None and best.macro_f1 is not None:
            lines.append(f"- macro_f1 delta vs baseline `{baseline.run_id}`: `{best.macro_f1 - baseline.macro_f1:+.4f}`")

    lines += ["", "### Top confusions", ""]
    for item in best.top_confusions:
        lines.append(f"- **{item['true']} → {item['pred']}**: {item['count']}")

    # Suggestions
    lines += ["", "## Next suggestions (heuristics)", ""]
    for s in _heuristic_suggestions(best, baseline):
        lines.append(f"- {s}")

    # AI-ready block
    lines += ["", _ai_ready_block(best, runners_up=runners_up, baseline=baseline)]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    this_file = Path(__file__).resolve()
    pi_agent_dir = this_file.parents[1]

    runs = _iter_runs(pi_agent_dir)
    # Write CSV first so it exists even if report is sparse.
    _write_index_csv(pi_agent_dir, runs)
    _write_report_md(pi_agent_dir, runs)
    print(f"[report] Wrote {pi_agent_dir / 'runs' / 'index.csv'}")
    print(f"[report] Wrote {pi_agent_dir / 'runs' / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

