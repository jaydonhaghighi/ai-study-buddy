from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"[label_quarantine] Expected JSON object in {path}")
    return data


def _norm_reasons(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def label_quarantine(
    *,
    report_json: Path,
    quarantine_dir: Path,
    out_csv: Path,
    write_sidecars: bool,
    only_if_exists: bool,
) -> dict[str, Any]:
    report = _load_json(report_json)
    summary = report.get("summary")
    items = report.get("items")
    if not isinstance(summary, dict) or not isinstance(items, list):
        raise SystemExit("[label_quarantine] report.json missing expected `summary` or `items` keys.")

    runs_dir = summary.get("runs_dir")
    if not isinstance(runs_dir, str) or not runs_dir.strip():
        raise SystemExit("[label_quarantine] Could not resolve `summary.runs_dir` from report.json.")
    runs_root = Path(runs_dir).expanduser().resolve()
    qroot = quarantine_dir.expanduser().resolve()

    rows: list[dict[str, Any]] = []
    wrote_sidecars = 0
    missing_files = 0
    total_flagged = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("status")) != "flagged":
            continue
        total_flagged += 1

        src_str = item.get("path")
        if not isinstance(src_str, str) or not src_str.strip():
            continue
        src = Path(src_str).expanduser().resolve()
        reasons = _norm_reasons(item.get("reasons"))
        reason_text = ", ".join(reasons) if reasons else "unknown"

        quarantine_path: Path | None
        try:
            rel = src.relative_to(runs_root)
            quarantine_path = qroot / rel
        except Exception:
            quarantine_path = None

        exists = bool(quarantine_path and quarantine_path.exists())
        if only_if_exists and not exists:
            missing_files += 1
            continue

        rows.append(
            {
                "source_path": str(src),
                "quarantine_path": str(quarantine_path) if quarantine_path else "",
                "exists_in_quarantine": exists,
                "participant": str(item.get("participant", "")),
                "raw_label": str(item.get("raw_label", "")),
                "effective_label": str(item.get("effective_label", "")),
                "status": "flagged",
                "reasons": "|".join(reasons),
                "blur_score": item.get("blur_score"),
                "yaw_deg": item.get("yaw_deg"),
                "pitch_deg": item.get("pitch_deg"),
                "predicted_label_from_pose": item.get("predicted_label_from_pose"),
            }
        )

        if write_sidecars and quarantine_path and exists:
            sidecar = quarantine_path.with_suffix(quarantine_path.suffix + ".why.txt")
            lines = [
                f"source_path: {src}",
                f"quarantine_path: {quarantine_path}",
                f"participant: {item.get('participant', '')}",
                f"raw_label: {item.get('raw_label', '')}",
                f"effective_label: {item.get('effective_label', '')}",
                f"status: flagged",
                f"reasons: {reason_text}",
                f"blur_score: {item.get('blur_score')}",
                f"yaw_deg: {item.get('yaw_deg')}",
                f"pitch_deg: {item.get('pitch_deg')}",
                f"predicted_label_from_pose: {item.get('predicted_label_from_pose')}",
            ]
            sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")
            wrote_sidecars += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "quarantine_path",
                "exists_in_quarantine",
                "participant",
                "raw_label",
                "effective_label",
                "status",
                "reasons",
                "blur_score",
                "yaw_deg",
                "pitch_deg",
                "predicted_label_from_pose",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return {
        "report_json": str(report_json),
        "quarantine_dir": str(qroot),
        "out_csv": str(out_csv),
        "total_flagged_in_report": int(total_flagged),
        "rows_written": int(len(rows)),
        "missing_quarantine_files": int(missing_files),
        "sidecars_written": int(wrote_sidecars),
        "write_sidecars": bool(write_sidecars),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write quarantine reason labels from clean_dataset report.json.")
    parser.add_argument("--report-json", default="artifacts/data_clean/report.json")
    parser.add_argument("--quarantine-dir", default="artifacts/data_clean/quarantine")
    parser.add_argument("--out-csv", default="artifacts/data_clean/quarantine_labels.csv")
    parser.add_argument("--no-sidecars", action="store_true", help="Only write CSV; do not write *.why.txt files.")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include rows even when mapped quarantine file is missing.",
    )
    args = parser.parse_args(argv)

    pi_agent_dir = Path(__file__).resolve().parents[1]
    report_json = (pi_agent_dir / args.report_json).resolve() if not Path(args.report_json).is_absolute() else Path(args.report_json).resolve()
    quarantine_dir = (pi_agent_dir / args.quarantine_dir).resolve() if not Path(args.quarantine_dir).is_absolute() else Path(args.quarantine_dir).resolve()
    out_csv = (pi_agent_dir / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv).resolve()

    summary = label_quarantine(
        report_json=report_json,
        quarantine_dir=quarantine_dir,
        out_csv=out_csv,
        write_sidecars=not bool(args.no_sidecars),
        only_if_exists=not bool(args.include_missing),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

