from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


KEEP_ARTIFACT_FILES = {
    "metrics_test.json",
    "focus_model_labels.json",
}


def _load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name not in {".git"}])


def _run_sort_key(run_dir: Path) -> str:
    """
    Prefer config.json createdAtUtc, fallback to directory name.
    """
    cfg = _load_json(run_dir / "config.json")
    if isinstance(cfg, dict):
        ts = cfg.get("createdAtUtc")
        if isinstance(ts, str) and ts.strip():
            return ts
    return run_dir.name


def _bytes_human(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f}{u}"
        v /= 1024.0
    return f"{n}B"


def _tree_size_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        try:
            return p.stat().st_size
        except Exception:
            return 0
    total = 0
    for child in p.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except Exception:
                pass
    return total


@dataclass
class DeleteAction:
    path: Path
    bytes: int
    reason: str


def _collect_artifact_deletes(run_dir: Path) -> list[DeleteAction]:
    actions: list[DeleteAction] = []
    art = run_dir / "artifacts"
    if not art.exists() or not art.is_dir():
        return actions

    for p in sorted(art.iterdir()):
        if p.is_file() and p.name in KEEP_ARTIFACT_FILES:
            continue
        # Keep small metrics-like json by default
        if p.is_file() and p.suffix.lower() == ".json" and p.stat().st_size < 2_000_000:
            continue
        size = _tree_size_bytes(p)
        actions.append(DeleteAction(path=p, bytes=size, reason="artifact_prune"))
    return actions


def _collect_run_dir_deletes(run_dirs: list[Path], *, keep: int, keep_run_ids: set[str]) -> list[DeleteAction]:
    # Sort oldest->newest then delete oldest beyond keep
    sorted_dirs = sorted(run_dirs, key=_run_sort_key)
    # Exclude explicitly kept
    filtered = [d for d in sorted_dirs if d.name not in keep_run_ids]
    # Keep the newest N from filtered
    to_delete = filtered[: max(0, len(filtered) - keep)]
    actions: list[DeleteAction] = []
    for d in to_delete:
        actions.append(DeleteAction(path=d, bytes=_tree_size_bytes(d), reason=f"prune_runs_keep_{keep}"))
    return actions


def _collect_pyc_deletes(*, roots: list[Path]) -> list[DeleteAction]:
    actions: list[DeleteAction] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("__pycache__"):
            if p.is_dir():
                actions.append(DeleteAction(path=p, bytes=_tree_size_bytes(p), reason="pycache"))
        for p in root.rglob("*.pyc"):
            if p.is_file():
                actions.append(DeleteAction(path=p, bytes=_tree_size_bytes(p), reason="pyc"))
    return actions


def _print_actions(actions: Iterable[DeleteAction]) -> None:
    total = 0
    for a in actions:
        total += a.bytes
        print(f"- {a.path}  ({_bytes_human(a.bytes)})  [{a.reason}]")
    print(f"\nTotal reclaimable (approx): {_bytes_human(total)}")


def _apply_actions(actions: list[DeleteAction]) -> None:
    for a in actions:
        p = a.path
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cleanup helper for pi-agent runs/artifacts.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Default is dry-run (preview only).",
    )
    parser.add_argument(
        "--prune-artifacts",
        action="store_true",
        help="Delete heavy files under runs/<run_id>/artifacts (keeps metrics_test.json + labels).",
    )
    parser.add_argument(
        "--prune-runs",
        action="store_true",
        help="Delete whole run folders under runs/ (use --keep).",
    )
    parser.add_argument("--keep", type=int, default=5, help="When pruning runs, keep the newest N (default: 5).")
    parser.add_argument(
        "--keep-run-ids",
        default="",
        help="Comma-separated run_ids to never delete (even when pruning runs).",
    )
    parser.add_argument(
        "--pycache",
        action="store_true",
        help="Delete __pycache__/ and *.pyc under pi-agent source folders (excludes .venv).",
    )
    parser.add_argument(
        "--pycache-all",
        action="store_true",
        help="Delete __pycache__/ and *.pyc under all of pi-agent/ (includes .venv).",
    )
    args = parser.parse_args(argv)

    pi_agent_dir = Path(__file__).resolve().parents[1]
    runs_dir = pi_agent_dir / "runs"

    keep_run_ids = {s.strip() for s in (args.keep_run_ids or "").split(",") if s.strip()}

    actions: list[DeleteAction] = []
    run_dirs = _iter_run_dirs(runs_dir)

    if args.prune_artifacts:
        for rd in run_dirs:
            actions.extend(_collect_artifact_deletes(rd))

    if args.prune_runs:
        actions.extend(_collect_run_dir_deletes(run_dirs, keep=int(args.keep), keep_run_ids=keep_run_ids))

    if args.pycache or args.pycache_all:
        if args.pycache_all:
            roots = [pi_agent_dir]
        else:
            roots = [pi_agent_dir / "studybuddy_pi", pi_agent_dir / "train", pi_agent_dir / "experiments"]
        actions.extend(_collect_pyc_deletes(roots=roots))

    if not actions:
        print("Nothing to do. Try --prune-artifacts and/or --prune-runs and/or --pycache.")
        return 0

    # De-dupe (prefer deleting parent dir over children)
    unique: dict[Path, DeleteAction] = {}
    for a in actions:
        unique[a.path] = a
    actions = sorted(unique.values(), key=lambda a: str(a.path))

    if not args.apply:
        print("Dry-run. Actions that would be performed:\n")
        _print_actions(actions)
        print("\nRe-run with --apply to actually delete.")
        return 0

    print("Applying deletes:\n")
    _print_actions(actions)
    _apply_actions(actions)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

