from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset local MLflow tracking (SQLite + artifact dir).")
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default: dry-run).")
    parser.add_argument(
        "--also-delete",
        default="",
        help="Comma-separated extra relative paths under pi-agent/ to delete (e.g. 'runs,mlruns').",
    )
    args = parser.parse_args(argv)

    pi_agent_dir = Path(__file__).resolve().parents[1]
    targets = [
        pi_agent_dir / "mlflow.db",
        pi_agent_dir / "mlflow_artifacts",
    ]

    extras = [p.strip() for p in (args.also_delete or "").split(",") if p.strip()]
    for rel in extras:
        targets.append((pi_agent_dir / rel).resolve())

    print("[reset_mlflow] Targets:")
    for t in targets:
        print(f"  - {t}")

    if not args.apply:
        print("\n[reset_mlflow] Dry-run only. Re-run with --apply to delete.")
        return 0

    for t in targets:
        # Safety: only allow deleting inside pi-agent/
        try:
            t.relative_to(pi_agent_dir)
        except Exception:
            raise SystemExit(f"Refusing to delete outside pi-agent/: {t}")
        _rm(t)

    print("\n[reset_mlflow] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

