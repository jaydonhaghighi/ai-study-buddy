from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def default_tracking_uri(pi_agent_dir: Path) -> str:
    # Absolute sqlite path (works regardless of cwd).
    db_path = (pi_agent_dir / "mlflow.db").resolve()
    return f"sqlite:///{db_path}"


def default_artifact_root(pi_agent_dir: Path) -> str:
    # Keep artifacts as files in the repo under pi-agent/.
    root = (pi_agent_dir / "mlflow_artifacts").resolve()
    return f"file:{root}"


def configure_mlflow(*, pi_agent_dir: Path, experiment_name: str) -> Any:
    """
    Configure MLflow to use a local SQLite backend by default.
    Returns the imported mlflow module.
    """
    import mlflow  # local import to avoid hard dependency at import time

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri(pi_agent_dir)
    mlflow.set_tracking_uri(tracking_uri)

    # Ensure experiment exists with a stable artifact root
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT") or default_artifact_root(pi_agent_dir)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None and getattr(exp, "lifecycle_stage", None) == "deleted":
        # If the experiment was deleted, restore it so we can keep a stable name.
        from mlflow.tracking import MlflowClient

        MlflowClient().restore_experiment(exp.experiment_id)
        exp = mlflow.get_experiment_by_name(experiment_name)

    if exp is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_root)
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_dict_as_json(mlflow: Any, data: dict, artifact_file: str) -> None:
    import json

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / artifact_file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(p), artifact_path=str(Path(artifact_file).parent) if "/" in artifact_file else None)


def log_text(mlflow: Any, text: str, artifact_file: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / artifact_file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        mlflow.log_artifact(str(p), artifact_path=str(Path(artifact_file).parent) if "/" in artifact_file else None)

