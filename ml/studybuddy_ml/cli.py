from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from .pipeline import (
    aggregate_loso,
    build_manifest,
    export_best_model,
    generate_loso_splits,
    load_participant_map,
    train_loso,
    write_manifest_and_report,
)
from .capstone_report import generate_capstone_report

app = typer.Typer(
    help="StudyBuddy ML pipeline commands (validate, split, train, eval, export, serve).",
    no_args_is_help=True,
)


@app.command("validate")
def validate(
    dataset_root: Path = typer.Option(
        ...,
        "--dataset-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Dataset root containing run_*/meta.jsonl files.",
    ),
    manifest_out: Path = typer.Option(
        Path("artifacts/data/manifest.csv"),
        "--manifest-out",
        help="Output CSV manifest path.",
    ),
    report_out: Path = typer.Option(
        Path("artifacts/reports/data_validation.json"),
        "--report-out",
        help="Validation report output path.",
    ),
    participant_map: Path | None = typer.Option(
        None,
        "--participant-map",
        help="Optional YAML mapping participant names to stable IDs.",
    ),
) -> None:
    mapped = load_participant_map(participant_map)
    manifest, report = build_manifest(dataset_root=dataset_root, participant_map=mapped)
    write_manifest_and_report(manifest, report, manifest_out=manifest_out, report_out=report_out)
    typer.echo(f"[validate] Wrote manifest: {manifest_out}")
    typer.echo(f"[validate] Wrote report:   {report_out}")
    typer.echo(f"[validate] rows={report['rows_kept']} participants={len(report['participants'])}")


@app.command("split-loso")
def split_loso(
    manifest_csv: Path = typer.Option(
        Path("artifacts/data/manifest.csv"),
        "--manifest-csv",
        exists=True,
        help="Manifest CSV from the validate step.",
    ),
    participant_column: str = typer.Option(
        "participant_id",
        "--participant-column",
        help="Column used for group-based LOSO split.",
    ),
    split_out_dir: Path = typer.Option(
        Path("artifacts/splits"),
        "--split-out-dir",
        help="Output directory for fold_*.json split files.",
    ),
) -> None:
    import pandas as pd

    frame = pd.read_csv(manifest_csv)
    splits = generate_loso_splits(
        manifest=frame,
        participant_column=participant_column,
        split_out_dir=split_out_dir,
    )
    typer.echo(f"[split-loso] Generated {len(splits)} folds in {split_out_dir}")


@app.command("train-loso")
def train_loso_cmd(
    manifest_csv: Path = typer.Option(
        Path("artifacts/data/manifest.csv"),
        "--manifest-csv",
        exists=True,
        help="Manifest CSV from validate step.",
    ),
    split_dir: Path = typer.Option(
        Path("artifacts/splits"),
        "--split-dir",
        exists=True,
        file_okay=False,
        help="Directory with fold_*.json split files.",
    ),
    config_path: Path = typer.Option(
        Path("configs/baseline.yaml"),
        "--config-path",
        exists=True,
        help="Training config YAML path.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/training"),
        "--output-dir",
        help="Output directory for fold checkpoints and reports.",
    ),
    tracking_uri: str | None = typer.Option(
        None,
        "--tracking-uri",
        help="Optional MLflow tracking URI override.",
    ),
) -> None:
    summary_csv = train_loso(
        manifest_csv=manifest_csv,
        split_dir=split_dir,
        config_path=config_path,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
    )
    typer.echo(f"[train-loso] Completed LOSO. Summary CSV: {summary_csv}")


@app.command("eval-loso")
def eval_loso(
    summary_csv: Path = typer.Option(
        Path("artifacts/training/loso_summary.csv"),
        "--summary-csv",
        exists=True,
        help="Summary CSV produced by train-loso.",
    ),
    aggregate_out: Path = typer.Option(
        Path("artifacts/reports/loso_aggregate.json"),
        "--aggregate-out",
        help="Output aggregate report path.",
    ),
) -> None:
    aggregate = aggregate_loso(summary_csv=summary_csv, aggregate_out=aggregate_out)
    typer.echo(f"[eval-loso] Wrote aggregate report: {aggregate_out}")
    if "test_macro_f1" in aggregate["metrics"]:
        macro = aggregate["metrics"]["test_macro_f1"]
        typer.echo(
            "[eval-loso] test_macro_f1 "
            f"mean={macro['mean']:.4f} std={macro['std']:.4f}"
        )


@app.command("export-best")
def export_best(
    summary_csv: Path = typer.Option(
        Path("artifacts/training/loso_summary.csv"),
        "--summary-csv",
        exists=True,
        help="Summary CSV produced by train-loso.",
    ),
    export_dir: Path = typer.Option(
        Path("artifacts/export"),
        "--export-dir",
        help="Output directory for exported model artifacts.",
    ),
    criterion: str = typer.Option(
        "test_macro_f1",
        "--criterion",
        help="Metric column from summary CSV used to pick best fold.",
    ),
) -> None:
    meta = export_best_model(
        summary_csv=summary_csv,
        export_dir=export_dir,
        criterion=criterion,
    )
    typer.echo(f"[export-best] Exported model to: {export_dir}")
    typer.echo(
        f"[export-best] fold={meta['best_fold_id']} "
        f"{meta['criterion']}={meta['criterion_value']:.4f}"
    )


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
) -> None:
    uvicorn.run("studybuddy_ml.serve_api:app", host=host, port=port, reload=False)


@app.command("show-summary")
def show_summary(
    summary_csv: Path = typer.Option(
        Path("artifacts/training/loso_summary.csv"),
        "--summary-csv",
        exists=True,
        help="Summary CSV produced by train-loso.",
    ),
) -> None:
    import pandas as pd

    frame = pd.read_csv(summary_csv)
    payload = {
        "num_folds": int(frame.shape[0]),
        "best_test_macro_f1": float(frame["test_macro_f1"].max()),
        "mean_test_macro_f1": float(frame["test_macro_f1"].mean()),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("capstone-report")
def capstone_report(
    summary_csv: Path = typer.Option(
        Path("artifacts/training/loso_summary.csv"),
        "--summary-csv",
        exists=True,
        help="Summary CSV produced by train-loso.",
    ),
    folds_dir: Path = typer.Option(
        Path("artifacts/training/folds"),
        "--folds-dir",
        exists=True,
        file_okay=False,
        help="Directory containing fold_*/ artifacts (history, confusion matrix, etc.).",
    ),
    out_md: Path = typer.Option(
        Path("artifacts/reports/capstone_report.md"),
        "--out-md",
        help="Output markdown report path.",
    ),
    plots_dir: Path = typer.Option(
        Path("artifacts/reports/plots"),
        "--plots-dir",
        help="Output directory for PNG plots referenced in the report.",
    ),
    criterion: str = typer.Option(
        "test_macro_f1",
        "--criterion",
        help="Metric column from summary CSV used to pick the best fold.",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config-path",
        help="Optional training config YAML to embed in the report.",
    ),
    data_validation_json: Path | None = typer.Option(
        None,
        "--data-validation-json",
        help="Optional data validation JSON to embed in the report.",
    ),
) -> None:
    meta = generate_capstone_report(
        summary_csv=summary_csv,
        folds_dir=folds_dir,
        out_md=out_md,
        plots_dir=plots_dir,
        criterion=criterion,
        config_path=config_path,
        data_validation_json=data_validation_json,
    )
    typer.echo(f"[capstone-report] Wrote: {meta['out_md']}")
    typer.echo(f"[capstone-report] Plots: {meta['plots_dir']}")


if __name__ == "__main__":
    app()
