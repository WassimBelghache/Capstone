from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from drone_mocap.pipeline.run import run_pipeline

app = typer.Typer(no_args_is_help=True)
console = Console()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@app.command("analyze")
def analyze(
    video: Path = typer.Option(..., exists=True, help="Path to sagittal video (mp4/mov)"),
    out_dir: Path = typer.Option(Path("runs"), help="Output directory"),
    mocap_txt: Optional[Path] = typer.Option(None, help="Optional MoCap / SPT ground-truth angles file"),
    side: str = typer.Option("right", help="Visible side: 'right' or 'left'"),
    max_frames: int = typer.Option(0, help="0 = all frames; >0 limits for quick tests"),
    cutoff_hz: float = typer.Option(6.0, help="Butterworth cutoff Hz: 6=walk, 10=jog, 12=sprint"),
    filter_order: int = typer.Option(4, help="Butterworth filter order (default 4)"),
    min_vis: float = typer.Option(0.3, help="Keypoint visibility anchor threshold (0–1)"),
    athlete_height_m: Optional[float] = typer.Option(
        None, help="Athlete height in metres — enables pixel→m/s velocity output",
    ),
    no_diagnostic: bool = typer.Option(
        False, "--no-diagnostic",
        help="Skip the annotated diagnostic video render (faster).",
    ),
):
    """
    Full kinematic analysis of a sagittal-view video (v1.2.0).

    Pipeline stages:
      1. MediaPipe pose estimation
      2. Confidence-weighted cubic spline gap-filling
      3. Zero-phase dual-pass Butterworth filter (FPS-aware cutoff)
      4. Signed sagittal joint angles (hip / knee / ankle)
      5. Anatomical range-of-motion outlier clamping
      6. Optional pixel→m/s velocity scaling
      7. Optional colour-coded diagnostic video overlay
    """
    out = run_pipeline(
        video=video,
        out_root=out_dir,
        mocap_txt=mocap_txt,
        visible_side=side,
        max_frames=max_frames,
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        min_vis=min_vis,
        athlete_height_m=athlete_height_m,
        diagnostic_video=not no_diagnostic,
    )
    print(f"[green]Done.[/green] Outputs in: {out}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@app.command("evaluate")
def evaluate(
    mocap: Path = typer.Option(
        ..., exists=True,
        help="Path to MoCap / SPT reference file (.csv or .txt). "
             "CSV format (M_Treadmill_Jogging.angles.csv) is detected automatically.",
    ),
    run_dir: Optional[Path] = typer.Option(
        None,
        help="Path to a previous 'analyze' run directory (contains derived/angles_sagittal.parquet). "
             "If omitted, --angles-file must be provided.",
    ),
    angles_file: Optional[Path] = typer.Option(
        None, exists=True,
        help="Direct path to an angles_sagittal.parquet or .csv file. "
             "Alternative to --run-dir.",
    ),
    side: str = typer.Option("right", help="Visible side used during analysis: 'right' or 'left'"),
    out_dir: Optional[Path] = typer.Option(
        None,
        help="Directory to write comparison outputs. "
             "Defaults to <run_dir>/reports or alongside --angles-file.",
    ),
):
    """
    Compare previously computed video angles against a MoCap / SPT reference.

    Produces:
      • reports/metrics_mocap.json   — RMSE, MAE, Pearson-r per joint
      • reports/compare_mocap.csv    — aligned signal table
      • reports/comparison_plot.png  — 3-subplot figure (hip / knee / ankle)

    Example usage:
      poetry run drone-mocap evaluate \\
          --mocap data/M_Treadmill_Jogging.angles.csv \\
          --run-dir runs/20240315_103200
    """
    import pandas as pd
    from drone_mocap.io.mocap_txt import read_mocap_angles_txt
    from drone_mocap.evaluation.compare_mocap import compare_video_to_mocap, save_compare_outputs

    # --- Resolve angles source ---
    if run_dir is not None:
        parquet_path = run_dir / "derived" / "angles_sagittal.parquet"
        csv_path     = run_dir / "derived" / "angles_sagittal.csv"
        if parquet_path.exists():
            df_angles = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df_angles = pd.read_csv(csv_path)
        else:
            print(f"[red]No angles file found in {run_dir}/derived/[/red]")
            raise typer.Exit(1)
        reports_dir = out_dir or (run_dir / "reports")

    elif angles_file is not None:
        if angles_file.suffix == ".parquet":
            df_angles = pd.read_parquet(angles_file)
        else:
            df_angles = pd.read_csv(angles_file)
        reports_dir = out_dir or angles_file.parent / "reports"

    else:
        print("[red]Provide either --run-dir or --angles-file.[/red]")
        raise typer.Exit(1)

    # --- Load MoCap reference ---
    print(f"[cyan]Loading MoCap reference:[/cyan] {mocap}")
    df_mocap = read_mocap_angles_txt(mocap)
    print(f"  → {len(df_mocap)} rows, columns: {list(df_mocap.columns)}")

    # --- Run comparison ---
    print("[cyan]Running comparison…[/cyan]")
    result = compare_video_to_mocap(df_angles, df_mocap, visible_side=side)

    # --- Save outputs ---
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_compare_outputs(result, reports_dir)

    # --- Print summary table ---
    table = Table(title="Kinematic Comparison Metrics", show_lines=True)
    table.add_column("Joint",       style="bold")
    table.add_column("Matched col", style="dim")
    table.add_column("RMSE (°)",    justify="right")
    table.add_column("MAE (°)",     justify="right")
    table.add_column("Pearson r",   justify="right")

    for j in ["HIP", "KNEE", "ANKLE"]:
        m = result.metrics.get(j, {})
        rmse = m.get("rmse", float("nan"))
        mae  = m.get("mae",  float("nan"))
        corr = m.get("corr", float("nan"))
        col  = m.get("matched_col") or result.axis_choice.get(j) or "—"

        def fmt(v: float) -> str:
            return f"{v:.3f}" if not (v != v) else "—"   # nan check

        table.add_row(j, str(col), fmt(rmse), fmt(mae), fmt(corr))

    console.print(table)
    shift = result.metrics.get("_time_shift_frames", 0)
    print(f"\n[dim]Time alignment shift: {shift} frames[/dim]")
    print(f"[green]Outputs written to:[/green] {reports_dir}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@app.command("doctor")
def doctor():
    """Print import / environment debug info."""
    import drone_mocap
    import drone_mocap.cli as cli_mod
    print("[cyan]drone_mocap package:[/cyan]", drone_mocap.__file__)
    print("[cyan]cli module:[/cyan]", cli_mod.__file__)
    print("[cyan]app:[/cyan]", app)


if __name__ == "__main__":
    app()
