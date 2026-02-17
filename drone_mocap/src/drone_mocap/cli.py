from __future__ import annotations

from pathlib import Path
import typer
from rich import print

from drone_mocap.pipeline.run import run_pipeline

app = typer.Typer(no_args_is_help=True)

@app.command("analyze")
def analyze(
    video: Path = typer.Option(..., exists=True, help="Path to sagittal video (mp4/mov)"),
    out_dir: Path = typer.Option(Path("runs"), help="Output directory"),
    mocap_txt: Path | None = typer.Option(None, help="Optional MoCap angles txt"),
    side: str = typer.Option("right", help="Visible side: right or left"),
    max_frames: int = typer.Option(0, help="0 = all frames, else limit for quick tests"),
):
    """Analyze a sagittal-view video and export angles."""
    out = run_pipeline(
        video=video,
        out_root=out_dir,
        mocap_txt=mocap_txt,
        visible_side=side,
        max_frames=max_frames,
    )
    print(f"[green]Done.[/green] Outputs in: {out}")

@app.command("doctor")
def doctor():
    """Prints import/debug info so we know we're running the right code."""
    import drone_mocap
    import drone_mocap.cli as cli_mod
    print("[cyan]drone_mocap package file:[/cyan]", drone_mocap.__file__)
    print("[cyan]cli module file:[/cyan]", cli_mod.__file__)
    print("[cyan]app object:[/cyan]", app)

if __name__ == "__main__":
    app()
