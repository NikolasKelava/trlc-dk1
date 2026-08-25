"""``dk1 study`` — the paperwork of the two-policy comparison. No motor is touched.

    dk1 study photo --scene 1    a still of one scene layout, from the top camera
    dk1 study scores A0          the rubric back, per scene, from the CSV

The scoring itself happens inside `dk1 policy session --study <row>`, as each
attempt ends. These two are what surrounds it: the photograph that makes a
layout reproducible on another day, and the reading of what has been scored so
far. `STUDY.md` is the protocol; :mod:`dk1lab.study` is the bookkeeping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..config import DEFAULT_CONFIG_PATH, load
from ..study import (
    DEFAULT_SCENE_DIR,
    DEFAULT_SCENES,
    DEFAULT_SCORES_DIR,
    grid,
    read,
    scores_path,
)

app = typer.Typer(no_args_is_help=True, help=__doc__)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]


HELP_PHOTO = """Photograph one scene layout with a camera on the cell.

The dice and the bowl sit on marks drawn on the desk; this is the picture of
what the marks mean, so a layout can be reproduced weeks later and on another
day's lighting. Written to study/scene/<n>.jpg.

Opens a video device and nothing else: no motor is energised, no arm moves.
The still carries the configured rotation, because all three cameras on this
cell are mounted upside down, and NOT the wrist crop — this is a picture of the
desk for a human, not the policy's view."""


@app.command("photo", help=HELP_PHOTO)
def photo(
    scene: Annotated[
        int | None,
        typer.Option("--scene", help="Which scene layout this is. Names the file."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write here instead of study/scene/<scene>.jpg."),
    ] = None,
    camera: Annotated[
        str, typer.Option("--camera", help="Which camera to shoot from.")
    ] = "top",
    profile: Annotated[
        str, typer.Option("--profile", help="Capture profile from dk1.toml.")
    ] = "policy",
    scene_dir: Annotated[
        Path, typer.Option("--scene-dir", help="Where scene photographs live.")
    ] = DEFAULT_SCENE_DIR,
    open_it: Annotated[
        bool, typer.Option("--open/--no-open", help="Show the still in an image viewer.")
    ] = True,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    from ..discovery.preview import CaptureError, capture_still, open_viewer, save_still

    if scene is None and out is None:
        raise typer.BadParameter("say which scene this is (--scene 1) or where it goes (--out)")

    settings = load(config)
    # An unknown camera or profile raises ConfigError, which `dk1`'s entry point
    # already turns into a clean message — no need to catch it here.
    device = settings.camera(camera)
    capture = settings.profile(profile)

    path = Path(out) if out is not None else Path(scene_dir) / f"{scene}.jpg"
    typer.echo(
        f"{camera} camera {device.path} at {capture.width}x{capture.height} "
        f"{capture.fourcc}, rotation {device.rotation}"
    )
    try:
        still = capture_still(
            device.path,
            width=capture.width,
            height=capture.height,
            fourcc=capture.fourcc,
            rotation=device.rotation,
        )
        written = save_still(still, path)
    except CaptureError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"wrote {written} ({written.stat().st_size / 1e3:.0f} kB)", fg=typer.colors.GREEN)
    if open_it:
        viewer = open_viewer(written)
        if viewer is None:
            typer.secho("no image viewer found — open the file yourself.", fg=typer.colors.YELLOW)


HELP_SCORES = """Read a scored row back: every attempt, then the per-scene grid.

The grid is the point of running three layouts. A row that reads 5 5 5 / 0 0 0 /
0 0 0 and one that reads 0 5 0 / 5 0 0 / 0 0 5 have the same success rate and are
not the same result."""


@app.command("scores", help=HELP_SCORES)
def scores(
    row: Annotated[str, typer.Argument(help="The configuration: R0, A0, A1, B0, B1.")],
    scores_dir: Annotated[
        Path, typer.Option("--scores-dir", help="Where the score CSVs live.")
    ] = DEFAULT_SCORES_DIR,
    scenes: Annotated[int, typer.Option("--scenes", help="How many scenes the row has.")] = (
        DEFAULT_SCENES
    ),
) -> None:
    path = scores_path(row, scores_dir)
    attempts = read(path)
    if not attempts:
        typer.secho(f"no attempts scored yet in {path}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    typer.secho(f"\n{row} — {len(attempts)} attempt(s) in {path}", bold=True)
    for attempt in attempts:
        episode = f"  [{attempt.episode}]" if attempt.episode else ""
        typer.echo(f"  {attempt.line()}{episode}")
    typer.secho("\nby scene", bold=True)
    for line in grid(attempts, scenes=scenes):
        typer.echo(line)
    typer.echo()


__all__ = ["app", "photo", "scores"]
