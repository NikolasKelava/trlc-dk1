"""``dk1 find`` — identify which device is which, and write it to dk1.toml.

``find arms`` and ``find cameras`` each rewrite only their own section of the
config. Running one can never disturb the other's settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..config import DEFAULT_CONFIG_PATH, load, write_arms
from ..discovery.arms import DiscoveryError, find_arms
from ..discovery.cameras import list_candidates

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]


@app.command("arms")
def arms(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Identify the four arm serial ports by unplugging each one in turn.

    You will be asked to unplug one arm at a time. Nothing is opened or
    energised — this only watches which /dev/ttyACM* node disappears.

    Writes only the arms.follower and arms.leader tables. The cameras section
    and every comment in dk1.toml are left exactly as they were.
    """
    typer.echo("Identifying the four arms. Unplug one at a time when asked.\n")
    typer.echo("Nothing is energised: this only watches /dev/ttyACM* appear and disappear.\n")
    try:
        ports = find_arms()
    except DiscoveryError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    write_arms(ports, config)
    typer.secho(f"\nwrote the arms section to {config}", fg=typer.colors.GREEN)
    typer.echo("(the cameras and capture sections were not touched)")


@app.command("cameras")
def cameras(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """List the cameras attached right now, next to what dk1.toml expects.

    Read-only: no camera is opened. Assigning the top / left / right labels needs
    a look at each camera's picture, which arrives with the rest of the camera
    tooling; for now this shows you whether the configured by-path nodes still
    match reality after a replug.
    """
    candidates = list_candidates()
    if not candidates:
        typer.secho("No cameras found under /dev/v4l/by-path.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    configured = {device.path: name for name, device in load(config).cameras.items()}

    typer.secho(f"{len(candidates)} camera(s) attached:\n", bold=True)
    for candidate in candidates:
        label = configured.get(candidate.by_path)
        tag = (
            typer.style(f"configured as {label}", fg=typer.colors.GREEN)
            if label
            else typer.style("not in dk1.toml", fg=typer.colors.YELLOW)
        )
        typer.echo(f"  hub port {candidate.hub_port:6s} -> {candidate.device or '?':12s} {tag}")
        typer.echo(f"      {candidate.by_path}")

    attached = {c.by_path for c in candidates}
    stale = [(name, path) for path, name in configured.items() if path not in attached]
    if stale:
        typer.secho("\nconfigured but not attached:", fg=typer.colors.RED)
        for name, path in stale:
            typer.echo(f"  {name}: {path}")
        raise typer.Exit(code=1)
