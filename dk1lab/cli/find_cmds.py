"""``dk1 find`` — identify which device is which, and write it to dk1.toml.

``find arms`` and ``find cameras`` each rewrite only their own section of the
config. Running one can never disturb the other's settings.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import click
import typer

from ..config import DEFAULT_CONFIG_PATH, CameraDevice, load, write_arms, write_cameras
from ..discovery.arms import DiscoveryError, find_arms
from ..discovery.cameras import CameraCandidate, LabelError, assign_labels, list_candidates
from ..discovery.ports import describe, list_ports, role_conflicts
from ..layout import CAMERA_NAMES
from .formats_report import report_formats

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]

#: All three DK1 cameras are mounted upside down (verified on hardware), so this
#: is the default rather than something the operator has to remember.
DEFAULT_ROTATION = 180


@app.command("arms")
def arms(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    inspect: Annotated[
        bool,
        typer.Option(
            "--inspect", help="Only print what each port is, by USB identity. No prompts."
        ),
    ] = False,
) -> None:
    """Identify the four arm serial ports by unplugging each one in turn.

    You will be asked to unplug one arm at a time. Nothing is opened or
    energised — this only watches which /dev/ttyACM* node disappears.

    USB identity already knows a follower adapter from a leader one, so what the
    unplugging actually settles is which side of each pair is the left arm; the
    result is cross-checked against the adapter families before anything is
    written. Use --inspect to see that identity table on its own.

    Writes only the arms.follower and arms.leader tables. The cameras section
    and every comment in dk1.toml are left exactly as they were.
    """
    present = list_ports()
    if inspect:
        if not present:
            typer.secho("No USB serial ports found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        typer.secho(f"{len(present)} USB serial port(s):\n", bold=True)
        for line in describe(present, load(config).arm_ports()):
            typer.echo(line)
        typer.echo(
            "\nUSB identity cannot say which side of a pair is the left arm — "
            "run `dk1 find arms` for that."
        )
        return

    typer.echo("Identifying the four arms. Unplug one at a time when asked.\n")
    typer.echo("Nothing is energised: this only watches /dev/ttyACM* appear and disappear.\n")
    try:
        ports = find_arms()
    except DiscoveryError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    assignments = {
        f"{role}_{side}": getattr(arm, side)
        for role, arm in ports.items()
        for side in ("left", "right")
    }
    if conflicts := role_conflicts(assignments, present):
        typer.secho(
            "\nthe result contradicts what the USB adapters are:", fg=typer.colors.RED, err=True
        )
        for conflict in conflicts:
            typer.secho(f"  {conflict}", fg=typer.colors.RED, err=True)
        typer.secho(
            "\nNothing has been written. A wrong cable was probably unplugged — "
            "run `dk1 find arms --inspect` to see the ports, then try again.\n",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    write_arms(ports, config)
    typer.secho(f"\nwrote the arms section to {config}", fg=typer.colors.GREEN)
    typer.echo("(the cameras and capture sections were not touched)")


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def _list_cameras(config: Path) -> None:
    """Show what is attached next to what dk1.toml expects. Opens nothing."""
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


@app.command("cameras")
def cameras(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    show_only: Annotated[
        bool,
        typer.Option(
            "--list", "-l", help="Only list what is attached; label nothing, write nothing."
        ),
    ] = False,
    rotation: Annotated[
        int, typer.Option("--rotation", help="Mounting rotation applied to every camera.")
    ] = DEFAULT_ROTATION,
    profile: Annotated[
        str, typer.Option("--profile", help="Capture profile used for the preview stills.")
    ] = "policy",
    check_formats: Annotated[
        bool, typer.Option("--probe/--no-probe", help="Check each camera offers every profile.")
    ] = True,
    outdir: Annotated[
        Path | None,
        typer.Option("--outdir", help="Where to keep the preview stills (default: a temp dir)."),
    ] = None,
) -> None:
    """Preview each camera and label it top / left / right, then write dk1.toml.

    Grabs one still per attached camera, opens it in an image viewer, and asks
    which view it is. The arms are not touched: this opens video devices only.

    The three names are not free choices — the MolmoAct2 BimanualYAM checkpoint's
    image keys are observation.images.{top,left,right} and must match exactly.

    Writes only the cameras tables; the arms and capture sections and every
    comment in dk1.toml are left as they were. Use --list to inspect without
    labelling or writing anything.
    """
    if show_only:
        _list_cameras(config)
        return

    from ..discovery.preview import CaptureError, capture_still, open_viewer, save_still

    settings = load(config)
    capture = settings.profile(profile)
    candidates = list_candidates()
    if not candidates:
        typer.secho("No cameras found under /dev/v4l/by-path.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    directory = Path(outdir) if outdir else Path(tempfile.mkdtemp(prefix="dk1-cameras-"))
    typer.echo(f"Labelling {len(candidates)} camera(s). Stills go to {directory}\n")
    typer.echo("Nothing is energised and the arms do not move: video devices only.\n")

    def show(candidate: CameraCandidate) -> None:
        typer.secho(f"\nhub port {candidate.hub_port} -> {candidate.device or '?'}", bold=True)
        still = capture_still(
            candidate.by_path,
            width=capture.width,
            height=capture.height,
            fourcc=capture.fourcc,
            rotation=rotation,
        )
        path = save_still(still, directory / f"hub-{candidate.hub_port}.png")
        viewer = open_viewer(path)
        typer.echo(f"      {path}" + (f"  (opened with {viewer})" if viewer else ""))
        if viewer is None:
            typer.secho(
                "      no image viewer found — open the file above yourself.",
                fg=typer.colors.YELLOW,
            )
        if check_formats:
            report_formats(candidate.by_path, settings.capture)

    def ask(candidate: CameraCandidate, remaining: list[str]) -> str:
        return typer.prompt(
            f"      which view is this? [{'/'.join(remaining)}]",
            type=click.Choice(remaining),
            show_choices=False,
        )

    try:
        labelled = assign_labels(
            candidates, names=CAMERA_NAMES, show=show, ask=ask, announce=typer.echo
        )
    except (LabelError, CaptureError) as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    devices = {
        name: CameraDevice(path=candidate.by_path, rotation=rotation)
        for name, candidate in labelled.items()
    }
    typer.echo("")
    for name in CAMERA_NAMES:
        typer.echo(f"  {name:6s} {devices[name].path}  rotation {rotation}")
    if not typer.confirm("\nwrite this to dk1.toml?", default=True):
        typer.echo("nothing written.")
        raise typer.Exit(code=1)

    write_cameras(devices, config)
    typer.secho(f"\nwrote the cameras section to {config}", fg=typer.colors.GREEN)
    typer.echo("(the arms and capture sections were not touched)")
