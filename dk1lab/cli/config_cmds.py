"""``dk1 config`` — inspect and validate ``dk1.toml``. Never touches hardware."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..cameras import cameras_cli_argument
from ..config import DEFAULT_CONFIG_PATH, ConfigError, check_devices, load
from .formats_report import report_formats

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]


@app.command("show")
def show(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Print the configured devices. Does not connect to anything."""
    cfg = load(config)
    typer.echo(f"config: {cfg.path}\n")

    typer.secho("arms", bold=True)
    for role in ("follower", "leader"):
        ports = getattr(cfg, role)
        typer.echo(f"  {role:9s} left  {ports.left}")
        typer.echo(f"  {'':9s} right {ports.right}")

    typer.secho("\ncameras", bold=True)
    for name, device in cfg.cameras.items():
        typer.echo(f"  {name:6s} rotation {device.rotation:3d}  {device.path}")

    if cfg.limits:
        typer.secho("\nspeed limits", bold=True)
        for name, limit in cfg.limits.items():
            rate = "none" if limit.max_joint_rate is None else f"{limit.max_joint_rate} rad/s"
            typer.echo(
                f"  {name:8s} joints {rate}  gripper {limit.max_gripper_rate}/s  "
                f"lag {limit.max_lag} rad  max dt {limit.max_dt}s"
            )

    typer.secho("\ncapture profiles", bold=True)
    for name, profile in cfg.capture.items():
        typer.echo(f"  {name:8s} {profile.width}x{profile.height} @ {profile.fps} fps  {profile.fourcc}")


@app.command("check")
def check(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    devices: Annotated[
        bool, typer.Option("--devices/--no-devices", help="Also require every device node to exist.")
    ] = True,
    formats: Annotated[
        bool,
        typer.Option(
            "--formats",
            help="Also ask each camera whether it really offers every capture profile.",
        ),
    ] = False,
) -> None:
    """Validate dk1.toml, and check every configured device is present.

    Read-only: opens no serial port, and nothing is energised. With --formats it
    additionally interrogates each camera's advertised capture modes, which does
    not open a stream either.
    """
    cfg = load(config)
    typer.secho(f"{cfg.path}: valid", fg=typer.colors.GREEN)
    if not devices:
        raise typer.Exit()
    try:
        check_devices(cfg)
    except ConfigError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho("all 4 arm ports and 3 cameras present", fg=typer.colors.GREEN)

    if not formats:
        return
    typer.secho("\ncapture profiles, as advertised by each camera", bold=True)
    complete = True
    for name, device in cfg.cameras.items():
        typer.echo(f"  {name}")
        complete &= report_formats(device.path, cfg.capture)
    if not complete:
        typer.secho(
            "\nat least one camera does not offer a configured profile — see above.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command("cameras-arg")
def cameras_arg(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    profile: Annotated[str, typer.Option("--profile", "-p", help="Capture profile.")] = "policy",
) -> None:
    """Print the --robot.cameras argument, for pasting into a raw lerobot command."""
    typer.echo(cameras_cli_argument(load(config), profile))
