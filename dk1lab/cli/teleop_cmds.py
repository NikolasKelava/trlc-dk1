"""``dk1 teleop`` — drive the followers from the leader arms.

The one teleoperation entry point. Everything it needs about the devices comes
from ``dk1.toml``; the only things this command decides are how fast, whether to
show you the cameras, and how tight the speed limit is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..config import DEFAULT_CONFIG_PATH, load
from ..limiter import DEFAULT_MAX_DT, DEFAULT_MAX_GRIPPER_RATE
from ..teleop import DEFAULT_FPS, TELEOP_MAX_JOINT_RATE, TELEOP_MAX_LAG, TeleopLimits, build, run
from .safety import LEADER_HELP, MOTION_HELP, confirm_motion

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]

HELP = (
    """Teleoperate: the followers track the leader arms.

Devices come from dk1.toml. Cameras are named top / left / right, which is what
the MolmoAct2 checkpoint requires and therefore what recording will need — the
naming is not a free choice, so it is not an option here.

Ctrl-C stops. Stopping disconnects and does nothing else: the arms are never
swept home, because sweeping them home is the last thing you want when you
stopped because something was wrong.

Use --dry-run to see exactly what would be built without connecting to anything,
which is worth doing before the first run on new hardware."""
    + MOTION_HELP
    + LEADER_HELP
)


def _report(follower, leader, *, fps: int, display: bool, duration_s: float | None) -> None:
    """Print what is about to run — shared by --dry-run and the real thing."""
    typer.secho("leader", bold=True)
    typer.echo(f"  left  {leader.config.left_arm_port}")
    typer.echo(f"  right {leader.config.right_arm_port}")

    config = follower.config
    typer.secho("\nfollower (bi_dk1_follower_safe)", bold=True)
    typer.echo(f"  left  {config.left_arm_port}")
    typer.echo(f"  right {config.right_arm_port}")
    typer.echo(f"  control mode  {config.control_mode}")

    typer.secho("\nspeed limit", bold=True)
    if config.max_joint_rate is None:
        typer.secho(
            "  DISABLED — the followers will track the leaders at full speed",
            fg=typer.colors.RED,
        )
    else:
        typer.echo(
            f"  joints    {config.max_joint_rate} rad/s "
            f"({config.max_joint_rate * 57.3:.0f} deg/s)"
        )
    typer.echo(f"  gripper   {config.max_gripper_rate} /s")
    typer.echo(f"  max lag   {config.max_lag} rad")

    typer.secho("\ncameras", bold=True)
    if not config.cameras:
        typer.echo("  none (--cameras to attach them)")
    for name, camera in config.cameras.items():
        typer.echo(
            f"  {name:6s} {camera.width}x{camera.height} @ {camera.fps} {camera.fourcc}"
            f"  rotation {int(camera.rotation)}  {camera.index_or_path}"
        )

    typer.secho("\nloop", bold=True)
    typer.echo(f"  target {fps} Hz" + (f", stopping after {duration_s}s" if duration_s else ""))
    typer.echo(f"  rerun visualisation {'on' if display else 'off'}")


def teleop(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    cameras: Annotated[
        bool, typer.Option("--cameras/--no-cameras", help="Attach the three cameras.")
    ] = True,
    display: Annotated[
        bool, typer.Option("--display", help="Stream observations to Rerun. Implies --cameras.")
    ] = False,
    fps: Annotated[int, typer.Option("--fps", help="Target loop rate, Hz.")] = DEFAULT_FPS,
    profile: Annotated[
        str, typer.Option("--profile", help="Capture profile for the cameras.")
    ] = "teleop",
    control_mode: Annotated[
        str, typer.Option("--control-mode", help="Follower control mode: impedance or pos_vel.")
    ] = "impedance",
    max_joint_rate: Annotated[
        float, typer.Option("--max-joint-rate", help="Joint speed cap, rad/s.")
    ] = TELEOP_MAX_JOINT_RATE,
    max_lag: Annotated[
        float, typer.Option("--max-lag", help="How far a command may lead the measurement, rad.")
    ] = TELEOP_MAX_LAG,
    no_limit: Annotated[
        bool,
        typer.Option("--no-limit", help="Remove the joint speed cap entirely. Say why out loud."),
    ] = False,
    duration_s: Annotated[
        float | None, typer.Option("--duration", help="Stop after this many seconds.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Build and print everything; connect to nothing.")
    ] = False,
    assume_yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y", help="Skip the confirmation prompt. The warning still prints."
        ),
    ] = False,
) -> None:
    if control_mode not in ("impedance", "pos_vel"):
        raise typer.BadParameter(f"control mode must be impedance or pos_vel, got {control_mode!r}")
    if fps <= 0:
        raise typer.BadParameter(f"--fps must be positive, got {fps}")
    if display and not cameras:
        raise typer.BadParameter("--display needs cameras; drop --no-cameras.")

    limits = TeleopLimits(
        max_joint_rate=None if no_limit else max_joint_rate,
        max_gripper_rate=DEFAULT_MAX_GRIPPER_RATE,
        max_lag=max_lag,
        max_dt=DEFAULT_MAX_DT,
    )
    leader, follower = build(
        load(config, require_devices=not dry_run),
        cameras=cameras,
        profile=profile,
        control_mode=control_mode,
        limits=limits,
    )
    _report(follower, leader, fps=fps, display=display, duration_s=duration_s)

    if dry_run:
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    confirm_motion(
        "teleoperate — the followers will track the leader arms",
        assume_yes=assume_yes,
        notes=["Connecting a LEADER also torques its gripper open: fingers out of the triggers."],
    )
    typer.secho("\nCtrl-C to stop. Stopping does not move the arms.\n", fg=typer.colors.GREEN)
    run(leader, follower, fps=fps, display=display, duration_s=duration_s)
    typer.secho("\nteleop ended; both devices disconnected.", fg=typer.colors.GREEN)


# Registered on the root app by ``dk1lab.cli.main`` as a plain command rather than
# a group: `dk1 teleop` takes no subcommand. HELP is passed there rather than left
# as this function's docstring so the motion warnings stay next to the text they
# belong to.
