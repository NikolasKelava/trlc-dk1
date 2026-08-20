"""``dk1 teleop`` — drive the followers from the leader arms.

The one teleoperation entry point. Everything it needs about the devices comes
from ``dk1.toml``; the only things this command decides are how fast, whether to
show you the cameras, and how tight the speed limit is.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from ..cameras import crop_summary
from ..modelview import DEFAULT_EVERY
from ..config import DEFAULT_CONFIG_PATH, load
from ..teleop import DEFAULT_FPS, TELEOP_LIMITS, build, run
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


def _report(
    follower,
    leader,
    *,
    fps: int,
    display: bool,
    duration_s: float | None,
    model_input: bool = False,
) -> None:
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
        typer.echo("  none — the followers track the leaders at full speed")
        typer.echo("  (gripper rate and max lag are inert while the cap is off)")
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
            # The crop is what --display shows, so it belongs in the banner you
            # read just before deciding whether the view looks right.
            + (f"  [{crop}]" if (crop := crop_summary(camera)) else "")
        )

    typer.secho("\nloop", bold=True)
    typer.echo(f"  target {fps} Hz" + (f", stopping after {duration_s}s" if duration_s else ""))
    typer.echo(f"  rerun visualisation {'on' if display else 'off'}")
    if model_input:
        typer.echo(
            f"  model's-eye view    on, sampled 1 tick in {DEFAULT_EVERY}"
            f" (~{fps / DEFAULT_EVERY:.0f} Hz) under policy_input/"
        )


def teleop(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    cameras: Annotated[
        bool, typer.Option("--cameras/--no-cameras", help="Attach the three cameras.")
    ] = True,
    display: Annotated[
        bool, typer.Option("--display", help="Stream observations to Rerun. Implies --cameras.")
    ] = False,
    display_policy_input: Annotated[
        bool,
        typer.Option(
            "--display-policy-input",
            help=(
                "Also show what the POLICY would be handed — the 378x378 tensors from "
                "the real checkpoint preprocessor — beside the camera view. "
                "Implies --display. No model weights are loaded and no GPU is used."
            ),
        ),
    ] = False,
    fps: Annotated[int, typer.Option("--fps", help="Target loop rate, Hz.")] = DEFAULT_FPS,
    profile: Annotated[
        str, typer.Option("--profile", help="Capture profile for the cameras.")
    ] = "teleop",
    control_mode: Annotated[
        str, typer.Option("--control-mode", help="Follower control mode: impedance or pos_vel.")
    ] = "impedance",
    max_joint_rate: Annotated[
        float | None,
        typer.Option("--max-joint-rate", help="Joint speed cap, rad/s. Overrides dk1.toml."),
    ] = None,
    max_lag: Annotated[
        float | None,
        typer.Option("--max-lag", help="How far a command may lead the measurement, rad."),
    ] = None,
    no_limit: Annotated[
        bool, typer.Option("--no-limit", help="Remove the joint speed cap for this run.")
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
    if display_policy_input:
        # It logs into the same Rerun session and the loop only calls the
        # observation processor when displaying, so on its own it would be inert.
        display = True
    if display and not cameras:
        raise typer.BadParameter("--display needs cameras; drop --no-cameras.")

    if no_limit and max_joint_rate is not None:
        raise typer.BadParameter("--no-limit and --max-joint-rate contradict each other.")

    settings = load(config, require_devices=not dry_run)
    # dk1.toml is the source of truth; the flags are a per-run override on top.
    limits = settings.limit("teleop", TELEOP_LIMITS)
    if no_limit:
        limits = limits.unlimited()
    elif max_joint_rate is not None:
        limits = replace(limits, max_joint_rate=max_joint_rate)
    if max_lag is not None:
        limits = replace(limits, max_lag=max_lag)

    leader, follower = build(
        settings,
        cameras=cameras,
        profile=profile,
        control_mode=control_mode,
        limits=limits,
    )
    _report(
        follower,
        leader,
        fps=fps,
        display=display,
        duration_s=duration_s,
        model_input=display_policy_input,
    )

    if dry_run:
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    confirm_motion(
        "teleoperate — the followers will track the leader arms",
        assume_yes=assume_yes,
        notes=["Connecting a LEADER also torques its gripper open: fingers out of the triggers."],
    )
    # Built before connecting: it reads a checkpoint off disk and that is a
    # second or so of work with the arms already energised if it is left later.
    model_view = None
    if display_policy_input:
        from ..modelview import ModelInputProbe, build_preprocessor

        capture = settings.profile(profile)
        typer.echo("loading the checkpoint's preprocessor (no model weights, no GPU) ...")
        preprocessor, features = build_preprocessor(
            str(settings.policy.checkpoint), width=capture.width, height=capture.height
        )
        model_view = ModelInputProbe(None, preprocessor, features)

    typer.secho("\nCtrl-C to stop. Stopping does not move the arms.\n", fg=typer.colors.GREEN)
    run(
        leader,
        follower,
        fps=fps,
        display=display,
        duration_s=duration_s,
        model_view=model_view,
    )
    if model_view is not None and model_view.failed:
        typer.secho(
            f"\nthe model-input view stopped after an error: {model_view.failed}",
            fg=typer.colors.YELLOW,
            err=True,
        )
    typer.secho("\nteleop ended; both devices disconnected.", fg=typer.colors.GREEN)


# Registered on the root app by ``dk1lab.cli.main`` as a plain command rather than
# a group: `dk1 teleop` takes no subcommand. HELP is passed there rather than left
# as this function's docstring so the motion warnings stay next to the text they
# belong to.
