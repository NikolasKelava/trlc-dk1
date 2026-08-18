"""``dk1 policy`` — evaluate MolmoAct2 on this cell, in four escalating steps.

    dk1 policy check     read the checkpoint. No GPU, no robot, no motion.
    dk1 policy smoke     load it and run inference on a synthetic frame. GPU only.
    dk1 policy dryrun    the full path with the arms attached — actions PRINTED.
    dk1 policy run       the rollout. The policy drives the arms.

Each step is a superset of the one before it. Do them in order: every failure
mode caught at step 1 is a failure mode that would otherwise have been found
with the arms live.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from .. import checkpoint as ckpt
from ..config import DEFAULT_CONFIG_PATH, load
from ..layout import ACTION_KEYS, GRIPPER_INDICES, IMAGE_KEYS
from ..policy import DEFAULT_EXECUTION_HORIZON, DEFAULT_FPS, POLICY_LIMITS
from .safety import ENERGISE_HELP, MOTION_HELP, confirm_motion

app = typer.Typer(no_args_is_help=True, help=__doc__)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]
CheckpointOpt = Annotated[
    str | None,
    typer.Option(
        "--checkpoint", help="Checkpoint dir or HF repo id. Default: [policy] in dk1.toml."
    ),
]
TaskOpt = Annotated[str, typer.Option("--task", help="The language instruction to condition on.")]
DeviceOpt = Annotated[str, typer.Option("--device", help="Torch device.")]

#: Instruction used by `smoke`, where the images are noise and the task text only
#: has to exist for the prompt to build.
PLACEHOLDER_TASK = "pick up the object"


def _checkpoint(config, override: str | None) -> str:
    """The checkpoint to use: the flag, else ``[policy]`` in dk1.toml."""
    return override if override else config.checkpoint()


def _echo_inversion() -> None:
    channels = ", ".join(ACTION_KEYS[i] for i in GRIPPER_INDICES)
    typer.secho("\ngripper inversion", bold=True)
    typer.echo(f"  ON for {channels}  (x -> 1 - x, both directions)")
    typer.echo("  the DK1 is 0=open/1=closed, the checkpoint is 1=open/0=closed")
    typer.echo("  applied to the loaded pipeline steps — --policy.joint_signs does nothing")


# --------------------------------------------------------------------------- #
# 1. check — reads JSON, nothing else
# --------------------------------------------------------------------------- #


@app.command("check")
def check(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    checkpoint: CheckpointOpt = None,
) -> None:
    """Inspect the checkpoint. No GPU, no robot, no motion, no downloads.

    Reads config.json and both saved processor pipelines and says whether they
    match what this cell provides: 14-D state and action, the yam_dual_molmoact2
    normalisation statistics, and the top/left/right image order. Exits non-zero
    if any of that is wrong.
    """
    settings = load(config)
    spec = _checkpoint(settings, checkpoint)

    try:
        info = ckpt.read(spec)
    except ckpt.CheckpointError as exc:
        typer.secho(f"\ncannot read checkpoint: {exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.secho("checkpoint", bold=True)
    typer.echo(f"  path          {info.path}")
    if info.weights_bytes:
        typer.echo(f"  weights       {info.weights_bytes / 2**30:.1f} GiB  ({info.model_dtype})")
    typer.echo(f"  type          {info.policy_type}")
    typer.echo(f"  norm_tag      {info.norm_tag}")
    typer.echo(f"  setup_type    {info.setup_type}")
    typer.echo(f"  control_mode  {info.control_mode}")
    typer.echo(f"  chunk         {info.chunk_size} steps, {info.n_action_steps} executed")
    typer.echo(f"  vectors       state {info.state_dim}-D, action {info.action_dim}-D")

    typer.secho("\nimage order (from the saved preprocessor, which is what runs)", bold=True)
    for key in info.pipeline_image_keys or ["(none pinned)"]:
        typer.echo(f"  {key}")
    typer.echo(f"  this cell provides: {', '.join(IMAGE_KEYS)}")

    _echo_inversion()

    for note in ckpt.notes(info):
        typer.secho(f"\nnote: {note}", fg=typer.colors.YELLOW)

    found = ckpt.problems(info)
    if found:
        typer.secho("\nproblems", fg=typer.colors.RED, bold=True, err=True)
        for problem in found:
            typer.secho(f"  * {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("\ncheckpoint is usable on this cell.", fg=typer.colors.GREEN)


# --------------------------------------------------------------------------- #
# 2. smoke — GPU only
# --------------------------------------------------------------------------- #


@app.command("smoke")
def smoke(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    checkpoint: CheckpointOpt = None,
    task: TaskOpt = PLACEHOLDER_TASK,
    steps: Annotated[int, typer.Option("--steps", help="Inference calls to make.")] = 5,
    device: DeviceOpt = "cuda",
) -> None:
    """Load the policy and run inference on a synthetic frame. GPU only, no robot.

    Nothing is connected and no /dev node is opened, so this is safe to run with
    the cell powered down. It proves the checkpoint loads, the processor
    pipelines build, the gripper inversion can be applied, and inference returns
    a 14-D action in this cell's key order — and it measures how long a call
    takes, which is what decides whether RTC is needed.

    The images are noise, so the action values mean nothing. Only their shape,
    their ordering and the latency do.
    """
    from ..policy import smoke as run_smoke

    settings = load(config)
    spec = _checkpoint(settings, checkpoint)
    capture = settings.profile("policy")

    typer.secho("smoke test — no robot is connected by this command", bold=True)
    typer.echo(f"  checkpoint  {spec}")
    typer.echo(f"  device      {device}")
    typer.echo(f"  frame       {capture.width}x{capture.height}, 3 views")
    typer.echo(f"  task        {task!r}")
    typer.echo("\nloading (the first call also pays CUDA graph capture) ...")

    result = run_smoke(
        spec,
        task=task,
        steps=steps,
        device=device,
        width=capture.width,
        height=capture.height,
    )

    typer.secho("\naction", bold=True)
    for key, value in zip(result.action_keys, result.action, strict=True):
        typer.echo(f"  {key:22s} {value:+.4f}")

    typer.secho("\nlatency", bold=True)
    typer.echo("  " + "  ".join(f"{ms:.0f}" for ms in result.latencies_ms) + "  ms")
    typer.echo(f"  steady state {result.steady_ms:.0f} ms")
    periods = result.steady_ms / (1000 / DEFAULT_FPS)
    typer.echo(f"  = {periods:.1f} control periods at {DEFAULT_FPS} Hz")
    if periods > 1:
        typer.echo("  more than one period: rollout wants --rtc, or it will stall the loop")
    typer.echo(f"\npeak GPU memory {result.peak_gpu_gib:.1f} GiB")
    typer.secho(f"\n{result.inversion.describe()}", fg=typer.colors.GREEN)
    typer.secho("smoke test passed. Nothing was connected.", fg=typer.colors.GREEN)


# --------------------------------------------------------------------------- #
# Shared rollout options
# --------------------------------------------------------------------------- #


def _limits(settings, max_joint_rate: float | None, no_limit: bool):
    limits = settings.limit("policy", POLICY_LIMITS)
    if no_limit and max_joint_rate is not None:
        raise typer.BadParameter("--no-limit and --max-joint-rate contradict each other.")
    if no_limit:
        return limits.unlimited()
    if max_joint_rate is not None:
        return replace(limits, max_joint_rate=max_joint_rate)
    return limits


def _report(cfg, spec: str, *, steps: int | None = None) -> None:
    """Print everything that was built, before anything is connected."""
    robot = cfg.robot

    typer.secho("policy", bold=True)
    typer.echo(f"  checkpoint    {spec}")
    typer.echo(f"  device        {cfg.policy.device}, {cfg.policy.model_dtype}")
    typer.echo(f"  action mode   {cfg.policy.inference_action_mode}")
    typer.echo(f"  task          {cfg.task!r}")
    typer.echo(f"  inference     {cfg.inference.type}")

    typer.secho("\nfollower (bi_dk1_follower_safe)", bold=True)
    typer.echo(f"  left   {robot.left_arm_port}")
    typer.echo(f"  right  {robot.right_arm_port}")
    typer.echo(f"  control mode  {robot.control_mode}")

    typer.secho("\nspeed limit", bold=True)
    if robot.max_joint_rate is None:
        typer.secho(
            "  NONE — the policy commands the arms at full speed", fg=typer.colors.RED
        )
    else:
        typer.echo(
            f"  joints    {robot.max_joint_rate} rad/s "
            f"({robot.max_joint_rate * 57.3:.0f} deg/s)"
        )
        typer.echo(f"  gripper   {robot.max_gripper_rate} /s")
        typer.echo(f"  max lag   {robot.max_lag} rad")

    typer.secho("\ncameras", bold=True)
    for name, camera in robot.cameras.items():
        typer.echo(
            f"  {name:6s} {camera.width}x{camera.height} @ {camera.fps} {camera.fourcc}"
            f"  rotation {int(camera.rotation)}  {camera.index_or_path}"
        )

    _echo_inversion()

    typer.secho("\nloop", bold=True)
    typer.echo(f"  target {cfg.fps} Hz, interpolation x{cfg.interpolation_multiplier}")
    if steps is not None:
        typer.echo(f"  {steps} inference steps, then stop")
    elif cfg.duration:
        typer.echo(f"  stopping after {cfg.duration}s")
    else:
        typer.echo("  until interrupted")
    typer.echo(
        f"  return to start pose on stop: {'YES' if cfg.return_to_initial_position else 'no'}"
    )


# --------------------------------------------------------------------------- #
# 3. dryrun — arms attached, nothing sent
# --------------------------------------------------------------------------- #


HELP_DRYRUN = (
    """Run the whole deployment path with the arms attached, and send nothing.

Cameras, robot state, processors, inference, action decoding — everything a
rollout does except the last step. Each tick prints where every joint is and
where the policy wants it. A large delta on the first tick means the policy
disagrees with your start pose, and a rollout would begin by driving there.

This is also what confirms the gripper convention: watch the two gripper
channels with the grippers open, then again with them closed."""
    + ENERGISE_HELP
)


@app.command("dryrun", help=HELP_DRYRUN)
def dryrun(
    task: TaskOpt,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    checkpoint: CheckpointOpt = None,
    steps: Annotated[int, typer.Option("--steps", help="Observations to run inference on.")] = 10,
    control_mode: Annotated[
        str, typer.Option("--control-mode", help="Follower control mode: impedance or pos_vel.")
    ] = "impedance",
    device: DeviceOpt = "cuda",
    build_only: Annotated[
        bool, typer.Option("--build-only", help="Build and print everything; connect nothing.")
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    from ..policy import dryrun as run_dryrun
    from ..policy import rollout_config

    settings = load(config, require_devices=not build_only)
    spec = _checkpoint(settings, checkpoint)

    cfg = rollout_config(
        settings,
        checkpoint=spec,
        task=task,
        control_mode=control_mode,
        device=device,
        return_home=False,
    )
    _report(cfg, spec, steps=steps)

    if build_only:
        typer.secho(
            "\n--build-only: nothing was connected and nothing moved.", fg=typer.colors.GREEN
        )
        return

    confirm_motion(
        "energise the arms for a policy dry run — no action is ever sent",
        assume_yes=assume_yes,
        notes=["Actions are PRINTED, never sent. The arms stay where they are."],
    )

    def show(step) -> None:
        key, delta = step.worst
        typer.secho(f"\nstep {step.index:2d}   worst: {key} {delta:+.4f} rad", bold=True)
        for name in step.commanded:
            measured, commanded = step.measured[name], step.commanded[name]
            typer.echo(
                f"    {name:22s} now={measured:+.4f}  cmd={commanded:+.4f}  "
                f"d={commanded - measured:+.4f}"
            )

    collected = run_dryrun(cfg, steps=steps, on_step=show)
    typer.secho(
        f"\ndry run complete: {len(collected)} actions computed, none sent.",
        fg=typer.colors.GREEN,
    )


# --------------------------------------------------------------------------- #
# 4. run — the rollout
# --------------------------------------------------------------------------- #


HELP_RUN = (
    """Deploy the policy: MolmoAct2 drives the follower arms.

The speed cap comes from [limits.policy] in dk1.toml and is on by default —
this is the case the limiter was written for. Ctrl-C stops; stopping
disconnects and nothing else, because the arms are never swept home unless you
ask for it with --return-home.

Do `dk1 policy check`, then `smoke`, then `dryrun` first. Keep a hand on the
e-stop."""
    + MOTION_HELP
)


@app.command("run", help=HELP_RUN)
def run(
    task: TaskOpt,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    checkpoint: CheckpointOpt = None,
    duration_s: Annotated[
        float, typer.Option("--duration", help="Stop after this many seconds. 0 = until stopped.")
    ] = 30.0,
    fps: Annotated[int, typer.Option("--fps", help="Control rate, Hz.")] = DEFAULT_FPS,
    interpolation: Annotated[
        int, typer.Option("--interpolation", help="Commands per policy action.")
    ] = 1,
    rtc: Annotated[
        bool,
        typer.Option("--rtc/--sync", help="Run inference in a background thread (recommended)."),
    ] = True,
    execution_horizon: Annotated[
        int, typer.Option("--execution-horizon", help="RTC: actions executed per chunk.")
    ] = DEFAULT_EXECUTION_HORIZON,
    control_mode: Annotated[
        str, typer.Option("--control-mode", help="Follower control mode: impedance or pos_vel.")
    ] = "impedance",
    max_joint_rate: Annotated[
        float | None,
        typer.Option("--max-joint-rate", help="Joint speed cap, rad/s. Overrides dk1.toml."),
    ] = None,
    no_limit: Annotated[
        bool,
        typer.Option("--no-limit", help="Remove the speed cap. Read the warning it prints."),
    ] = False,
    display: Annotated[
        bool, typer.Option("--display", help="Stream observations to Rerun.")
    ] = False,
    device: DeviceOpt = "cuda",
    return_home: Annotated[
        bool,
        typer.Option("--return-home", help="Sweep the arms to their start pose at shutdown."),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Build and print everything; connect nothing.")
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    from ..policy import rollout_config
    from ..policy import run as run_rollout

    if control_mode not in ("impedance", "pos_vel"):
        raise typer.BadParameter(f"control mode must be impedance or pos_vel, got {control_mode!r}")
    if fps <= 0:
        raise typer.BadParameter(f"--fps must be positive, got {fps}")
    if interpolation < 1:
        raise typer.BadParameter(f"--interpolation must be at least 1, got {interpolation}")

    settings = load(config, require_devices=not dry_run)
    spec = _checkpoint(settings, checkpoint)
    limits = _limits(settings, max_joint_rate, no_limit)

    cfg = rollout_config(
        settings,
        checkpoint=spec,
        task=task,
        limits=limits,
        control_mode=control_mode,
        fps=fps,
        duration_s=duration_s,
        interpolation=interpolation,
        rtc=rtc,
        execution_horizon=execution_horizon,
        device=device,
        display=display,
        return_home=return_home,
    )
    _report(cfg, spec)

    if dry_run:
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    notes = ["The POLICY commands the arms. Nobody has verified what it does on this cell."]
    if limits.max_joint_rate is None:
        notes.append("The speed cap is OFF for this run (--no-limit).")
    if return_home:
        notes.append("--return-home: stopping will sweep both arms to their start pose.")
    confirm_motion(
        f"run MolmoAct2 on the follower arms — {task!r}",
        assume_yes=assume_yes,
        notes=notes,
    )
    typer.secho("\nCtrl-C to stop. Stopping does not move the arms.\n", fg=typer.colors.GREEN)
    run_rollout(cfg, display=display)
    typer.secho("\nrollout ended; the robot is disconnected.", fg=typer.colors.GREEN)


__all__ = ["app"]
