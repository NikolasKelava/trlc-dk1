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
from ..policy import DEFAULT_EXECUTION_HORIZON, DEFAULT_FPS, HOME_AT_START_POSE, POLICY_LIMITS
from .safety import ENERGISE_HELP, MOTION_HELP, confirm_motion

app = typer.Typer(no_args_is_help=True, help=__doc__)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]
CheckpointOpt = Annotated[
    str | None,
    typer.Option(
        "--checkpoint", help="Checkpoint dir or HF repo id. Default: \\[policy] in dk1.toml."
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


def _echo_rtc(cfg) -> None:
    """The RTC delay-vs-horizon relationship, which is what decides smoothness."""
    from ..policy import MEASURED_RTC_LATENCY_S, MIN_RTC_BLEND_STEPS, rtc_headroom

    horizon = getattr(getattr(cfg, "inference", None), "rtc", None)
    if horizon is None:
        typer.secho(
            "  inference SYNC — the model runs inline every 30th tick and stalls the loop",
            fg=typer.colors.YELLOW,
        )
        return
    horizon = horizon.execution_horizon
    delay, ok = rtc_headroom(MEASURED_RTC_LATENCY_S, fps=cfg.fps, execution_horizon=horizon)
    typer.echo(
        f"  inference RTC, execution horizon {horizon}; "
        f"~{MEASURED_RTC_LATENCY_S * 1000:.0f} ms = {delay} ticks of delay"
    )
    if ok:
        typer.echo(f"  prefix blend over {horizon - delay} steps (re-measured at startup)")
    else:
        typer.secho(
            f"  NO BLEND: a {delay}-tick delay against a horizon of {horizon} leaves "
            f"{horizon - delay} steps of ramp. RTC's prefix weights collapse towards a step "
            f"function, consecutive chunks meet with a discontinuity, and the arms judder. "
            f"Raise --execution-horizon to at least {delay + MIN_RTC_BLEND_STEPS}.",
            fg=typer.colors.RED,
        )


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
    period_ms = 1000 / DEFAULT_FPS
    periods = result.inference_ms / period_ms
    typer.echo(f"  first call     {result.warmup_ms:.0f} ms   (warmup + CUDA graph capture)")
    typer.echo(
        f"  model call     {result.inference_ms:.0f} ms   "
        f"= {periods:.1f} control periods at {DEFAULT_FPS} Hz"
    )
    typer.echo(
        f"  cached call    {result.cached_ms:.0f} ms   "
        f"(29 calls in 30 — the chunk is already computed)"
    )
    from ..policy import DEFAULT_EXECUTION_HORIZON, MIN_RTC_BLEND_STEPS, rtc_headroom

    delay, ok = rtc_headroom(
        result.rtc_inference_ms / 1000,
        fps=DEFAULT_FPS,
        execution_horizon=DEFAULT_EXECUTION_HORIZON,
    )
    typer.echo(
        f"  RTC call       {result.rtc_inference_ms:.0f} ms   "
        f"= {delay} ticks of inference delay  <- what a rollout actually pays"
    )
    if periods > 1:
        typer.echo(
            "\n  A model call costs more than one control period, so with --sync the loop\n"
            "  stalls every 30th tick. --rtc runs inference in a background thread and is\n"
            "  the default for rollout."
        )
    if ok:
        typer.echo(
            f"  With --execution-horizon {DEFAULT_EXECUTION_HORIZON}, RTC blends consecutive "
            f"chunks over {DEFAULT_EXECUTION_HORIZON - delay} steps."
        )
    else:
        typer.secho(
            f"  A {delay}-tick delay leaves no blend inside the default horizon of "
            f"{DEFAULT_EXECUTION_HORIZON}. Raise --execution-horizon to at least "
            f"{delay + MIN_RTC_BLEND_STEPS}, or the arms will judder.",
            fg=typer.colors.RED,
        )
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


def _report(cfg, spec: str, *, steps: int | None = None, home=None) -> None:
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
    _echo_rtc(cfg)
    if steps is not None:
        typer.echo(f"  {steps} inference steps, then stop")
    elif cfg.duration:
        typer.echo(f"  stopping after {cfg.duration}s")
    else:
        typer.echo("  until interrupted")
    _echo_home(home)


def _echo_home(home) -> None:
    """What will happen when the loop ends. The last thing printed before it acts."""
    typer.secho("\nwhen the run ends", bold=True)
    if home is None:
        typer.echo("  disconnect only — the arms stay where they are, and the motors")
        typer.echo("  are disabled, so support anything holding itself up")
        return
    if home is HOME_AT_START_POSE:
        typer.secho(
            "  HOME to the pose captured at connect (no [home] in dk1.toml)",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho("  HOME to the [home] pose in dk1.toml:", fg=typer.colors.YELLOW)
        for side in ("left", "right"):
            values = ", ".join(f"{v:+.3f}" for v in getattr(home, side))
            typer.echo(f"    {side:5s} [{values}]")
    typer.echo("  on the duration limit and on Ctrl-C; not after an error.")
    typer.echo("  Ctrl-C during the sweep stops it where the arms are.")


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

The speed cap comes from \\[limits.policy] in dk1.toml and is on by default —
this is the case the limiter was written for. Ctrl-C stops; stopping
disconnects and nothing else, unless you ask for --home.

With --home, the arms are swept back to the \\[home] pose in dk1.toml when the
run ends — on the duration limit and on Ctrl-C alike, but never after an error.
The sweep runs at the same speed cap the policy ran under and stops when the
arms arrive, not after a fixed time. A second Ctrl-C stops it where they are.
Set the pose with `dk1 policy home --capture`; without one, --home falls back
to the pose the arms were in when the run connected.

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
    home: Annotated[
        bool,
        typer.Option("--home", help="Sweep the arms to the \\[home] pose when the run ends."),
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
    )
    # LeRobot's own return_to_initial_position stays off whatever --home says:
    # it fires from teardown on every exit path including a crash, sweeps for a
    # fixed 3 s whether or not the arms arrive, and targets the connect-time
    # pose. dk1lab.home does the job on our terms. See dk1lab/home.py.
    home_pose = None
    if home:
        home_pose = settings.home if settings.home is not None else HOME_AT_START_POSE
    _report(cfg, spec, home=home_pose)

    if dry_run:
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    notes = ["The POLICY commands the arms. Nobody has verified what it does on this cell."]
    if limits.max_joint_rate is None:
        notes.append("The speed cap is OFF for this run (--no-limit).")
    if home_pose is not None:
        notes.append("--home: when the run ends, BOTH ARMS SWEEP to the home pose.")
        if home_pose is HOME_AT_START_POSE:
            notes.append("        no [home] in dk1.toml, so home = the pose at connect.")
    confirm_motion(
        f"run MolmoAct2 on the follower arms — {task!r}",
        assume_yes=assume_yes,
        notes=notes,
    )
    if home_pose is None:
        typer.secho("\nCtrl-C to stop. Stopping does not move the arms.\n", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "\nCtrl-C to stop. Stopping then SWEEPS THE ARMS HOME; "
            "a second Ctrl-C stops the sweep.\n",
            fg=typer.colors.YELLOW,
        )
    report = run_rollout(cfg, display=display, home=home_pose)
    typer.secho("\nrollout ended; the robot is disconnected.", fg=typer.colors.GREEN)
    if report is not None:
        colour = typer.colors.GREEN if report.reached else typer.colors.YELLOW
        typer.secho(report.summary(), fg=colour)
        if not report.reached:
            typer.secho(
                "the motors are now disabled — support anything holding itself up.",
                fg=typer.colors.YELLOW,
            )


# --------------------------------------------------------------------------- #
# 5. home — the pose a run ends at, and how to drive there now
# --------------------------------------------------------------------------- #


HELP_HOME = (
    """Drive both follower arms to the \\[home] pose in dk1.toml.

This is the same sweep `dk1 policy run --home` does when a run ends, without
loading the model: ramp at the \\[limits.policy] speed cap, stop when the arms
arrive rather than after a fixed time, and report where they got to. No
cameras are opened. Ctrl-C stops the sweep where the arms are.

  --capture   the opposite direction: read where the arms are RIGHT NOW and
              write that into \\[home] in dk1.toml. Commands no pose, but still
              energises the arms. Position them by hand first (with everything
              powered down, or through teleoperation), then capture."""
    + MOTION_HELP
)


@app.command("home", help=HELP_HOME)
def home(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    capture: Annotated[
        bool,
        typer.Option("--capture", help="Write the arms' current pose into \\[home] in dk1.toml."),
    ] = False,
    control_mode: Annotated[
        str, typer.Option("--control-mode", help="Follower control mode: impedance or pos_vel.")
    ] = "impedance",
    max_joint_rate: Annotated[
        float | None,
        typer.Option("--max-joint-rate", help="Sweep speed, rad/s. Overrides dk1.toml."),
    ] = None,
    show: Annotated[
        bool, typer.Option("--show", help="Print the configured home pose and exit. No motion.")
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    from ..config import check_devices, write_home
    from ..home import DEFAULT_HOME_RATE, capture_pose, sweep_to_home

    if control_mode not in ("impedance", "pos_vel"):
        raise typer.BadParameter(f"control mode must be impedance or pos_vel, got {control_mode!r}")

    # Devices are checked below, once it is clear something will actually be
    # connected: `--show` reads the file, and "there is no home pose" is a better
    # message than "the cameras are missing" when both are true.
    settings = load(config)

    if show:
        if settings.home is None:
            typer.secho(
                f"{config}: no [home] section — `dk1 policy run --home` would fall back to "
                f"the pose captured at connect. Set one with `dk1 policy home --capture`.",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)
        _echo_home(settings.home)
        return

    if capture:
        check_devices(settings)
        confirm_motion(
            "energise both follower arms and record their current pose as home",
            assume_yes=assume_yes,
            notes=["No pose is commanded — but the grippers self-zero OPEN, so that is"
                   " what gets captured for them."],
        )
        pose = capture_pose(settings, control_mode=control_mode)
        write_home(pose, config)
        typer.secho(f"\nwrote [home] to {config}:", fg=typer.colors.GREEN)
        _echo_home(pose)
        return

    if settings.home is None:
        typer.secho(
            f"{config}: no [home] section, so there is nothing to drive to. "
            f"Position the arms and run `dk1 policy home --capture` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    check_devices(settings)
    limits = _limits(settings, max_joint_rate, no_limit=False)
    rate = limits.max_joint_rate or DEFAULT_HOME_RATE
    _echo_home(settings.home)
    confirm_motion(
        f"sweep BOTH follower arms to the home pose at {rate} rad/s",
        assume_yes=assume_yes,
        notes=["Both arms move. Ctrl-C stops the sweep where they are."],
    )
    report = sweep_to_home(
        settings,
        target=settings.home.as_action_dict(),
        limits=limits,
        control_mode=control_mode,
    )
    typer.secho(
        f"\n{report.summary()}",
        fg=typer.colors.GREEN if report.reached else typer.colors.YELLOW,
    )
    if not report.reached:
        typer.secho(
            "the motors are now disabled — support anything holding itself up.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)


__all__ = ["app"]
