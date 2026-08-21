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

import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from .. import checkpoint as ckpt
from ..cameras import crop_summary
from ..config import DEFAULT_CONFIG_PATH, load
from ..layout import ACTION_KEYS, GRIPPER_INDICES, IMAGE_KEYS
from ..fifo import DEFAULT_BLEND_STEPS, DEFAULT_REPLAN_AT
from ..record import DEFAULT_RECORD_DIR
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
InvertOpt = Annotated[
    bool,
    typer.Option(
        "--invert-gripper/--no-invert-gripper",
        help=(
            "Flip the two gripper channels (x -> 1-x) in both directions. ON by "
            "default: the DK1 is 0=open and the checkpoint is 1=open, and that was "
            "confirmed on the arms. --no-invert-gripper is for testing it again."
        ),
    ),
]
TraceOpt = Annotated[
    bool,
    typer.Option(
        "--trace/--no-trace",
        help="Print the policy's own action per chunk, plus where the time went.",
    ),
]
FifoOpt = Annotated[
    bool,
    typer.Option(
        "--fifo/--no-fifo",
        help=(
            "Serve the whole action chunk from a queue instead of rebuilding the "
            "input pipeline on every tick. ON by default under --sync, where it "
            "is worth ~22 ms of a 33.3 ms control period; ignored under --rtc. "
            "--no-fifo is for measuring the difference."
        ),
    ),
]
AsyncOpt = Annotated[
    bool,
    typer.Option(
        "--async-fifo/--blocking-fifo",
        help=(
            "Compute each chunk on a worker thread while the previous one is still "
            "being served, so the loop never waits for the model. ON by default. "
            "--blocking-fifo is the old behaviour: the arms freeze for one model "
            "call (~310 ms) per chunk. For comparison, not for use."
        ),
    ),
]
ReplanAtOpt = Annotated[
    int,
    typer.Option(
        "--replan-at",
        help=(
            "Queue depth, in rows, at which the next chunk is started. Must exceed "
            "the inference latency in ticks (~10) or the queue can run dry. Higher "
            "is fresher and safer, at the cost of running the GPU harder."
        ),
    ),
]
BlendOpt = Annotated[
    int,
    typer.Option(
        "--blend",
        help=(
            "Rows over which a new chunk is cross-faded into the one it replaces. "
            "0 splices hard. Keep it well under the replan interval."
        ),
    ),
]
RecordOpt = Annotated[
    bool,
    typer.Option(
        "--record",
        help=(
            "Write the episode to a Rerun .rrd: the camera images, the policy's own "
            "plan, the command the arms were given (post-blend, post-limiter) and the "
            "measured positions. The same four streams --display draws, kept."
        ),
    ),
]
RecordDirOpt = Annotated[
    Path,
    typer.Option("--record-dir", help="Where recordings are written."),
]
WatchInputOpt = Annotated[
    bool,
    typer.Option(
        "--display-policy-input",
        help=(
            "Open Rerun and log the images and actions as the MODEL sees them: the "
            "378x378 tensors unpacked from pixel_values, not the robot-side view "
            "--display shows. This is where to check camera orientation."
        ),
    ),
]

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
        typer.echo(
            "  inference SYNC — the whole 30-step chunk is executed, with no prefix\n"
            "                   blending and no trimming. This is the arrangement that\n"
            "                   scored 100% in the simulator; --rtc is the one that\n"
            "                   starved the queue on the arms. How the chunk reaches\n"
            "                   the arms is the next line."
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


def _echo_inversion(invert: bool | None = None) -> None:
    """What the gripper channels will do. ``None`` = describing the option itself."""
    channels = ", ".join(ACTION_KEYS[i] for i in GRIPPER_INDICES)
    typer.secho("\ngripper inversion", bold=True)
    if invert is None:
        typer.echo(f"  available for {channels}  (x -> 1 - x, both directions)")
    elif invert:
        typer.echo(f"  ON for {channels}  (x -> 1 - x, both directions) — the default")
    else:
        typer.secho(
            f"  OFF — {channels} pass through unchanged. The checkpoint speaks YAM "
            f"(1=open) and this cell is 0=open, so the grippers will work backwards "
            f"unless you are deliberately re-testing that.",
            fg=typer.colors.YELLOW,
        )
    typer.echo("  the DK1 is 0=open/1=closed, the checkpoint is 1=open/0=closed;")
    typer.echo("  confirmed on the arms, and confirmed a fourth way by the simulator")
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

    _echo_inversion(None)

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
    invert_gripper: InvertOpt = True,
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
        invert_gripper=invert_gripper,
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
    typer.secho(
        f"\n{result.inversion.describe() if result.inversion else 'gripper inversion off'}",
        fg=typer.colors.GREEN,
    )
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


def _echo_fifo(cfg, *, fifo: bool, asynchronous: bool, replan_at: int, blend: int) -> None:
    """How the chunk reaches the arms — the thing that decides whether they pause.

    Only meaningful under sync inference; RTC serves chunks whole from its own
    background thread and this engine is not installed at all.
    """
    if getattr(getattr(cfg, "inference", None), "rtc", None) is not None:
        return
    if not fifo:
        typer.secho(
            "  chunk FIFO OFF — the whole input pipeline re-runs every tick and is "
            "thrown away on 29 of 30 (~22 ms of a 33 ms budget)",
            fg=typer.colors.YELLOW,
        )
        return
    if not asynchronous:
        typer.secho(
            "  chunk FIFO BLOCKING — the loop waits for each model call, so the arms "
            "freeze for ~310 ms once per chunk (--blocking-fifo)",
            fg=typer.colors.YELLOW,
        )
        return
    typer.echo(
        f"  chunk FIFO async — the model runs on a worker thread; the loop never\n"
        f"                   waits for it. Next chunk starts at {replan_at} rows queued,\n"
        f"                   {blend}-row cross-fade at the splice."
    )


def _report(
    cfg,
    spec: str,
    *,
    steps: int | None = None,
    home=None,
    invert: bool = False,
    fifo: bool = True,
    asynchronous: bool = True,
    replan_at: int = DEFAULT_REPLAN_AT,
    blend: int = DEFAULT_BLEND_STEPS,
    home_when: str = "when the run ends",
) -> None:
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
            # What the policy is actually shown. A wrist view cropped to the
            # trained field of view and one left at the lens's own 105 degrees
            # are the same size and the same file; this is what tells them apart.
            + (f"  [{crop}]" if (crop := crop_summary(camera)) else "")
        )

    _echo_inversion(invert)

    typer.secho("\nloop", bold=True)
    typer.echo(f"  target {cfg.fps} Hz, interpolation x{cfg.interpolation_multiplier}")
    _echo_rtc(cfg)
    _echo_fifo(cfg, fifo=fifo, asynchronous=asynchronous, replan_at=replan_at, blend=blend)
    if steps is not None:
        typer.echo(f"  {steps} inference steps, then stop")
    elif cfg.duration:
        typer.echo(f"  stopping after {cfg.duration}s")
    else:
        typer.echo("  until interrupted")
    _echo_home(home, when=home_when)


def _echo_home(home, *, when: str = "when the run ends") -> None:
    """What will happen when the loop ends. The last thing printed before it acts."""
    typer.secho(f"\n{when}", bold=True)
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


def _make_trace(*, fps: float, enabled: bool, display_policy_input: bool):
    """A :class:`~dk1lab.trace.RolloutTrace`, or ``None`` if nothing was asked for.

    One line per chunk — not per tick — so it stays readable while the loop
    runs. Printed from whichever thread the record was cut on: the RTC thread
    under ``--rtc``, and the control loop under the chunk FIFO, which reports a
    chunk at the tick it is spliced in.
    """
    if not (enabled or display_policy_input):
        return None
    from ..trace import RolloutTrace

    def show(record) -> None:
        colour = typer.colors.RED if record.starved else None
        typer.secho(record.line(), fg=colour)
        for line in record.action_lines():
            typer.echo(line)

    return RolloutTrace(
        fps=fps,
        on_chunk=show if enabled else None,
        display=display_policy_input,
    )


def _make_recorder(enabled: bool, directory, *, task: str, notes: dict):
    """An :class:`~dk1lab.record.EpisodeRecorder`, or ``None``.

    Built here rather than inside the rollout so the path is decided — and can
    be printed — before anything is connected.
    """
    if not enabled:
        return None
    from ..record import EpisodeRecorder, episode_path, next_index

    index = next_index(directory)
    return EpisodeRecorder(
        episode_path(directory, task, index), task=task, notes={**notes, "episode": index}
    )


def _keep_recording(recording) -> bool:
    """Ask whether to keep the episode just recorded, and delete it if not.

    Asked **after** the rollout because that is when you know: the episode ends
    when the task is done or has visibly failed, and only then is it clear
    whether the file is worth keeping. The recording itself has to be written as
    the arms move — there is nowhere to put three minutes of video otherwise —
    so declining is a delete, not a decision made in advance.

    Keeping is the default: an accidental Enter should not throw away an attempt
    that cannot be repeated. Non-interactive runs keep everything for the same
    reason.
    """
    if recording is None:
        return False
    typer.secho(recording.summary(), fg=typer.colors.GREEN)
    if not sys.stdin.isatty():
        return True
    if typer.confirm("  keep this episode?", default=True):
        return True
    if recording.discard():
        typer.secho(f"  discarded {recording.path}", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"  could not delete {recording.path}", fg=typer.colors.RED, err=True)
    return False


def _echo_trace_summary(trace) -> None:
    """The end-of-run reading of the trace. Printed after the arms have stopped."""
    if trace is None:
        return
    # Under sync the last window is still open when the run ends; close it so the
    # final chunk is counted rather than silently dropped.
    trace.close()
    summary = trace.summary()
    typer.secho("\ninference and queue", bold=True)
    for line in summary.lines():
        typer.echo(f"  {line}")
    for verdict in summary.verdicts():
        typer.secho(f"\n  {verdict}", fg=typer.colors.RED)


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
    invert_gripper: InvertOpt = True,
    display_policy_input: WatchInputOpt = False,
    fifo: FifoOpt = True,
    asynchronous: AsyncOpt = True,
    replan_at: ReplanAtOpt = DEFAULT_REPLAN_AT,
    blend: BlendOpt = DEFAULT_BLEND_STEPS,
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
        invert_gripper=invert_gripper,
    )
    _report(
        cfg, spec, steps=steps, invert=invert_gripper,
        fifo=fifo, asynchronous=asynchronous, replan_at=replan_at, blend=blend,
    )
    if display_policy_input:
        typer.secho(
            "\n  --display-policy-input: Rerun will show the 378x378 images the MODEL "
            "is handed,\n  unpacked from pixel_values, under policy_input/ — alongside "
            "the policy's own\n  action values. Nothing is sent.",
            fg=typer.colors.YELLOW,
        )

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

    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

    trace = _make_trace(fps=cfg.fps, enabled=True, display_policy_input=display_policy_input)
    if display_policy_input:
        init_visualization("rerun", session_name="dk1-policy-input")
    try:
        collected = run_dryrun(
            cfg,
            steps=steps,
            on_step=show,
            invert_gripper=invert_gripper,
            trace=trace,
            fifo=fifo,
            asynchronous=asynchronous,
            replan_at=replan_at,
            blend=blend,
        )
    finally:
        if display_policy_input:
            shutdown_visualization("rerun")
    _echo_trace_summary(trace)
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

Inference runs SYNCHRONOUSLY by default: the full 30-step chunk is executed,
then the loop blocks for one model call. That is a deliberate change from
--rtc, made on 2026-08-20. RTC's own diagnostics on the second rollout showed a
900 ms chunk latency, which made it discard 27 of every 30 actions and deliver
100 ms of motion per second — the stall. Sync cannot discard anything: it pays
one visible pause per chunk and executes all 30. It is also exactly the
arrangement `sim_eval` uses, which scored 100% on this checkpoint in ManiSkill.
--rtc is still there, and is the right answer once inference is well under
chunk/fps = 1 s in situ.

--duration defaults to 180 s. The policy is SLOW: in simulation it barely acts
for the first ~30 s and successful episodes averaged 54 s. A 30 s rollout does
not give it enough time to do anything, which is worth knowing before reading a
short run as a failure.

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
    ] = 180.0,
    fps: Annotated[int, typer.Option("--fps", help="Control rate, Hz.")] = DEFAULT_FPS,
    interpolation: Annotated[
        int, typer.Option("--interpolation", help="Commands per policy action.")
    ] = 1,
    rtc: Annotated[
        bool,
        typer.Option(
            "--rtc/--sync",
            help="RTC runs inference in a background thread. Default is --sync; see HELP_RUN.",
        ),
    ] = False,
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
        bool,
        typer.Option(
            "--display",
            help=(
                "Stream to Rerun: the cameras and the robot state as always, plus one "
                "panel per joint overlaying the policy's own plan, the command the arms "
                "were given, and where they got to."
            ),
        ),
    ] = False,
    display_policy_input: WatchInputOpt = False,
    record: RecordOpt = False,
    record_dir: RecordDirOpt = DEFAULT_RECORD_DIR,
    trace: TraceOpt = True,
    fifo: FifoOpt = True,
    asynchronous: AsyncOpt = True,
    replan_at: ReplanAtOpt = DEFAULT_REPLAN_AT,
    blend: BlendOpt = DEFAULT_BLEND_STEPS,
    device: DeviceOpt = "cuda",
    invert_gripper: InvertOpt = True,
    home: Annotated[
        bool,
        typer.Option(
            "--home/--no-home",
            help=(
                "Sweep the arms to the \\[home] pose when the run ends. ON by default: "
                "leaving them wherever the policy stopped is what wears them."
            ),
        ),
    ] = True,
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
        invert_gripper=invert_gripper,
    )
    # LeRobot's own return_to_initial_position stays off whatever --home says:
    # it fires from teardown on every exit path including a crash, sweeps for a
    # fixed 3 s whether or not the arms arrive, and targets the connect-time
    # pose. dk1lab.home does the job on our terms. See dk1lab/home.py.
    home_pose = None
    if home:
        home_pose = settings.home if settings.home is not None else HOME_AT_START_POSE
    _report(
        cfg, spec, home=home_pose, invert=invert_gripper,
        fifo=fifo, asynchronous=asynchronous, replan_at=replan_at, blend=blend,
    )

    if dry_run:
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    notes = ["The POLICY commands the arms. Nobody has verified what it does on this cell."]
    notes.append(
        "Gripper inversion is ON, as it should be."
        if invert_gripper
        else "Gripper inversion is OFF (--no-invert-gripper) — the grippers will work BACKWARDS."
    )
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
    tracer = _make_trace(
        fps=cfg.fps, enabled=trace, display_policy_input=display_policy_input
    )
    recorder = _make_recorder(
        record,
        record_dir,
        task=task,
        notes={
            "checkpoint": spec,
            "max_joint_rate": limits.max_joint_rate,
            "invert_gripper": invert_gripper,
            "fps": cfg.fps,
        },
    )
    report = run_rollout(
        cfg,
        display=display,
        home=home_pose,
        invert_gripper=invert_gripper,
        trace=tracer,
        fifo=fifo,
        asynchronous=asynchronous,
        replan_at=replan_at,
        blend=blend,
        recorder=recorder,
    )
    typer.secho("\nrollout ended; the robot is disconnected.", fg=typer.colors.GREEN)
    _echo_trace_summary(tracer)
    if recorder is not None:
        _keep_recording(recorder.stop())
    if report is not None:
        colour = typer.colors.GREEN if report.reached else typer.colors.YELLOW
        typer.secho(report.summary(), fg=colour)
        if not report.reached:
            typer.secho(
                "the motors are now disabled — support anything holding itself up.",
                fg=typer.colors.YELLOW,
            )


# --------------------------------------------------------------------------- #
# session — load once, roll out many times
# --------------------------------------------------------------------------- #


HELP_SESSION = """Load the policy once, then run rollout after rollout, task by task.

Everything `dk1 policy run` does, minus the minute it spends loading 10 GiB of
weights, building the CUDA graph, opening three cameras and energising four arms
— because it does that once, at the start, and keeps it. After that a rollout is
one line at the prompt:

  task> pick up the dice          run it
  task>                           run the same instruction again
  task> :record on                write the next episodes to a .rrd
  task> :home                     sweep both arms to the \\[home] pose
  task> :duration 60              change the per-episode limit
  task> :quit                     disconnect and leave

Ctrl-C ends the current rollout and returns you to the prompt; it does not end
the session. A second Ctrl-C during one rollout interrupts for real.

THE ARMS STAY CONNECTED AND ENERGISED BETWEEN ROLLOUTS. That is what makes a
rollout one command — but it means live motors sit in the room while you type.
Nothing is commanded between rollouts, so each arm holds its last target and
says so once ("No command for 0.50 s"); that warning is expected here. Quitting
disconnects, which disables every motor, so support anything held up.

This is the command for SCORING: same policy, same cell, same settings, one
instruction after another, with a success count kept on paper. Everything that
would otherwise differ between attempts is held fixed by construction."""  # noqa: E501


@app.command("session", help=HELP_SESSION + MOTION_HELP)
def session(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    checkpoint: CheckpointOpt = None,
    task: Annotated[
        str | None,
        typer.Option("--task", help="First instruction. Press Enter at the prompt to run it."),
    ] = None,
    duration_s: Annotated[
        float, typer.Option("--duration", help="Seconds per episode. 0 = until stopped.")
    ] = 180.0,
    fps: Annotated[int, typer.Option("--fps", help="Control rate, Hz.")] = DEFAULT_FPS,
    interpolation: Annotated[
        int, typer.Option("--interpolation", help="Commands per policy action.")
    ] = 1,
    rtc: Annotated[
        bool,
        typer.Option("--rtc/--sync", help="RTC runs inference in a background thread."),
    ] = False,
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
        bool, typer.Option("--no-limit", help="Remove the speed cap. Read the warning it prints.")
    ] = False,
    display: Annotated[
        bool,
        typer.Option("--display", help="Stream to Rerun, with one panel per joint."),
    ] = False,
    display_policy_input: WatchInputOpt = False,
    record: RecordOpt = False,
    record_dir: RecordDirOpt = DEFAULT_RECORD_DIR,
    trace: TraceOpt = True,
    fifo: FifoOpt = True,
    asynchronous: AsyncOpt = True,
    replan_at: ReplanAtOpt = DEFAULT_REPLAN_AT,
    blend: BlendOpt = DEFAULT_BLEND_STEPS,
    device: DeviceOpt = "cuda",
    invert_gripper: InvertOpt = True,
    home: Annotated[
        bool,
        typer.Option(
            "--home/--no-home",
            help=(
                "Sweep to the \\[home] pose after every episode. ON by default: leaving "
                "the arms wherever the policy stopped is what wears them."
            ),
        ),
    ] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Build and print everything; connect nothing.")
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    from ..policy import rollout_config
    from ..session import PolicySession

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
        # The task is set per episode; this is only what the banner prints before
        # the first one, and PolicySession.set_task overwrites it every time.
        task=task or "",
        limits=limits,
        control_mode=control_mode,
        fps=fps,
        duration_s=duration_s,
        interpolation=interpolation,
        rtc=rtc,
        execution_horizon=execution_horizon,
        device=device,
        display=display,
        invert_gripper=invert_gripper,
    )
    home_pose = None
    if home:
        home_pose = settings.home if settings.home is not None else HOME_AT_START_POSE
    _report(
        cfg, spec, home=home_pose, invert=invert_gripper,
        fifo=fifo, asynchronous=asynchronous, replan_at=replan_at, blend=blend,
        home_when="after each episode",
    )
    typer.secho("\nsession", bold=True)
    typer.echo("  the policy is loaded ONCE; each episode is one line at the prompt")
    if record:
        from ..record import next_index

        typer.echo(f"  recording ON -> {record_dir}, next episode {next_index(record_dir)}")
        typer.echo("  you are asked whether to keep each episode when it ends")
    else:
        typer.echo(f"  recording off (`:record on` at the prompt) -> {record_dir}")

    if dry_run:
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    notes = [
        "The POLICY commands the arms, once per episode, until you quit.",
        "The arms stay ENERGISED between episodes, holding their last target.",
    ]
    notes.append(
        "Gripper inversion is ON, as it should be."
        if invert_gripper
        else "Gripper inversion is OFF (--no-invert-gripper) — the grippers will work BACKWARDS."
    )
    if limits.max_joint_rate is None:
        notes.append("The speed cap is OFF for this session (--no-limit).")
    if home_pose is not None:
        notes.append("--home: BOTH ARMS SWEEP home after every episode that ends cleanly.")
    confirm_motion(
        "open a policy session on the follower arms",
        assume_yes=assume_yes,
        notes=notes,
    )

    tracer = _make_trace(fps=cfg.fps, enabled=trace, display_policy_input=display_policy_input)
    live = PolicySession(
        cfg,
        display=display,
        invert_gripper=invert_gripper,
        fifo=fifo,
        asynchronous=asynchronous,
        replan_at=replan_at,
        blend=blend,
        trace=tracer,
        record_dir=record_dir,
        record=record,
        duration_s=duration_s,
        notes={
            "checkpoint": spec,
            "max_joint_rate": limits.max_joint_rate,
            "invert_gripper": invert_gripper,
            "fps": cfg.fps,
        },
    )
    typer.secho("\nloading the policy and connecting the cell...", fg=typer.colors.YELLOW)
    live.open()
    if task:
        live.set_task(task)
    typer.secho("\nready. `:help` lists the commands, `:quit` leaves.\n", fg=typer.colors.GREEN)
    try:
        _session_loop(live, tracer, home=home_pose)
    finally:
        typer.secho("\nclosing the session...", fg=typer.colors.YELLOW)
        live.close()
        typer.secho(
            "disconnected. The motors are now disabled — support anything held up.",
            fg=typer.colors.GREEN,
        )


SESSION_HELP_LINES = (
    "  <instruction>   run one episode with that task",
    "  <empty>         run the last task again",
    "  :record on|off  write each episode to a .rrd",
    "  :duration <s>   seconds per episode; 0 runs until Ctrl-C",
    "  :home           sweep both arms to the home pose now",
    "  :help  :quit",
)


def _session_loop(live, tracer, *, home=None) -> None:
    """Read a line, run an episode, print what it did. Until `:quit` or EOF.

    The reading is here and the deciding is in
    :func:`dk1lab.session.parse_command`, which is why the grammar can be tested
    without a robot attached.
    """
    from ..session import DURATION, HELP, HOME, NOTHING, QUIT, RECORD, RUN, parse_command

    while True:
        try:
            line = input(_prompt(live))
        except EOFError:
            typer.echo()
            return
        except KeyboardInterrupt:
            # At the prompt, not in a rollout: nothing to stop. Say so rather
            # than quitting, since quitting from here disables the motors.
            typer.echo("\n(`:quit` to leave)")
            continue

        command = parse_command(line, last_task=live.task)
        if command.error:
            typer.secho(command.error, fg=typer.colors.RED)
            continue
        if command.kind == QUIT:
            return
        if command.kind == NOTHING:
            typer.echo("type an instruction, or `:help`")
            continue
        if command.kind == HELP:
            for text in SESSION_HELP_LINES:
                typer.echo(text)
            continue
        if command.kind == RECORD:
            live.record = bool(command.value)
            typer.echo(f"recording {'ON' if live.record else 'off'} -> {live.record_dir}")
            continue
        if command.kind == DURATION:
            live.duration_s = float(command.value)
            typer.echo(
                f"{live.duration_s:.0f} s per episode"
                if live.duration_s
                else "each episode runs until Ctrl-C"
            )
            continue
        if command.kind == HOME:
            _run_home(live)
            continue
        if command.kind == RUN:
            _run_episode(live, tracer, command.task, home=home)


def _episode_number(live) -> int:
    """The number the next episode will carry.

    While recording that is the **file's** index, read off the recordings
    directory, so the prompt names the file that is about to appear rather than
    a count that restarts with every session. Otherwise there is no file, and
    the session's own count is all there is to show.
    """
    if not live.record:
        return live.episodes + 1
    from ..record import next_index

    return next_index(live.record_dir)


def _prompt(live) -> str:
    """The prompt line: what the next episode will do, before you commit to it."""
    limit = f"{live.duration_s:.0f}s" if live.duration_s else "no limit"
    flags = f"episode {_episode_number(live)} | {limit}"
    if live.record:
        flags += " | rec"
    return f"\n[{flags}] task> "


def _run_episode(live, tracer, task: str, *, home=None) -> None:
    """One rollout, with everything it produced printed after it."""
    typer.secho(
        f"\n>>> episode {_episode_number(live)}: {task!r} — Ctrl-C stops it\n",
        fg=typer.colors.YELLOW,
    )
    try:
        outcome = live.rollout(task, home=home)
    except Exception as exc:  # noqa: BLE001 - one bad episode must not end the session
        typer.secho(f"\nepisode failed: {exc}", fg=typer.colors.RED, err=True)
        typer.secho(
            "the arms are still connected and energised; `:quit` to disconnect.",
            fg=typer.colors.YELLOW,
        )
        return
    typer.secho(f"\n{outcome.summary()}", fg=typer.colors.GREEN)
    _echo_trace_summary(tracer)
    _keep_recording(outcome.recording)
    if outcome.home is not None:
        _echo_home_report(outcome.home)


def _run_home(live) -> None:
    """The home sweep, from the prompt. **Moves the arms.**"""
    typer.secho("\nsweeping both arms home — Ctrl-C stops it where they are\n",
                fg=typer.colors.YELLOW)
    try:
        report = live.home()
    except Exception as exc:  # noqa: BLE001 - a failed sweep must not end the session
        typer.secho(f"home sweep failed: {exc}", fg=typer.colors.RED, err=True)
        return
    _echo_home_report(report)


def _echo_home_report(report) -> None:
    """What the sweep did, and the warning that matters when it did not arrive."""
    typer.secho(
        report.summary(), fg=typer.colors.GREEN if report.reached else typer.colors.YELLOW
    )
    if not report.reached:
        typer.secho(
            "the arms did not reach home — they are still energised, but a disconnect "
            "now would disable the motors.",
            fg=typer.colors.YELLOW,
        )



# --------------------------------------------------------------------------- #
# serve — the same policy, over HTTP, for the ManiSkill sim
# --------------------------------------------------------------------------- #


HELP_SERVE = """Serve the rollout checkpoint over the MolmoAct2 /act protocol.

No robot, no /dev node, no motion — this is a GPU-only command, like `smoke`.
It exists so `sai-prasanna/molmoact2`'s `sim_eval` can drive THE SAME policy the
arms run, without modifying a line of it. Point it here:

  dk1 policy serve                                    # this terminal
  uv run python -m sim_eval.run_eval \
      --policy-type remote-yam \
      --remote-url http://localhost:8202/act \
      -e BimanualYAMPutEverythingInBox-v1             # the other terminal

Two structural differences from a rollout, both because the sim has no real-time
deadline and speaks the checkpoint's own conventions: there is no RTC (sim_eval
blocks on each response, so there is no latency to compensate), and the gripper
inversion is off (sim_eval already sends 1=open, which is what the checkpoint
expects). Behaviour transfers between sim and hardware; timing does not."""


@app.command("serve", help=HELP_SERVE)
def serve(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    checkpoint: CheckpointOpt = None,
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8202,
    device: DeviceOpt = "cuda",
    task: Annotated[
        str | None,
        typer.Option("--task", help="Override the instruction sim_eval sends. Default: use it."),
    ] = None,
    invert_gripper: InvertOpt = False,
    warmup: Annotated[
        bool, typer.Option("--warmup/--no-warmup", help="Run one inference before listening.")
    ] = True,
) -> None:
    from ..serve import serve as run_server

    settings = load(config, require_devices=False)
    spec = _checkpoint(settings, checkpoint)
    capture = settings.profile("policy")

    typer.secho("/act server — no robot is connected by this command", bold=True)
    typer.echo(f"  checkpoint  {spec}")
    typer.echo(f"  device      {device}")
    typer.echo(f"  listening   http://{host}:{port}/act")
    typer.echo(f"  cameras     {', '.join(IMAGE_KEYS)}")
    if task:
        typer.secho(f"  task        OVERRIDDEN to {task!r}", fg=typer.colors.YELLOW)
    if invert_gripper:
        typer.secho(
            "  gripper     INVERTED — note sim_eval already sends 1=open, so this\n"
            "              is very likely wrong here. See dk1lab/serve.py.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo("  gripper     pass-through (sim_eval speaks the YAM convention)")
    typer.echo("\nloading the checkpoint; Ctrl-C to stop the server.\n")

    run_server(
        spec,
        host=host,
        port=port,
        device=device,
        width=capture.width,
        height=capture.height,
        invert_gripper=invert_gripper,
        task=task,
        warmup=warmup,
    )


# --------------------------------------------------------------------------- #
# 5. home — the pose a run ends at, and how to drive there now
# --------------------------------------------------------------------------- #


HELP_HOME = (
    """Drive both follower arms to the \\[home] pose in dk1.toml.

This is the same sweep `dk1 policy run --home` does when a run ends, without
loading the model: ease up to 0.3 rad/s and back down again, stop when the arms
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
        typer.Option(
            "--max-joint-rate",
            help="Peak sweep speed, rad/s. The sweep eases in and out of it. "
            "Default: the slower of 0.3 and the dk1.toml cap.",
        ),
    ] = None,
    show: Annotated[
        bool, typer.Option("--show", help="Print the configured home pose and exit. No motion.")
    ] = False,
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    from ..config import check_devices, write_home
    from ..home import capture_pose, home_rate, sweep_to_home

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
    # An explicit --max-joint-rate is the operator naming a speed, so it is used
    # as given; without one the sweep is slower than the policy cap on purpose.
    rate = float(max_joint_rate) if max_joint_rate else home_rate(limits.max_joint_rate)
    _echo_home(settings.home)
    confirm_motion(
        f"sweep BOTH follower arms to the home pose, easing up to {rate} rad/s",
        assume_yes=assume_yes,
        notes=["Both arms move. Ctrl-C stops the sweep where they are."],
    )
    report = sweep_to_home(
        settings,
        target=settings.home.as_action_dict(),
        limits=limits,
        rate=rate,
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
