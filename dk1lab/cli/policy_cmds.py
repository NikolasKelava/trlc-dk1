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

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

import typer

from .. import checkpoint as ckpt
from ..cameras import crop_summary
from ..config import DEFAULT_CONFIG_PATH, load
from ..layout import ACTION_KEYS, CAMERA_NAMES, GRIPPER_INDICES, IMAGE_KEYS
from ..fifo import DEFAULT_BLEND_STEPS, DEFAULT_REPLAN_AT
from ..record import DEFAULT_RECORD_DIR
from ..dataset import DEFAULT_DATASET_DIR, DEFAULT_DEMO_DIR, DEFAULT_VCODEC, default_repo_id, one
from ..policy import DEFAULT_EXECUTION_HORIZON, DEFAULT_FPS, HOME_AT_START_POSE
from ..finetune import ADAPT_DEFAULT, DEFAULT_RUNS_DIR
from ..runprofile import COMMON, DEFAULT_PROFILE, ProfileError
from ..study import (
    DEFAULT_ATTEMPTS,
    DEFAULT_SCENE_DIR,
    DEFAULT_SCENES,
    DEFAULT_SCORES_DIR,
    DEFAULT_STUDY_RRD_DIR,
    RUBRIC,
    Attempt,
    ScenePlan,
    ScoreError,
)
from .. import study as study_sheet
from ..runprofile import resolve as resolve_profile
from .safety import ENERGISE_HELP, MOTION_HELP, confirm_motion

app = typer.Typer(no_args_is_help=True, help=__doc__)

logger = logging.getLogger(__name__)

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
SimOpt = Annotated[
    bool,
    typer.Option(
        "--sim",
        help=(
            "Drive the MuJoCo cell instead of the arms: no /dev node is opened, no "
            "motor is energised, and nothing in the room moves. Everything else — "
            "the rollout loop, the chunk FIFO, the speed limiter, the home sweep — "
            "is the same code. A sim rollout is evidence that the pipeline runs and "
            "nothing more; it produces no episodes and no scores."
        ),
    ),
]
SimRealtimeOpt = Annotated[
    bool,
    typer.Option(
        "--realtime/--free-run",
        help=(
            "--sim only: keep the sim in step with the wall clock, or let it run as "
            "fast as the model allows. Neither changes what the policy sees."
        ),
    ),
]
SimViewOpt = Annotated[
    bool,
    typer.Option("--view/--no-view", help="--sim only: open mujoco.viewer."),
]
DatasetOpt = Annotated[
    bool,
    typer.Option(
        "--record-dataset",
        help=(
            "Write each episode into a LeRobot dataset v3.0 under --dataset-dir: "
            "the 14-D state, the action the arms were GIVEN, and the three camera "
            "streams as video. This is the format STUDY.md scores and fine-tunes "
            "from, and it is viewable with lerobot-dataset-viz. Independent of "
            "--record, which writes the .rrd carrying the policy's own plan; use "
            "both when an attempt is worth diagnosing as well as keeping."
        ),
    ),
]
DatasetDirOpt = Annotated[
    Path,
    typer.Option(
        "--dataset-dir",
        help=(
            "The dataset directory. Episodes are APPENDED to it, so a scored run of "
            "five attempts is one dataset of five episodes; an existing directory is "
            "resumed rather than replaced."
        ),
    ),
]
LogOpt = Annotated[
    bool,
    typer.Option(
        "--log/--no-log",
        help=(
            "Write everything this run does to logs/<time>-<what>.log, fsynced "
            "line by line. ON by default: this machine has frozen hard twice "
            "mid-session and the terminal does not survive a reset."
        ),
    ),
]
TelemetryOpt = Annotated[
    bool,
    typer.Option(
        "--telemetry/--no-telemetry",
        help=(
            "Sample PSU power and the +12 V rail, CPU and GPU temperature, GPU "
            "power and memory once a second into logs/<time>-<what>.jsonl, each "
            "line fsynced. ON by default, for the same reason: the last line is "
            "the state the machine was in when it stopped."
        ),
    ),
]

VcodecOpt = Annotated[
    str,
    typer.Option(
        "--vcodec",
        help=(
            "Video codec for --record-dataset. `auto` (the default) takes the "
            "hardware encoder when there is one — NVENC here — instead of "
            "LeRobot's SVT-AV1 on the CPU, which spends minutes per episode with "
            "the operator waiting. `libsvtav1` is the old behaviour."
        ),
    ),
]
StreamVideoOpt = Annotated[
    bool,
    typer.Option(
        "--stream-video/--no-stream-video",
        help=(
            "Encode the cameras AS THE ARMS MOVE rather than from a cache of PNG "
            "frames when the episode is kept. Keeping an episode then takes "
            "seconds instead of minutes — but it is paid for out of the CONTROL "
            "LOOP, and on the arms that bill came to a 984 ms tick and six "
            "starved ones. OFF by default, and it should stay off for anything "
            "being scored: the loop is the experiment, and waiting is cheaper "
            "than an attempt the arms stuttered through."
        ),
    ),
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

StudyOpt = Annotated[
    str | None,
    typer.Option(
        "--study",
        help=(
            "Score this session as one row of STUDY.md — R0, A0, A1, B0, B1. The "
            "session then walks the scene layouts in order, three attempts each, "
            "asks for the 0-5 rubric as every attempt ends, and appends it to "
            "study/scores/<row>.csv. Recordings are KEPT without asking: in a "
            "scored row a failure is evidence."
        ),
    ),
]
ScenesOpt = Annotated[
    int, typer.Option("--scenes", help="--study: how many marked scene layouts.")
]
AttemptsOpt = Annotated[
    int, typer.Option("--attempts", help="--study: attempts at each scene.")
]
ScoresDirOpt = Annotated[
    Path, typer.Option("--scores-dir", help="--study: where the score CSVs are written.")
]
SceneDirOpt = Annotated[
    Path, typer.Option("--scene-dir", help="--study: where the scene photographs live.")
]

ProfileOpt = Annotated[
    str,
    typer.Option(
        "--profile",
        help=(
            "How the cell is set up for this run. `optimized` (the default) is "
            "what it already runs: the wrist crop and \\[limits.policy]. `common` "
            "is the level playing field for the two-policy comparison — no wrist "
            "crop, no offset, the full lens on all three cameras, and "
            "\\[limits.study]. Neither writes dk1.toml."
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
    if info.policy_type != "pi05":
        typer.echo(f"  norm_tag      {info.norm_tag}")
        typer.echo(f"  setup_type    {info.setup_type}")
        typer.echo(f"  control_mode  {info.control_mode}")
    typer.echo(f"  chunk         {info.chunk_size} steps, {info.n_action_steps} executed")
    typer.echo(f"  vectors       state {info.state_dim}-D, action {info.action_dim}-D")
    if info.max_state_dim:
        typer.echo(
            f"                (padded widths; narrowed to {len(ACTION_KEYS)}-D at load)"
        )

    if info.policy_type == "pi05":
        _echo_pi05_adaptation(info)
    else:
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


def _family(spec: str) -> str:
    """Which policy a checkpoint holds, read off its own ``config.json``.

    Detected rather than asked for. A ``--policy pi05`` flag would let the flag
    and the directory disagree, and the two are adapted to this cell in different
    ways — 32-D padding and borrowed statistics for one, a gripper inversion for
    the other. A checkpoint that cannot be read locally (a bare Hugging Face repo
    id) is assumed to be MolmoAct2, which is what the [policy] section of dk1.toml points at.
    """
    try:
        return ckpt.read(spec).policy_type or ckpt.EXPECTED_TYPE
    except ckpt.CheckpointError:
        return ckpt.EXPECTED_TYPE


def _smoke_pi05(spec: str, *, task: str, steps: int, device: str, capture, source):
    """π0.5's smoke test, with the adaptation printed before the weights load."""
    from ..pi05 import Pi05Error, load_norm_stats
    from ..pi05 import smoke as run_pi05_smoke

    _echo_pi05_adaptation()
    try:
        stats = load_norm_stats(source)
    except Pi05Error as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.secho(f"\n  {stats.describe()}", fg=typer.colors.GREEN)

    typer.echo("\nchecking the gated tokenizer before loading 14 GB of weights ...")
    try:
        typer.echo("loading ...")
        return run_pi05_smoke(
            spec,
            task=task,
            steps=steps,
            device=device,
            width=capture.width,
            height=capture.height,
            stats=stats,
        )
    except Pi05Error as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _echo_smoke_tail(result, periods: float) -> None:
    """The end of a smoke report for a policy with no RTC path to time."""
    if periods > 1:
        typer.echo(
            "\n  A model call costs more than one control period. The chunk FIFO serves\n"
            "  the whole chunk from a queue while the next one is computed on a worker\n"
            "  thread, which is what keeps the loop at 30 Hz — see `dk1 policy run`."
        )
    typer.echo(f"\npeak GPU memory {result.peak_gpu_gib:.1f} GiB")
    typer.secho("smoke test passed. Nothing was connected.", fg=typer.colors.GREEN)


def _echo_pi05_adaptation(info=None) -> None:
    """What is done to π0.5 to make it drive this cell, and where it came from.

    Printed by every π0.5 command. The borrowed-normalisation line is not a
    footnote: a π0.5 number quoted without it is a number about a different
    experiment.
    """
    from ..pi05 import IMAGE_RENAME, NORM_STATS_REPO

    typer.secho("\nimage keys (renamed at load — the model embeds the views by position)", bold=True)
    for ours, theirs in IMAGE_RENAME.items():
        typer.echo(f"  {ours}  ->  {theirs}")

    typer.secho("\nnormalisation", bold=True)
    typer.secho(
        f"  BORROWED from {NORM_STATS_REPO} (meta/stats.json)",
        fg=typer.colors.YELLOW,
    )
    typer.echo("  π0.5's base checkpoint has no statistics for this robot and cannot make")
    typer.echo("  any, so its saved normalizer covers no features at all. These are 1823")
    typer.echo("  episodes of this exact bi_dk1_follower 14-D layout, and the channel")
    typer.echo("  names are checked against dk1lab.layout before they are used.")
    typer.secho(
        "  Quote every π0.5 zero-shot result as ZERO-SHOT WEIGHTS, BORROWED NORMALISATION.",
        fg=typer.colors.YELLOW,
    )

    typer.secho("\ngripper inversion", bold=True)
    typer.echo("  OFF, always. π0.5's normalisation comes from DK1 data in DK1")
    typer.echo("  convention (0=open), so there is nothing to flip. Inverting here")
    typer.echo("  would introduce the sign error, not remove it.")


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
    norm_stats: Annotated[
        str | None,
        typer.Option(
            "--norm-stats",
            help=(
                "π0.5 only: a local LeRobot dataset to take 14-D normalisation "
                "statistics from. Default: the dk1-merge-2026-03 dataset on the Hub, "
                "which covers this exact layout. Ignored for MolmoAct2, which brings "
                "its own."
            ),
        ),
    ] = None,
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
    settings = load(config)
    spec = _checkpoint(settings, checkpoint)
    capture = settings.profile("policy")
    family = _family(spec)

    typer.secho("smoke test — no robot is connected by this command", bold=True)
    typer.echo(f"  checkpoint  {spec}")
    typer.echo(f"  policy      {family}")
    typer.echo(f"  device      {device}")
    typer.echo(f"  frame       {capture.width}x{capture.height}, 3 views")
    typer.echo(f"  task        {task!r}")

    if family == "pi05":
        result = _smoke_pi05(
            spec, task=task, steps=steps, device=device, capture=capture, source=norm_stats
        )
    else:
        from ..policy import smoke as run_smoke

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
    if not result.rtc_ms:
        # π0.5 is deployed through --sync plus the chunk FIFO, as MolmoAct2 is;
        # timing the RTC path would be measuring something nothing runs.
        _echo_smoke_tail(result, periods)
        return

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


def _profile(name: str):
    """The :class:`~dk1lab.runprofile.RunProfile` ``--profile`` asked for.

    A misspelled name is a :class:`typer.BadParameter` rather than a silent
    fallback to the default: a run under a configuration nobody chose is evidence
    about the wrong thing, which is the whole reason the profiles exist.
    """
    try:
        return resolve_profile(name)
    except ProfileError as exc:
        raise typer.BadParameter(str(exc)) from None


def _limits(settings, max_joint_rate: float | None, no_limit: bool, profile=None):
    """The speed cap this run will use: the profile's table, then the overrides.

    ``profile`` is the run profile, which decides *which* ``[limits.*]`` table is
    read — ``[limits.policy]`` under ``optimized`` and ``[limits.study]`` under
    ``common``. ``None`` means the default profile, which is what the commands
    that have no ``--profile`` of their own (``dk1 policy home``) want.
    """
    profile = resolve_profile(DEFAULT_PROFILE) if profile is None else profile
    limits = profile.limits(settings)
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
    profile=None,
    fifo: bool = True,
    asynchronous: bool = True,
    replan_at: int = DEFAULT_REPLAN_AT,
    blend: int = DEFAULT_BLEND_STEPS,
    home_when: str = "when the run ends",
) -> None:
    """Print everything that was built, before anything is connected."""
    robot = cfg.robot

    _echo_profile(profile)

    pi05 = getattr(cfg.policy, "type", None) == "pi05"

    typer.secho("policy", bold=True)
    typer.echo(f"  checkpoint    {spec}")
    typer.echo(f"  type          {getattr(cfg.policy, 'type', '?')}")
    # MolmoAct2 spells it `model_dtype` and pi0.5 spells it `dtype`; only one of
    # them has an action mode at all.
    dtype = getattr(cfg.policy, "model_dtype", None) or getattr(cfg.policy, "dtype", "?")
    typer.echo(f"  device        {cfg.policy.device}, {dtype}")
    if not pi05:
        typer.echo(f"  action mode   {cfg.policy.inference_action_mode}")
    typer.echo(f"  chunk         {cfg.policy.chunk_size} steps")
    typer.echo(f"  task          {cfg.task!r}")
    typer.echo(f"  inference     {cfg.inference.type}")

    simulated = type(robot).__name__ == "SimRobotConfig"
    if simulated:
        typer.secho("\nrobot: THE SIMULATOR (dk1_sim)", bold=True, fg=typer.colors.GREEN)
        typer.echo("  MuJoCo. No /dev node is opened, no motor is energised, and")
        typer.echo("  nothing in the room moves. A sim rollout is evidence that the")
        typer.echo("  pipeline runs — it is not a score and not a task result.")
        typer.echo(
            f"  {'realtime' if robot.realtime else 'free-run'}, "
            f"{'viewer' if robot.view else 'headless'}, "
            f"{robot.width}x{robot.height} rendered"
        )
    else:
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
    if simulated:
        # Rendered, not opened. The names still have to be this cell's, because
        # they are what the checkpoint's image keys are built from.
        for name in CAMERA_NAMES:
            typer.echo(f"  {name:6s} rendered by MuJoCo at {robot.width}x{robot.height}")
    for name, camera in robot.cameras.items():
        typer.echo(
            f"  {name:6s} {camera.width}x{camera.height} @ {camera.fps} {camera.fourcc}"
            f"  rotation {int(camera.rotation)}  {camera.index_or_path}"
            # What the policy is actually shown. A wrist view cropped to the
            # trained field of view and one left at the lens's own 105 degrees
            # are the same size and the same file; this is what tells them apart.
            + (f"  [{crop}]" if (crop := crop_summary(camera)) else "")
        )

    if pi05:
        _echo_pi05_adaptation()
    else:
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


def _echo_profile(profile) -> None:
    """Which configuration this run is, printed first because it decides the rest.

    ``optimized`` is stated as plainly as ``common`` on purpose. The two differ in
    what the policy is shown and how fast it may move, and a banner that only
    mentioned the unusual one would leave every ordinary run's evidence unlabelled.
    """
    if profile is None:
        return
    colour = None if profile.cropped else typer.colors.YELLOW
    typer.secho("profile", bold=True)
    typer.secho(f"  {profile.name}", fg=colour)
    typer.echo(f"  {profile.summary}")


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


def _start_recording_the_machine(what: str, *, log: bool, telemetry: bool, context: dict):
    """The session log and the machine telemetry. Returns the telemetry, or None.

    Both exist because of the hard freezes of 2026-08-25..26, which left nothing
    behind: the terminal died with the machine and the kernel journal simply
    stops. That fault was the platform firmware and is fixed (`docs/CRASH.md`,
    closed); these files are kept for whatever is unexplained next.
    """
    if log:
        from ..logs import start as start_log

        path = start_log(what)
        typer.echo(f"  log -> {path}")
    if not telemetry:
        return None
    from ..telemetry import DEFAULT_TELEMETRY_DIR, Telemetry

    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    monitor = Telemetry(DEFAULT_TELEMETRY_DIR / f"{stamp}-{what}.jsonl", context=context)
    monitor.start()
    typer.echo(f"  telemetry -> {monitor.path} (PSU, CPU, GPU, once a second)")
    return monitor


def _make_dataset(
    enabled: bool, directory, *, capture, fps: int, profile: str,
    vcodec: str = DEFAULT_VCODEC, streaming: bool = False,
):
    """A :class:`~dk1lab.dataset.DatasetSession`, or ``None``.

    Opened here, before anything is connected, so an unwritable directory or a
    dataset that cannot be resumed is a message rather than a surprise thirty
    seconds into a rollout. The frame size comes from the capture profile the
    run will actually use, because it is declared in the dataset's metadata.
    """
    if not enabled:
        return None
    from ..dataset import DatasetError, DatasetSession

    session = DatasetSession(
        directory,
        fps=fps,
        width=capture.width,
        height=capture.height,
        vcodec=vcodec,
        streaming=streaming,
    )
    try:
        session.open()
    except DatasetError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.secho("\ndataset", bold=True)
    typer.echo(f"  LeRobot v3.0 -> {session.root}  ({session.repo_id})")
    typer.echo(f"  {session.episodes} episode(s) already there; this run appends")
    typer.echo(f"  frames {capture.width}x{capture.height} at {fps} Hz, profile {profile}")
    encoder = session._encoder()
    if encoder is not None:
        typer.echo(
            f"  video {encoder.vcodec}"
            + (", encoded AS THE ARMS MOVE (--stream-video)" if streaming else ", encoded when the episode is kept")
        )
    return session


def _keep_recording(recording, *, ask: bool = True) -> bool:
    """Keep the episode just recorded — asking first, unless a row is being scored.

    Asked **after** the rollout because that is when you know: the episode ends
    when the task is done or has visibly failed, and only then is it clear
    whether the file is worth keeping. The recording itself has to be written as
    the arms move — there is nowhere to put three minutes of video otherwise —
    so declining is a delete, not a decision made in advance.

    Keeping is the default: an accidental Enter should not throw away an attempt
    that cannot be repeated. Non-interactive runs keep everything for the same
    reason, and so does a scored row (``ask=False``) — there the operator is
    asked for a rubric score instead, and an attempt that scored 0 is exactly as
    much evidence as one that scored 5.
    """
    if recording is None:
        return False
    typer.secho(recording.summary(), fg=typer.colors.GREEN)
    keep = True
    if ask and sys.stdin.isatty():
        keep = _confirm("  keep this episode?", default=True)
    if keep:
        # An .rrd is already on disk and needs nothing; a LeRobot episode is
        # buffered until it is written, which is what lets declining one work at
        # all. `keep` exists only on the latter, so this is the whole difference
        # between the two recorders as far as the prompt is concerned.
        if hasattr(recording, "keep"):
            typer.echo("  writing the episode (encoding video) ...")
            recording.keep()
        return True
    if recording.discard():
        typer.secho(f"  discarded {_where(recording)}", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"  could not discard {_where(recording)}", fg=typer.colors.RED, err=True)
    return False


def _where(recording) -> str:
    """What to name in a message about a recording: its file, or its dataset."""
    return str(getattr(recording, "path", None) or getattr(recording, "root", ""))


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

--profile chooses how the cell is set up. `optimized`, the default, is what it
already runs: the wrist crop and \\[limits.policy] at 1.0 rad/s. `common` is the
level playing field STUDY.md defines for the two-policy comparison — no wrist
crop, no offset, the full 105 degree lens on all three cameras, and
\\[limits.study] at 0.6 rad/s. Neither writes dk1.toml; the profile is applied to
the loaded config and printed at the top of the banner.

Do `dk1 policy check`, then `smoke`, then `dryrun` first. Keep a hand on the
e-stop."""
    + MOTION_HELP
)


@app.command("run", help=HELP_RUN)
def run(
    task: TaskOpt,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    checkpoint: CheckpointOpt = None,
    profile: ProfileOpt = DEFAULT_PROFILE,
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
    record_dataset: DatasetOpt = False,
    dataset_dir: DatasetDirOpt = DEFAULT_DATASET_DIR,
    vcodec: VcodecOpt = DEFAULT_VCODEC,
    stream_video: StreamVideoOpt = False,
    log: LogOpt = True,
    telemetry: TelemetryOpt = True,
    sim: SimOpt = False,
    realtime: SimRealtimeOpt = True,
    view: SimViewOpt = True,
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

    chosen = _profile(profile)
    # The profile is applied to the loaded config rather than written to the
    # file: under `common` every camera loses its crop here, once, and every
    # consumer downstream — the camera builders, the banner, the recorder's
    # notes — reads the same derived config. dk1.toml is never touched.
    settings = chosen.apply(load(config, require_devices=not dry_run))
    spec = _checkpoint(settings, checkpoint)
    limits = _limits(settings, max_joint_rate, no_limit, profile=chosen)
    # The gripper inversion is MolmoAct2's, and applying it to pi0.5 would
    # introduce the sign error rather than remove it: pi0.5's normalisation comes
    # from DK1 data already in DK1 convention. Detected off the checkpoint rather
    # than left to the operator, because the default is on and the failure is
    # silent. STUDY.md § The gripper convention.
    invert_gripper = invert_gripper and _family(spec) != "pi05"

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
        sim=sim,
        realtime=realtime,
        view=view,
    )
    # LeRobot's own return_to_initial_position stays off whatever --home says:
    # it fires from teardown on every exit path including a crash, sweeps for a
    # fixed 3 s whether or not the arms arrive, and targets the connect-time
    # pose. dk1lab.home does the job on our terms. See dk1lab/home.py.
    home_pose = None
    if home:
        home_pose = settings.home if settings.home is not None else HOME_AT_START_POSE
    _report(
        cfg, spec, home=home_pose, invert=invert_gripper, profile=chosen,
        fifo=fifo, asynchronous=asynchronous, replan_at=replan_at, blend=blend,
    )

    dataset = _make_dataset(
        record_dataset, dataset_dir, capture=settings.profile("policy"),
        fps=cfg.fps, profile=chosen.name, vcodec=vcodec, streaming=stream_video,
    ) if not dry_run else None

    if dry_run:
        if record_dataset:
            typer.echo(f"\n  --record-dataset would append to {dataset_dir}")
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    if sim:
        typer.secho(
            "\n--sim: the MuJoCo cell. Ctrl-C stops. Nothing in the room moves.\n",
            fg=typer.colors.GREEN,
        )
    notes = ["The POLICY commands the arms. Nobody has verified what it does on this cell."]
    if not chosen.cropped:
        notes.append("--profile common: NO wrist crop. The policy sees the full 105 deg lens.")
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
    if not sim:
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
    typer.secho("\nthe machine itself", bold=True)
    monitor = _start_recording_the_machine(
        "run", log=log, telemetry=telemetry,
        context={"command": "run", "profile": chosen.name, "checkpoint": spec,
                 "task": task, "fps": cfg.fps},
    )
    tracer = _make_trace(
        fps=cfg.fps, enabled=trace, display_policy_input=display_policy_input
    )
    notes_for_recording = {
        "checkpoint": spec,
        "profile": chosen.name,
        "max_joint_rate": limits.max_joint_rate,
        "invert_gripper": invert_gripper,
        "fps": cfg.fps,
        # As in `session`: the one setting here that changes the control loop.
        "stream_video": stream_video,
    }
    recorder = one(
        _make_recorder(record, record_dir, task=task, notes=notes_for_recording),
        dataset.episode(task, notes=notes_for_recording) if dataset is not None else None,
    )
    try:
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
    finally:
        if monitor is not None:
            monitor.stop()
    typer.secho("\nrollout ended; the robot is disconnected.", fg=typer.colors.GREEN)
    _echo_trace_summary(tracer)
    if recorder is not None:
        _keep_recording(recorder.stop())
    if dataset is not None:
        # Finalises the metadata. A v3.0 dataset that is never closed can be
        # missing its last episode even though the frames are on disk.
        dataset.close()
        typer.secho(f"dataset closed: {dataset.root}", fg=typer.colors.GREEN)
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

With --study <row> it also keeps the score sheet. The session walks the marked
scene layouts in order — three attempts at scene 1, then scene 2, then scene 3 —
prints which layout to set up before the first attempt at each, and asks for the
0-5 rubric the moment an attempt ends, appending it to study/scores/<row>.csv.
Recordings are kept without asking: in a scored row a failure is evidence. Use
`:scene <n>` to go back to a layout after a VOID attempt.

Ctrl-C ends the current rollout and returns you to the prompt; it does not end
the session. A second Ctrl-C during one rollout interrupts for real.

THE ARMS STAY CONNECTED AND ENERGISED BETWEEN ROLLOUTS. That is what makes a
rollout one command — but it means live motors sit in the room while you type.
Nothing is commanded between rollouts, so each arm holds its last target and
says so once ("No command for 0.50 s"); that warning is expected here. Quitting
disconnects, which disables every motor, so support anything held up.

This is the command for SCORING: same policy, same cell, same settings, one
instruction after another, and with --study the count is kept in a file rather
than on paper. Everything that
would otherwise differ between attempts is held fixed by construction — the
--profile included, which is fixed for the life of the session and named in
every recording's notes."""  # noqa: E501


@app.command("session", help=HELP_SESSION + MOTION_HELP)
def session(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    checkpoint: CheckpointOpt = None,
    profile: ProfileOpt = DEFAULT_PROFILE,
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
    record_dir: Annotated[
        Path | None,
        typer.Option(
            "--record-dir",
            help=(
                "Where .rrd recordings are written. Default: recordings/, or "
                "study/rrd/<row> under --study, which is what keeps a scored row's "
                "recordings apart from every other row's and from the old ones."
            ),
        ),
    ] = None,
    record_dataset: DatasetOpt = False,
    dataset_dir: DatasetDirOpt = DEFAULT_DATASET_DIR,
    vcodec: VcodecOpt = DEFAULT_VCODEC,
    stream_video: StreamVideoOpt = False,
    log: LogOpt = True,
    telemetry: TelemetryOpt = True,
    study: StudyOpt = None,
    scenes: ScenesOpt = DEFAULT_SCENES,
    attempts: AttemptsOpt = DEFAULT_ATTEMPTS,
    scores_dir: ScoresDirOpt = DEFAULT_SCORES_DIR,
    scene_dir: SceneDirOpt = DEFAULT_SCENE_DIR,
    sim: SimOpt = False,
    realtime: SimRealtimeOpt = True,
    view: SimViewOpt = True,
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

    # A scored row keeps its recordings in its own directory. R0 runs under
    # `optimized`, so its frames must not join any other row's dataset — but they
    # are still the only visual record of the reference row, so they are written
    # and kept rather than mixed into recordings/ or thrown away.
    scored = _study(study, scenes=scenes, attempts=attempts, scores_dir=scores_dir,
                    scene_dir=scene_dir)
    if record_dir is None:
        record_dir = DEFAULT_STUDY_RRD_DIR / study if scored is not None else DEFAULT_RECORD_DIR

    if control_mode not in ("impedance", "pos_vel"):
        raise typer.BadParameter(f"control mode must be impedance or pos_vel, got {control_mode!r}")
    if fps <= 0:
        raise typer.BadParameter(f"--fps must be positive, got {fps}")
    if interpolation < 1:
        raise typer.BadParameter(f"--interpolation must be at least 1, got {interpolation}")

    chosen = _profile(profile)
    # The profile is applied to the loaded config rather than written to the
    # file: under `common` every camera loses its crop here, once, and every
    # consumer downstream — the camera builders, the banner, the recorder's
    # notes — reads the same derived config. dk1.toml is never touched.
    settings = chosen.apply(load(config, require_devices=not dry_run))
    spec = _checkpoint(settings, checkpoint)
    limits = _limits(settings, max_joint_rate, no_limit, profile=chosen)
    # The gripper inversion is MolmoAct2's, and applying it to pi0.5 would
    # introduce the sign error rather than remove it: pi0.5's normalisation comes
    # from DK1 data already in DK1 convention. Detected off the checkpoint rather
    # than left to the operator, because the default is on and the failure is
    # silent. STUDY.md § The gripper convention.
    invert_gripper = invert_gripper and _family(spec) != "pi05"

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
        sim=sim,
        realtime=realtime,
        view=view,
    )
    home_pose = None
    if home:
        home_pose = settings.home if settings.home is not None else HOME_AT_START_POSE
    _report(
        cfg, spec, home=home_pose, invert=invert_gripper, profile=chosen,
        fifo=fifo, asynchronous=asynchronous, replan_at=replan_at, blend=blend,
        home_when="after each episode",
    )
    typer.secho("\nsession", bold=True)
    typer.echo("  the policy is loaded ONCE; each episode is one line at the prompt")
    if record:
        from ..record import next_index

        typer.echo(f"  .rrd ON -> {record_dir}, next episode {next_index(record_dir)}")
        typer.echo(
            "  every episode is kept — this row is scored"
            if scored is not None
            else "  you are asked whether to keep each episode when it ends"
        )
    else:
        typer.echo(f"  .rrd off (`:record on` at the prompt) -> {record_dir}")
    if record_dataset:
        typer.echo(f"  dataset ON -> {dataset_dir}")
    else:
        typer.echo(f"  dataset off (`:dataset on` at the prompt) -> {dataset_dir}")

    if scored is not None:
        _report_study(scored)
        if not record and not record_dataset:
            typer.secho(
                "  nothing is being recorded — a scored row with no frames cannot be "
                "reviewed. Use --record (R0) or --record-dataset (every other row).",
                fg=typer.colors.YELLOW,
            )

    if dry_run:
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    notes = [
        "The POLICY commands the arms, once per episode, until you quit.",
        "The arms stay ENERGISED between episodes, holding their last target.",
    ]
    if not chosen.cropped:
        notes.append("--profile common: NO wrist crop. The policy sees the full 105 deg lens.")
    notes.append(
        "Gripper inversion is ON, as it should be."
        if invert_gripper
        else "Gripper inversion is OFF (--no-invert-gripper) — the grippers will work BACKWARDS."
    )
    if limits.max_joint_rate is None:
        notes.append("The speed cap is OFF for this session (--no-limit).")
    if home_pose is not None:
        notes.append("--home: BOTH ARMS SWEEP home after every episode that ends cleanly.")
    if sim:
        typer.secho(
            "\n--sim: the MuJoCo cell. Nothing in the room moves.\n", fg=typer.colors.GREEN
        )
    else:
        confirm_motion(
            "open a policy session on the follower arms",
            assume_yes=assume_yes,
            notes=notes,
        )

    typer.secho("\nthe machine itself", bold=True)
    monitor = _start_recording_the_machine(
        f"session-{study}" if study else "session",
        log=log,
        telemetry=telemetry,
        context={
            "command": "session", "profile": chosen.name, "study": study,
            "checkpoint": spec, "duration_s": duration_s, "fps": cfg.fps,
        },
    )
    tracer = _make_trace(fps=cfg.fps, enabled=trace, display_policy_input=display_policy_input)
    dataset = _make_dataset(
        record_dataset and not dry_run,
        dataset_dir,
        capture=settings.profile("policy"),
        fps=cfg.fps,
        profile=chosen.name,
        vcodec=vcodec,
        streaming=stream_video,
    )
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
        dataset=dataset,
        record_dataset=record_dataset,
        duration_s=duration_s,
        notes={
            "checkpoint": spec,
            "profile": chosen.name,
            "max_joint_rate": limits.max_joint_rate,
            "invert_gripper": invert_gripper,
            "fps": cfg.fps,
            # The encoding mode belongs in the record because it is the one
            # setting here that changes the CONTROL LOOP, and a row whose
            # episodes were not all recorded the same way is worth knowing about.
            "stream_video": stream_video,
        },
    )
    typer.secho("\nloading the policy and connecting the cell...", fg=typer.colors.YELLOW)
    live.open()
    if task:
        live.set_task(task)
    typer.secho("\nready. `:help` lists the commands, `:quit` leaves.\n", fg=typer.colors.GREEN)
    try:
        _session_loop(live, tracer, home=home_pose, study=scored, monitor=monitor)
    finally:
        typer.secho("\nclosing the session...", fg=typer.colors.YELLOW)
        live.close()
        if monitor is not None:
            monitor.stop()
            typer.echo(f"telemetry written to {monitor.path}")
        if dataset is not None and dataset.failures:
            typer.secho(
                f"\n{len(dataset.failures)} EPISODE(S) WERE NOT WRITTEN:", fg=typer.colors.RED,
                bold=True, err=True,
            )
            for line in dataset.failures:
                typer.secho(f"  {line}", fg=typer.colors.RED, err=True)
        typer.secho(
            "disconnected. The motors are now disabled — support anything held up.",
            fg=typer.colors.GREEN,
        )


@dataclass
class StudyRun:
    """A scored row in progress: where it is, and where its scores go.

    Held here rather than on :class:`~dk1lab.session.PolicySession` because none
    of it reaches the robot — it decides what the prompt says and what lands in
    the CSV, and the session is already the thing that must not grow opinions
    about the study.
    """

    row: str
    plan: ScenePlan
    path: Path
    scene_dir: Path
    #: The scene whose set-up banner has already been printed, so it is printed
    #: once per scene rather than before every prompt.
    announced: int | None = None


def _study(
    row: str | None,
    *,
    scenes: int,
    attempts: int,
    scores_dir: Path,
    scene_dir: Path,
) -> StudyRun | None:
    """Build the scored row, resuming from whatever its CSV already holds."""
    if row is None:
        return None
    path = study_sheet.scores_path(row, scores_dir)
    try:
        done = study_sheet.read(path)
    except ScoreError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        plan = ScenePlan(scenes=scenes, attempts=attempts, done=done)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return StudyRun(row=row, plan=plan, path=path, scene_dir=scene_dir)


def _report_study(scored: StudyRun) -> None:
    """What the scored row is, before anything is connected."""
    plan = scored.plan
    typer.secho("\nscored row", bold=True)
    typer.echo(
        f"  {scored.row}: {plan.scenes} scene(s) x {plan.attempts} attempts "
        f"= {plan.wanted}, scored into {scored.path}"
    )
    if plan.total:
        typer.echo(f"  {plan.total} already scored — resuming at {plan.label()}")
    typer.echo("  every attempt is KEPT: in a scored row a failure is evidence")


def _study_banner(scored: StudyRun) -> None:
    """Ask for the desk to be set up, once per scene."""
    plan = scored.plan
    if not plan.scene_starting or scored.announced == plan.scene:
        return
    typer.secho(f"\n{plan.banner(scored.scene_dir)}", fg=typer.colors.CYAN, bold=True)
    scored.announced = plan.scene


def _score_attempt(scored: StudyRun, outcome) -> None:
    """Ask for the rubric and write the row. Called the moment an episode ends.

    Written here and now rather than reconstructed later, because the only
    person who can score an attempt is the one who just watched it, and the only
    time they can is before the next one.
    """
    plan = scored.plan
    reference = study_sheet.episode_reference(outcome.recording)
    if not sys.stdin.isatty():
        typer.secho(
            "  not a terminal — this attempt was NOT scored. Add it to "
            f"{scored.path} by hand.",
            fg=typer.colors.YELLOW,
        )
        return

    typer.secho(f"\n  score this attempt ({plan.label()})", bold=True)
    for line in RUBRIC:
        typer.echo(f"    {line}")
    typer.echo(
        "    e.g. `3 left dropped it short of the bowl`, `5 right`, `skip`. "
        "A 5 takes its time from the episode."
    )

    while True:
        try:
            text = _ask("  score> ").strip()
        except EOFError:
            typer.secho("\n  not scored.", fg=typer.colors.YELLOW)
            return
        except KeyboardInterrupt:
            # Ctrl-C here is not "stop the rollout" — there is nothing running.
            # The attempt happened; refusing to lose it is the whole point.
            typer.secho("\n  the attempt still needs a score — or type `skip`.",
                        fg=typer.colors.YELLOW)
            continue
        if text.lower() in ("skip", ":skip"):
            typer.secho(
                "  not scored, and not counted. The attempt was void.",
                fg=typer.colors.YELLOW,
            )
            return
        try:
            scoring = study_sheet.parse_score(text)
        except ScoreError as exc:
            typer.secho(f"  {exc}", fg=typer.colors.RED)
            continue
        break

    # Time to success is READ OFF THE EPISODE, not typed: the episode ends when
    # the operator stops it, which is at the success, so asking them for a number
    # from the same clock is a prompt and a chance to mistype. A time typed
    # inline still wins, for the attempt where the arms kept moving afterwards.
    seconds = scoring.seconds
    if scoring.success and seconds is None:
        seconds = outcome.seconds
    attempt = replace(
        scoring,
        scene=plan.scene,
        attempt=plan.next_attempt,
        episode=reference,
        seconds=seconds,
    )
    study_sheet.append(scored.path, attempt)
    plan.record(attempt)
    typer.secho(f"  {attempt.line()} -> {scored.path}", fg=typer.colors.GREEN)
    if plan.complete:
        typer.secho(
            f"\n{scored.row} is complete: {plan.total} attempt(s). "
            f"`dk1 study scores {scored.row}` reads it back.",
            fg=typer.colors.GREEN,
            bold=True,
        )
    else:
        typer.echo(f"  {plan.total}/{plan.wanted} done — next: {plan.label()}")


SESSION_HELP_LINES = (
    "  <instruction>   run one episode with that task",
    "  <empty>         run the last task again",
    "  :record on|off  write each episode to a .rrd",
    "  :dataset on|off append each episode to the LeRobot dataset",
    "  :duration <s>   seconds per episode; 0 runs until Ctrl-C",
    "  :home           sweep both arms to the home pose now",
    "  :scene <n>      --study: go back to a scene, for a VOID attempt",
    "  :help  :quit",
)


@contextmanager
def _quiet_stderr():
    """Silence file descriptor 2 for as long as the block runs.

    The three cameras stream MJPG, and their firmware pads a few bytes before a
    restart marker, so libjpeg prints `Corrupt JPEG data: N extraneous bytes` on
    a good frame. It goes to fd 2 from C, under no logger and past any Python
    redirect — and while the operator is typing at the prompt it lands in the
    middle of the line they are typing. Nothing is commanded here and the arms
    are idle, so there is nothing to miss; everything printed during a rollout,
    where it matters, still goes to the terminal.
    """
    saved = None
    try:
        sys.stderr.flush()
        saved = os.dup(2)
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 2)
        yield
    finally:
        if saved is not None:
            os.dup2(saved, 2)
            os.close(saved)


def _ask(prompt: str) -> str:
    """Read one line with the cameras' JPEG chatter kept off the line.

    **The prompt is written here, to stdout, and not handed to ``input()``.**
    ``input()`` writes its prompt to **file descriptor 2** — the one
    :func:`_quiet_stderr` is pointing at ``/dev/null`` — so passing it through
    sends the prompt to the void. That is what silently removed the session's
    ``[A0 scene 1/3, attempt 1/3 | episode 0 | 120s | dataset] task>`` line and
    the ``score>`` line with it: the operator was left typing at a blank screen,
    with no way to see which scene and attempt were live. Found 2026-08-27.

    Writing it ourselves is what click does on Windows, and for the same reason.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    with _quiet_stderr():
        return input()


def _confirm(question: str, *, default: bool = True) -> bool:
    """``typer.confirm``, but with the question actually visible.

    Click hands the whole prompt to ``input()`` on this platform, so it goes to
    fd 2 and :func:`_quiet_stderr` eats it exactly as it ate the ``task>`` line.
    The parsing is click's rules: empty accepts the default, ``y``/``n`` decide,
    anything else asks again, and EOF takes the default rather than raising —
    a non-interactive keep must not become an abort.
    """
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            answer = _ask(question + suffix).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        typer.secho("  answer y or n", fg=typer.colors.RED)


def _session_loop(live, tracer, *, home=None, study=None, monitor=None) -> None:
    """Read a line, run an episode, print what it did. Until `:quit` or EOF.

    The reading is here and the deciding is in
    :func:`dk1lab.session.parse_command`, which is why the grammar can be tested
    without a robot attached.
    """
    from ..session import (
        DATASET, DURATION, HELP, HOME, NOTHING, QUIT, RECORD, RUN, SCENE, parse_command,
    )

    while True:
        if study is not None:
            _study_banner(study)
        try:
            line = _ask(_prompt(live, study))
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
            typer.echo(f".rrd {'ON' if live.record else 'off'} -> {live.record_dir}")
            continue
        if command.kind == DATASET:
            if live.dataset is None:
                typer.secho(
                    "this session has no dataset open — start it with --record-dataset, "
                    "since the directory has to be resolved before anything connects",
                    fg=typer.colors.YELLOW,
                )
                continue
            live.record_dataset = bool(command.value)
            state = "ON" if live.record_dataset else "off"
            typer.echo(f"dataset {state} -> {live.dataset.root}")
            continue
        if command.kind == DURATION:
            live.duration_s = float(command.value)
            typer.echo(
                f"{live.duration_s:.0f} s per episode"
                if live.duration_s
                else "each episode runs until Ctrl-C"
            )
            continue
        if command.kind == SCENE:
            if study is None:
                typer.secho(
                    "this session is not scoring a row — start it with --study <row>",
                    fg=typer.colors.YELLOW,
                )
                continue
            try:
                study.plan.jump(int(command.value))
            except ValueError as exc:
                typer.secho(str(exc), fg=typer.colors.RED)
                continue
            study.announced = None
            typer.echo(f"back to {study.plan.label()} — for a VOID attempt; say so in the note")
            continue
        if command.kind == HOME:
            _run_home(live)
            continue
        if command.kind == RUN:
            _run_episode(live, tracer, command.task, home=home, study=study, monitor=monitor)


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


def _prompt(live, study=None) -> str:
    """The prompt line: what the next episode will do, before you commit to it."""
    limit = f"{live.duration_s:.0f}s" if live.duration_s else "no limit"
    flags = f"episode {_episode_number(live)} | {limit}"
    if study is not None:
        flags = f"{study.row} {study.plan.label()} | {flags}"
    if live.record:
        flags += " | rrd"
    if live.record_dataset:
        flags += " | dataset"
    return f"\n[{flags}] task> "


def _run_episode(live, tracer, task: str, *, home=None, study=None, monitor=None) -> None:
    """One rollout, with everything it produced printed after it."""
    typer.secho(
        f"\n>>> episode {_episode_number(live)}: {task!r} — Ctrl-C stops it\n",
        fg=typer.colors.YELLOW,
    )
    if monitor is not None:
        # An event line in the telemetry, so a freeze can be placed against what
        # the operator was doing: mid-rollout, encoding, or sitting at the prompt.
        monitor.mark(event="episode_start", episode=_episode_number(live), task=task)
    logger.info("episode %d starting: %r", _episode_number(live), task)
    try:
        outcome = live.rollout(task, home=home)
    except Exception:  # noqa: BLE001 - one bad episode must not end the session
        logger.exception("episode failed")
        exc = sys.exc_info()[1]
        typer.secho(f"\nepisode failed: {exc}", fg=typer.colors.RED, err=True)
        typer.secho(
            "the arms are still connected and energised; `:quit` to disconnect.",
            fg=typer.colors.YELLOW,
        )
        return
    typer.secho(f"\n{outcome.summary()}", fg=typer.colors.GREEN)
    logger.info("%s", outcome.summary())
    _echo_trace_summary(tracer)
    written = _keep_recording(outcome.recording, ask=study is None)
    if monitor is not None:
        monitor.mark(event="episode_end", seconds=round(outcome.seconds, 1), kept=written)
    _warn_if_nothing_was_written(live, outcome, written)
    if study is not None:
        _score_attempt(study, outcome)
    if outcome.home is not None:
        _echo_home_report(outcome.home)


def _warn_if_nothing_was_written(live, outcome, written: bool) -> None:
    """Say it in red, immediately, when an attempt produced no file.

    On 2026-08-26 a scored session recorded nothing at all: the first episode's
    video encode failed, the failure was one log line among many, and five more
    attempts were run and scored against a dataset that stayed empty. The
    operator has to learn this while the arms are still warm.
    """
    dataset = getattr(live, "dataset", None)
    failures = list(getattr(dataset, "failures", ()) or ())
    if written and not failures:
        return
    if outcome.recording is None and not live.record and not live.record_dataset:
        return  # nothing was asked for; nothing missing
    typer.secho(
        "\n  !! THIS ATTEMPT WAS NOT WRITTEN. The score will have no frames behind it.",
        fg=typer.colors.RED, bold=True, err=True,
    )
    if failures:
        typer.secho(f"  !! {failures[-1]}", fg=typer.colors.RED, err=True)
    typer.secho(
        "  !! the log file has the traceback. Stop and fix this before scoring more.",
        fg=typer.colors.RED, err=True,
    )


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


# --------------------------------------------------------------------------- #
# The fine-tune
# --------------------------------------------------------------------------- #

HELP_FINETUNE = """LoRA fine-tune a policy on the recorded demonstrations. GPU only.

STUDY.md Phase 4. Runs LeRobot's own trainer through its PEFT path, over the
demonstrations recorded by `dk1 teleop --record-dataset`, and writes the run
directory the protocol requires: the base checkpoint's hash, the dk1.toml in
force, the command line, and the git SHA.

    dk1 policy finetune --row R1        MolmoAct2 on the tuned rig  (the wrist crop)
    dk1 policy finetune --row A1        MolmoAct2 on the level playing field
    dk1 policy finetune --row B1        pi0.5, same recipe, same bytes

The recipe is fixed by the protocol and is the same for every row: LoRA r=32,
alpha=16, dropout=0.05, no fully-trained modules, a STEP budget rather than
epochs, and ten episodes held out for validation. The hold-out is spread evenly
across the three scene configurations, because the demonstrations are recorded
grouped by scene and the last ten of them are all one layout.

R1 trains on the CROPPED copy of the demonstrations — `dk1 dataset crop` makes
it — and this command refuses to start if the dataset's lens does not match the
row's profile. A1 and B1 train on the demonstrations as recorded.

The arms are not involved and nothing is energised. It holds the GPU for hours."""


@app.command("finetune", help=HELP_FINETUNE)
def finetune(
    row: Annotated[
        str, typer.Option("--row", help="Which STUDY.md row to train: R1, A1 or B1.")
    ] = "R1",
    dataset_dir: Annotated[
        Path | None,
        typer.Option(
            "--dataset-dir",
            help="The demonstrations. Default: study/demos-optimized for R1, study/demos otherwise.",
        ),
    ] = None,
    checkpoint: CheckpointOpt = None,
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Where run directories go.")
    ] = DEFAULT_RUNS_DIR,
    steps: Annotated[int | None, typer.Option("--steps", help="Optimiser updates. NOT epochs.")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", help="Frames per update.")] = None,
    lr: Annotated[
        float | None,
        typer.Option("--lr", help="Peak learning rate. Default 1e-4, not the checkpoint's 1e-5."),
    ] = None,
    holdout: Annotated[
        int | None, typer.Option("--holdout", help="Episodes reserved for validation.")
    ] = None,
    eval_every: Annotated[
        int | None,
        typer.Option(
            "--eval-every",
            help="Steps between held-out evaluations. A checkpoint is written at each one.",
        ),
    ] = None,
    num_workers: Annotated[int | None, typer.Option("--num-workers", help="Dataloader workers.")] = None,
    max_eval_samples: Annotated[
        int | None,
        typer.Option(
            "--max-eval-samples",
            help=(
                "Cap the frames one evaluation reads. 0 reads the whole hold-out, which "
                "is ~1400 forward passes EVERY --eval-every steps. LeRobot takes the "
                "first N frames of the hold-out, so a cap buys time with a smaller and "
                "more front-loaded validation set."
            ),
        ),
    ] = None,
    adapt: Annotated[
        str,
        typer.Option(
            "--adapt",
            help=(
                "What the adapter attaches to. 'default' is each policy's own choice, "
                "which for MolmoAct2 is the VLM ONLY and leaves its 578M action expert "
                "frozen; 'vlm+expert' extends MolmoAct2's own regex over the action "
                "expert, which is what pi0.5's default already covers."
            ),
        ),
    ] = ADAPT_DEFAULT,
    wandb_project: Annotated[
        str | None, typer.Option("--wandb", help="Log to this Weights & Biases project.")
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Train even though the dataset check found problems."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Write the run directory and print the command line; train nothing.",
        ),
    ] = False,
    log: Annotated[bool, typer.Option("--log/--no-log", help="Write train.log into the run dir.")] = True,
    assume_yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    from dataclasses import replace as _replace

    from ..dataset import DatasetError, summarise
    from ..finetune import (
        DEFAULT_BUDGET,
        RECIPE,
        FinetuneError,
        describe_adapt,
        run_name,
        split_episodes,
        target_modules,
        train_argv,
        train_dir,
        write_run_dir,
    )
    from ..finetune import row as resolve_row
    from ..finetune import run as run_training
    from ..recrop import lens_profile

    settings = load(config, require_devices=False)
    try:
        what = resolve_row(row)
    except FinetuneError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    directory = dataset_dir or (
        DEFAULT_DEMO_DIR.with_name(DEFAULT_DEMO_DIR.name + "-optimized")
        if what.cropped
        else DEFAULT_DEMO_DIR
    )
    spec = ckpt.resolve(_checkpoint(settings, checkpoint))

    budget = _replace(
        DEFAULT_BUDGET,
        **{
            name: value
            for name, value in (
                ("steps", steps),
                ("batch_size", batch_size),
                ("lr", lr),
                ("holdout", holdout),
                ("eval_steps", eval_every),
                ("save_freq", eval_every),
                ("num_workers", num_workers),
                ("max_eval_samples", max_eval_samples),
            )
            if value is not None
        },
    )

    try:
        target_modules(what.family, adapt)  # a bad --adapt is a typo, not a run
        summary = summarise(directory)
        split = split_episodes(summary.scenes, budget.holdout)
    except (DatasetError, FinetuneError) as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # The lens is the invariant STUDY.md names as the one that will be violated
    # if any is: train on frames the policy will never be shown again and the
    # fine-tune is for a camera that does not exist.
    lens = lens_profile(summary.root, summary.notes)
    if lens != what.profile:
        typer.secho(
            f"\n{directory} carries the {lens or 'unknown'} lens, but {what.name} rolls out "
            f"under {what.profile}.\n"
            + (
                "  Make the cropped copy first:\n"
                f"    dk1 dataset crop {directory} {directory}-optimized\n"
                if what.cropped and lens == COMMON
                else "  Point --dataset-dir at the dataset recorded for this profile.\n"
            ),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    directory_name = run_name(what)
    output = Path(runs_dir) / directory_name

    typer.secho(f"\n{what.name} — {what.summary}", bold=True)
    typer.echo(f"  checkpoint  {spec}")
    typer.echo(f"  dataset     {directory} — {summary.episodes} episodes, {summary.frames} frames")
    typer.echo(f"  lens        {lens}")
    typer.echo(f"  split       {split.describe()}")
    typer.echo(f"  recipe      {RECIPE.describe()}")
    typer.echo(f"  adapting    {describe_adapt(what.family, adapt)}")
    typer.echo(
        f"  budget      {budget.steps} steps, batch {budget.batch_size}, lr {budget.lr:g}, "
        f"warmup {budget.warmup}, eval+checkpoint every {budget.eval_steps}, "
        f"eval reads {budget.max_eval_samples or 'the whole hold-out'}"
    )
    typer.echo(f"  run dir     {output}")
    # Said here rather than only once training starts, so it is on screen before
    # the operator walks away and in the --dry-run they read first.
    typer.echo(
        f"  stopping    dk1 policy pause {output} — stops at the next checkpoint, "
        f"losing nothing; Ctrl-C stops now, losing at most {budget.save_freq} steps"
    )

    if summary.problems:
        typer.secho(f"\n{len(summary.problems)} problem(s) with the dataset:", fg=typer.colors.RED)
        for problem in summary.problems:
            typer.secho(f"  - {problem}", fg=typer.colors.RED)
        if not force:
            typer.secho(
                "\nrefusing to train. Run `dk1 dataset check` and fix them, or pass "
                "--force if they are understood and acceptable.\n",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        typer.secho("--force: training anyway.\n", fg=typer.colors.YELLOW)

    argv = train_argv(
        dataset_root=directory,
        repo_id=summary.repo_id or default_repo_id(directory),
        checkpoint=spec,
        output_dir=train_dir(output),
        job_name=f"dk1-{what.name.lower()}",
        split=split,
        budget=budget,
        recipe=RECIPE,
        family=what.family,
        adapt=adapt,
        wandb_project=wandb_project,
    )

    typer.echo("\nlerobot-train \\\n  " + " \\\n  ".join(argv))

    typer.secho(
        f"\nafter this, {what.name} is rolled out with --no-invert-gripper: the "
        f"demonstrations are in DK1 convention, so the fine-tuned weights speak DK1 "
        f"and inverting would flip every grasp.",
        fg=typer.colors.YELLOW,
    )

    typer.echo("\nhashing the base checkpoint ...")
    write_run_dir(
        output,
        what=what,
        argv=argv,
        checkpoint=spec,
        config_path=config,
        dataset_root=directory,
        split=split,
        budget=budget,
        recipe=RECIPE,
        notes={
            "dataset_episodes": summary.episodes,
            "dataset_frames": summary.frames,
            "dataset_lens": lens,
            "dataset_settings": {name: values for name, values in summary.settings.items()},
            "dataset_problems": summary.problems,
            "adapt": adapt,
            "adapting": describe_adapt(what.family, adapt),
            "wandb_project": wandb_project,
        },
    )
    typer.secho(f"wrote {output}/dk1_run.json, dk1.toml, command.txt", fg=typer.colors.GREEN)

    if dry_run:
        typer.secho("\n--dry-run: nothing was trained.\n", fg=typer.colors.GREEN)
        return

    if not assume_yes:
        typer.secho(
            f"\nThis loads {spec} onto the GPU and trains for {budget.steps} steps — hours. "
            f"Nothing is energised and no arm moves.",
            fg=typer.colors.YELLOW,
        )
        if not _confirm("start the training run?", default=False):
            typer.secho("nothing was trained.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

    if log:
        from ..logs import start as start_log

        typer.echo(f"log -> {start_log('finetune', path=output / 'train.log')}")

    typer.secho(f"\ntraining {what.name} ...\n", fg=typer.colors.GREEN)
    outcome = run_training(argv, checkpoint=spec, say=typer.echo, run_dir=output)
    _echo_finetune_outcome(outcome, output, budget.steps)


def _echo_finetune_outcome(outcome: str, run_dir: Path, budget_steps: int | None = None) -> None:
    """What happened, and the one command that follows from it.

    A paused or interrupted run is an ordinary outcome with a checkpoint behind
    it, not a failure — and the difference between the two is only how many steps
    were lost, so both are reported the same way and neither exits non-zero.

    A stop that lands on the **last** checkpoint is a finished run, not a paused
    one: telling the operator to resume a run with no steps left is how somebody
    spends a morning waiting for a training job that exits immediately.
    """
    from ..finetune import COMPLETED, PAUSED, checkpoint_steps, clear_stop

    saved = checkpoint_steps(run_dir)
    if outcome != COMPLETED and budget_steps and saved and max(saved) >= budget_steps:
        clear_stop(run_dir)
        outcome = COMPLETED

    if outcome == COMPLETED:
        typer.secho("\ntrained to the end of the budget.", fg=typer.colors.GREEN)
    else:
        if outcome == PAUSED:
            clear_stop(run_dir)
            typer.secho(
                "\npaused at a checkpoint — nothing was lost, and the stop request "
                "has been cleared.",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(
                "\ninterrupted. The last checkpoint written is intact; the steps "
                "since it are gone.",
                fg=typer.colors.YELLOW,
            )
        typer.echo(f"  continue with:  dk1 policy resume {run_dir}")
    typer.secho("\nread the curve back with:", bold=True)
    typer.echo(f"  dk1 policy curve {run_dir}\n")


HELP_CURVE = """Read a fine-tune's held-out loss back, and name the checkpoint to deploy.

There is no early stop: LeRobot 0.6.1 measures a held-out loss every --eval-every
steps and logs it, and nothing acts on it. So the budget runs to the end, a
checkpoint is written at every evaluation, and this picks the one with the lowest
held-out loss out of the ones that exist.

Reads two text files — the training log and the checkpoint directory listing. No
GPU, no model, no robot."""


@app.command("curve", help=HELP_CURVE)
def curve(
    run_dir: Annotated[Path, typer.Argument(help="A run directory written by `dk1 policy finetune`.")],
) -> None:
    import json as _json

    from ..finetune import best_checkpoint, checkpoint_steps, deployable, eval_losses

    log_path = Path(run_dir) / "train.log"
    record_path = Path(run_dir) / "dk1_run.json"
    if not log_path.is_file():
        typer.secho(
            f"\n{log_path} does not exist — the run was started with --no-log, or "
            f"{run_dir} is not a run directory.\n",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    record = _json.loads(record_path.read_text()) if record_path.is_file() else {}
    total = int((record.get("budget") or {}).get("steps") or 0)
    saved = checkpoint_steps(run_dir)
    losses = eval_losses(log_path.read_text(encoding="utf-8", errors="replace"))

    typer.secho(f"\n{run_dir}", bold=True)
    if record:
        typer.echo(f"  row {record.get('row')} — {record.get('summary')}")
        typer.echo(f"  base {record.get('checkpoint')}")
        typer.echo(f"  sha256 {record.get('checkpoint_sha256') or '(not hashed)'}")
        typer.echo(f"  git {(record.get('git') or {}).get('sha') or '?'} "
                   f"{'(clean)' if (record.get('git') or {}).get('clean') else '(DIRTY)'}")

    if not losses:
        typer.secho(
            "\nno held-out loss was logged. Either --eval-every was never reached, or "
            "the hold-out was empty.\n",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    typer.secho("\nheld-out loss", bold=True)
    for step, loss in losses:
        mark = "  *" if step in set(saved) else "   "
        typer.echo(f"  {step:>8} {loss:.5f}{mark}")
    typer.echo("  * a checkpoint exists at this step")

    best = best_checkpoint(log_path.read_text(encoding="utf-8", errors="replace"), available=saved)
    if best is None:
        typer.secho(
            "\nno step has both a measured loss and a checkpoint on disk. Set "
            "--eval-every equal to the checkpoint interval and run again.\n",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    step, loss = best
    where = deployable(run_dir, step, total or step)
    typer.secho(f"\nbest: step {step}, held-out loss {loss:.5f}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {where}")
    if losses and best[0] == losses[-1][0]:
        typer.secho(
            "  the last evaluation is the best one — the curve was still improving, so "
            "the budget may be short rather than the recipe wrong.",
            fg=typer.colors.YELLOW,
        )
    row = str(record.get("row") or "A1")
    profile = str(record.get("profile") or COMMON)
    typer.secho("\nscore it with:", bold=True)
    typer.echo(
        f"  dk1 policy session --study {row} --profile {profile} --duration 120 \\\n"
        f"    --checkpoint {where} --no-invert-gripper \\\n"
        f"    --record-dataset --dataset-dir study/rollouts/{row}\n"
    )
    typer.secho(
        "--no-invert-gripper is not optional: the fine-tune learned DK1 convention "
        "from the demonstrations, and inverting it flips every grasp. Confirm it on "
        "the FIRST episode before spending nine attempts.\n",
        fg=typer.colors.YELLOW,
    )


HELP_RESUME = """Continue a fine-tune from its last checkpoint. GPU only.

A run that was paused (`dk1 policy pause`) or interrupted picks up where its
last checkpoint left off — same dataset, same split, same recipe, same budget and
the same learning-rate schedule, because all of it comes back from the
checkpoint's own train_config.json rather than from what is typed here.

    dk1 policy resume study/finetune/R1-<stamp>
    dk1 policy resume study/finetune/R1-<stamp> --step 4000    # an earlier one

Each session is appended to dk1_resume.jsonl, so a checkpoint can say it came
from more than one sitting. The arms are not involved and nothing is energised."""


@app.command("resume", help=HELP_RESUME)
def resume(
    run_dir: Annotated[Path, typer.Argument(help="A run directory from `dk1 policy finetune`.")],
    step: Annotated[
        int | None,
        typer.Option("--step", help="Resume from this checkpoint instead of the last one."),
    ] = None,
    steps: Annotated[
        int | None,
        typer.Option(
            "--steps",
            help=(
                "Raise the budget. Past the original, the learning rate is already at "
                "its floor and the extra steps barely move the adapter."
            ),
        ),
    ] = None,
    log: Annotated[bool, typer.Option("--log/--no-log", help="Append to the run's train.log.")] = True,
    assume_yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    from ..finetune import (
        FinetuneError,
        checkpoint_steps,
        clear_stop,
        decay_warning,
        latest_checkpoint,
        note_resume,
        read_run,
        resume_argv,
    )
    from ..finetune import run as run_training

    try:
        checkpoint = latest_checkpoint(run_dir, step)
        argv = resume_argv(run_dir, step=step, steps=steps)
    except FinetuneError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    record = read_run(run_dir)
    budget = record.get("budget") or {}
    saved = checkpoint_steps(run_dir)
    at = int(checkpoint.name) if checkpoint.name.isdigit() else None

    typer.secho(f"\nresuming {record.get('row') or run_dir.name}", bold=True)
    typer.echo(f"  run dir     {run_dir}")
    typer.echo(f"  from        {checkpoint}" + (f"  (step {at})" if at else ""))
    typer.echo(f"  checkpoints {saved or 'none'}")
    typer.echo(f"  budget      {steps or budget.get('steps') or '?'} steps "
               f"(recorded {budget.get('steps') or '?'})")
    typer.echo(f"  adapting    {record.get('adapting') or 'as recorded'}")
    typer.echo("\nlerobot-train \\\n  " + " \\\n  ".join(argv))

    # A stop request left behind would stop the resumed run at its first
    # checkpoint, which looks exactly like it refusing to run.
    if clear_stop(run_dir):
        typer.secho("\ncleared the pending stop request.", fg=typer.colors.YELLOW)

    if steps is not None and (warning := decay_warning(run_dir, steps)):
        typer.secho(f"\n{warning}", fg=typer.colors.YELLOW)

    if not assume_yes and not _confirm("continue this run?", default=True):
        typer.secho("nothing was trained.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    if log:
        from ..logs import start as start_log

        typer.echo(f"log -> {start_log('finetune', path=Path(run_dir) / 'train.log')}")

    note_resume(run_dir, argv=argv, checkpoint=checkpoint)
    typer.secho(
        f"\ncontinuing ...\n  pause it again with:  dk1 policy pause {run_dir}\n",
        fg=typer.colors.GREEN,
    )
    outcome = run_training(
        argv,
        checkpoint=checkpoint / "pretrained_model",
        say=typer.echo,
        run_dir=run_dir,
    )
    _echo_finetune_outcome(outcome, Path(run_dir), steps or budget.get("steps"))


HELP_PAUSE = """Ask a running fine-tune to stop at its next checkpoint.

Writes a STOP file into the run directory. The training loop reads it right after
each checkpoint is written, so it stops having lost nothing — unlike Ctrl-C, which
stops now and gives up everything since the last save.

    dk1 policy pause study/finetune/R1-<stamp>            ask it to stop
    dk1 policy pause study/finetune/R1-<stamp> --cancel   change your mind

Resume with `dk1 policy resume`. Writes one small file and nothing else."""


@app.command("pause", help=HELP_PAUSE)
def pause(
    run_dir: Annotated[Path, typer.Argument(help="A run directory from `dk1 policy finetune`.")],
    cancel: Annotated[
        bool, typer.Option("--cancel", help="Withdraw a stop request instead of making one.")
    ] = False,
) -> None:
    from ..finetune import clear_stop, request_stop

    if not Path(run_dir).is_dir():
        typer.secho(f"\n{run_dir} does not exist.\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if cancel:
        if clear_stop(run_dir):
            typer.secho(f"\nstop request withdrawn; {run_dir} runs on.\n", fg=typer.colors.GREEN)
            return
        typer.secho(f"\n{run_dir} had no stop request.\n", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    path = request_stop(run_dir)
    typer.secho(f"\nwrote {path}", fg=typer.colors.GREEN)
    typer.echo(
        "  training stops after the next checkpoint is written — nothing is lost.\n"
        f"  cancel with:  dk1 policy pause {run_dir} --cancel\n"
        f"  continue with: dk1 policy resume {run_dir}\n"
    )


__all__ = ["app"]
