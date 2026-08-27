"""``dk1 teleop`` — drive the followers from the leader arms.

The one teleoperation entry point. Everything it needs about the devices comes
from ``dk1.toml``; the only things this command decides are how fast, whether to
show you the cameras, how tight the speed limit is — and, with
``--record-dataset``, whether what the operator does with their hands is written
down.

**Two words that were the same word, and are not any more.** ``--capture`` is the
``[capture.*]`` table the cameras run at; ``--profile`` is
``optimized``/``common``, meaning here exactly what it means on ``dk1 policy run``
and ``dk1 policy session``. It used to be ``--profile`` for the first of those,
which put two meanings on one word on the command line that records the dataset
both fine-tunes are built from.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from ..cameras import crop_summary
from ..modelview import DEFAULT_EVERY
from ..config import DEFAULT_CONFIG_PATH, load
from ..dataset import DEFAULT_DATASET_DIR, DEFAULT_VCODEC
from ..runprofile import COMMON, DEFAULT_PROFILE, ProfileError, resolve
from ..teleop import DEFAULT_FPS, TELEOP_LIMITS, build, run
from .safety import LEADER_HELP, MOTION_HELP, confirm_motion

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]

#: Where the demonstrations go. ``STUDY.md`` names this directory; it is not in
#: git — it is a LeRobot dataset and belongs on the Hugging Face Hub.
DEFAULT_DEMO_DIR = DEFAULT_DATASET_DIR / "demos"

#: The loop rate a recorded demonstration runs at. **The policy's rate, not
#: teleoperation's.** It is written into the dataset's metadata and it is what
#: gives every action chunk its time scale, so recording at 60 Hz and fine-tuning
#: a 30 Hz policy on it would teach the model a cell that moves at half speed.
DEMO_FPS = 30

HELP = (
    """Teleoperate: the followers track the leader arms.

Devices come from dk1.toml. Cameras are named top / left / right, which is what
the MolmoAct2 checkpoint requires and therefore what recording will need — the
naming is not a free choice, so it is not an option here.

Ctrl-C stops. Stopping disconnects and does nothing else: the arms are never
swept home, because sweeping them home is the last thing you want when you
stopped because something was wrong.

Use --dry-run to see exactly what would be built without connecting to anything,
which is worth doing before the first run on new hardware.

--record-dataset turns this into the demonstration recorder STUDY.md Phase 3
needs: it asks for the task once, then Enter starts an episode and Enter ends it,
`again` throws the last one away, `scene <n>` says which layout the ones that
follow are of, and `done` finishes. Teleoperation does not stop between
episodes — the arms keep tracking the leaders throughout, which is what keeps the
first tick of the next episode from being a jump.

Recording changes three defaults, because a demonstration is only worth having if
it matches the rollout it will be fine-tuned for: --profile common (the full
lens), --capture policy (1280x720) and --fps 30 (the policy's rate, which is
written into the dataset). All three are printed below; pass them explicitly to
say otherwise."""
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
    profile=None,
    capture: str = "teleop",
) -> None:
    """Print what is about to run — shared by --dry-run and the real thing."""
    if profile is not None:
        typer.secho(f"profile {profile.name}", bold=True)
        typer.echo(f"  {profile.summary}")
        typer.echo(f"  capture [{capture}] — the limits are teleop's, whatever the profile")
        typer.echo("")
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


def _task(given: str | None) -> str:
    """The instruction for the whole session — the flag, or asked for once.

    Written to stdout rather than handed to ``input()``, for the reason
    ``policy_cmds._ask`` documents: ``input()`` puts its prompt on fd 2, which is
    the descriptor the cameras' libjpeg chatter is silenced through, so a prompt
    passed to it disappears. An empty answer is asked again — an episode recorded
    with no instruction is an episode nothing can be conditioned on.
    """
    import sys

    task = (given or "").strip()
    while not task:
        sys.stdout.write("\ntask (recorded on every frame of every episode)> ")
        sys.stdout.flush()
        try:
            task = input().strip()
        except EOFError:
            raise typer.BadParameter("no task given; pass --task") from None
    return task


def _open_dataset(directory: Path, *, capture, fps: int, vcodec: str, streaming: bool):
    """The :class:`~dk1lab.dataset.DatasetSession`, opened before anything moves.

    Opening here rather than at the first episode is what makes an unwritable
    directory, or one holding somebody else's dataset, a message on a cold cell
    instead of a surprise with four arms energised.
    """
    from ..dataset import DatasetError, DatasetSession

    live = DatasetSession(
        directory,
        fps=fps,
        width=capture.width,
        height=capture.height,
        vcodec=vcodec,
        streaming=streaming,
    )
    try:
        live.open()
    except DatasetError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"\ndataset {live.root} ({live.repo_id}) — {live.episodes} episode(s) already there")
    return live


def _report_dataset(live, session) -> None:
    """What the session actually left behind. Failures in red, and by name."""
    typer.secho("\ndemonstrations", bold=True)
    typer.echo(f"  {session.written} of {session.attempts} episode(s) written to {live.root}")
    for failure in live.failures:
        typer.secho(f"  FAILED {failure}", fg=typer.colors.RED, err=True)
    if live.failures:
        typer.secho(
            "  those episodes are NOT in the dataset — the traceback is in the log file.",
            fg=typer.colors.RED,
            err=True,
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
    fps: Annotated[
        int | None,
        typer.Option(
            "--fps",
            help=(
                f"Target loop rate, Hz. Defaults to {DEFAULT_FPS}, or to {DEMO_FPS} with "
                "--record-dataset: the rate goes into the dataset's metadata and has to "
                "be the rate the policy will run at."
            ),
        ),
    ] = None,
    capture: Annotated[
        str | None,
        typer.Option(
            "--capture",
            help=(
                "Which \\[capture.*] table the cameras run at. Defaults to `teleop`, or "
                "to `policy` with --record-dataset, so the demonstrations are the size "
                "the rollout will be."
            ),
        ),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help=(
                "How the cell is set up: `optimized` (the wrist crop) or `common` (the "
                "full lens on all three cameras). The same word it is on `dk1 policy "
                "run`. Defaults to `optimized`, or to `common` with --record-dataset — "
                "STUDY.md records the demonstrations uncropped and applies the crop at "
                "training time, so one dataset serves both the cropped and the "
                "uncropped row. It selects no speed limit here: teleoperation stays "
                "uncapped, because the commands come from a human hand."
            ),
        ),
    ] = None,
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            help=(
                "--record-dataset: the instruction recorded on every frame of every "
                "episode. ONE string for the whole session — it is the prompt the "
                "fine-tuned policy is rolled out under, so it must be byte-identical "
                "everywhere. Asked for at the start if it is not given."
            ),
        ),
    ] = None,
    record_dataset: Annotated[
        bool,
        typer.Option(
            "--record-dataset",
            help=(
                "Record demonstrations into a LeRobot dataset v3.0 under --dataset-dir: "
                "the 14-D state, the action the arms were GIVEN, and the three camera "
                "streams as video. Enter starts and ends an episode, `again` deletes "
                "the last one, `scene <n>` labels the ones that follow, `done` ends the "
                "session."
            ),
        ),
    ] = False,
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            help=(
                "Where the demonstrations go. Episodes are APPENDED, so an existing "
                "directory is resumed rather than replaced and a second day of "
                "recording continues the same dataset."
            ),
        ),
    ] = DEFAULT_DEMO_DIR,
    scene: Annotated[
        int,
        typer.Option(
            "--scene",
            help=(
                "Which marked scene layout the demonstrations start on. Recorded per "
                "episode in dk1_notes.jsonl and never in the task string; change it "
                "mid-session with `scene <n>` at the prompt."
            ),
        ),
    ] = 1,
    vcodec: Annotated[
        str,
        typer.Option(
            "--vcodec",
            help=(
                "Video codec for --record-dataset. `auto` takes the hardware encoder "
                "when there is one — NVENC here — instead of LeRobot's SVT-AV1 on the "
                "CPU, which spends minutes per episode."
            ),
        ),
    ] = DEFAULT_VCODEC,
    stream_video: Annotated[
        bool,
        typer.Option(
            "--stream-video/--no-stream-video",
            help=(
                "Encode the cameras AS THE ARMS MOVE rather than from a cache of PNG "
                "frames when the episode is written, which takes writing one from "
                "minutes to seconds. ON by default HERE and nowhere else: it is paid "
                "for out of the control loop, and the reason it must stay off for a "
                "scored rollout — the loop is the experiment — does not apply to a "
                "human hand on a leader arm, with no policy holding the GPU."
            ),
        ),
    ] = True,
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
    log: Annotated[
        bool,
        typer.Option("--log/--no-log", help="Write this run to logs/<time>-teleop.log, fsynced."),
    ] = True,
    telemetry: Annotated[
        bool,
        typer.Option(
            "--telemetry/--no-telemetry",
            help=(
                "Sample PSU power, the +12 V rail, CPU and GPU once a second into "
                "logs/<time>-teleop.jsonl. ON by default: this machine has frozen "
                "during teleoperation too, and the last line is what it was doing."
            ),
        ),
    ] = True,
    assume_yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y", help="Skip the confirmation prompt. The warning still prints."
        ),
    ] = False,
) -> None:
    if control_mode not in ("impedance", "pos_vel"):
        raise typer.BadParameter(f"control mode must be impedance or pos_vel, got {control_mode!r}")
    # Three defaults that move together with --record-dataset, and are printed
    # rather than assumed: a demonstration recorded through a different lens, at a
    # different size or at a different rate than the rollout is a demonstration of
    # a cell that does not exist.
    fps = (DEMO_FPS if record_dataset else DEFAULT_FPS) if fps is None else fps
    capture = ("policy" if record_dataset else "teleop") if capture is None else capture
    profile = (COMMON if record_dataset else DEFAULT_PROFILE) if profile is None else profile
    if fps <= 0:
        raise typer.BadParameter(f"--fps must be positive, got {fps}")
    try:
        chosen = resolve(profile)
    except ProfileError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if scene < 1:
        raise typer.BadParameter(f"a scene is numbered from 1, got {scene}")
    if display_policy_input:
        # It logs into the same Rerun session and the loop only calls the
        # observation processor when displaying, so on its own it would be inert.
        display = True
    if display and not cameras:
        raise typer.BadParameter("--display needs cameras; drop --no-cameras.")

    if no_limit and max_joint_rate is not None:
        raise typer.BadParameter("--no-limit and --max-joint-rate contradict each other.")

    if record_dataset:
        # Each of these would produce a dataset that looks recorded and is not
        # usable, which is the failure worth spending a line of validation on.
        if not cameras:
            raise typer.BadParameter("--record-dataset needs the cameras; drop --no-cameras.")
        if duration_s is not None:
            raise typer.BadParameter(
                "--record-dataset has no duration: an episode ends when you press Enter."
            )
        if display_policy_input:
            raise typer.BadParameter(
                "--display-policy-input is for checking the view, not for recording; "
                "run it on its own first."
            )

    # The profile is a DERIVED config, never a write: under `common` every camera
    # loses its crop, and every consumer below — the camera builders, the banner,
    # the recorder's notes — reads the same object and therefore agrees about what
    # the lens does. Its [limits.*] table is deliberately NOT read: teleoperation
    # is uncapped whatever profile it runs under, because the cap exists to bound
    # a policy and these commands come from a human hand.
    settings = chosen.apply(load(config, require_devices=not dry_run))
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
        profile=capture,
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
        profile=chosen,
        capture=capture,
    )

    if record_dataset:
        typer.secho("\nrecording", bold=True)
        typer.echo(f"  LeRobot v3.0 -> {dataset_dir}  (episodes are appended)")
        typer.echo(f"  frames {settings.profile(capture).width}x"
                   f"{settings.profile(capture).height} at {fps} Hz")
        typer.echo(f"  scene {scene} to start; `scene <n>` at the prompt changes it")
        typer.echo(
            "  video " + vcodec
            + (", encoded AS THE ARMS MOVE (--stream-video)" if stream_video
               else ", encoded when the episode is written")
        )

    if dry_run:
        typer.secho("\n--dry-run: nothing was connected and nothing moved.", fg=typer.colors.GREEN)
        return

    if record_dataset:
        # Asked before anything is energised, and once for the whole session: the
        # string is the prompt the fine-tuned policy will be rolled out under, so
        # it has to be byte-identical on every frame of every episode.
        task = _task(task)
        live = _open_dataset(
            dataset_dir,
            capture=settings.profile(capture),
            fps=fps,
            vcodec=vcodec,
            streaming=stream_video,
        )

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

        frame = settings.profile(capture)
        typer.echo("loading the checkpoint's preprocessor (no model weights, no GPU) ...")
        preprocessor, features = build_preprocessor(
            str(settings.policy.checkpoint), width=frame.width, height=frame.height
        )
        model_view = ModelInputProbe(None, preprocessor, features)

    monitor = None
    if log:
        from ..logs import start as start_log

        typer.echo(f"log -> {start_log('teleop')}")
    if telemetry:
        from datetime import datetime

        from ..telemetry import DEFAULT_TELEMETRY_DIR, Telemetry

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        monitor = Telemetry(
            DEFAULT_TELEMETRY_DIR / f"{stamp}-teleop.jsonl",
            context={"command": "teleop", "fps": fps, "cameras": cameras, "profile": profile},
        )
        monitor.start()
        typer.echo(f"telemetry -> {monitor.path} (PSU, CPU, GPU, once a second)")

    if record_dataset:
        typer.secho(
            f"\nrecording demonstrations of {task!r}.\n"
            "Enter starts an episode and Enter ends it; `again` throws the last one "
            "away;\n`scene <n>` labels the ones that follow; `done` finishes.\n"
            "The arms track the leaders throughout, including while you type.\n",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho("\nCtrl-C to stop. Stopping does not move the arms.\n", fg=typer.colors.GREEN)
    try:
        if record_dataset:
            from ..demos import run as run_demos

            session = run_demos(
                leader,
                follower,
                live,
                task=task,
                fps=fps,
                scene=scene,
                display=display,
                notes={
                    "profile": chosen.name,
                    "capture": capture,
                    "fps": fps,
                    "teleop": True,
                    "max_joint_rate": follower.config.max_joint_rate,
                    "stream_video": stream_video,
                    "vcodec": live.resolved_vcodec or vcodec,
                },
            )
        else:
            run(
                leader,
                follower,
                fps=fps,
                display=display,
                duration_s=duration_s,
                model_view=model_view,
            )
    finally:
        if monitor is not None:
            monitor.stop()
            typer.echo(f"telemetry written to {monitor.path}")
    if record_dataset:
        _report_dataset(live, session)
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
