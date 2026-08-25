"""One loaded policy, many rollouts: the task becomes a prompt, not a process.

`dk1 policy run` loads 10 GiB of weights, builds the CUDA graph, opens three
cameras and energises four arms — about a minute — drives one task, and throws
all of it away. Trying a second phrasing of the instruction, or a second attempt
at the same one, pays that minute again. Scoring a policy means doing it twenty
times.

This module splits the two halves that `run` fused:

``PolicySession.open()``
    load the checkpoint, connect the cell, prewarm. Once.
``PolicySession.rollout(task)``
    set the instruction, run the control loop, stop. As often as you like.

**What actually changes between rollouts is one string.** The instruction reaches
the policy through the inference engine's ``_task``, which
``prepare_observation_for_inference`` reads at every model call — so writing it
is enough, and :meth:`PolicySession.set_task` writes it on the engine *and* on
the config the banner prints, then refuses if the engine has no such attribute
rather than running the old task quietly. That is the whole of the coupling.

**The arms stay connected and energised between rollouts.** That is the point —
reconnecting costs seconds and re-zeroes both grippers against their open stop —
and it is also the hazard: live motors sit in the room while the operator types.
Between rollouts nothing is commanded, so the motor chain holds the last target
and warns once per arm that it is doing so; that warning is expected here.
:meth:`close` disconnects, which **disables every motor**, so a raised arm sags.

**Ctrl-C ends the rollout, not the session.** LeRobot's ``ProcessSignalHandler``
counts signals for the life of the process and calls ``sys.exit(1)`` on the
second one — which in a session is the second rollout you stop. So the session
owns SIGINT for the length of each rollout instead, the same trick and for the
same reason as :func:`dk1lab.home.interrupt_aborts`.

**The instruments attach once, and are reset per episode.** The trace and the
per-joint action view wrap methods on the live objects; wrapping them again
every rollout would stack a new layer per episode. An
:class:`~dk1lab.record.EpisodeRecorder`, which is per-episode by nature,
attaches and detaches around one rollout — last in, first out, so the
restoration puts the chain back exactly as it was.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

from .fifo import DEFAULT_BLEND_STEPS, DEFAULT_REPLAN_AT

logger = logging.getLogger(__name__)

#: What a prompt line was asking for. The parsing is separated from the reading
#: for the reason every decision in this package is separated from its I/O: it
#: is the part worth testing, and it tests without a robot.
RUN = "run"
QUIT = "quit"
HOME = "home"
HELP = "help"
RECORD = "record"
DATASET = "dataset"
DURATION = "duration"
SCENE = "scene"
NOTHING = "nothing"


@dataclass(frozen=True)
class Command:
    """One line from the session prompt, understood.

    ``kind`` is one of the constants above. ``task`` carries the instruction for
    :data:`RUN`, ``value`` the argument of :data:`RECORD` / :data:`DURATION`, and
    ``error`` a complaint to print instead of acting.
    """

    kind: str
    task: str = ""
    value: Any = None
    error: str | None = None


def parse_command(line: str, *, last_task: str = "") -> Command:
    """A prompt line as a :class:`Command`.

    Anything that is not a ``:``-prefixed word is the **instruction**, which is
    what makes the common case one keystroke past typing it. An empty line
    repeats the last task, because the ordinary way to score a policy is to run
    the same instruction over and over.
    """
    text = line.strip()
    if not text:
        if last_task:
            return Command(RUN, task=last_task)
        return Command(NOTHING)
    if not text.startswith(":"):
        return Command(RUN, task=text)

    word, _, rest = text[1:].partition(" ")
    word, rest = word.lower(), rest.strip()
    if word in ("q", "quit", "exit"):
        return Command(QUIT)
    if word in ("h", "help", "?"):
        return Command(HELP)
    if word == "home":
        return Command(HOME)
    if word == "record":
        if rest.lower() in ("on", "yes", "true"):
            return Command(RECORD, value=True)
        if rest.lower() in ("off", "no", "false"):
            return Command(RECORD, value=False)
        return Command(RECORD, error=f"say `:record on` or `:record off`, not {rest!r}")
    if word == "dataset":
        if rest.lower() in ("on", "yes", "true"):
            return Command(DATASET, value=True)
        if rest.lower() in ("off", "no", "false"):
            return Command(DATASET, value=False)
        return Command(DATASET, error=f"say `:dataset on` or `:dataset off`, not {rest!r}")
    if word == "scene":
        # Only meaningful in a scored session; the prompt loop says so when
        # there is no study, rather than the grammar knowing about one.
        try:
            number = int(rest)
        except ValueError:
            return Command(SCENE, error=f"`:scene` wants a scene number, got {rest!r}")
        return Command(SCENE, value=number)
    if word == "duration":
        try:
            seconds = float(rest)
        except ValueError:
            return Command(DURATION, error=f"`:duration` wants a number of seconds, got {rest!r}")
        if seconds < 0:
            return Command(DURATION, error="a duration cannot be negative; 0 means until stopped")
        return Command(DURATION, value=seconds)
    return Command(NOTHING, error=f"no such command: `:{word}` — `:help` lists them")


@dataclass(frozen=True)
class EpisodeOutcome:
    """What one rollout in a session did. Everything here is read, not decided."""

    index: int
    task: str
    seconds: float
    #: Why the loop left: ``"duration"``, ``"interrupted"``, or the exception's name.
    ended: str
    recording: Any = None
    home: Any = None

    def summary(self) -> str:
        return (
            f"episode {self.index} ({self.task!r}) ran {self.seconds:.1f} s "
            f"and ended: {self.ended}"
        )


class PolicySession:
    """A loaded policy, a connected cell, and a rollout you can run again.

    **Opening connects and energises the arms; every rollout moves them.**

    Args:
        cfg: the ``RolloutConfig`` from :func:`dk1lab.policy.rollout_config`. Its
            ``task`` and ``duration`` are overwritten per rollout; everything
            else — the limits, the capture profile, the checkpoint — is fixed
            for the life of the session, because changing them would mean
            reloading the thing this exists to keep loaded.
        display: stream to Rerun, with the per-joint panels of
            :mod:`dk1lab.actionview`.
        trace: a :class:`~dk1lab.trace.RolloutTrace`, attached once and reset
            between episodes so each one gets its own summary.
        record_dir: where episode recordings are written when recording is on.
        record: whether to write an ``.rrd`` from the first rollout. Toggled at
            the prompt.
        dataset: an open :class:`~dk1lab.dataset.DatasetSession` to append
            episodes to, or ``None``. Unlike the ``.rrd`` path it is one object
            for the whole session, so five scored attempts are five episodes of
            one dataset.
        record_dataset: whether to write each rollout into ``dataset``. Also
            toggled at the prompt, and inert without a ``dataset``.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        display: bool = False,
        invert_gripper: bool = False,
        fifo: bool = True,
        asynchronous: bool = True,
        replan_at: int = DEFAULT_REPLAN_AT,
        blend: int = DEFAULT_BLEND_STEPS,
        trace: Any = None,
        record_dir: Path | str | None = None,
        record: bool = False,
        dataset: Any = None,
        record_dataset: bool = False,
        duration_s: float = 0.0,
        notes: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.display = display
        self.invert_gripper = invert_gripper
        self.fifo = fifo
        self.asynchronous = asynchronous
        self.replan_at = replan_at
        self.blend = blend
        self.trace = trace
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self.record = record
        #: An open :class:`~dk1lab.dataset.DatasetSession`, or ``None``. Opened by
        #: the caller before anything is connected, and closed by :meth:`close` —
        #: it spans the whole session, because a scored run of five attempts is
        #: five episodes of one dataset rather than five datasets.
        self.dataset = dataset
        self.record_dataset = record_dataset and dataset is not None
        self.duration_s = duration_s
        self.notes = dict(notes or {})
        self.task = ""
        self.episodes = 0

        self.ctx: Any = None
        self.shutdown_event = Event()
        self._strategy: Any = None
        self._view: Any = None
        self._watching = False

    # -- lifecycle ---------------------------------------------------------- #

    def open(self) -> None:
        """Load the policy and connect the cell. **Energises the arms.**

        Everything expensive happens here and only here: the weights, the CUDA
        graph, the cameras, the two CAN adapters. What is left for a rollout is
        a string and a control loop.
        """
        from lerobot.rollout.strategies import BaseStrategy
        from lerobot.utils.visualization_utils import init_visualization

        from .actionview import ActionView, pin_blueprint
        from .policy import build_context, use_chunk_fifo

        self._watching = self.display or (self.trace is not None and self.trace.display)
        if self._watching:
            init_visualization("rerun", session_name="dk1-policy")
            # Before the first log_rerun_data call: it caches a blueprint off the
            # first observation it sees, and whichever layout gets there first is
            # the one the operator looks at for the whole session.
            pin_blueprint(model_input=self.trace is not None and self.trace.display)

        ctx, _ = build_context(
            self.cfg, self.shutdown_event, invert_gripper=self.invert_gripper
        )
        use_chunk_fifo(
            ctx,
            enabled=self.fifo,
            asynchronous=self.asynchronous,
            replan_at=self.replan_at,
            blend=self.blend,
            fps=self.cfg.fps,
        )
        self.ctx = ctx
        self._strategy = BaseStrategy(self.cfg.strategy)
        # After the FIFO swap, so the trace wraps the engine that will be driven.
        if self.trace is not None:
            self.trace.attach(ctx)
        # After the trace, so its timing wrappers are inside this one and the
        # cost of drawing is not charged to the robot or to the engine.
        if self.display:
            self._view = ActionView()
            self._view.attach(ctx)

    def close(self) -> None:
        """Disconnect the cell and tear the session down. **Disables the motors.**

        Nothing is commanded — but a disconnect in impedance mode reaches
        ``DK1MotorChain.stop()``, which disables every motor, so a raised arm
        sags. Support anything holding itself up.
        """
        from lerobot.utils.visualization_utils import shutdown_visualization

        if self._view is not None:
            self._view.detach()
            self._view = None
        if self._strategy is not None and self.ctx is not None:
            self._strategy.teardown(self.ctx)
        self._strategy = None
        if self._watching:
            shutdown_visualization("rerun")
            self._watching = False
        if self.dataset is not None:
            # Finalises the metadata, and keeps any episode nobody answered for.
            self.dataset.close()
        self.ctx = None

    def __enter__(self) -> PolicySession:
        self.open()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- the instruction ---------------------------------------------------- #

    def set_task(self, task: str) -> None:
        """Point the loaded policy at a new instruction.

        Raises rather than warns when the engine has no ``_task``: a session that
        went on running the previous instruction while reporting the new one
        would produce evidence about the wrong thing, and every engine LeRobot
        ships — sync, RTC — and both of ours keep it under that name.
        """
        engine = self.ctx.policy.inference
        if not hasattr(engine, "_task"):
            raise AttributeError(
                f"{type(engine).__name__} has no _task, so the instruction cannot be "
                "changed without rebuilding the engine. Do not run this session: it "
                "would keep executing the previous task."
            )
        engine._task = task
        self.cfg.task = task
        self.task = task

    # -- one rollout -------------------------------------------------------- #

    def rollout(
        self,
        task: str | None = None,
        *,
        duration_s: float | None = None,
        record: bool | None = None,
        dataset: bool | None = None,
        home: Any = None,
    ) -> EpisodeOutcome:
        """Drive the arms with the policy, once. **Moves the arms.**

        The same LeRobot ``BaseStrategy`` loop `dk1 policy run` uses, set up and
        torn down around this call rather than around the process: ``setup``
        resets the policy and starts the engine, ``run`` is the loop, and the
        engine is stopped afterwards so no worker thread keeps burning the GPU
        while the operator is thinking.

        Args:
            task: the instruction. Defaults to the one already set.
            duration_s: seconds, or 0 for until-stopped. Defaults to the
                session's.
            record: write this episode to a ``.rrd``. Defaults to the session's.
            dataset: append this episode to the LeRobot dataset. Defaults to the
                session's, and is inert without one.
            home: sweep to this pose afterwards, on a clean end only.

        Returns:
            An :class:`EpisodeOutcome`. Never raises for an ordinary stop —
            Ctrl-C is an ordinary stop.
        """
        from .dataset import one as combine
        from .record import DEFAULT_RECORD_DIR, EpisodeRecorder, episode_path, next_index

        if task:
            self.set_task(task)
        if not self.task:
            raise ValueError("no task set: a rollout needs an instruction to condition on")
        self.cfg.duration = self.duration_s if duration_s is None else duration_s
        recording_wanted = self.record if record is None else record

        self.episodes += 1
        self.shutdown_event.clear()
        if self.trace is not None:
            self.trace.reset()

        rrd_recorder = None
        if recording_wanted:
            # The index comes off the directory, not off this session's count, so
            # it keeps rising across sessions rather than overwriting yesterday's
            # first episode with today's.
            directory = self.record_dir or DEFAULT_RECORD_DIR
            index = next_index(directory)
            rrd_recorder = EpisodeRecorder(
                episode_path(directory, self.task, index),
                task=self.task,
                notes={**self.notes, "episode": index},
            )
        dataset_wanted = self.record_dataset if dataset is None else dataset
        dataset_recorder = None
        if dataset_wanted and self.dataset is not None:
            dataset_recorder = self.dataset.episode(
                self.task, notes={**self.notes, "episode": self.dataset.episodes}
            )
        recorder = combine(rrd_recorder, dataset_recorder)
        if recorder is not None:
            # After every session-level instrument, so it is the innermost
            # wrapper and detaching it puts the chain back as it was.
            recorder.attach(self.ctx)
            recorder.start()

        started = time.perf_counter()
        ended = "duration"
        recording = None
        error: BaseException | None = None
        try:
            with interrupt_stops(self.shutdown_event):
                self._strategy.setup(self.ctx)
                # After setup: the RTC action queue does not exist until start().
                if self.trace is not None:
                    self.trace.attach_queue(self._strategy)
                self._strategy.run(self.ctx)
            if self.shutdown_event.is_set():
                ended = "interrupted"
        except KeyboardInterrupt as exc:
            error = exc
            ended = "interrupted"
            self.shutdown_event.set()
        except BaseException as exc:
            error = exc
            ended = type(exc).__name__
            raise
        finally:
            engine = getattr(self._strategy, "_engine", None)
            if engine is not None:
                engine.stop()
            if self.trace is not None:
                self.trace.close()
            if recorder is not None:
                recording = recorder.stop()
                recorder.detach()

        report = None
        if home is not None and _clean(error):
            report = self.home(home)
        return EpisodeOutcome(
            index=self.episodes,
            task=self.task,
            seconds=time.perf_counter() - started,
            ended=ended,
            recording=recording,
            home=report,
        )

    # -- homing between rollouts -------------------------------------------- #

    def home(self, pose: Any = None) -> Any:
        """Sweep both arms to ``pose``, or to the pose captured at connect. **Moves the arms.**

        The same sweep `dk1 policy run --home` ends on, run between rollouts
        instead of at shutdown: put the cell back where it started, then give the
        policy the next instruction from the same place. A second Ctrl-C stops
        the sweep where the arms are.
        """
        from .policy import HOME_AT_START_POSE, _home_rate, go_home_before_teardown, home_target

        target = home_target(self.ctx, None if pose is HOME_AT_START_POSE else pose)
        return go_home_before_teardown(
            self.ctx,
            self._strategy,
            target=target,
            rate=_home_rate(self.cfg),
            fps=self.cfg.fps,
        )


def _clean(error: BaseException | None) -> bool:
    """Whether a finished episode earns a home sweep. See ``policy.ended_cleanly``."""
    from .policy import ended_cleanly

    return ended_cleanly(error)


@contextmanager
def interrupt_stops(event: Event) -> Generator[None, None, None]:
    """Make Ctrl-C end the current rollout for the length of the block.

    The first signal sets the shutdown event, which the control loop checks at
    the top of every tick — so the rollout ends within ~33 ms, cleanly, with
    nothing commanded afterwards. The **second** raises ``KeyboardInterrupt`` on
    the main thread the ordinary way, so an operator whose loop is genuinely
    stuck is not trapped by an instrument that swallows their interrupt.

    This replaces LeRobot's ``ProcessSignalHandler`` inside a session rather than
    complementing it: that handler counts signals for the life of the process and
    exits on the second, which in a session is the second rollout you stop —
    killing the process with the arms energised and the cameras open.

    Outside the main thread, where handlers cannot be installed, it does nothing
    and leaves signal handling alone.
    """
    signals = 0

    def handler(_signum: int, _frame: object) -> None:
        nonlocal signals
        signals += 1
        if signals > 1:
            raise KeyboardInterrupt
        event.set()
        logger.warning("interrupt — ending this rollout; the arms stay connected")

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
