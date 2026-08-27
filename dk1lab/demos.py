"""Teleoperated demonstrations, recorded episode by episode into one dataset.

``STUDY.md`` Phase 3 is 45 demonstrations by hand — the one thing in this study
that cannot be bought back once the day is spent. This module is what records
them: :func:`run` connects the leader and follower pair, teleoperates
continuously, and turns one keypress into the start and the end of an episode.

**The operator's whole vocabulary is four things**, read from stdin while the
arms are live::

    <Enter>   start recording an episode; press it again to stop
    again     throw the last episode away (or abort the one being recorded)
    scene N   the demonstrations from here on are of scene layout N
    done      commit what is held and end the session (also: quit)

Everything else about it follows from three decisions.

**The teleoperation loop never stops.** Not between episodes, not while the
operator is typing. That is a safety property, not a convenience: teleop runs
**uncapped** here (`STUDY.md`: *the cap exists to bound a policy, and
demonstrations come from a human hand*), and a loop that paused between episodes
would leave the followers holding their last target while the passive leader arms
sagged under gravity — so the first tick after the pause would command the arms
to the sagged pose at full speed. A loop that never stops has no such step in it.
The cost is that stdin has to be read without blocking, which is
:class:`TerminalConsole`.

**The loop is ours, and this is the one place that is true.** Every other loop in
this fork is LeRobot's, imported rather than reimplemented — see
:mod:`dk1lab.teleop`. ``teleop_loop`` takes a duration and nothing else: it
cannot be ended on a keypress, and it prints a line per tick over whatever the
operator is typing. What is reimplemented is six calls in tick order, and
:func:`tick` is deliberately the whole of it, so the thing that could drift from
``teleop_loop`` is small enough to read side by side with it.

**An episode is committed one episode late.** :meth:`stop` leaves the episode in
the dataset's buffer; the *next* start commits it, and so does ``done`` and any
exit. That is what makes ``again`` a real deletion rather than an apology —
``save_episode`` cannot be undone, and LeRobot v3.0 has no way to take an episode
back out. What it costs is bounded and known: every episode before the current
one is sealed on disk and readable (:meth:`dk1lab.dataset.DatasetSession._make_durable`),
so a crash costs at most the attempt just made, which is the one the operator was
still deciding about.

Nothing here decides what the cameras show. The demonstrations are recorded under
``--profile common`` — the full lens, no wrist crop — and the ``optimized`` crop
is applied at training time; that is a :class:`~dk1lab.runprofile.RunProfile`
applied to the config before the devices are built, and this module never sees
it.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: What a line typed at the demo prompt was asking for. Separated from the
#: reading for the same reason every decision in this package is: it is the part
#: worth testing, and it tests with no robot and no terminal.
RECORD = "record"
AGAIN = "again"
DONE = "done"
SCENE = "scene"
NOTHING = "nothing"

#: How often the recording status line is redrawn, seconds.
STATUS_EVERY_S = 1.0


@dataclass(frozen=True)
class DemoCommand:
    """One line typed at the demo prompt, understood."""

    kind: str
    value: Any = None
    error: str | None = None


def parse_command(line: str) -> DemoCommand:
    """A typed line as a :class:`DemoCommand`.

    An **empty line is the common case** and means "start or stop", because the
    operator has one hand on a leader arm and the other on the keyboard; every
    other word is a rare correction. A ``:`` prefix is accepted on all of them so
    that the muscle memory from ``dk1 policy session`` works here too.
    """
    text = line.strip()
    if not text:
        return DemoCommand(RECORD)
    word, _, rest = text.lstrip(":").partition(" ")
    word, rest = word.lower(), rest.strip()
    if word in ("again", "a", "redo"):
        return DemoCommand(AGAIN)
    if word in ("done", "quit", "q", "exit", "stop"):
        return DemoCommand(DONE)
    if word == "scene":
        try:
            number = int(rest)
        except ValueError:
            return DemoCommand(SCENE, error=f"`scene` wants a number, got {rest!r}")
        if number < 1:
            return DemoCommand(SCENE, error=f"a scene is numbered from 1, got {number}")
        return DemoCommand(SCENE, value=number)
    return DemoCommand(
        NOTHING,
        error=f"no such command: {text!r} — Enter records, `again`, `scene <n>`, `done`",
    )


# --------------------------------------------------------------------------- #
# The console
# --------------------------------------------------------------------------- #


class TerminalConsole:
    """Reads whole lines from a live terminal **without blocking the loop**.

    The arms are being teleoperated while the operator types, so nothing here may
    wait: :meth:`poll` asks whether a complete line is available and returns
    ``None`` when it is not. The terminal stays in its ordinary line mode, so the
    line only becomes available when Enter is pressed — which is exactly the
    event this session is driven by — and what is typed is echoed by the terminal
    as usual.

    **fd 2 is silenced while the operator is between episodes.** The three
    cameras stream MJPG and their firmware pads a few bytes before a restart
    marker, so libjpeg prints ``Corrupt JPEG data: N extraneous bytes`` on a
    perfectly good frame, from C, past every Python redirect — and it lands in
    the middle of the line being typed. It is restored for the length of an
    episode, where a warning from the motor chain is worth more than a tidy
    screen. Same trick and same reason as ``policy_cmds._quiet_stderr``.
    """

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self._saved: int | None = None

    # -- output ------------------------------------------------------------- #

    def say(self, text: str = "") -> None:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def status(self, text: str) -> None:
        """Redraw the one-line recording status in place."""
        sys.stdout.write("\r" + text + "\x1b[K")
        sys.stdout.flush()

    def end_status(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # -- input -------------------------------------------------------------- #

    def interactive(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def poll(self) -> str | None:
        """One complete line if the operator has typed it, else ``None``."""
        if not self.interactive():
            return None
        ready, _, _ = select.select([self.stream], [], [], 0)
        if not ready:
            return None
        line = self.stream.readline()
        if line == "":  # EOF: the terminal went away
            raise EOFError
        return line

    # -- the JPEG chatter --------------------------------------------------- #

    def mute(self) -> None:
        if self._saved is not None:
            return
        try:
            sys.stderr.flush()
            self._saved = os.dup(2)
            with open(os.devnull, "wb") as sink:
                os.dup2(sink.fileno(), 2)
        except OSError:  # pragma: no cover - no fd 2 to speak of
            self._saved = None

    def unmute(self) -> None:
        if self._saved is None:
            return
        try:
            os.dup2(self._saved, 2)
            os.close(self._saved)
        finally:
            self._saved = None


# --------------------------------------------------------------------------- #
# One tick
# --------------------------------------------------------------------------- #


def tick(leader: Any, follower: Any, processors: Any, *, display: bool = False) -> dict:
    """One teleoperation tick: read, process, send. Returns what was sent.

    The six calls, in the order ``lerobot.scripts.lerobot_teleoperate.teleop_loop``
    makes them, and nothing else. Kept as its own function so the one loop in this
    fork that is not LeRobot's is small enough to check against theirs.

    The recorder wraps ``follower.get_observation`` and ``follower.send_action``,
    so a frame is written by these two calls without this function knowing that
    recording exists.
    """
    teleop_action_processor, robot_action_processor, robot_observation_processor = processors

    observation = follower.get_observation()
    raw_action = leader.get_action()
    teleop_action = teleop_action_processor((raw_action, observation))
    to_send = robot_action_processor((teleop_action, observation))
    sent = follower.send_action(to_send)

    if display:
        from lerobot.utils.visualization_utils import log_visualization_data

        log_visualization_data(
            "rerun",
            observation=robot_observation_processor(observation),
            action=teleop_action,
        )
    return sent if sent is not None else to_send


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


class DemoSession:
    """Teleoperation with a dataset recorder the operator switches on and off.

    **Constructing connects to nothing; :meth:`loop` moves the arms.** The
    devices are connected by the caller (:func:`run`), which is also what
    disconnects them.

    Args:
        leader: the connected :class:`BiDK1Leader`.
        follower: the connected :class:`~dk1lab.robot.SafeBiDK1Follower`.
        dataset: an open :class:`~dk1lab.dataset.DatasetSession`. Episodes are
            appended to it; an existing directory is resumed, so a session that
            ended yesterday continues rather than starting a second dataset.
        task: the instruction recorded on every frame of every episode. **One
            string for the whole session** — it is the prompt the fine-tuned
            policy will be rolled out under, and ``STUDY.md`` requires it
            byte-identical everywhere.
        fps: the control rate. It is written into the dataset's metadata, so it
            has to be the rate the policy will run at.
        scene: which marked scene layout the demonstrations start on. Recorded
            per episode in ``dk1_notes.jsonl`` and **never** in the task string.
        notes: anything else worth keeping per episode — the profile, the
            capture, the codec.
    """

    def __init__(
        self,
        leader: Any,
        follower: Any,
        dataset: Any,
        *,
        task: str,
        fps: int = 30,
        scene: int = 1,
        display: bool = False,
        notes: dict[str, Any] | None = None,
        console: Any = None,
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.dataset = dataset
        self.task = task
        self.fps = int(fps)
        self.scene = int(scene)
        self.display = display
        self.notes = dict(notes or {})
        self.console = console if console is not None else TerminalConsole()

        #: The recorder of the episode being recorded, or ``None`` while idle.
        self.recorder: Any = None
        #: How many episodes this session started. Not the dataset's count —
        #: a dropped one is counted here and not there.
        self.attempts = 0
        #: How many were actually written. Counted here rather than read off the
        #: dataset afterwards, because reporting happens after :meth:`close`,
        #: which lets the dataset object go.
        self.written = 0
        self._started = 0.0
        self._status_at = 0.0
        self._ticks = 0
        self._drew_status = False

    # -- what the prompt says ----------------------------------------------- #

    def next_index(self) -> int:
        """The dataset episode index the next recording will be written as.

        The held episode has not been committed yet, so it does not show in the
        dataset's count; it will be there by the time the next one is written.
        """
        held = 1 if getattr(self.dataset, "pending", None) is not None else 0
        return int(self.dataset.episodes) + held

    def prompt(self) -> str:
        return (
            f"[scene {self.scene} | next episode {self.next_index()}] "
            f"Enter to record, `again`, `scene <n>`, `done`> "
        )

    # -- an episode --------------------------------------------------------- #

    def start_episode(self) -> None:
        """Begin recording. Commits the episode held from last time first."""
        self.commit_held()
        self.attempts += 1
        self.recorder = self.dataset.episode(
            self.task,
            notes={**self.notes, "scene": self.scene, "episode": self.next_index()},
        )
        self.recorder.attach_robot(self.follower)
        self.recorder.start()
        self._started = time.perf_counter()
        self._status_at = 0.0
        self._ticks = 0
        self._drew_status = False
        # The motor chain's warnings are worth more than a tidy screen while the
        # arms are being driven for the record.
        self.console.unmute()
        self.console.say(f"  recording episode {self.next_index()} — Enter to stop")

    def stop_episode(self) -> Any:
        """End the episode. **Does not write it** — see the module docstring."""
        recorder = self.recorder
        if recorder is None:
            return None
        self.recorder = None
        report = recorder.stop()
        recorder.detach()
        # Only if there is a status line to close: a stray newline before every
        # summary reads as a blank line nobody wrote.
        if self._drew_status:
            self.console.end_status()
            self._drew_status = False
        seconds = time.perf_counter() - self._started
        rate = report.ticks / seconds if seconds > 0 else 0.0
        self.console.say(
            f"  stopped: {report.ticks} frames, {seconds:.1f} s ({rate:.1f} Hz)"
            + (f" — RECORDING FAILED: {report.failed}" if report.failed else "")
        )
        if report.failed:
            logger.error("episode %d recorded nothing usable: %s", report.index, report.failed)
        self.console.mute()
        return report

    def commit_held(self) -> bool:
        """Write the episode held from the previous recording, if there is one."""
        held = getattr(self.dataset, "pending", None)
        if held is None:
            return False
        self.console.say(f"  writing episode {held.index} ({held.ticks} frames) ...")
        started = time.perf_counter()
        written = self.dataset.commit()
        if written:
            self.written += 1
            self.console.say(
                f"  episode {held.index} written in {time.perf_counter() - started:.1f} s "
                f"-> {self.dataset.root}"
            )
        else:
            self.console.say(f"  COULD NOT WRITE episode {held.index} — see the log")
        return written

    def drop(self) -> bool:
        """Throw away the episode being recorded, or the one held. Nothing else."""
        if self.recorder is not None:
            self.stop_episode()
        if self.dataset.drop():
            self.console.say("  dropped — nothing was written")
            return True
        self.console.say("  nothing to drop: the episodes before the last one are on disk")
        return False

    # -- the loop ----------------------------------------------------------- #

    def handle(self, line: str) -> bool:
        """Act on one typed line. Returns whether the session should continue."""
        command = parse_command(line)
        if command.error:
            self.console.say(f"  {command.error}")
            return True
        if command.kind == RECORD:
            if self.recorder is not None:
                self.stop_episode()
            else:
                self.start_episode()
        elif command.kind == AGAIN:
            self.drop()
        elif command.kind == SCENE:
            if self.recorder is not None:
                self.console.say("  stop the episode first (Enter), then say `scene <n>`")
                return True
            self.scene = int(command.value)
            self.console.say(f"  scene {self.scene} — set the layout, then Enter to record")
        elif command.kind == DONE:
            return False
        if command.kind != DONE and self.recorder is None:
            self.console.say(self.prompt())
        return True

    def loop(self) -> None:
        """Teleoperate until ``done``, recording the episodes asked for. **Moves the arms.**

        Ctrl-C and EOF end the session the same way ``done`` does: whatever is
        being recorded is stopped and whatever is held is written. A day of
        demonstrations must not be lost to the way it was ended.
        """
        from lerobot.processor import make_default_processors
        from lerobot.utils.robot_utils import precise_sleep

        processors = make_default_processors()
        period = 1.0 / self.fps
        self.console.mute()
        self.console.say(self.prompt())
        try:
            while True:
                loop_start = time.perf_counter()
                tick(self.leader, self.follower, processors, display=self.display)
                if self.recorder is not None:
                    self._ticks += 1
                    self._show_status()
                try:
                    line = self.console.poll()
                except EOFError:
                    break
                if line is not None and not self.handle(line):
                    break
                precise_sleep(max(period - (time.perf_counter() - loop_start), 0.0))
        except KeyboardInterrupt:
            if self._drew_status:
                self.console.end_status()
                self._drew_status = False
            self.console.say("\n  interrupt — ending the session and keeping what was recorded")
        finally:
            self.console.unmute()
            self.stop_episode()
            self.commit_held()

    def _show_status(self) -> None:
        now = time.perf_counter()
        if now - self._status_at < STATUS_EVERY_S:
            return
        self._status_at = now
        self._drew_status = True
        seconds = now - self._started
        rate = self._ticks / seconds if seconds > 0 else 0.0
        self.console.status(
            f"  recording  {seconds:6.1f} s  {self._ticks:6d} frames  {rate:4.1f} Hz"
        )


# --------------------------------------------------------------------------- #
# Connect, run, disconnect
# --------------------------------------------------------------------------- #


def run(
    leader: Any,
    follower: Any,
    dataset: Any,
    *,
    task: str,
    fps: int = 30,
    scene: int = 1,
    display: bool = False,
    notes: dict[str, Any] | None = None,
) -> DemoSession:
    """Connect both devices, run the demo session, disconnect. **Moves the arms.**

    Connecting is itself motion — see :mod:`dk1lab.cli.safety`. The dataset is
    closed here too, which commits anything still held and finalises the metadata:
    a dataset that is never finalised can be missing its last episodes even though
    their frames are on disk.
    """
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

    if display:
        init_visualization("rerun", session_name="dk1-demos")

    # Leader first, as in dk1lab.teleop: it is the passive half, and connecting it
    # before the followers are live means a hand already resting on a leader arm
    # cannot command anything yet.
    leader.connect()
    follower.connect()
    session = DemoSession(
        leader,
        follower,
        dataset,
        task=task,
        fps=fps,
        scene=scene,
        display=display,
        notes=notes,
    )
    try:
        session.loop()
    finally:
        if display:
            shutdown_visualization("rerun")
        follower.disconnect()
        leader.disconnect()
        # After the arms are safe: finalising encodes nothing but can take a
        # moment, and it must happen whatever the loop did.
        dataset.close()
    return session
