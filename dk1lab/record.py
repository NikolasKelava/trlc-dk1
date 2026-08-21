"""Record what a rollout actually did, to a Rerun ``.rrd`` file.

Four streams, which are the four `--display` already draws and no fewer:

=====================  ======================================================
``observation.<cam>``  the camera images, as the cell produced them — after
                       the rotation and after the wrist crop, because that is
                       what :mod:`dk1lab.crop` does inside the camera
``observation.<key>``  every measured joint position, from the same
                       ``get_observation`` the control loop reads
``policy/<key>``       the model's own row for this tick, in robot units,
                       before the cross-fade and before the speed limiter
``command/<key>``      what ``send_action`` returned — after both
=====================  ======================================================

The layout it writes is :func:`dk1lab.actionview.build_blueprint`, the same one
the live panel uses, so a replay is the thing that was watched rather than a
second thing to learn.

**It is not a LeRobot dataset, and that is deliberate.** There is no standard
slot in that format for "the policy's own plan" — the thing a rollout has to be
diagnosed against — and this is not the Phase 4 recorder. What it is for is
answering "what did that run actually do" an hour later, with the same three
lines per joint that made the sixth rollout's verdict readable.

**It attaches by wrapping**, the rule every instrument here follows
(:mod:`dk1lab.trace`, :mod:`dk1lab.actionview`, :mod:`dk1lab.modelview`). Three
wrappers — the robot's ``get_observation`` and ``send_action``, and the engine's
``get_action`` — each of which forwards its call untouched and logs as a side
effect. :meth:`EpisodeRecorder.detach` restores them, so a session can record
one rollout and not the next.

**Its own recording stream, not the global one.** Two reasons, both load-bearing.
LeRobot's ``log_rerun_data`` logs images with ``static=True``, and a static
entity keeps only its latest value — a file written through it would hold three
final frames rather than three streams. And a dedicated stream means recording
does not require ``--display``, does not fight upstream's blueprint cache, and
cannot be switched off by whatever the viewer is doing.

**The images are encoded on a worker thread.** Three 720p JPEGs cost ~5 ms
against a 33.3 ms control period, and the encode is not the only cost — running
it inline pushed the control thread's share from 1.8 ms to 5.1 ms, measured
here. The scalars are cheap (42 logs, well under a millisecond) and stay inline,
which is what keeps every stream on the same tick index. A frame that arrives
while the worker is busy waits in a bounded queue; one that arrives when the
queue is full is **dropped and counted** — and reported at the end, because a
recording with a hole in it must say so rather than look complete.

**A recorder must never take a rollout down.** Every call is guarded, and a
failure switches recording off for the rest of the episode rather than raising
into the control loop.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .layout import ACTION_KEYS, CAMERA_NAMES

logger = logging.getLogger(__name__)

#: Where recordings go when nothing else is said. Relative to the working
#: directory, next to `dk1.toml`, and git-ignored.
DEFAULT_RECORD_DIR = Path("recordings")

#: JPEG quality for the camera streams. 75 keeps a 180 s three-camera episode in
#: the low hundreds of MB; the images are for watching, not for training.
DEFAULT_JPEG_QUALITY = 75

#: Frames the encoder may fall behind by before frames start being dropped.
#: 60 ticks is two seconds at 30 Hz — enough to ride out a hiccup, short enough
#: that a worker which has genuinely stopped keeping up says so quickly.
DEFAULT_QUEUE_TICKS = 60

#: The recording's own timelines. ``tick`` is the control tick, which is what
#: every one of the four streams is indexed by; ``elapsed`` is seconds since the
#: episode started, so a replay runs at the speed the arms did.
TICK_TIMELINE = "tick"
TIME_TIMELINE = "elapsed"

_SLUG = re.compile(r"[^a-z0-9]+")

#: A recording's filename: a four-digit index, then the task it ran.
_INDEXED = re.compile(r"^(\d+)_")


def slug(text: str, *, limit: int = 40) -> str:
    """``text`` as a filename fragment: lowercase, hyphenated, trimmed.

    Empty when there is nothing usable left, which the caller is expected to
    handle by leaving the fragment out rather than writing a bare separator.
    """
    return _SLUG.sub("-", text.lower()).strip("-")[:limit].strip("-")


def next_index(directory: Path | str) -> int:
    """One past the highest episode index already in ``directory``.

    The index just counts up. It is read off the directory rather than kept in
    the session, so it keeps counting across sessions rather than starting again
    at 1 every time the process does — which would overwrite yesterday's first
    episode with today's. An episode that was discarded is gone, so its number
    is free again; every episode that was *kept* is still there and is counted.
    """
    highest = 0
    try:
        names = [path.name for path in Path(directory).glob("*.rrd")]
    except OSError:  # pragma: no cover - an unreadable directory is start-of-day
        return 1
    for name in names:
        if (match := _INDEXED.match(name)) is not None:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def episode_path(directory: Path | str, task: str, index: int) -> Path:
    """Where one episode is written: ``<dir>/<index>_<task>.rrd``.

    The index leads so a listing is in the order the episodes were run, and the
    task follows so a file is identifiable without opening it — which is what a
    colleague handed one needs. Both are also written inside the file.
    """
    fragment = slug(task)
    stem = f"{index:04d}_{fragment}" if fragment else f"{index:04d}"
    return Path(directory) / f"{stem}.rrd"


@dataclass(frozen=True)
class EpisodeRecording:
    """What one recording turned out to be. Printed when the episode ends."""

    path: Path
    ticks: int
    frames: int
    dropped: int
    seconds: float
    #: The reason recording stopped early, or ``None`` if it ran to the end.
    failed: str | None = None

    @property
    def megabytes(self) -> float:
        """Size on disk, or 0.0 if the file is not there — which is a finding."""
        try:
            return self.path.stat().st_size / 1e6
        except OSError:
            return 0.0

    def discard(self) -> bool:
        """Delete the file. Returns whether there was one to delete.

        For the "keep this episode?" prompt: the recording is written as the
        rollout runs — there is nowhere else to put 180 s of video — so
        declining it is a delete afterwards rather than a decision beforehand.
        """
        try:
            self.path.unlink()
            return True
        except OSError:
            return False

    def summary(self) -> str:
        """One line for the operator, with the drops stated rather than implied."""
        if self.ticks == 0:
            nothing = f"recorded nothing to {self.path}"
            return f"{nothing}\n  recording stopped early: {self.failed}" if self.failed else nothing
        rate = self.ticks / self.seconds if self.seconds > 0 else 0.0
        line = (
            f"recorded {self.ticks} ticks ({self.seconds:.1f} s at {rate:.1f} Hz), "
            f"{self.frames} camera frames, {self.megabytes:.1f} MB -> {self.path}"
        )
        if self.dropped:
            line += f"\n  {self.dropped} frames DROPPED — the encoder fell behind"
        if self.failed:
            line += f"\n  recording stopped early: {self.failed}"
        return line


class EpisodeRecorder:
    """Writes the four streams of one rollout to a ``.rrd``. **Read-only.**

    Lifecycle, and it is the same shape as every other instrument here::

        recorder = EpisodeRecorder(path, task=task)
        recorder.attach(ctx)      # wrap the three calls
        recorder.start()          # open the file, send the layout
        ...                       # the rollout runs
        recording = recorder.stop()   # flush, close, report
        recorder.detach()

    Args:
        path: the ``.rrd`` to write. Parent directories are created.
        task: the instruction this episode ran, written into the file so a
            recording is self-describing.
        notes: anything else worth keeping with it — the checkpoint, the speed
            cap, whether the gripper was inverted. Written as one static text
            document; a recording that cannot say what settings produced it is
            evidence of nothing.
        keys: the action keys, in this cell's order.
        camera_names: the observation keys that hold images.
        jpeg_quality: 0-100. Lower is smaller and faster to encode.
        queue_ticks: how far the image encoder may fall behind before frames are
            dropped.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        task: str = "",
        notes: dict[str, Any] | None = None,
        keys: tuple[str, ...] = ACTION_KEYS,
        camera_names: tuple[str, ...] = CAMERA_NAMES,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        queue_ticks: int = DEFAULT_QUEUE_TICKS,
    ) -> None:
        self.path = Path(path)
        self.task = task
        self.notes = dict(notes or {})
        self.keys = tuple(keys)
        self.camera_names = tuple(camera_names)
        self.jpeg_quality = int(jpeg_quality)
        self.ticks = 0
        self.frames = 0
        self.dropped = 0
        #: Set when logging raises, which switches recording off for the rest of
        #: the episode. One failure at 30 Hz is thirty failures a second.
        self.failed: str | None = None

        self._stream: Any = None
        self._engine: Any = None
        self._restore: list[tuple[Any, str, Any]] = []
        self._images: queue.Queue = queue.Queue(maxsize=max(1, int(queue_ticks)))
        self._worker: threading.Thread | None = None
        self._started: float = 0.0
        self._report: EpisodeRecording | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Open the file, send the layout, start the encoder. Idempotent.

        Guarded like everything else here: an unwritable path or a missing Rerun
        must leave the rollout running and unrecorded, not stop it. The failure
        is logged and carried into the report, because "no file appeared" is a
        thing the operator has to be told rather than left to notice.
        """
        if self._stream is not None or self.failed is not None:
            return
        self._started = time.perf_counter()
        try:
            import rerun as rr

            from .actionview import build_blueprint

            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A stream of our own, never the default one: the live viewer's
            # stream belongs to LeRobot, and make_default would redirect its
            # logging here.
            stream = rr.RecordingStream("dk1-rollout")
            stream.save(self.path)
            stream.send_blueprint(build_blueprint(self.keys, self.camera_names))
            stream.log(
                "run", rr.TextDocument(self._describe(), media_type="text/markdown"), static=True
            )
        except Exception as exc:  # noqa: BLE001 - a recorder must never stop a rollout
            self._fail(exc)
            return
        self._stream = stream
        self._worker = threading.Thread(target=self._encode, name="dk1-record", daemon=True)
        self._worker.start()
        logger.info("recording this episode to %s", self.path)

    def stop(self, timeout: float = 30.0) -> EpisodeRecording:
        """Drain the encoder, close the file, and report. Safe without :meth:`start`.

        The timeout is generous on purpose: the queue holds at most two seconds
        of frames, and losing them at the end of a run — the part that usually
        matters most — to save a fraction of a second would be a bad trade.

        Idempotent, and it returns the *same* report every time: two callers
        wanting to know what was recorded is normal — the rollout that closes it
        and the operator who is told about it — and the second one must not be
        handed a longer duration measured from a stopped clock.
        """
        if self._report is not None:
            return self._report
        seconds = time.perf_counter() - self._started if self._started else 0.0
        worker, self._worker = self._worker, None
        if worker is not None:
            self._images.put(None)
            worker.join(timeout=timeout)
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.flush()
            except Exception as exc:  # noqa: BLE001 - closing must not raise
                logger.warning("could not flush the recording: %s", exc)
        self._report = EpisodeRecording(
            path=self.path,
            ticks=self.ticks,
            frames=self.frames,
            dropped=self.dropped,
            seconds=seconds,
            failed=self.failed,
        )
        return self._report

    def _describe(self) -> str:
        """The run's own settings, as the document written into the file."""
        lines = [f"# {self.task or 'rollout'}", ""]
        lines += [f"- **{name}**: {value}" for name, value in self.notes.items()]
        lines.append(f"- **recorded**: {datetime.now().isoformat(timespec='seconds')}")
        return "\n".join(lines)

    # -- attach ------------------------------------------------------------- #

    def attach(self, ctx: Any) -> None:
        """Wrap the robot's two calls and the engine's, in tick order.

        ``get_observation`` opens the tick — it is where the tick index advances,
        so everything logged afterwards lands on the same index as the frame it
        was computed from. ``get_action`` gives the policy's own plan, and
        ``send_action`` what the arms were finally told.
        """
        engine = ctx.policy.inference
        self._engine = engine
        robot = getattr(getattr(ctx, "hardware", None), "robot_wrapper", None)

        inner_get = engine.get_action

        def get_action(obs_frame: Any) -> Any:
            action = inner_get(obs_frame)
            self._log_policy()
            return action

        engine.get_action = get_action
        self._restore.append((engine, "get_action", inner_get))

        if robot is None:
            logger.debug("no robot wrapper: recording the policy's plan only")
            return

        inner_obs = robot.get_observation

        def get_observation(*args: Any, **kwargs: Any) -> Any:
            observation = inner_obs(*args, **kwargs)
            self._log_observation(observation)
            return observation

        robot.get_observation = get_observation
        self._restore.append((robot, "get_observation", inner_obs))

        inner_send = robot.send_action

        def send_action(action: Any) -> Any:
            sent = inner_send(action)
            self._log_command(sent if sent is not None else action)
            return sent

        robot.send_action = send_action
        self._restore.append((robot, "send_action", inner_send))

    def detach(self) -> None:
        """Put the wrapped methods back, in reverse order of wrapping."""
        for owner, name, inner in reversed(self._restore):
            try:
                setattr(owner, name, inner)
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("could not restore %s.%s", type(owner).__name__, name)
        self._restore.clear()

    # -- the control loop's half -------------------------------------------- #

    def _log_observation(self, observation: Any) -> None:
        """Advance the tick, log every measured position, queue the images.

        The images are handed over by reference, which is safe for the same
        reason :mod:`dk1lab.modelview` relies on: ``OpenCVCamera`` publishes each
        frame as a **new** array rather than writing into the one it handed out,
        so the worker's copy stays the frame this tick saw.
        """
        if self._stream is None or self.failed is not None:
            return
        if not isinstance(observation, dict):
            return
        self.ticks += 1
        tick = self.ticks
        self._mark(tick)
        scalars = {
            key: value
            for key, value in observation.items()
            if key in self.keys and _scalar(value) is not None
        }
        self._log_scalars("observation", scalars, prefixed=False)
        frames = [
            (name, observation[name])
            for name in self.camera_names
            if observation.get(name) is not None
        ]
        if frames:
            self._offer(tick, time.perf_counter() - self._started, frames)

    def _log_policy(self) -> None:
        """The model's own row for this tick, if the engine keeps one.

        ``planned`` exists only on :mod:`dk1lab.fifo`'s engines; under ``--rtc``
        or ``--no-fifo`` there is nothing to record and the file carries the
        other three streams rather than an invented fourth.
        """
        row = getattr(self._engine, "planned", None)
        if row is None:
            return
        self._log_scalars("policy", dict(zip(self.keys, row, strict=False)))

    def _log_command(self, action: Any) -> None:
        """What ``send_action`` returned: the limited action the arms were given."""
        if isinstance(action, dict):
            self._log_scalars("command", action)

    def _offer(self, tick: int, elapsed: float, frames: list[tuple[str, Any]]) -> None:
        """Hand the frames to the encoder, or drop them and count it."""
        try:
            self._images.put_nowait((tick, elapsed, frames))
        except queue.Full:
            self.dropped += len(frames)

    def _mark(self, tick: int) -> None:
        """Put this thread's stream clock on ``tick``, and on the wall clock."""
        self._stream.set_time(TICK_TIMELINE, sequence=tick)
        self._stream.set_time(TIME_TIMELINE, duration=time.perf_counter() - self._started)

    def _log_scalars(self, prefix: str, values: dict, *, prefixed: bool = True) -> None:
        """One ``log`` per named scalar, under ``<prefix>/<key>`` or ``observation.<key>``.

        Per key rather than one batched ``Scalars`` for the same reason
        :mod:`dk1lab.actionview` does it: the layout overlays three series on a
        per-joint axis, and a batch under one path cannot be split across views.
        """
        if self._stream is None or self.failed is not None:
            return
        try:
            import rerun as rr

            for key, value in values.items():
                if key not in self.keys:
                    continue
                if (number := _scalar(value)) is None:
                    continue
                path = f"{prefix}/{key}" if prefixed else f"{prefix}.{key}"
                self._stream.log(path, rr.Scalars(number))
        except Exception as exc:  # noqa: BLE001 - a recorder must never stop a rollout
            self._fail(exc)

    # -- the encoder's half -------------------------------------------------- #

    def _encode(self) -> None:
        """JPEG-encode and log queued frames until :meth:`stop` sends the sentinel."""
        while (item := self._images.get()) is not None:
            tick, elapsed, frames = item
            self._write(tick, elapsed, frames)

    def _write(self, tick: int, elapsed: float, frames: list[tuple[str, Any]]) -> None:
        stream = self._stream
        if stream is None or self.failed is not None:
            return
        try:
            import cv2
            import rerun as rr

            # The clock is per-thread state on the stream, so the worker sets its
            # own — the tick the frame came from, not the tick now.
            stream.set_time(TICK_TIMELINE, sequence=tick)
            stream.set_time(TIME_TIMELINE, duration=elapsed)
            for name, image in frames:
                # cv2 rather than rerun's own ``Image.compress``: 1.8 ms against
                # 4.8 for the same 720p frame and the same quality, measured
                # here. The encoder is a background thread but the CPU it burns
                # is not free, and the swap is a colour-order flip — cv2 writes
                # its input as BGR, and LeRobot's cameras deliver RGB.
                ok, buffer = cv2.imencode(
                    ".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    continue
                stream.log(
                    f"observation.{name}",
                    rr.EncodedImage(contents=buffer.tobytes(), media_type="image/jpeg"),
                )
                self.frames += 1
        except Exception as exc:  # noqa: BLE001 - a recorder must never stop a rollout
            self._fail(exc)

    def _fail(self, exc: BaseException) -> None:
        """Switch recording off, once, and say why."""
        if self.failed is None:
            self.failed = str(exc)
            logger.warning("recording disabled after an error: %s", exc)


def _scalar(value: Any) -> float | None:
    """``value`` as a float, or ``None`` if it is not one number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
