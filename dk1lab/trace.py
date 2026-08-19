"""Watch a rollout from the inside: what the policy sees, what it says, and where the time goes.

This module exists because of two symptoms the first rollouts produced that
nothing in the offline measurements explains:

``Indexes diff is not equal to real delay. indexes_diff=10, real_delay=27``
    RTC's own arithmetic, telling you that the chunk it just produced took 27
    ticks (900 ms) of wall time while the control loop managed to consume only 10
    actions. Both halves are bad. 900 ms is more than three times the 271 ms
    measured offline, and a 30-step chunk trimmed by 27 leaves **three** actions
    — a tenth of a second of motion, followed by another 900 ms of silence.

``No command for 0.50 s — holding last commanded target``
    The other end of the same fact, reported by the motor chain: the queue ran
    dry and the control loop had nothing to send. That is the stall.

Neither number can be recovered from LeRobot's logs after the fact, and the one
that matters most — *why* a chunk costs 900 ms in situ when it costs 271 ms on
the bench — is not logged at all. So this instruments the four places the time
can go and records them per chunk:

===================  =========================================================
``model_ms``         inside ``predict_action_chunk`` — the GPU work
``pre_ms``           the policy preprocessor: normalise, tokenise, to device
``post_ms``          the policy postprocessor: unnormalise, gripper transform
``total_ticks``      what RTC itself measured, in ticks, and trimmed the chunk by
===================  =========================================================

If ``model_ms`` alone accounts for 900 ms the problem is the GPU path; if it is
270 ms and the total is 900, the cost is in the pre/post-processing or in the
background thread losing the CPU to the control loop — a completely different
fix. Measuring only the path you are not going to run is what caused the
previous mis-diagnosis; this measures all of them, on the arms.

It also answers the second open question of Phase 3 — *does the policy ever
actually move the gripper* — by recording the model's **own** output, before the
postprocessor unnormalises it and before the gripper inversion flips it. What
the arms are told and what the policy said are two different vectors, and only
one of them is evidence about the policy.

Nothing here changes behaviour. Every wrapper forwards, times, and returns; a
trace that is not attached costs nothing.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from .layout import ACTION_KEYS, GRIPPER_INDICES

logger = logging.getLogger(__name__)

#: Below this fraction of ticks served, the control loop spent more time holding
#: its last command than executing new ones, and "stalling" is the right word.
STARVED_FRACTION = 0.9


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkRecord:
    """One action chunk: what it cost, what it said, and what RTC did with it.

    ``raw`` is the policy's own output — the flow-matching head's 14 numbers,
    still in normalised space, before ``clamp_action``, before the unnormaliser
    and before the gripper transform. ``robot`` is the same row after all of
    those, which is what the arms are told (up to the limiter). They are kept
    separately because a question about the *policy* cannot be answered with a
    vector the postprocessor rewrote.
    """

    index: int
    model_ms: float
    pre_ms: float
    post_ms: float
    #: RTC's own latency measurement for this chunk, in control ticks. It is
    #: also the number of actions it discarded from the front of the chunk.
    total_ticks: int
    #: Actions the control loop actually consumed while this chunk was computing.
    consumed: int
    #: Actions left in the queue after the merge. This is how much motion the
    #: arms have available until the *next* chunk lands.
    queue_after: int
    raw: tuple[float, ...]
    robot: tuple[float, ...]

    @property
    def total_ms(self) -> float:
        """RTC's latency for this chunk in milliseconds, as it measured it."""
        return self.total_ticks * self._tick_ms

    #: Set by :class:`RolloutTrace` from the run's fps so ``total_ms`` can exist.
    _tick_ms: float = 1000.0 / 30.0

    @property
    def unaccounted_ms(self) -> float:
        """Wall time the chunk cost that none of the three timers explain.

        Preprocessing, the forward pass and postprocessing are the whole of what
        the RTC thread does per iteration. A large remainder means the thread was
        not running — descheduled, or waiting on the GIL behind the control
        loop's camera and serial work — and no amount of GPU tuning will fix it.
        """
        return self.total_ms - (self.model_ms + self.pre_ms + self.post_ms)

    @property
    def starved(self) -> int:
        """Ticks the control loop had nothing to send while this chunk computed.

        The loop consumed ``consumed`` actions over ``total_ticks`` ticks. The
        difference is time the arms spent holding their last commanded target —
        which is what ``No command for 0.50 s`` in the motor-chain log is
        reporting, seen from the other side.
        """
        return max(0, self.total_ticks - self.consumed)

    def grippers(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """The two gripper channels out of a 14-vector."""
        return tuple(values[i] for i in GRIPPER_INDICES if i < len(values))

    def line(self) -> str:
        """One dense line per chunk, for a live console."""
        return (
            f"chunk {self.index:3d}  {self.total_ms:5.0f} ms = {self.total_ticks:2d} ticks "
            f"(model {self.model_ms:.0f} · pre {self.pre_ms:.0f} · post {self.post_ms:.0f} · "
            f"other {self.unaccounted_ms:.0f})  "
            f"consumed {self.consumed:2d}  queue -> {self.queue_after:2d}"
            + (f"  STARVED {self.starved} ticks" if self.starved else "")
        )

    def action_lines(self) -> list[str]:
        """The policy's own first action, and the same row after LeRobot."""
        raw_l, raw_r = (self.grippers(self.raw) + (float("nan"),) * 2)[:2]
        rob_l, rob_r = (self.grippers(self.robot) + (float("nan"),) * 2)[:2]
        return [
            "  policy  " + " ".join(f"{v:+.3f}" for v in self.raw)
            + f"   grip L {raw_l:+.3f} R {raw_r:+.3f}",
            "  robot   " + " ".join(f"{v:+.3f}" for v in self.robot)
            + f"   grip L {rob_l:+.3f} R {rob_r:+.3f}",
        ]


@dataclass
class TraceSummary:
    """What a whole run's chunks add up to. Printed once, at the end."""

    chunks: int
    ticks_asked: int
    ticks_served: int
    model_ms: float
    pre_ms: float
    post_ms: float
    total_ms: float
    total_ticks: int
    consumed: int
    queue_after: int
    gripper_span: tuple[float, float]
    #: Whether these chunks came through the RTC queue. Under ``--sync`` there is
    #: no queue, no trimming and no starvation, so half of this is not a reading
    #: that exists and is not printed.
    rtc: bool = True

    @property
    def served_fraction(self) -> float:
        return self.ticks_served / self.ticks_asked if self.ticks_asked else 1.0

    def lines(self) -> list[str]:
        out = [
            f"chunks              {self.chunks}",
            f"median model call   {self.model_ms:.0f} ms",
            f"  preprocessor      {self.pre_ms:.0f} ms",
            f"  postprocessor     {self.post_ms:.0f} ms",
        ]
        if self.rtc:
            out[1:1] = [
                f"median chunk cost   {self.total_ms:.0f} ms = {self.total_ticks} ticks "
                f"(RTC's own measurement)"
            ]
            out += [
                f"  unaccounted       "
                f"{self.total_ms - self.model_ms - self.pre_ms - self.post_ms:.0f} ms",
                f"median consumed     {self.consumed} actions per chunk",
                f"median queue depth  {self.queue_after} actions after each merge",
                f"loop served         {self.ticks_served}/{self.ticks_asked} ticks "
                f"({self.served_fraction * 100:.0f}%)",
            ]
        out += [
            f"gripper range       policy commanded {self.gripper_span[0]:+.3f} .. "
            f"{self.gripper_span[1]:+.3f}"
        ]
        return out

    def verdicts(self) -> list[str]:
        """Plain readings of the numbers. Empty when nothing is wrong."""
        out: list[str] = []
        if self.chunks == 0:
            return ["no actions were produced at all — the policy never ran"]
        if self.rtc and self.served_fraction < STARVED_FRACTION:
            out.append(
                f"THE QUEUE RAN DRY: the loop had an action to send on only "
                f"{self.served_fraction * 100:.0f}% of ticks. The arms held their last "
                f"commanded target for the rest, which is the stall you are watching."
            )
        if self.rtc and self.total_ticks >= 20:
            out.append(
                f"INFERENCE IS TOO SLOW FOR RTC: {self.total_ms:.0f} ms is {self.total_ticks} "
                f"ticks, and RTC discards that many actions from the front of every "
                f"30-step chunk. Only {max(0, 30 - self.total_ticks)} actions survive, so the "
                f"arms get {max(0, 30 - self.total_ticks) / 30:.2f} s of motion per chunk and "
                f"wait {self.total_ms:.0f} ms for the next one. It also pins the prefix "
                f"weights at 1.0 across the whole execution horizon, which is why the policy "
                f"stops reacting to the scene."
            )
        unaccounted = self.total_ms - self.model_ms - self.pre_ms - self.post_ms
        if self.rtc and unaccounted > max(50.0, 0.3 * self.total_ms):
            out.append(
                f"{unaccounted:.0f} ms per chunk is not in the model, the preprocessor or the "
                f"postprocessor. The RTC thread is not running during that time — CPU or GIL "
                f"contention with the control loop's camera and serial reads, not GPU cost. "
                f"Tuning the model will not help; reducing per-tick work will."
            )
        span = self.gripper_span[1] - self.gripper_span[0]
        if span < 0.05:
            out.append(
                f"the policy never moved a gripper: its own output spanned {span:.3f} across "
                f"the whole run. Nothing here says the inversion is right or wrong — the "
                f"channel simply never changed, so it is still untested."
            )
        return out


# --------------------------------------------------------------------------- #
# Proxies
# --------------------------------------------------------------------------- #


class _TimedPipeline:
    """A processor pipeline that records how long each call took.

    A proxy rather than a subclass because the pipeline objects are already
    built, already patched by :func:`~dk1lab.policy.apply_gripper_inversion`, and
    already referenced by the inference engine. Everything except ``__call__``
    forwards, so ``reset()``, ``.steps`` and anything else LeRobot reaches for
    still lands on the real object.
    """

    def __init__(self, inner: Any, on_call: Any) -> None:
        self._inner = inner
        self._on_call = on_call

    def __call__(self, value: Any) -> Any:
        start = time.perf_counter()
        result = self._inner(value)
        self._on_call((time.perf_counter() - start) * 1000.0, value, result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _first_row(tensor: Any) -> tuple[float, ...]:
    """The first action of a chunk, whatever leading batch dimensions it has.

    ``predict_action_chunk`` returns ``[1, chunk, dim]``; the sync engine's
    ``select_action`` returns ``[1, dim]``. Both reduce to one 14-vector.
    """
    try:
        while getattr(tensor, "ndim", 1) > 1:
            tensor = tensor[0]
        return tuple(float(v) for v in tensor)
    except Exception:  # noqa: BLE001 - a trace must never break a rollout
        return ()


# --------------------------------------------------------------------------- #
# The trace
# --------------------------------------------------------------------------- #


@dataclass
class RolloutTrace:
    """Instrumentation attached to a live rollout. **Read-only** — it never acts.

    Attach in two stages, because the pieces come into existence at different
    times:

    ``attach(ctx)``
        after :func:`~dk1lab.policy.build_context` (and therefore after
        ``prewarm``, so the cold call is not recorded) — wraps the policy call
        and both pipelines.
    ``attach_queue(strategy)``
        after ``strategy.setup``, because the RTC action queue does not exist
        until ``engine.start()`` creates it.

    Args:
        fps: the run's control rate, used to turn RTC's tick counts into
            milliseconds.
        on_chunk: called with each :class:`ChunkRecord` as it completes, from the
            RTC background thread. Keep it cheap — printing is fine, blocking is
            not.
        display: log the model's actual inputs and outputs to Rerun. See
            :meth:`_log_rerun`.
    """

    fps: float = 30.0
    on_chunk: Any = None
    display: bool = False
    chunks: list[ChunkRecord] = field(default_factory=list)
    #: Ticks on which the control loop asked for an action, and got one.
    ticks_asked: int = 0
    ticks_served: int = 0
    #: True once an RTC action queue has been found and wrapped. Under ``--sync``
    #: — which is what ``dk1 policy dryrun`` runs — there is no queue, so records
    #: are cut at the postprocessor instead and the queue readings are absent
    #: rather than zero.
    rtc: bool = False

    _model_ms: float = 0.0
    _pre_ms: float = 0.0
    _post_ms: float = 0.0
    _raw: tuple[float, ...] = ()
    _robot: tuple[float, ...] = ()
    _images: dict = field(default_factory=dict)

    # -- attach ------------------------------------------------------------- #

    def attach(self, ctx: Any) -> None:
        """Wrap the policy call and both processor pipelines on the live engine.

        The engine holds its **own** references to the pipelines (captured in its
        constructor), so replacing ``ctx.policy.preprocessor`` alone would not
        reach it. Both are replaced, and the engine's private attributes with
        them, so sync and RTC are instrumented the same way.
        """
        engine = ctx.policy.inference
        policy = ctx.policy.policy

        pre = _TimedPipeline(ctx.policy.preprocessor, self._record_pre)
        post = _TimedPipeline(ctx.policy.postprocessor, self._record_post)
        ctx.policy.preprocessor, ctx.policy.postprocessor = pre, post
        for name, proxy in (("_preprocessor", pre), ("_postprocessor", post)):
            if hasattr(engine, name):
                setattr(engine, name, proxy)

        inner_predict = policy.predict_action_chunk

        def predict(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return inner_predict(*args, **kwargs)
            finally:
                self._model_ms = (time.perf_counter() - start) * 1000.0

        policy.predict_action_chunk = predict

        inner_get = engine.get_action

        def get_action(obs_frame: Any) -> Any:
            action = inner_get(obs_frame)
            self.ticks_asked += 1
            self.ticks_served += action is not None
            return action

        engine.get_action = get_action

    def attach_queue(self, strategy: Any) -> None:
        """Wrap the RTC action queue's ``merge``, which is where a chunk lands.

        This is the only place that knows all three of RTC's own latency estimate,
        how many actions the control loop got through in the meantime, and how
        much motion is left afterwards. LeRobot computes the first two, compares
        them, logs a warning when they disagree — and then throws the comparison
        away. Nothing else records the queue depth at all.

        A no-op under ``--sync``, which has no queue.
        """
        engine = getattr(strategy, "_engine", None)
        queue = getattr(engine, "_action_queue", None)
        if queue is None:
            logger.debug("no RTC action queue to trace (sync inference?)")
            return
        self.rtc = True

        inner_merge = queue.merge

        def merge(
            original_actions: Any,
            processed_actions: Any,
            real_delay: int,
            action_index_before_inference: int | None = None,
        ) -> Any:
            before = action_index_before_inference
            consumed = 0 if before is None else max(0, queue.get_action_index() - before)
            result = inner_merge(
                original_actions, processed_actions, real_delay, action_index_before_inference
            )
            self._record_chunk(int(real_delay), consumed, queue.qsize())
            return result

        queue.merge = merge

    # -- recording ---------------------------------------------------------- #

    def _record_pre(self, ms: float, _value: Any, result: Any) -> None:
        self._pre_ms = ms
        if self.display:
            self._images = {
                key: value
                for key, value in (result or {}).items()
                if isinstance(key, str) and "image" in key
            }

    def _record_post(self, ms: float, value: Any, result: Any) -> None:
        self._post_ms = ms
        self._raw = _first_row(value)
        self._robot = _first_row(result)
        if not self.rtc:
            # Sync inference: the postprocessor is the last thing that happens to
            # an action, so it is where a record has to be cut. There is no queue
            # to have trimmed it and no delay to have been measured.
            self._record_chunk(0, 0, 0)

    def _record_chunk(self, real_delay: int, consumed: int, queue_after: int) -> None:
        record = ChunkRecord(
            index=len(self.chunks),
            model_ms=self._model_ms,
            pre_ms=self._pre_ms,
            post_ms=self._post_ms,
            total_ticks=real_delay,
            consumed=consumed,
            queue_after=queue_after,
            raw=self._raw,
            robot=self._robot,
            _tick_ms=1000.0 / self.fps if self.fps else 1000.0 / 30.0,
        )
        self.chunks.append(record)
        if self.display:
            self._log_rerun(record)
        if self.on_chunk is not None:
            try:
                self.on_chunk(record)
            except Exception:  # noqa: BLE001 - never let reporting kill the run
                logger.exception("trace callback failed")

    # -- rerun -------------------------------------------------------------- #

    def _log_rerun(self, record: ChunkRecord) -> None:
        """Log what the model actually received and said, to Rerun.

        The point of logging the **preprocessed** images rather than the robot's
        is that they are not the same picture. Teleoperation already showed that
        the robot-side view is right way up — that is `--display`, and it is not
        in question. What is in question is everything the policy pipeline does
        after that: the rename, the ordering of the three keys, the channel
        layout, the resize. This logs the tensors as the model sees them, under
        ``policy_input/``, so "correct in Rerun during teleop" and "correct at
        the model's input" stop being the same claim.

        Called from the RTC background thread; Rerun's recording stream is
        global and thread-safe, so this needs no handshake with the control loop.
        """
        try:
            import numpy as np
            import rerun as rr
        except ImportError:  # pragma: no cover - display is opt-in
            return

        for key, tensor in self._images.items():
            array = _as_image(tensor, np)
            if array is not None:
                rr.log(f"policy_input/{key.rsplit('.', 1)[-1]}", rr.Image(array))

        for name, values in (("policy_output", record.raw), ("to_robot", record.robot)):
            for key, value in zip(ACTION_KEYS, values, strict=False):
                rr.log(f"{name}/{key}", rr.Scalars(float(value)))

        rr.log("rtc/queue_after_merge", rr.Scalars(float(record.queue_after)))
        rr.log("rtc/chunk_ms", rr.Scalars(float(record.total_ms)))
        rr.log("rtc/starved_ticks", rr.Scalars(float(record.starved)))

    # -- reporting ---------------------------------------------------------- #

    def summary(self) -> TraceSummary:
        """Medians rather than means: one 3-second hitch should not set the story."""

        def median(values: list[float]) -> float:
            return statistics.median(values) if values else 0.0

        grippers = [
            value
            for chunk in self.chunks
            for value in chunk.grippers(chunk.raw)
        ]
        return TraceSummary(
            chunks=len(self.chunks),
            ticks_asked=self.ticks_asked,
            ticks_served=self.ticks_served,
            model_ms=median([c.model_ms for c in self.chunks]),
            pre_ms=median([c.pre_ms for c in self.chunks]),
            post_ms=median([c.post_ms for c in self.chunks]),
            total_ms=median([c.total_ms for c in self.chunks]),
            total_ticks=int(median([float(c.total_ticks) for c in self.chunks])),
            consumed=int(median([float(c.consumed) for c in self.chunks])),
            queue_after=int(median([float(c.queue_after) for c in self.chunks])),
            gripper_span=(min(grippers), max(grippers)) if grippers else (0.0, 0.0),
            rtc=self.rtc,
        )


def _as_image(tensor: Any, np: Any) -> Any:
    """A torch or numpy image in whatever layout, as HWC uint8. ``None`` if it isn't one."""
    try:
        array = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        while array.ndim > 3:
            array = array[0]
        if array.ndim != 3:
            return None
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = array.transpose(1, 2, 0)
        if array.dtype != np.uint8:
            # Normalised floats live in [0, 1]; anything wider is already 0..255.
            scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        return array
    except Exception:  # noqa: BLE001 - a trace must never break a rollout
        return None
