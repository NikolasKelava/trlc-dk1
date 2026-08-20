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

**Under ``--sync`` none of that shape applies, and pretending it did produced
nonsense.** There is no background thread, no queue and no RTC latency estimate;
``SyncInferenceEngine`` runs ``preprocessor -> select_action -> postprocessor``
inline on *every tick*, and ``select_action`` serves from MolmoAct2's cached
30-step chunk on 29 ticks out of 30. Cutting a record at the postprocessor
therefore cut one per tick, called it a chunk, handed it RTC's ``real_delay`` of
zero as a total, and printed ``0 ms = 0 ticks ... other -160``: a span measured
across something that is not a chunk boundary, and a remainder of three timers
minus a total of nothing.

So sync is traced on its own terms. The unit is the window between two real
inference calls — detectable because ``predict_action_chunk`` runs only when the
policy's own queue is empty — and what it records is the thing that matters at
27.7 Hz: where each **tick** goes.

===================  =========================================================
``wall_ms``          measured wall clock across the window, not inferred
``ticks``            ticks in it — 30 for MolmoAct2: one inference, 29 cached
``model_ms``         the one real forward pass, at the head of the window
``pre_ms``           the preprocessor, per *cached* tick — it re-runs every tick
``post_ms``          the postprocessor, per cached tick
``select_ms``        ``select_action`` popping a cached row, per cached tick
``outside_ms``       tick period minus the engine call: cameras, serial, limiter
===================  =========================================================

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
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from .layout import ACTION_KEYS, GRIPPER_INDICES, IMAGE_KEYS

logger = logging.getLogger(__name__)

#: Below this fraction of ticks served, the control loop spent more time holding
#: its last command than executing new ones, and "stalling" is the right word.
STARVED_FRACTION = 0.9


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TickCosts:
    """One control tick under ``--sync``, timed from inside the inference engine.

    ``period_ms`` is measured to the *next* tick's start, so it is unknown until
    that tick arrives and is filled in then. It is the only number here that
    includes what the control loop does outside the engine — reading three
    cameras, reading twelve motors, the limiter, the dataset write — which is
    exactly the part a "Record loop is running slower" warning is about and the
    part no timer inside the engine can see.
    """

    start: float
    engine_ms: float
    pre_ms: float
    post_ms: float
    #: Nonzero only on the tick where ``predict_action_chunk`` actually ran.
    model_ms: float = 0.0
    period_ms: float = 0.0

    @property
    def inferred(self) -> bool:
        return self.model_ms > 0.0

    @property
    def select_ms(self) -> float:
        """The engine call minus the two pipelines: ``select_action`` itself."""
        return max(0.0, self.engine_ms - self.pre_ms - self.post_ms - self.model_ms)

    @property
    def outside_ms(self) -> float:
        """Tick period minus the engine call — everything else in the loop."""
        return max(0.0, self.period_ms - self.engine_ms)


@dataclass(frozen=True)
class SyncWindow:
    """The ticks between two inference calls: one chunk's worth of execution.

    Medians are over the **cached** ticks only. The inference tick is a different
    animal — it carries the whole forward pass and is reported separately as
    ``infer_ms`` — and averaging it in would hide the per-tick cost that the
    other 29 ticks are actually paying.
    """

    ticks: int
    wall_ms: float
    #: Sum of every ``get_action`` call in the window, inference included.
    engine_total_ms: float
    #: The engine call on the inference tick: the visible pause under sync.
    infer_ms: float
    period_ms: float
    period_p95_ms: float
    engine_ms: float
    pre_ms: float
    post_ms: float
    select_ms: float
    outside_ms: float

    @property
    def hz(self) -> float:
        """The rate the window actually ran at, pause included."""
        return 1000.0 * self.ticks / self.wall_ms if self.wall_ms else 0.0

    @property
    def cached_hz(self) -> float:
        """The rate the cached ticks ran at, i.e. with the pause taken out."""
        return 1000.0 / self.period_ms if self.period_ms else 0.0


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

    #: Present only under ``--sync``, where none of ``total_ticks``, ``consumed``
    #: and ``queue_after`` exist and the window between two inference calls is
    #: what a record spans instead.
    sync: SyncWindow | None = None

    @property
    def total_ms(self) -> float:
        """How long this chunk took, in milliseconds.

        Under RTC that is RTC's own latency measurement, which is the number it
        trimmed the chunk by. Under sync nobody measures it, so the trace does:
        wall clock from one inference call to the next.
        """
        if self.sync is not None:
            return self.sync.wall_ms
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

        Under sync the three timers cover only the inference *engine*, and the
        window deliberately contains 29 further ticks of loop work they were
        never meant to explain. So the remainder there is wall clock minus every
        engine call: the cameras, the motor reads, the limiter and the dataset
        write, which is where the missing milliseconds per tick have to be.
        """
        if self.sync is not None:
            return self.sync.wall_ms - self.sync.engine_total_ms
        return self.total_ms - (self.model_ms + self.pre_ms + self.post_ms)

    @property
    def starved(self) -> int:
        """Ticks the control loop had nothing to send while this chunk computed.

        The loop consumed ``consumed`` actions over ``total_ticks`` ticks. The
        difference is time the arms spent holding their last commanded target —
        which is what ``No command for 0.50 s`` in the motor-chain log is
        reporting, seen from the other side.
        """
        if self.sync is not None:
            return 0
        return max(0, self.total_ticks - self.consumed)

    def grippers(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """The two gripper channels out of a 14-vector."""
        return tuple(values[i] for i in GRIPPER_INDICES if i < len(values))

    def line(self) -> str:
        """One dense line per chunk, for a live console.

        Two shapes, because the two engines produce genuinely different readings
        and a single format could only be right about one of them. Under RTC the
        question is what the queue got; under sync there is no queue, and the
        question is where a 33.3 ms tick went.
        """
        if self.sync is not None:
            w = self.sync
            return (
                f"chunk {self.index:3d}  {w.wall_ms:6.0f} ms over {w.ticks:3d} ticks  "
                f"= {w.hz:4.1f} Hz  (pause {w.infer_ms:.0f} ms, model {self.model_ms:.0f})\n"
                f"  per cached tick  {w.period_ms:5.1f} ms = {w.cached_hz:4.1f} Hz  "
                f"(pre {w.pre_ms:.1f} · select {w.select_ms:.1f} · post {w.post_ms:.1f} · "
                f"loop {w.outside_ms:.1f})"
            )
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
    #: The per-tick reading, medians of the per-chunk medians. Sync only.
    sync: SyncWindow | None = None
    #: The rate the loop was asked to run at, so the sync verdict can say by how
    #: much it fell short rather than just quoting a number.
    fps: float = 30.0

    @property
    def served_fraction(self) -> float:
        return self.ticks_served / self.ticks_asked if self.ticks_asked else 1.0

    @property
    def budget_ms(self) -> float:
        """The tick period the loop was asked to hold."""
        return 1000.0 / self.fps if self.fps else 1000.0 / 30.0

    def lines(self) -> list[str]:
        if self.sync is not None:
            return self._sync_lines()
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
        out += [self._gripper_line()]
        return out

    def _gripper_line(self) -> str:
        return (
            f"gripper range       policy commanded {self.gripper_span[0]:+.3f} .. "
            f"{self.gripper_span[1]:+.3f}"
        )

    def _sync_lines(self) -> list[str]:
        """The sync reading: the tick budget, and what is spending it.

        Everything here is per cached tick except the model call and the pause,
        because 29 ticks in 30 are cached ticks and they are the ones that have
        to fit in the budget.
        """
        w = self.sync
        assert w is not None
        return [
            f"chunks              {self.chunks} of {w.ticks} ticks",
            f"loop rate           {w.cached_hz:.1f} Hz between pauses, "
            f"{w.hz:.1f} Hz including them (target {self.fps:.0f} Hz)",
            f"median tick         {w.period_ms:.1f} ms of a {self.budget_ms:.1f} ms budget"
            + (f", p95 {w.period_p95_ms:.1f} ms" if w.period_p95_ms else ""),
            f"  inference engine  {w.engine_ms:.1f} ms",
            f"    preprocessor    {w.pre_ms:.1f} ms   (re-runs every tick)",
            f"    select_action   {w.select_ms:.1f} ms   (serves the cached chunk)",
            f"    postprocessor   {w.post_ms:.1f} ms",
            f"  rest of the loop  {w.outside_ms:.1f} ms   "
            f"(cameras, motor reads, limiter, dataset)",
            f"median model call   {self.model_ms:.0f} ms, once per chunk",
            f"  chunk pause       {w.infer_ms:.0f} ms of held position, once per "
            f"{w.ticks} ticks",
            self._gripper_line(),
        ]

    def verdicts(self) -> list[str]:
        """Plain readings of the numbers. Empty when nothing is wrong."""
        out: list[str] = []
        if self.chunks == 0:
            return ["no actions were produced at all — the policy never ran"]
        out += self._sync_verdicts()
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

    def _sync_verdicts(self) -> list[str]:
        """Readings that only exist under sync. Empty when the loop held its rate.

        The threshold is one whole millisecond of overrun rather than a
        percentage: at 30 Hz the budget is 33.3 ms and jitter of a few tenths is
        the normal state of a loop that is keeping up.
        """
        w = self.sync
        if w is None or w.period_ms <= self.budget_ms + 1.0:
            return []
        over = w.period_ms - self.budget_ms
        out = [
            f"THE LOOP IS SLOW: {w.period_ms:.1f} ms per cached tick against a "
            f"{self.budget_ms:.1f} ms budget — {over:.1f} ms over, i.e. {w.cached_hz:.1f} Hz "
            f"instead of {self.fps:.0f}. The inference engine is {w.engine_ms:.1f} ms of that "
            f"and the rest of the loop is {w.outside_ms:.1f} ms."
        ]
        if w.pre_ms >= over:
            out.append(
                f"THE PREPROCESSOR ALONE COVERS THE OVERRUN: {w.pre_ms:.1f} ms per tick, "
                f"against {over:.1f} ms of overrun. It re-packs all three camera views every "
                f"tick, while select_action serves a cached row on 29 ticks in 30, so that "
                f"work is thrown away 29 times per chunk. Lowering [capture.policy], or "
                f"serving postprocessed actions from a local FIFO instead of calling "
                f"select_action per tick, both remove it."
            )
        elif w.outside_ms >= over:
            out.append(
                f"THE COST IS OUTSIDE THE INFERENCE ENGINE: {w.outside_ms:.1f} ms per tick of "
                f"cameras, motor reads, the limiter and the dataset write, against "
                f"{over:.1f} ms of overrun. Nothing about the policy or the capture "
                f"resolution will fix this one."
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
    #: The sync ticks since the last real inference call. Empty under RTC.
    _window: list[TickCosts] = field(default_factory=list)

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
            # Under sync this call *is* the tick: the whole pipeline runs inside
            # it. Reset the model timer first so it reads nonzero only when
            # ``select_action`` found its queue empty and really inferred.
            syncing = not self.rtc
            if syncing:
                self._model_ms = 0.0
            start = time.perf_counter()
            action = inner_get(obs_frame)
            engine_ms = (time.perf_counter() - start) * 1000.0
            self.ticks_asked += 1
            self.ticks_served += action is not None
            if syncing:
                self._record_tick(start, engine_ms)
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
            self._images = model_input_images(result)

    def _record_post(self, ms: float, value: Any, result: Any) -> None:
        self._post_ms = ms
        self._raw = _first_row(value)
        self._robot = _first_row(result)

    # -- sync windowing ----------------------------------------------------- #

    def _record_tick(self, start: float, engine_ms: float) -> None:
        """Fold one sync tick into the open window, closing it on an inference.

        The postprocessor used to be where a record was cut, and under sync it
        runs on every tick — so a 180 s run reported 4390 "chunks", each with a
        total of zero. A chunk boundary under sync is the tick that actually ran
        the model, and nothing else is.
        """
        tick = TickCosts(
            start=start,
            engine_ms=engine_ms,
            pre_ms=self._pre_ms,
            post_ms=self._post_ms,
            model_ms=self._model_ms,
        )
        if tick.inferred and self._window:
            # This tick opens a new window, so it also closes the previous one:
            # its start is where the previous window's wall clock stops.
            self._close_window(start)
        self._window.append(tick)

    def _close_window(self, end: float) -> None:
        """Turn the open window into a :class:`ChunkRecord` and start a new one."""
        window, self._window = self._window, []
        head = window[0]
        # The period of the last tick runs to ``end`` — the next window's first
        # tick — which is the one period no later tick can supply.
        starts = [tick.start for tick in window[1:]] + [end]
        timed = [
            TickCosts(
                start=tick.start,
                engine_ms=tick.engine_ms,
                pre_ms=tick.pre_ms,
                post_ms=tick.post_ms,
                model_ms=tick.model_ms,
                period_ms=(nxt - tick.start) * 1000.0,
            )
            for tick, nxt in zip(window, starts, strict=True)
        ]
        cached = [tick for tick in timed if not tick.inferred] or timed
        periods = sorted(tick.period_ms for tick in cached)
        self._record_chunk(
            0,
            0,
            0,
            sync=SyncWindow(
                ticks=len(timed),
                wall_ms=(end - head.start) * 1000.0,
                engine_total_ms=sum(tick.engine_ms for tick in timed),
                infer_ms=head.engine_ms,
                period_ms=statistics.median(periods),
                period_p95_ms=periods[min(len(periods) - 1, int(0.95 * len(periods)))],
                engine_ms=statistics.median([t.engine_ms for t in cached]),
                pre_ms=statistics.median([t.pre_ms for t in cached]),
                post_ms=statistics.median([t.post_ms for t in cached]),
                select_ms=statistics.median([t.select_ms for t in cached]),
                outside_ms=statistics.median([t.outside_ms for t in cached]),
            ),
            model_ms=head.model_ms,
        )

    def close(self) -> None:
        """Close any window still open, so the last chunk is not silently lost.

        Called at the end of a run. A no-op under RTC, and a no-op when the run
        stopped exactly on a chunk boundary.
        """
        if self._window:
            self._close_window(time.perf_counter())

    def _record_chunk(
        self,
        real_delay: int,
        consumed: int,
        queue_after: int,
        *,
        sync: SyncWindow | None = None,
        model_ms: float | None = None,
    ) -> None:
        record = ChunkRecord(
            index=len(self.chunks),
            model_ms=self._model_ms if model_ms is None else model_ms,
            pre_ms=self._pre_ms,
            post_ms=self._post_ms,
            total_ticks=real_delay,
            consumed=consumed,
            queue_after=queue_after,
            raw=self._raw,
            robot=self._robot,
            sync=sync,
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

        The images come from :func:`model_input_images`, which reconstructs them
        from ``pixel_values`` — the tensor the VLM is handed. That is a stronger
        claim than the observation dict can make: ``observation.images.*``
        survives the pipeline **unchanged**, at the camera's own size, so logging
        those would show a picture that looks like the model input and is not
        one. Teleoperation already showed the robot-side view is right way up;
        that is ``--display`` and it was never in question. What is in question
        is the rename, the key order, the channel layout and the 378x378 resize,
        and only the packed tensor has been through all four.

        Called from the RTC background thread; Rerun's recording stream is
        global and thread-safe, so this needs no handshake with the control loop.
        """
        try:
            import numpy as np
            import rerun as rr
        except ImportError:  # pragma: no cover - display is opt-in
            return

        for key, array in self._images.items():
            rr.log(f"policy_input/{key}", rr.Image(array))

        for name, values in (("policy_output", record.raw), ("to_robot", record.robot)):
            for key, value in zip(ACTION_KEYS, values, strict=False):
                rr.log(f"{name}/{key}", rr.Scalars(float(value)))

        if record.sync is not None:
            rr.log("sync/loop_hz", rr.Scalars(record.sync.cached_hz))
            rr.log("sync/tick_ms", rr.Scalars(record.sync.period_ms))
            rr.log("sync/pause_ms", rr.Scalars(record.sync.infer_ms))
        else:
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
            sync=self._sync_summary(),
            fps=self.fps or 30.0,
        )

    def _sync_summary(self) -> SyncWindow | None:
        """Medians of the per-chunk medians, so one long hitch cannot set the story."""
        windows = [c.sync for c in self.chunks if c.sync is not None]
        if not windows:
            return None

        def med(pick: Any) -> float:
            return statistics.median([pick(w) for w in windows])

        return SyncWindow(
            ticks=int(med(lambda w: float(w.ticks))),
            wall_ms=med(lambda w: w.wall_ms),
            engine_total_ms=med(lambda w: w.engine_total_ms),
            infer_ms=med(lambda w: w.infer_ms),
            period_ms=med(lambda w: w.period_ms),
            period_p95_ms=med(lambda w: w.period_p95_ms),
            engine_ms=med(lambda w: w.engine_ms),
            pre_ms=med(lambda w: w.pre_ms),
            post_ms=med(lambda w: w.post_ms),
            select_ms=med(lambda w: w.select_ms),
            outside_ms=med(lambda w: w.outside_ms),
        )


#: What ``molmoact2_pack_inputs`` normalises with — ``image_mean`` / ``image_std``
#: from the checkpoint's ``processor_config.json``. Undoing it is what turns the
#: packed tensor back into a picture.
_IMAGE_MEAN, _IMAGE_STD = 0.5, 0.5


def model_input_images(batch: Any) -> dict[str, Any]:
    """The images the VLM is actually handed, as ``HxWx3`` uint8, keyed by camera.

    ``molmoact2_pack_inputs`` leaves ``observation.images.*`` in the batch
    untouched — same size, same dtype as the robot produced — and puts what the
    model consumes in ``pixel_values``: one row per image, patchified into
    ``patch_size x patch_size x 3`` blocks and normalised. So the two disagree
    about resolution, aspect ratio and value range, and only the second one is
    evidence about what the policy sees.

    This undoes the packing: un-patchify to ``378x378`` (27x27 patches of 14),
    un-normalise, and scale back to bytes. Camera names come from
    :data:`dk1lab.layout.IMAGE_KEYS`, which is the order the checkpoint's
    preprocessor pins, so row *i* really is that camera.

    Returns an empty dict rather than raising for anything unexpected: this
    feeds a display, and a display must never take a rollout down.
    """
    try:
        import numpy as np

        pixel_values = batch.get("pixel_values") if hasattr(batch, "get") else None
        if pixel_values is None:
            return {}
        values = (
            pixel_values.detach().float().cpu().numpy()
            if hasattr(pixel_values, "detach")
            else np.asarray(pixel_values, dtype=np.float32)
        )
        if values.ndim == 4 and values.shape[0] == 1:  # a batch dimension, if any
            values = values[0]
        if values.ndim != 3:
            return {}
        count, patches, depth = values.shape
        side = int(round(math.sqrt(patches)))
        patch = int(round(math.sqrt(depth / 3)))
        if side * side != patches or patch * patch * 3 != depth:
            return {}
        names = [key.rsplit(".", 1)[-1] for key in IMAGE_KEYS]
        images: dict[str, Any] = {}
        for index in range(min(count, len(names))):
            image = (
                values[index]
                .reshape(side, side, patch, patch, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(side * patch, side * patch, 3)
            )
            image = image * _IMAGE_STD + _IMAGE_MEAN
            images[names[index]] = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return images
    except Exception:  # noqa: BLE001 - a trace must never break a rollout
        return {}
