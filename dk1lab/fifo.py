"""Serve a whole action chunk from a queue, and compute the next one while it drains.

Two engines live here. Both run the model **once per chunk** rather than once
per control tick; they differ in whether the control loop waits for it.

``ChunkFIFOInferenceEngine``
    Blocking. Runs the model inline when the queue empties.
``AsyncChunkFIFOInferenceEngine``
    Runs the model on a background thread while the queue is still draining,
    and splices the result in when it lands. The loop never blocks.

The first problem, and why the queue exists at all
==================================================

Measured on this cell with the real checkpoint and the real cameras, paced at
30 Hz, driving ``SyncInferenceEngine``:

============================================  =========
``MolmoAct2PackInputsProcessorStep``           15.9 ms
the rest of the preprocessor                    0.8 ms
``select_action`` and the engine's own work     5.5 ms
the postprocessor                               0.2 ms
**per control tick, of a 33.3 ms budget**    **~22 ms**
============================================  =========

and on **29 of every 30 ticks every millisecond of it is discarded**.
:meth:`MolmoAct2Policy.select_action` looks at its own 30-deep queue first,
finds it non-empty, and returns ``popleft()`` without reading the batch at all.
So three camera views are resized to 378x378, patchified and normalised thirty
times per chunk, and twenty-nine of those are thrown away.

:class:`~lerobot.rollout.inference.sync.SyncInferenceEngine` is built that way
because it drives every policy through ``select_action``, and LeRobot's own
comment block above the class says what the fix is and why upstream has not
taken it: SAC raises from ``predict_action_chunk``, ACT's temporal ensembler
lives inside ``select_action``, and the Diffusion family fills its
observation-history queues as a side effect. **MolmoAct2 has none of the three**
— its ``select_action`` *is* a ``predict_action_chunk`` call plus a ``deque`` —
so the fix is available to us and not to upstream.

That is what ``ChunkFIFOInferenceEngine`` does, and it worked: the engine costs
**0.02 ms** on a cached tick instead of 23.2, and the thirty rows it serves are
**bit-identical** to the sync engine's (max absolute difference ``0.000e+00``
over all 14 channels, real weights, fixed seed).

The second problem, which the queue alone does not touch
========================================================

Measured on the arms, one trace line per chunk::

    chunk 144   1297 ms over 30 ticks = 23.1 Hz  (pause 313 ms, model 220)

Twenty-nine ticks of ~33 ms, and one tick of **313 ms** in which the model runs
and the arms are sent nothing. LeRobot's control loop reports the same event as
``Record loop is running slower (3.4 Hz)``, once per chunk, because it measures
one iteration and that iteration contained a model call. It is not the cameras,
the crop or the capture resolution — the whole line reproduces on this machine
with **no robot attached** from nothing but "sleep 220 ms once per 30 ticks".

Three things follow from that one pause, and all three are what the arms look
like:

- a **freeze of about a third of a second, once a second**;
- the chunk **plays back 23% slow**, because 30 rows meant for 1000 ms are
  spread over 1297;
- the policy **re-plans only every 1.3 s**, and each new chunk is anchored on
  the *measured* pose, which lags the commanded one (speed limit plus impedance
  compliance). The longer the gap, the more lag has built up, and the bigger the
  correction when the plan lands. Under ``--sync`` nothing blends that seam.

The fix is to overlap: compute the next chunk while the current one is still
being served. That is the async engine.

What the async engine does, in order
====================================

1. Every tick, publish the observation the loop just took, and serve one row.
2. When the queue falls to ``replan_at`` rows, wake the worker thread.
3. The worker runs preprocessor -> ``predict_action_chunk`` -> postprocessor on
   the newest published observation, and hands back thirty rows.
4. On the next tick the loop **splices**: the chunk describes the world as it
   was when that observation was taken, so the first ``ceil(latency / period)``
   rows describe time that has already gone — they are **dropped**. The rest
   replace the queue, cross-faded over ``blend`` rows so the seam is a ramp
   rather than a step.

**Measured on the arms, 335 chunks:** in-situ latency 212 ms = 7 rows dropped,
the worker woken with 15 rows in hand and 8 (270 ms) still there when the chunk
landed, **zero starved ticks**, and the loop at 29.9 Hz against a 30 Hz target
with the only over-budget tick being the cold start. Raise ``replan_at`` for
fresher plans and more margin at the cost of running the GPU harder — 30 would
mean back to back, which is what RTC does.

**This is not action-identical, and it is not meant to be.** The blocking engine
was a pure speed change; this one re-plans four to five times more often, drops
rows that describe the past and blends the seam. That is the point: it is the
same trade RTC makes, made on the postprocessed actions instead of inside the
flow model, and without RTC's requirement that inference fit inside
``chunk / fps``.

What both engines still depend on
=================================

Actions must be **absolute**. A relative-action policy re-anchors to the current
state on every call, so serving a precomputed chunk would drift — upstream's
fourth caveat, and also an argument for doing this rather than against it. It is
checked at construction rather than assumed, and this checkpoint is absolute
joint pose.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)

#: Queue depth, in rows, at which the async engine starts the next chunk.
#:
#: Half a chunk. At the 310 ms in-situ latency this leaves ~190 ms of margin
#: before the queue could run dry, and lands a fresh plan every ~510 ms against
#: the 1300 ms the blocking engine managed. Higher is fresher *and* safer — 30
#: means the model runs back to back — at the cost of GPU duty cycle and of
#: executing fewer rows of each chunk before the next replaces it.
DEFAULT_REPLAN_AT: int = 15

#: Rows over which a newly spliced chunk is cross-faded into the one it replaces.
#:
#: The seam between two chunks is a step: consecutive chunks are independent
#: samples from a flow-matching model, and the new one is anchored on a measured
#: pose that lags the commanded one. Ramping over four ticks (133 ms) turns that
#: step into a slope. It must stay well under the replan interval — blending
#: most of what gets executed drags the new plan onto the old one, which is the
#: ``--execution-horizon 30`` failure mode arriving through a different door.
DEFAULT_BLEND_STEPS: int = 4

#: Consecutive worker failures tolerated before the error is raised on the
#: control thread. One transient CUDA hiccup should not end a rollout; a policy
#: that has stopped producing actions must not be allowed to look like one that
#: is holding still.
MAX_CONSECUTIVE_ERRORS: int = 5

#: How long the worker waits for a wake-up before looking again. Only bounds how
#: quickly ``stop()`` is noticed; the trigger itself is an event, not a poll.
_WORKER_POLL_S: float = 0.05

#: Attributes copied off the engine being replaced. They are private on
#: ``SyncInferenceEngine`` and there is no public accessor; naming them in one
#: place makes the coupling to an upstream implementation detail explicit rather
#: than scattering ``_engine._task`` through the code.
_CARRIED = (
    "_policy",
    "_preprocessor",
    "_postprocessor",
    "_dataset_features",
    "_ordered_action_keys",
    "_task",
    "_device",
    "_robot_type",
)


# --------------------------------------------------------------------------- #
# What one model call cost, and what it produced
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkReport:
    """One chunk: what it cost, what happened to it, and what the policy said.

    Emitted by the engine itself rather than inferred by wrapping, because with
    a worker thread there is no longer a tick to attribute the cost to. The
    blocking engine emits the same record with ``dropped``, ``blended`` and
    ``starved`` at zero, so one reader serves both.

    ``raw`` is the policy's own output — the flow-matching head's 14 numbers,
    still normalised, before ``clamp_action``, the unnormaliser and the gripper
    transform. ``robot`` is the same row after all of them, which is what the
    arms are told (up to the limiter). A question about the *policy* cannot be
    answered with a vector the postprocessor rewrote.
    """

    index: int
    #: Preprocessor, model and postprocessor, timed inside the call.
    pre_ms: float
    model_ms: float
    post_ms: float
    #: Observation timestamp to splice, i.e. how stale the plan is on arrival.
    #: Equals ``pre + model + post`` plus scheduling for the blocking engine.
    latency_ms: float
    #: Rows discarded from the front because they describe time already spent.
    dropped: int
    #: Rows of this chunk that were actually queued. Zero means the whole chunk
    #: described the past — inference took longer than a chunk is worth of
    #: motion — and the previous plan was kept instead.
    served: int
    #: Rows queued after the splice, and the depth it replaced.
    queue_before: int
    queue_after: int
    #: Rows cross-faded into the chunk they replaced.
    blended: int
    #: Ticks since the previous chunk on which the queue had nothing to serve.
    starved: int
    #: Control ticks between this chunk and the previous one.
    ticks: int
    raw: tuple[float, ...]
    robot: tuple[float, ...]

    @property
    def latency_ticks(self) -> int:
        """``dropped``, restated: how much of the chunk the wait consumed."""
        return self.dropped


def _relative_action_steps(preprocessor: Any) -> list[str]:
    """Any pipeline step that makes actions relative to the current state.

    Matched by class name because the classes live in different modules across
    LeRobot versions and importing them all to ``isinstance`` against would make
    this file fail to import rather than fail to find one.
    """
    steps = getattr(preprocessor, "steps", ()) or ()
    return [type(s).__name__ for s in steps if "Relative" in type(s).__name__]


def build(
    engine: Any,
    *,
    asynchronous: bool = True,
    replan_at: int = DEFAULT_REPLAN_AT,
    blend: int = DEFAULT_BLEND_STEPS,
    fps: float = 30.0,
    on_chunk: Any = None,
) -> Any:
    """A chunk-FIFO engine standing in for ``engine``.

    Takes the already-built sync engine rather than the pieces, so there is no
    second place that has to know how a policy, its pipelines and this cell's
    action key order are wired together. Everything is carried across by
    reference — the same policy object, the same two pipeline objects — so a
    gripper inversion already applied to them is still applied.

    Args:
        asynchronous: run the model on a worker thread while the queue drains.
            ``False`` gives the blocking engine, which is what to compare
            against; it is the one that pauses the arms once per chunk.
        replan_at, blend, fps: async only; see
            :class:`AsyncChunkFIFOInferenceEngine`.
        on_chunk: called with each :class:`ChunkReport`, on the control thread.
    """
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    if not isinstance(engine, SyncInferenceEngine):
        raise TypeError(
            f"the chunk FIFO replaces a SyncInferenceEngine, not a {type(engine).__name__}. "
            "Under --rtc the chunk is already served whole and there is nothing to fix."
        )
    carried = {name.lstrip("_"): getattr(engine, name) for name in _CARRIED}
    if not asynchronous:
        return ChunkFIFOInferenceEngine(**carried, on_chunk=on_chunk)
    return AsyncChunkFIFOInferenceEngine(
        **carried, on_chunk=on_chunk, replan_at=replan_at, blend=blend, fps=fps
    )


class ChunkFIFOInferenceEngine:
    """Synchronous inference that runs the policy once per chunk, not once per tick.

    A drop-in for ``SyncInferenceEngine``: same ``InferenceEngine`` lifecycle,
    same ``get_action`` contract, same tensor out. It is deliberately **not** a
    subclass — the only method it would inherit is the one it replaces, and
    inheriting the rest would suggest the two share a per-tick path that they do
    not.

    Args:
        policy: the loaded policy. Must produce absolute actions.
        preprocessor, postprocessor: the pipelines the sync engine held, by
            reference, so anything patched onto them still applies.
        dataset_features: what names the action slots.
        ordered_action_keys: this cell's key order, from ``dk1lab.layout``.
        task: the instruction handed to the policy.
        device: where the policy lives.
        robot_type: passed through to ``prepare_observation_for_inference``.
        on_chunk: called with a :class:`ChunkReport` as each chunk is served,
            on the control thread. Keep it cheap — printing is fine.
    """

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: Any,
        robot_type: str,
        on_chunk: Any = None,
    ) -> None:
        import torch

        relative = _relative_action_steps(preprocessor)
        if relative:
            raise ValueError(
                f"the chunk FIFO serves precomputed actions, so it is only correct for a policy "
                f"with absolute actions; this pipeline has {', '.join(relative)}. Use the plain "
                f"sync engine, which re-anchors every tick."
            )
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = list(ordered_action_keys)
        self._task = task
        self._device = device if isinstance(device, torch.device) else torch.device(device or "cpu")
        self._robot_type = robot_type
        self.on_chunk = on_chunk
        self._fifo: deque = deque()
        #: The same rows **before** the async engine's cross-fade, so a display
        #: can show what the policy planned next to what the arms were told and
        #: the difference between them is attributable. Identical to ``_fifo``
        #: under the blocking engine, which never blends.
        self._plan: deque = deque()
        self._planned: Any = None
        self._chunks = 0
        self._ticks = 0
        self._starved = 0
        self._permutation: list[int] | None = None
        #: Held for the length of a model call, so a reset cannot land in the
        #: middle of one and so the control thread and the worker can never both
        #: be inside the policy.
        self._compute_lock = Lock()

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """No background resources; put the policy in eval mode once, here.

        ``select_action`` calls ``self.eval()`` on every call, which walks 1737
        submodules of a 7B model for 1.8 ms to set a flag that is already set.
        Doing it once at the start of the run is the whole of what that achieved.
        """
        self._policy.eval()
        logger.info(
            "ChunkFIFOInferenceEngine started — one model call per %d-step chunk, "
            "not one pipeline pass per tick",
            self._chunk_steps(),
        )

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("ChunkFIFOInferenceEngine stopped")

    def reset(self) -> None:
        """Drop the queued chunk and reset the policy and both pipelines.

        The local queue has to go with the policy's own: a reset means the
        episode boundary moved, and actions computed for the old one describe a
        situation that no longer exists.
        """
        with self._compute_lock:
            self._fifo.clear()
            self._plan.clear()
            self._planned = None
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()
            self._policy.eval()
        logger.info("Resetting chunk FIFO inference state (policy + processors + queue)")

    # -- the InferenceEngine optional hooks --------------------------------- #

    def notify_observation(self, observation: dict) -> None:
        """No-op: this engine is driven by ``get_action``, like the sync one."""

    def pause(self) -> None:
        """No-op."""

    def resume(self) -> None:
        """No-op."""

    # -- actions ------------------------------------------------------------ #

    def _chunk_steps(self) -> int:
        """How many rows of a chunk to serve — the policy's own ``n_action_steps``."""
        return int(getattr(self._policy.config, "n_action_steps", 0) or 0)

    def get_action(self, obs_frame: dict | None) -> Any:
        """The next action: from the queue, or from one model call that fills it.

        Returns ``None`` only when there is nothing queued *and* no observation
        to compute from, which is the same condition the sync engine returns
        ``None`` under.
        """
        self._ticks += 1
        if self._fifo:
            return self._serve()
        if obs_frame is None:
            return None
        rows, timings = self._compute(obs_frame, time.perf_counter())
        self._replace(rows, rows)
        self._emit(timings, rows, dropped=0, blended=0, queue_before=0)
        return self._serve() if self._fifo else None

    def _replace(self, served: Any, planned: Any) -> None:
        """Install a chunk: what will be sent, and what the policy planned."""
        self._fifo.clear()
        self._fifo.extend(served)
        self._plan.clear()
        self._plan.extend(planned)

    def _serve(self) -> Any:
        """Pop one row, remembering the policy's own row for the same tick."""
        self._planned = self._plan.popleft() if self._plan else None
        return self._fifo.popleft()

    # -- the model call ----------------------------------------------------- #

    def _compute(self, obs_frame: dict, captured_at: float) -> tuple[list[Any], dict]:
        """Run the model once. Returns the chunk's rows and its raw timings.

        Called on the control thread by the blocking engine and on the worker
        thread by the async one, never on both at once.
        """
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        if getattr(self._policy, "_rtc_enabled", lambda: False)():
            raise RuntimeError(
                "the policy has RTC enabled, so predict_action_chunk expects RTC's delay and "
                "previous-chunk arguments. Run the RTC engine, or build the FIFO on a "
                "policy configured for sync inference."
            )

        # A shallow copy for the same reason the sync engine takes one: the
        # caller rebuilds obs_frame per tick, so nothing else reads these values.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with self._compute_lock, torch.inference_mode(), autocast_ctx:
            start = time.perf_counter()
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            mid = time.perf_counter()
            chunk = self._policy.predict_action_chunk(observation)
            steps = self._chunk_steps()
            if steps:
                chunk = chunk[:, :steps]
            # The policy's own rows, kept whole rather than just the head: the
            # row a splice ends up serving first is not row 0, and comparing the
            # model's output against what the arms were told is only meaningful
            # when the two describe the same instant. Reordered like the
            # postprocessed rows so channel *i* is the same joint in both.
            #
            # This ``.cpu()`` is also where the device synchronises, so
            # ``model_ms`` below is the real GPU cost. Doing it after the
            # postprocessor instead — which is what the previous version did —
            # charged the sync to ``post_ms`` and reported 46 ms of
            # "postprocessing" that was the GPU still working.
            raw_rows = self._reorder(chunk.squeeze(0).float().cpu())
            after_model = time.perf_counter()
            processed = self._postprocessor(chunk)
            # ``[1, steps, dim]`` -> ``[steps, dim]``, reordered to this cell's
            # action keys. ``.cpu()`` once for the whole chunk rather than once
            # per row: each call synchronises the device. The last postprocessor
            # step already moves it, so this is usually a no-op.
            rows = self._reorder(processed.squeeze(0).cpu())
            done = time.perf_counter()

        return list(rows), {
            "pre_ms": (mid - start) * 1000.0,
            "model_ms": (after_model - mid) * 1000.0,
            "post_ms": (done - after_model) * 1000.0,
            "latency_ms": (done - captured_at) * 1000.0,
            "raw_rows": raw_rows,
        }

    def _reorder(self, chunk: Any) -> Any:
        """``[steps, dim]`` in the policy's slot order -> this cell's key order.

        The per-row alternative is ``make_robot_action`` thirty times, each
        building a dict of 14 Python floats and a fresh tensor — ~14 ms per chunk
        on the arms, all of it inside the pause. The mapping is a fixed
        permutation of the action feature names, so it is worked out once and
        applied to the whole chunk as one indexing operation.
        """
        import torch

        if self._permutation is None:
            self._permutation = self._build_permutation()
        if self._permutation is None:  # shape we did not recognise — do it the slow way
            from lerobot.policies.utils import make_robot_action

            return [
                torch.tensor([make_robot_action(row, self._dataset_features)[k]
                              for k in self._ordered_action_keys])
                for row in chunk
            ]
        return chunk[:, self._permutation]

    def _build_permutation(self) -> list[int] | None:
        """Indices into the policy's action vector, in this cell's key order.

        ``None`` when the feature dict does not name every key we send, which
        means something upstream changed shape and the slow path should decide
        rather than a silently wrong index list.
        """
        from lerobot.utils.constants import ACTION

        names = (self._dataset_features.get(ACTION) or {}).get("names")
        if not names:
            return None
        try:
            return [list(names).index(key) for key in self._ordered_action_keys]
        except ValueError:
            logger.warning(
                "action features %s do not name every key this cell sends; falling back to "
                "make_robot_action per row", list(names),
            )
            return None

    # -- reporting ---------------------------------------------------------- #

    def _emit(
        self,
        timings: dict,
        served: Any,
        *,
        dropped: int,
        blended: int,
        queue_before: int,
    ) -> None:
        """Build a :class:`ChunkReport` and hand it to ``on_chunk``.

        ``served`` is the rows that were actually queued, so ``robot`` is the
        next action the arms will be sent and ``raw`` is the policy's own output
        for that same instant — row ``dropped``, not row 0.
        """
        raw_rows = timings.get("raw_rows")
        in_range = raw_rows is not None and dropped < len(raw_rows)
        report = ChunkReport(
            index=self._chunks,
            pre_ms=timings["pre_ms"],
            model_ms=timings["model_ms"],
            post_ms=timings["post_ms"],
            latency_ms=timings["latency_ms"],
            dropped=dropped,
            served=len(served),
            queue_before=queue_before,
            queue_after=len(self._fifo),
            blended=blended,
            starved=self._starved,
            ticks=self._ticks,
            raw=_first_row(raw_rows[dropped]) if in_range else (),
            robot=_first_row(served[0]) if len(served) else (),
        )
        self._chunks += 1
        self._ticks = 0
        self._starved = 0
        if self.on_chunk is not None:
            try:
                self.on_chunk(report)
            except Exception:  # noqa: BLE001 - never let reporting kill a rollout
                logger.exception("chunk report callback failed")

    @property
    def queued(self) -> int:
        """Rows left to serve before the next model call. For a banner or a test."""
        return len(self._fifo)

    @property
    def planned(self) -> Any:
        """The policy's own row for the tick just served, before any cross-fade.

        ``None`` before the first row. Read by :mod:`dk1lab.actionview`, which
        logs it next to what the arms were actually told: the gap between the two
        is ours — the blend and the speed limiter — and is the first thing to
        look at when the motion is rougher than the plan.
        """
        return self._planned


class AsyncChunkFIFOInferenceEngine(ChunkFIFOInferenceEngine):
    """The same chunk FIFO, filled by a worker thread instead of by the loop.

    The control loop never waits for the model. It publishes each observation,
    serves one row, and — when the queue falls to ``replan_at`` — wakes the
    worker. The chunk the worker returns is spliced in on a later tick, minus
    the rows that describe time already spent, cross-faded over ``blend`` rows.

    Only the **first** chunk is computed inline: at that point there is nothing
    to serve and nothing to overlap with, so the run pays one visible pause at
    the start and none after it.

    Args:
        replan_at: queue depth, in rows, at which to start the next chunk. Must
            exceed the inference latency in ticks or the queue can run dry.
        blend: rows over which to cross-fade a new chunk into the one it
            replaces. ``0`` splices hard.
    """

    def __init__(
        self,
        *,
        replan_at: int = DEFAULT_REPLAN_AT,
        blend: int = DEFAULT_BLEND_STEPS,
        fps: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if replan_at < 1:
            raise ValueError(f"replan_at must be at least 1 row, got {replan_at}")
        if blend < 0:
            raise ValueError(f"blend must not be negative, got {blend}")
        self._replan_at = int(replan_at)
        self._blend = int(blend)
        self._period = 1.0 / float(fps or 30.0)

        self._lock = Lock()
        self._latest: tuple[dict, float] | None = None
        self._ready: tuple[list, dict, int] | None = None
        self._error: BaseException | None = None
        self._inflight = False
        self._generation = 0
        self._errors = 0
        self._last_row: Any = None
        #: Nothing queued and nothing on the way, so the next call has to run the
        #: model inline. True at the start of a run and again after every reset —
        #: an episode boundary leaves the queue as empty as a cold start does.
        self._cold = True
        self._wake = Event()
        self._shutdown = Event()
        self._worker: Thread | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Put the policy in eval mode and launch the worker thread."""
        super().start()
        self._shutdown.clear()
        self._worker = Thread(target=self._work, daemon=True, name="ChunkFIFOInference")
        self._worker.start()
        logger.info(
            "async chunk FIFO: next chunk starts at %d rows queued, %d-row cross-fade",
            self._replan_at, self._blend,
        )

    def stop(self) -> None:
        """Signal the worker and wait for it. Idempotent."""
        self._shutdown.set()
        self._wake.set()
        worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning("chunk FIFO worker did not stop within 5 s")
        logger.info("AsyncChunkFIFOInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and drop everything in flight.

        ``_compute_lock`` is taken by :meth:`ChunkFIFOInferenceEngine.reset`, so
        this waits for any model call already running rather than resetting the
        policy underneath it. The generation counter discards the chunk that
        call is about to produce: it describes an episode that has ended.
        """
        with self._lock:
            self._generation += 1
            self._ready = None
            self._latest = None
            self._last_row = None
            self._inflight = False
            self._error = None
            self._errors = 0
            self._cold = True
        super().reset()

    # -- actions ------------------------------------------------------------ #

    def get_action(self, obs_frame: dict | None) -> Any:
        """Serve one row, splice anything the worker finished, ask for more.

        Never blocks on the model except on the very first call. When the queue
        is empty and a chunk is in flight it repeats the last row it served —
        which is what the arms are doing anyway, since the motor chain holds the
        last commanded target — and counts the tick as starved.
        """
        self._ticks += 1
        self._raise_worker_error()
        if obs_frame is not None:
            with self._lock:
                self._latest = (copy(obs_frame), time.perf_counter())

        self._splice()
        if self._cold and not self._fifo and obs_frame is not None:
            # Cold start: nothing to serve and nothing to overlap with. Also the
            # first tick after a reset, where the episode boundary has left the
            # queue exactly as empty as the start of a run does.
            self._cold = False
            rows, timings = self._compute(obs_frame, time.perf_counter())
            self._replace(rows, rows)
            self._emit(timings, rows, dropped=0, blended=0, queue_before=0)

        self._request()

        if self._fifo:
            self._last_row = self._serve()
            return self._last_row
        self._starved += 1
        return self._last_row

    def _request(self) -> None:
        """Wake the worker if the queue is low and nothing is already coming."""
        if len(self._fifo) > self._replan_at:
            return
        with self._lock:
            if self._inflight or self._ready is not None or self._latest is None:
                return
            self._inflight = True
        self._wake.set()

    def _splice(self) -> None:
        """Merge a finished chunk into the queue, dropping what it missed.

        The chunk was computed from the world as it stood ``latency_ms`` ago, so
        its leading rows describe time that has already been executed. They are
        dropped by wall clock rather than by rows consumed, because a chunk is a
        plan parameterised by time and that is the quantity that has passed;
        ``ceil`` rather than ``round`` so the first row served is never one the
        arms have already gone past.
        """
        with self._lock:
            pending, self._ready = self._ready, None
        if pending is None:
            return
        rows, timings, generation = pending
        if generation != self._generation:
            logger.debug("discarding a chunk computed before the last reset")
            return

        dropped = min(len(rows), math.ceil(timings["latency_ms"] / 1000.0 / self._period))
        fresh = rows[dropped:]
        queue_before = len(self._fifo)
        if not fresh:
            # The whole chunk describes time that has already gone: inference
            # took longer than the chunk is worth of motion. Keep what is queued
            # rather than emptying it — stale motion beats none — and let the
            # report say so, because this is the failure RTC hit at 900 ms.
            logger.warning(
                "a chunk arrived %.0f ms late, past its own %d rows; keeping the previous plan",
                timings["latency_ms"], len(rows),
            )
            self._emit(timings, [], dropped=dropped, blended=0, queue_before=queue_before)
            return
        served, blended = self._crossfade(fresh)
        # ``fresh`` is the policy's own plan and ``served`` is that plan faded
        # into the one it replaces. Both are kept: the gap between them is ours,
        # and :mod:`dk1lab.actionview` draws it.
        self._replace(served, fresh)
        self._emit(timings, served, dropped=dropped, blended=blended, queue_before=queue_before)

    def _crossfade(self, fresh: list) -> tuple[list, int]:
        """The rows to serve: ``fresh``, ramped onto the rows it replaces.

        ``fresh[i]`` and the queue's ``i``-th row describe the same instant, so
        they line up without further bookkeeping. Weight rises from
        ``1/(blend+1)`` to ``blend/(blend+1)`` — never 0 and never 1, so the
        first blended row still moves toward the new plan and the last is nearly
        it. Nothing to fade into when the queue is empty, which is the starved
        case, and there a hard splice is the only option anyway.

        The gripper channels are faded like every other, which is a choice and
        not an oversight: four ticks is 133 ms, and ``[limits.policy]`` already
        caps the gripper at 1.0 unit/s — a full open-to-closed takes a second, so
        the limiter dominates this ramp by an order of magnitude anyway.
        """
        span = min(self._blend, len(fresh), len(self._fifo))
        served = list(fresh)
        for i in range(span):
            weight = (i + 1) / (self._blend + 1)
            served[i] = self._fifo[i] * (1.0 - weight) + fresh[i] * weight
        return served, span

    # -- the worker --------------------------------------------------------- #

    def _work(self) -> None:
        """Compute a chunk whenever the control loop asks for one."""
        while not self._shutdown.is_set():
            if not self._wake.wait(_WORKER_POLL_S):
                continue
            self._wake.clear()
            if self._shutdown.is_set():
                break
            with self._lock:
                latest, generation = self._latest, self._generation
            if latest is None:
                with self._lock:
                    self._inflight = False
                continue
            obs_frame, captured_at = latest
            try:
                rows, timings = self._compute(obs_frame, captured_at)
            except Exception as exc:  # noqa: BLE001 - reported to the control thread
                with self._lock:
                    self._errors += 1
                    self._inflight = False
                    if self._errors >= MAX_CONSECUTIVE_ERRORS:
                        self._error = exc
                    count = self._errors
                logger.error(
                    "chunk inference failed (%d/%d): %s", count, MAX_CONSECUTIVE_ERRORS, exc
                )
                continue
            with self._lock:
                self._errors = 0
                self._ready = (rows, timings, generation)
                self._inflight = False

    def _raise_worker_error(self) -> None:
        """Re-raise a persistent worker failure on the control thread.

        A rollout whose policy has stopped producing actions must not look like
        one whose policy is holding still: without this the arms would sit at
        their last target for the rest of the duration and the trace would say
        ``starved``, with the actual exception only in the log.
        """
        with self._lock:
            error, self._error = self._error, None
        if error is not None:
            raise RuntimeError(
                f"chunk inference failed {MAX_CONSECUTIVE_ERRORS} times in a row; the arms "
                f"have been holding their last commanded target"
            ) from error


def _first_row(tensor: Any) -> tuple[float, ...]:
    """The first action of a chunk, whatever leading dimensions it has.

    ``predict_action_chunk`` returns ``[1, chunk, dim]`` and one row is
    ``[dim]``; both reduce to a tuple of 14 floats. Never raises: this feeds a
    report, and a report must not take a rollout down.
    """
    try:
        while getattr(tensor, "ndim", 1) > 1:
            tensor = tensor[0]
        return tuple(float(v) for v in tensor)
    except Exception:  # noqa: BLE001
        return ()
