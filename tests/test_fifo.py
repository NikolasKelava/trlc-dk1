"""The chunk FIFO: the same actions as the sync engine, at a thirtieth of the cost.

No model and no hardware — a fake policy shaped like ``MolmoAct2Policy``'s
``select_action`` / ``predict_action_chunk`` pair is enough, because the claim
under test is about *plumbing*: that serving a precomputed chunk yields the same
rows in the same order as recomputing one every tick.

The equivalence test is the important one. Everything else here is a guard
around the conditions under which that equivalence holds.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.utils.feature_utils import build_dataset_frame

from dk1lab.fifo import AsyncChunkFIFOInferenceEngine, ChunkFIFOInferenceEngine, build
from dk1lab.layout import ACTION_KEYS, CAMERA_NAMES, DOF, STATE_KEYS
from dk1lab.policy import ROBOT_TYPE, dataset_features

CHUNK = 30
WIDTH, HEIGHT = 64, 36


class FakePipeline:
    """A processor pipeline: callable, resettable, counting its calls."""

    def __init__(self, transform=None):
        self.transform = transform or (lambda value: value)
        self.calls = 0
        self.resets = 0
        self.steps: list = []

    def __call__(self, value):
        self.calls += 1
        return self.transform(value)

    def reset(self):
        self.resets += 1


class FakePolicy:
    """``MolmoAct2Policy``'s action path, and nothing else.

    Faithful in the three respects the FIFO depends on: ``predict_action_chunk``
    returns ``[1, chunk, dim]``, ``select_action`` serves it one row at a time
    from its own queue, and ``select_action`` calls ``eval()`` every time.
    """

    def __init__(self, chunk: int = CHUNK, *, rtc: bool = False):
        self.config = SimpleNamespace(n_action_steps=chunk, use_amp=False)
        self.chunk = chunk
        self.predictions = 0
        self.evals = 0
        self._rtc = rtc
        self._queue: list = []

    def _next_chunk(self) -> torch.Tensor:
        base = float(self.predictions)
        rows = torch.arange(self.chunk, dtype=torch.float32).unsqueeze(1) / 100.0
        cols = torch.arange(DOF, dtype=torch.float32).unsqueeze(0) / 1000.0
        return (base + rows + cols).unsqueeze(0)

    def predict_action_chunk(self, _batch, **_kwargs):
        chunk = self._next_chunk()
        self.predictions += 1
        return chunk

    def select_action(self, batch, **kwargs):
        self.eval()
        if not self._queue:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._queue = list(actions.transpose(0, 1))
        return self._queue.pop(0)

    def _rtc_enabled(self) -> bool:
        return self._rtc

    def eval(self):
        self.evals += 1
        return self

    def reset(self):
        self._queue = []


def frames(count: int = 4):
    """Observation frames shaped exactly as the control loop builds them."""
    features = dataset_features(width=WIDTH, height=HEIGHT)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(count):
        values: dict = dict.fromkeys(STATE_KEYS, 0.0)
        values.update(
            {
                name: rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
                for name in CAMERA_NAMES
            }
        )
        out.append(build_dataset_frame(features, values, prefix="observation"))
    return features, out


def engines(chunk: int = CHUNK, *, rtc: bool = False, asynchronous: bool = False, **kwargs):
    """A sync engine and a FIFO engine over equivalent, separate fake policies.

    Blocking by default: the equivalence claim below is that engine's contract.
    The async one deliberately does *not* serve the same rows — it re-plans four
    to five times more often — and is exercised in its own section.
    """
    features, obs = frames()
    made = []
    for _ in range(2):
        pipelines = (FakePipeline(), FakePipeline())
        made.append(
            SyncInferenceEngine(
                policy=FakePolicy(chunk, rtc=rtc),
                preprocessor=pipelines[0],
                postprocessor=pipelines[1],
                dataset_features=features,
                ordered_action_keys=list(ACTION_KEYS),
                task="pick up the dice",
                device="cpu",
                robot_type=ROBOT_TYPE,
            )
        )
    return made[0], build(made[1], asynchronous=asynchronous, **kwargs), obs


# --------------------------------------------------------------------------- #
# The claim: identical actions
# --------------------------------------------------------------------------- #


def test_the_fifo_serves_exactly_what_the_sync_engine_computes():
    """Row for row, over three whole chunks, in the same order."""
    sync, fifo, obs = engines()
    fifo.start()
    for tick in range(3 * CHUNK):
        frame = obs[tick % len(obs)]
        assert torch.allclose(sync.get_action(frame), fifo.get_action(frame)), f"tick {tick}"


def test_the_model_runs_once_per_chunk_instead_of_never_being_asked_twice():
    sync, fifo, obs = engines()
    for tick in range(2 * CHUNK):
        sync.get_action(obs[0])
        fifo.get_action(obs[0])
    assert sync._policy.predictions == 2
    assert fifo._policy.predictions == 2


def test_the_input_pipeline_runs_once_per_chunk_rather_than_once_per_tick():
    """This is the whole point: ~22 ms of a 33.3 ms tick, on 29 ticks in 30."""
    sync, fifo, obs = engines()
    for _ in range(2 * CHUNK):
        sync.get_action(obs[0])
        fifo.get_action(obs[0])
    assert sync._preprocessor.calls == 2 * CHUNK
    assert sync._postprocessor.calls == 2 * CHUNK
    assert fifo._preprocessor.calls == 2
    assert fifo._postprocessor.calls == 2


def test_eval_is_not_walked_over_the_model_on_every_tick():
    """``select_action`` calls ``self.eval()`` per call; 1737 submodules, 1.8 ms."""
    sync, fifo, obs = engines()
    fifo.start()
    for _ in range(CHUNK):
        sync.get_action(obs[0])
        fifo.get_action(obs[0])
    assert sync._policy.evals == CHUNK
    assert fifo._policy.evals == 1


# --------------------------------------------------------------------------- #
# The queue
# --------------------------------------------------------------------------- #


def test_a_chunk_is_queued_whole_and_drained_one_row_at_a_time():
    _sync, fifo, obs = engines()
    fifo.get_action(obs[0])
    assert fifo.queued == CHUNK - 1
    for _ in range(CHUNK - 1):
        fifo.get_action(obs[0])
    assert fifo.queued == 0


def test_a_shorter_action_horizon_queues_only_that_many():
    """``n_action_steps`` slices the chunk, exactly as ``select_action`` does."""
    _sync, fifo, obs = engines(chunk=8)
    fifo.get_action(obs[0])
    assert fifo.queued == 7


def test_resetting_drops_the_queued_chunk_and_the_policys_own():
    _sync, fifo, obs = engines()
    fifo.get_action(obs[0])
    assert fifo.queued > 0
    fifo.reset()
    assert fifo.queued == 0
    assert fifo._preprocessor.resets == 1
    assert fifo._postprocessor.resets == 1
    # The next call has to run the model again rather than serve stale actions.
    fifo.get_action(obs[0])
    assert fifo._policy.predictions == 2


def test_no_observation_and_nothing_queued_yields_no_action():
    _sync, fifo, _obs = engines()
    assert fifo.get_action(None) is None
    assert fifo._policy.predictions == 0


def test_a_queued_action_is_served_even_without_a_fresh_observation():
    """The rows are already computed; a missing frame is not a reason to stall."""
    _sync, fifo, obs = engines()
    first = fifo.get_action(obs[0])
    assert first is not None
    assert fifo.get_action(None) is not None


# --------------------------------------------------------------------------- #
# The conditions the equivalence rests on, checked rather than assumed
# --------------------------------------------------------------------------- #


def test_a_relative_action_policy_is_refused_rather_than_drifted():
    """Serving precomputed rows is only correct for absolute actions."""
    features, _obs = frames()
    preprocessor = FakePipeline()
    # Matched by class name, so a class with the right name is the whole fake.
    preprocessor.steps = [type("RelativeActionsProcessorStep", (), {})()]
    with pytest.raises(ValueError, match="absolute actions"):
        ChunkFIFOInferenceEngine(
            policy=FakePolicy(),
            preprocessor=preprocessor,
            postprocessor=FakePipeline(),
            dataset_features=features,
            ordered_action_keys=list(ACTION_KEYS),
            task="t",
            device="cpu",
            robot_type=ROBOT_TYPE,
        )


def test_an_rtc_enabled_policy_is_refused_rather_than_called_wrongly():
    """``predict_action_chunk`` needs RTC's extra arguments when RTC is on."""
    _sync, fifo, obs = engines(rtc=True)
    with pytest.raises(RuntimeError, match="RTC"):
        fifo.get_action(obs[0])


def test_building_over_something_that_is_not_a_sync_engine_is_refused():
    with pytest.raises(TypeError, match="SyncInferenceEngine"):
        build(SimpleNamespace())


def test_the_pipelines_are_carried_across_by_reference_not_rebuilt():
    """A gripper inversion patched onto them must survive the swap."""
    sync = engines()[0]
    fifo = build(sync)
    assert fifo._preprocessor is sync._preprocessor
    assert fifo._postprocessor is sync._postprocessor
    assert fifo._policy is sync._policy


def test_it_answers_the_whole_inference_engine_lifecycle():
    """``BaseStrategy`` calls all of these unconditionally."""
    _sync, fifo, _obs = engines()
    fifo.start()
    fifo.notify_observation({})
    fifo.pause()
    fifo.resume()
    fifo.stop()


# --------------------------------------------------------------------------- #
# The async engine: the loop never waits for the model
# --------------------------------------------------------------------------- #
#
# The claim here is *not* that the rows match the blocking engine's — they must
# not, since the whole point is re-planning four to five times more often. It is
# that the queue never runs dry, that a chunk which arrives late has its stale
# rows dropped rather than executed, and that the seam is a ramp.


class SlowPolicy(FakePolicy):
    """A policy whose ``predict_action_chunk`` takes real wall-clock time.

    The async engine's whole behaviour is timing, so a fake that returns
    instantly would exercise none of it. ``rows`` are constant per chunk so a
    splice is visible as a step in the served value.
    """

    def __init__(self, chunk: int = CHUNK, *, seconds: float = 0.05):
        super().__init__(chunk)
        self.seconds = seconds

    def _next_chunk(self) -> torch.Tensor:
        # Every row of chunk *n* is the constant ``n``: a splice then shows up as
        # a step from one integer to the next, and a cross-fade as the ramp
        # between them.
        return torch.full((1, self.chunk, DOF), float(self.predictions))

    def predict_action_chunk(self, batch, **kwargs):
        time.sleep(self.seconds)
        return super().predict_action_chunk(batch, **kwargs)


def async_engine(*, seconds: float = 0.05, fps: float = 100.0, **kwargs):
    """An async FIFO over a policy that takes ``seconds`` per chunk.

    ``fps`` of 100 keeps a test tick at 10 ms, so a 50 ms model call is five
    ticks and the whole exercise runs in well under a second.
    """
    features, obs = frames()
    engine = SyncInferenceEngine(
        policy=SlowPolicy(seconds=seconds),
        preprocessor=FakePipeline(),
        postprocessor=FakePipeline(),
        dataset_features=features,
        ordered_action_keys=list(ACTION_KEYS),
        task="pick up the dice",
        device="cpu",
        robot_type=ROBOT_TYPE,
    )
    return build(engine, asynchronous=True, fps=fps, **kwargs), obs


def drive(engine, obs, ticks: int, *, period: float = 0.01):
    """Run ``ticks`` control ticks at ``period``, as the rollout loop would."""
    served = []
    for tick in range(ticks):
        start = time.perf_counter()
        served.append(engine.get_action(obs[tick % len(obs)]))
        remaining = period - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)
    return served


def test_the_loop_never_waits_for_the_model_after_the_first_chunk():
    """The visible pause is what this engine exists to remove."""
    engine, obs = async_engine(seconds=0.05)
    engine.start()
    try:
        engine.get_action(obs[0])  # the cold start, which does block
        costs = []
        for tick in range(60):
            start = time.perf_counter()
            engine.get_action(obs[tick % len(obs)])
            costs.append(time.perf_counter() - start)
            time.sleep(0.01)
        # Nothing like a 50 ms model call ever lands on the control thread.
        assert max(costs) < 0.02, f"worst tick {max(costs) * 1000:.1f} ms"
    finally:
        engine.stop()


def test_the_queue_is_refilled_before_it_empties():
    engine, obs = async_engine(seconds=0.05, replan_at=15)
    engine.start()
    try:
        engine.get_action(obs[0])
        depths = []
        for tick in range(90):
            engine.get_action(obs[tick % len(obs)])
            depths.append(engine.queued)
            time.sleep(0.01)
        assert min(depths) > 0, f"the queue ran dry: {depths}"
        assert engine._policy.predictions > 2, "no replanning happened at all"
    finally:
        engine.stop()


def test_rows_that_describe_time_already_spent_are_dropped():
    """A plan is parameterised by time; the arms cannot execute its past."""
    reports = []
    engine, obs = async_engine(seconds=0.05, replan_at=20)
    engine.on_chunk = reports.append
    engine.start()
    try:
        drive(engine, obs, 80)
    finally:
        engine.stop()
    spliced = [r for r in reports if r.index > 0]
    assert spliced, "nothing was spliced"
    # 50 ms of latency at 100 Hz is five ticks, so about five rows go.
    assert all(3 <= r.dropped <= 9 for r in spliced), [r.dropped for r in spliced]
    assert all(r.served == CHUNK - r.dropped for r in spliced)


def test_the_seam_between_two_chunks_is_a_ramp_not_a_step():
    """Consecutive chunks are independent samples; the join has to be smoothed."""
    engine, obs = async_engine(seconds=0.05, replan_at=20, blend=4)
    engine.start()
    try:
        served = drive(engine, obs, 80)
    finally:
        engine.stop()
    values = [float(row[0]) for row in served if row is not None]
    # Every chunk is a different integer, so an unblended splice is a jump of a
    # whole 1.0. With a four-row ramp no single step may exceed 1/(blend+1).
    steps = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
    assert max(steps) <= 1.0 / 5 + 1e-6, f"worst step {max(steps):.3f}"
    assert max(values) >= 2.0, "no splice happened, so nothing was smoothed"


def test_without_a_blend_the_seam_is_a_hard_step():
    """The contrast that makes the previous test mean something."""
    engine, obs = async_engine(seconds=0.05, replan_at=20, blend=0)
    engine.start()
    try:
        served = drive(engine, obs, 80)
    finally:
        engine.stop()
    values = [float(row[0]) for row in served if row is not None]
    steps = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
    assert max(steps) >= 0.9


def test_a_chunk_that_arrives_past_its_own_last_row_keeps_the_old_plan():
    """Stale motion beats none. This is the 900 ms RTC failure, reported."""
    reports = []
    # 0.6 s per chunk against a 30-row chunk at 100 Hz = 0.3 s of motion.
    engine, obs = async_engine(seconds=0.6, replan_at=29, fps=100.0)
    engine.on_chunk = reports.append
    engine.start()
    try:
        drive(engine, obs, 120)
    finally:
        engine.stop()
    late = [r for r in reports if r.served == 0]
    assert late, [(r.dropped, r.served) for r in reports]
    assert all(r.dropped == CHUNK for r in late)


def test_the_worker_stops_and_the_engine_survives_being_stopped_twice():
    engine, obs = async_engine()
    engine.start()
    engine.get_action(obs[0])
    engine.stop()
    engine.stop()
    assert engine._worker is None


def test_a_reset_discards_a_chunk_computed_for_the_episode_that_ended():
    engine, obs = async_engine(seconds=0.05, replan_at=29)
    engine.start()
    try:
        engine.get_action(obs[0])
        drive(engine, obs, 3)          # a chunk is in flight by now
        engine.reset()
        assert engine.queued == 0
        time.sleep(0.15)               # long enough for that chunk to land
        engine.get_action(obs[0])
        # It was discarded rather than spliced, so this had to compute a new one.
        assert engine.queued == CHUNK - 1
    finally:
        engine.stop()


def test_a_persistent_worker_failure_is_raised_on_the_control_thread():
    """Silence would look exactly like a policy that has decided to hold still."""
    engine, obs = async_engine(seconds=0.0, replan_at=29)
    engine.start()
    try:
        engine.get_action(obs[0])

        def explode(*_args, **_kwargs):
            raise RuntimeError("the GPU fell over")

        engine._policy.predict_action_chunk = explode
        with pytest.raises(RuntimeError, match="failed .* times in a row"):
            drive(engine, obs, 200)
    finally:
        engine.stop()


def test_replan_at_and_blend_are_validated_rather_than_trusted():
    with pytest.raises(ValueError, match="replan_at"):
        async_engine(replan_at=0)
    with pytest.raises(ValueError, match="blend"):
        async_engine(blend=-1)
