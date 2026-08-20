"""The chunk FIFO: the same actions as the sync engine, at a thirtieth of the cost.

No model and no hardware — a fake policy shaped like ``MolmoAct2Policy``'s
``select_action`` / ``predict_action_chunk`` pair is enough, because the claim
under test is about *plumbing*: that serving a precomputed chunk yields the same
rows in the same order as recomputing one every tick.

The equivalence test is the important one. Everything else here is a guard
around the conditions under which that equivalence holds.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.utils.feature_utils import build_dataset_frame

from dk1lab.fifo import ChunkFIFOInferenceEngine, build
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


def engines(chunk: int = CHUNK, *, rtc: bool = False):
    """A sync engine and a FIFO engine over equivalent, separate fake policies."""
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
    return made[0], build(made[1]), obs


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
