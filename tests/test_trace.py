"""The rollout trace: it must record everything and change nothing.

No hardware, no lerobot, no torch. The trace only ever wraps callables, so a
handful of fakes shaped like the engine, the pipelines and the RTC action queue
exercise the whole of it — including the arithmetic that turns RTC's own numbers
into a diagnosis.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dk1lab import trace as trace_module
from dk1lab.layout import ACTION_KEYS, DOF, GRIPPER_INDICES
from dk1lab.trace import (
    ChunkRecord,
    RolloutTrace,
    SyncWindow,
    TraceSummary,
    _first_row,
    model_input_images,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakePipeline:
    """A processor pipeline: callable, resettable, and carrying steps."""

    def __init__(self, transform=None):
        self.transform = transform or (lambda value: value)
        self.steps = ["a step"]
        self.resets = 0

    def __call__(self, value):
        return self.transform(value)

    def reset(self):
        self.resets += 1


class FakeQueue:
    """Just enough of ``lerobot.policies.rtc.ActionQueue`` to be merged into."""

    def __init__(self):
        self.index = 0
        self.depth = 0
        self.merges = []

    def get_action_index(self):
        return self.index

    def qsize(self):
        return self.depth

    def merge(self, original, processed, real_delay, action_index_before_inference=None):
        self.merges.append((real_delay, action_index_before_inference))
        return "merged"


def chunk(value: float, steps: int = 30):
    """A ``[1, steps, DOF]`` action chunk whose first row is all ``value``."""
    return np.full((1, steps, DOF), value, dtype=np.float32)


def build(*, actions=(1.0,), fps: float = 30.0, display: bool = False):
    """A trace attached to a fake context, plus the fakes, ready to drive."""
    served = list(actions)
    pre = FakePipeline()
    post = FakePipeline(lambda value: value * -1.0 + 1.0)
    queue = FakeQueue()

    engine = SimpleNamespace(
        _preprocessor=pre,
        _postprocessor=post,
        _action_queue=queue,
        get_action=lambda frame: served.pop(0) if served else None,
    )
    policy = SimpleNamespace(predict_action_chunk=lambda *a, **k: chunk(0.5))
    ctx = SimpleNamespace(
        policy=SimpleNamespace(
            policy=policy, preprocessor=pre, postprocessor=post, inference=engine
        )
    )
    trace = RolloutTrace(fps=fps, display=display)
    trace.attach(ctx)
    trace.attach_queue(SimpleNamespace(_engine=engine))
    return trace, ctx, engine, queue, pre, post


# --------------------------------------------------------------------------- #
# The trace does not change what runs
# --------------------------------------------------------------------------- #


def test_the_wrapped_pipelines_still_transform_the_same_way():
    trace, ctx, engine, _q, _pre, _post = build()
    assert ctx.policy.postprocessor(np.array([0.25])) == pytest.approx([0.75])
    assert engine._postprocessor(np.array([0.25])) == pytest.approx([0.75])


def test_the_wrapped_pipeline_forwards_everything_it_does_not_time():
    """``reset()`` and ``.steps`` are reached by LeRobot on the object we replaced."""
    _trace, ctx, _e, _q, pre, _post = build()
    ctx.policy.preprocessor.reset()
    assert pre.resets == 1
    assert ctx.policy.preprocessor.steps == ["a step"]


def test_the_wrapped_queue_still_merges():
    _trace, _ctx, _e, queue, _pre, _post = build()
    assert queue.merge(chunk(0.0), chunk(1.0), 9, 0) == "merged"
    assert queue.merges == [(9, 0)]


def test_the_wrapped_engine_still_returns_its_action():
    _trace, _ctx, engine, _q, _pre, _post = build(actions=(7.0,))
    assert engine.get_action(None) == 7.0
    assert engine.get_action(None) is None


# --------------------------------------------------------------------------- #
# What it records
# --------------------------------------------------------------------------- #


def test_a_merge_becomes_a_chunk_record():
    trace, _ctx, engine, queue, _pre, _post = build()
    engine._preprocessor({"observation.state": 0})
    engine._postprocessor(chunk(0.9))
    queue.index, queue.depth = 10, 3
    queue.merge(chunk(0.0), chunk(1.0), 27, 0)

    assert len(trace.chunks) == 1
    record = trace.chunks[0]
    assert record.total_ticks == 27
    assert record.consumed == 10
    assert record.queue_after == 3


def test_the_policys_own_action_is_kept_apart_from_the_one_the_robot_gets():
    """The whole point: ``robot`` is what LeRobot made of ``raw``, not the policy's word."""
    trace, _ctx, engine, queue, _pre, _post = build()
    engine._postprocessor(chunk(0.9))  # post is x -> 1 - x, standing in for the pipeline
    queue.merge(chunk(0.0), chunk(1.0), 9, 0)

    record = trace.chunks[0]
    assert record.raw == pytest.approx((0.9,) * DOF)
    assert record.robot == pytest.approx((0.1,) * DOF)
    assert record.grippers(record.raw) == pytest.approx((0.9, 0.9))


def test_ticks_the_loop_asked_for_and_ticks_it_got():
    trace, _ctx, engine, _q, _pre, _post = build(actions=(1.0, 2.0))
    for _ in range(5):
        engine.get_action(None)
    assert (trace.ticks_asked, trace.ticks_served) == (5, 2)


def test_the_first_row_of_a_chunk_survives_any_leading_batch_dimensions():
    assert _first_row(chunk(0.4)) == pytest.approx((0.4,) * DOF)
    assert _first_row(np.full((1, DOF), 0.4, dtype=np.float32)) == pytest.approx((0.4,) * DOF)
    assert _first_row("not a tensor at all") == ()


def test_a_failing_callback_never_stops_the_run():
    """The trace reports; it must not be able to take the arms down with it."""
    trace, _ctx, _e, queue, _pre, _post = build()

    def explode(_record):
        raise RuntimeError("the console went away")

    trace.on_chunk = explode
    queue.merge(chunk(0.0), chunk(1.0), 9, 0)
    assert len(trace.chunks) == 1


def test_sync_inference_has_no_queue_and_attaching_is_a_no_op():
    trace = RolloutTrace()
    trace.attach_queue(SimpleNamespace(_engine=SimpleNamespace(_action_queue=None)))
    assert trace.chunks == []


# --------------------------------------------------------------------------- #
# The arithmetic that turns RTC's numbers into a reading
# --------------------------------------------------------------------------- #


def record(**overrides) -> ChunkRecord:
    fields = {
        "index": 0,
        "model_ms": 270.0,
        "pre_ms": 8.0,
        "post_ms": 2.0,
        "total_ticks": 9,
        "consumed": 9,
        "queue_after": 21,
        "raw": (0.0,) * DOF,
        "robot": (0.0,) * DOF,
    }
    fields.update(overrides)
    return ChunkRecord(**fields)


def test_starvation_is_the_ticks_the_loop_could_not_fill():
    """The observed run: 27 ticks of inference, 10 actions consumed."""
    assert record(total_ticks=27, consumed=10).starved == 17
    assert record(total_ticks=9, consumed=12).starved == 0


def test_time_the_three_timers_do_not_explain_is_named_as_such():
    """900 ms of chunk with 270 ms of model in it is 620 ms of thread not running."""
    slow = record(total_ticks=27, model_ms=270.0, pre_ms=8.0, post_ms=2.0)
    assert slow.total_ms == pytest.approx(900.0, abs=1.0)
    assert slow.unaccounted_ms == pytest.approx(620.0, abs=1.0)


def test_the_summary_reads_a_healthy_run_without_complaint():
    trace = RolloutTrace(rtc=True)
    trace.chunks = [record(index=i, raw=(0.2,) * DOF) for i in range(5)]
    trace.chunks += [record(index=9, raw=(0.9,) * DOF)]
    trace.ticks_asked, trace.ticks_served = 100, 100
    summary = trace.summary()
    assert summary.chunks == 6
    assert summary.total_ticks == 9
    assert summary.verdicts() == []


def test_the_summary_names_the_stall_the_starvation_and_the_lost_time():
    """Feed it the run that was actually observed and check it says all four things."""
    trace = RolloutTrace(rtc=True)
    trace.chunks = [
        record(index=i, total_ticks=27, consumed=10, queue_after=3, raw=(0.99,) * DOF)
        for i in range(8)
    ]
    trace.ticks_asked, trace.ticks_served = 100, 40
    verdicts = " ".join(trace.summary().verdicts())
    assert "QUEUE RAN DRY" in verdicts
    assert "TOO SLOW FOR RTC" in verdicts
    assert "not in the model" in verdicts
    assert "never moved a gripper" in verdicts


def test_a_gripper_that_does_move_is_not_reported_as_stuck():
    trace = RolloutTrace()
    trace.chunks = [
        record(index=0, raw=(0.05,) * DOF),
        record(index=1, raw=(0.95,) * DOF),
    ]
    trace.ticks_asked = trace.ticks_served = 60
    summary = trace.summary()
    assert summary.gripper_span == pytest.approx((0.05, 0.95))
    assert summary.verdicts() == []


def test_an_empty_trace_says_the_policy_never_ran():
    assert RolloutTrace().summary().verdicts() == [
        "no actions were produced at all — the policy never ran"
    ]


def test_under_sync_the_queue_readings_are_absent_rather_than_zero():
    """``dryrun`` runs sync inference: there is no queue to starve, so say nothing."""
    trace = RolloutTrace()
    trace.chunks = [record(index=0, total_ticks=0, consumed=0, queue_after=0)]
    summary = trace.summary()
    assert summary.rtc is False
    assert not any("QUEUE RAN DRY" in v or "TOO SLOW FOR RTC" in v for v in summary.verdicts())
    assert not any("queue" in line for line in summary.lines())
    assert any("model call" in line for line in summary.lines())


# --------------------------------------------------------------------------- #
# Sync inference: a record is a window between inference calls, not a tick
# --------------------------------------------------------------------------- #


class FakeSyncEngine:
    """``SyncInferenceEngine`` in miniature: pre -> select_action -> post, per tick.

    Faithful in the one respect the trace depends on: ``select_action`` runs the
    model only when its own queue is empty, so ``predict_action_chunk`` fires on
    one tick in ``chunk_steps``. Everything is looked up through the namespaces
    at call time, so the trace's wrappers are the things that actually run.
    """

    def __init__(self, ctx, *, chunk_steps=30, tick_ms=1.0, model_ms=8.0):
        self.ctx = ctx
        self.chunk_steps = chunk_steps
        self.tick_ms = tick_ms
        self.model_ms = model_ms
        self.clock = SimpleNamespace(now=0.0)
        self._queue = 0
        self._preprocessor = ctx.policy.preprocessor
        self._postprocessor = ctx.policy.postprocessor

    def get_action(self, frame):
        self._preprocessor({"observation.state": 0})
        if self._queue == 0:
            self.ctx.policy.policy.predict_action_chunk()
            self._queue = self.chunk_steps
        self._queue -= 1
        self._postprocessor(chunk(0.4, steps=1))
        return "an action"


def build_sync(*, chunk_steps=30, tick_ms=1.0, model_ms=8.0, fps=30.0):
    """A trace attached to a sync engine, plus a monotonic fake clock to drive it.

    Real ``perf_counter`` would make the assertions timing-dependent, so the
    clock is driven by hand: every tick advances it by ``tick_ms``, and the tick
    that infers advances it by ``model_ms`` more.
    """
    pre = FakePipeline()
    post = FakePipeline(lambda value: value * -1.0 + 1.0)
    clock = SimpleNamespace(now=0.0)

    def predict_action_chunk(*_a, **_k):
        # The whole cost of a chunk, paid on the tick that finds its queue empty.
        clock.now += model_ms / 1000.0
        return chunk(0.5)

    policy = SimpleNamespace(predict_action_chunk=predict_action_chunk)
    ctx = SimpleNamespace(
        policy=SimpleNamespace(
            policy=policy, preprocessor=pre, postprocessor=post, inference=None
        )
    )
    engine = FakeSyncEngine(ctx, chunk_steps=chunk_steps, tick_ms=tick_ms, model_ms=model_ms)
    engine.clock = clock
    ctx.policy.inference = engine
    trace = RolloutTrace(fps=fps)
    trace.attach(ctx)
    # No attach_queue: sync has no queue, so ``trace.rtc`` stays False.
    return trace, engine


def drive(trace, engine, ticks):
    """Run ``ticks`` control ticks against a monotonic fake clock.

    The clock is a stand-in for ``perf_counter`` so the assertions are about
    arithmetic rather than about how fast the test machine happens to be.
    """
    clock = engine.clock
    real = trace_module.time
    trace_module.time = SimpleNamespace(perf_counter=lambda: clock.now)
    try:
        for _ in range(ticks):
            engine.get_action(None)
            clock.now += engine.tick_ms / 1000.0
    finally:
        trace_module.time = real


def test_sync_cuts_one_record_per_inference_not_per_tick():
    """The bug this replaced: 4390 "chunks" in a 180 s run, one per tick."""
    trace, engine = build_sync(chunk_steps=30)
    drive(trace, engine, 90)
    # Three inferences happened; the third window is still open until close().
    assert len(trace.chunks) == 2
    trace.close()
    assert len(trace.chunks) == 3
    assert [c.sync.ticks for c in trace.chunks] == [30, 30, 30]


def test_the_sync_window_measures_wall_clock_rather_than_rtcs_missing_delay():
    """``total_ms`` used to be ``real_delay * tick``, and sync has no real_delay."""
    trace, engine = build_sync(chunk_steps=10, tick_ms=2.0, model_ms=50.0)
    drive(trace, engine, 20)
    record = trace.chunks[0]
    # 10 ticks of 2 ms, one of which also paid a 50 ms model call.
    assert record.total_ms == pytest.approx(70.0, abs=1e-6)
    assert record.model_ms == pytest.approx(50.0, abs=1e-6)
    assert record.sync.infer_ms == pytest.approx(50.0, abs=0.5)


def test_the_time_outside_the_engine_is_never_negative():
    """``other -160`` was the old sync line's headline symptom."""
    trace, engine = build_sync(chunk_steps=10, tick_ms=2.0, model_ms=50.0)
    drive(trace, engine, 20)
    for record in trace.chunks:
        assert record.unaccounted_ms >= 0.0
        assert "other" not in record.line()


def test_the_per_tick_costs_exclude_the_tick_that_ran_the_model():
    """29 ticks in 30 are cached ticks, and they are the ones that must fit."""
    trace, engine = build_sync(chunk_steps=10, tick_ms=2.0, model_ms=50.0)
    drive(trace, engine, 20)
    window = trace.chunks[0].sync
    assert window.period_ms == pytest.approx(2.0, abs=1e-6)
    assert window.cached_hz == pytest.approx(500.0, rel=1e-3)
    # The whole window, pause included, is much slower than its cached ticks.
    assert window.hz == pytest.approx(1000.0 * 10 / 70.0, rel=1e-3)


def test_sync_records_carry_no_queue_reading_and_cannot_be_starved():
    trace, engine = build_sync(chunk_steps=10)
    drive(trace, engine, 20)
    record = trace.chunks[0]
    assert record.starved == 0
    assert "queue" not in record.line()
    assert record.total_ticks == 0


def test_the_sync_summary_names_the_budget_and_what_is_spending_it():
    trace, engine = build_sync(chunk_steps=30, tick_ms=36.1, model_ms=270.0, fps=30.0)
    drive(trace, engine, 60)
    summary = trace.summary()
    text = "\n".join(summary.lines())
    assert "target 30 Hz" in text
    assert "33.3 ms budget" in text
    assert "rest of the loop" in text
    assert any("THE LOOP IS SLOW" in v for v in summary.verdicts())


def test_a_loop_that_holds_its_rate_is_not_complained_about():
    trace, engine = build_sync(chunk_steps=30, tick_ms=33.0, model_ms=270.0, fps=30.0)
    drive(trace, engine, 60)
    assert not any("THE LOOP IS SLOW" in v for v in trace.summary().verdicts())


def test_the_preprocessor_is_named_when_it_covers_the_overrun():
    """The suspect: it re-runs on all three views every tick, cached or not."""
    summary = TraceSummary(
        chunks=4,
        ticks_asked=120,
        ticks_served=120,
        model_ms=270.0,
        pre_ms=11.0,
        post_ms=0.3,
        total_ms=0.0,
        total_ticks=0,
        consumed=0,
        queue_after=0,
        gripper_span=(-1.0, 1.0),
        rtc=False,
        fps=30.0,
        sync=SyncWindow(
            ticks=30,
            wall_ms=1083.0,
            engine_total_ms=600.0,
            infer_ms=281.0,
            period_ms=36.1,
            period_p95_ms=38.0,
            engine_ms=11.4,
            pre_ms=11.0,
            post_ms=0.3,
            select_ms=0.1,
            outside_ms=24.7,
        ),
    )
    verdicts = "\n".join(summary.verdicts())
    assert "THE LOOP IS SLOW" in verdicts
    assert "PREPROCESSOR ALONE COVERS THE OVERRUN" in verdicts


def test_when_the_loop_not_the_engine_is_the_cost_it_says_so():
    summary = TraceSummary(
        chunks=4,
        ticks_asked=120,
        ticks_served=120,
        model_ms=270.0,
        pre_ms=2.0,
        post_ms=0.3,
        total_ms=0.0,
        total_ticks=0,
        consumed=0,
        queue_after=0,
        gripper_span=(-1.0, 1.0),
        rtc=False,
        fps=30.0,
        sync=SyncWindow(
            ticks=30,
            wall_ms=1500.0,
            engine_total_ms=400.0,
            infer_ms=281.0,
            period_ms=45.0,
            period_p95_ms=50.0,
            engine_ms=2.5,
            pre_ms=2.0,
            post_ms=0.3,
            select_ms=0.2,
            outside_ms=42.5,
        ),
    )
    verdicts = "\n".join(summary.verdicts())
    assert "COST IS OUTSIDE THE INFERENCE ENGINE" in verdicts
    assert "PREPROCESSOR ALONE" not in verdicts


def test_closing_a_trace_twice_adds_nothing():
    trace, engine = build_sync(chunk_steps=10)
    drive(trace, engine, 20)
    trace.close()
    before = len(trace.chunks)
    trace.close()
    assert len(trace.chunks) == before


# --------------------------------------------------------------------------- #
# The model-eye view
# --------------------------------------------------------------------------- #

#: What MolmoAct2 actually packs: 27x27 patches of 14x14x3, i.e. 378x378.
SIDE, PATCH = 27, 14


def packed(count: int = 3, value: float = 0.0) -> np.ndarray:
    """A ``pixel_values`` tensor shaped exactly as the pack step emits one."""
    return np.full((count, SIDE * SIDE, PATCH * PATCH * 3), value, dtype=np.float32)


def test_the_packed_tensor_unpacks_to_the_size_the_model_sees():
    images = model_input_images({"pixel_values": packed()})
    assert list(images) == ["top", "left", "right"]
    for image in images.values():
        assert image.shape == (SIDE * PATCH, SIDE * PATCH, 3) == (378, 378, 3)
        assert image.dtype == np.uint8


def test_the_normalisation_is_undone():
    """mean 0.5 / std 0.5, so 0.0 in the tensor is mid grey and +1.0 is white."""
    assert int(model_input_images({"pixel_values": packed(value=0.0)})["top"][0, 0, 0]) == 127
    assert int(model_input_images({"pixel_values": packed(value=1.0)})["top"][0, 0, 0]) == 255
    assert int(model_input_images({"pixel_values": packed(value=-1.0)})["top"][0, 0, 0]) == 0


def test_patches_are_laid_back_out_in_raster_order():
    """A single lit patch must come back at that patch's place in the picture."""
    values = packed(count=1)
    values[0, SIDE + 2] = 1.0  # row 1, column 2 of the patch grid
    image = model_input_images({"pixel_values": values})["top"]
    assert int(image[PATCH + 1, 2 * PATCH + 1, 0]) == 255
    assert int(image[1, 1, 0]) == 127  # patch (0,0) untouched


def test_the_camera_names_come_from_the_pinned_key_order():
    """Row i is camera i only because the checkpoint pins top/left/right."""
    values = packed()
    values[1] = 1.0
    images = model_input_images({"pixel_values": values})
    assert int(images["left"][0, 0, 0]) == 255
    assert int(images["top"][0, 0, 0]) == 127
    assert int(images["right"][0, 0, 0]) == 127


def test_a_batch_dimension_is_dropped():
    assert model_input_images({"pixel_values": packed()[None]})["top"].shape == (378, 378, 3)


def test_anything_unexpected_yields_nothing_rather_than_raising():
    """A display must never take a rollout down."""
    assert model_input_images({}) == {}
    assert model_input_images({"pixel_values": np.zeros((3, 5, 7), dtype=np.float32)}) == {}
    assert model_input_images(None) == {}
    assert model_input_images({"pixel_values": "not a tensor"}) == {}


def test_the_pass_through_observation_images_are_not_what_gets_logged():
    """They survive the pipeline unchanged, so they would show a picture that
    looks like the model's input and is not one — different size, different
    aspect ratio. Only pixel_values has been through the resize."""
    trace, _ctx, engine, _q, _pre, _post = build(display=True)
    engine._preprocessor(None)
    trace._record_pre(
        0.0,
        None,
        {
            "observation.images.top": np.zeros((360, 640, 3), dtype=np.uint8),
            "observation.state": np.zeros((DOF,), dtype=np.float32),
            "pixel_values": packed(),
        },
    )
    assert list(trace._images) == ["top", "left", "right"]
    assert trace._images["top"].shape == (378, 378, 3)


def test_the_gripper_channels_come_from_the_shared_layout():
    """Never a literal: the record slices the same indices everything else derives."""
    values = tuple(float(i) for i in range(DOF))
    assert record().grippers(values) == tuple(float(i) for i in GRIPPER_INDICES)
    assert len(ACTION_KEYS) == DOF
