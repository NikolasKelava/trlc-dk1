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

from dk1lab.layout import ACTION_KEYS, DOF, GRIPPER_INDICES
from dk1lab.trace import ChunkRecord, RolloutTrace, _as_image, _first_row


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


def test_a_sync_postprocessor_call_is_itself_a_record():
    """With no queue to hang a record on, the postprocessor is where one is cut."""
    trace, _ctx, engine, _q, _pre, _post = build()
    trace.rtc = False
    engine._postprocessor(chunk(0.4))
    assert len(trace.chunks) == 1
    assert trace.chunks[0].raw == pytest.approx((0.4,) * DOF)


# --------------------------------------------------------------------------- #
# The model-eye view
# --------------------------------------------------------------------------- #


def test_a_chw_float_image_becomes_hwc_uint8():
    """What the preprocessor hands the model is CHW and normalised; Rerun wants neither."""
    array = _as_image(np.full((3, 4, 6), 0.5, dtype=np.float32), np)
    assert array.shape == (4, 6, 3)
    assert array.dtype == np.uint8
    assert int(array[0, 0, 0]) == 127


def test_an_already_hwc_uint8_image_is_left_alone():
    source = np.zeros((4, 6, 3), dtype=np.uint8)
    source[1, 2] = 200
    array = _as_image(source, np)
    assert array.shape == (4, 6, 3)
    assert int(array[1, 2, 0]) == 200


def test_a_batched_image_loses_its_batch_dimension():
    assert _as_image(np.zeros((1, 3, 4, 6), dtype=np.float32), np).shape == (4, 6, 3)


def test_something_that_is_not_an_image_is_skipped_rather_than_raised():
    assert _as_image(np.zeros((14,), dtype=np.float32), np) is None
    assert _as_image("not an array", np) is None


def test_only_image_keys_are_kept_for_display():
    trace, _ctx, engine, _q, _pre, _post = build(display=True)
    engine._preprocessor(None)
    trace._record_pre(
        0.0,
        None,
        {
            "observation.images.top": np.zeros((3, 4, 6), dtype=np.float32),
            "observation.state": np.zeros((DOF,), dtype=np.float32),
        },
    )
    assert list(trace._images) == ["observation.images.top"]


def test_the_gripper_channels_come_from_the_shared_layout():
    """Never a literal: the record slices the same indices everything else derives."""
    values = tuple(float(i) for i in range(DOF))
    assert record().grippers(values) == tuple(float(i) for i in GRIPPER_INDICES)
    assert len(ACTION_KEYS) == DOF
