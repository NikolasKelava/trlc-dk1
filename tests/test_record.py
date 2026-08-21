"""Recording an episode: the four streams, on the right ticks, with the drops counted.

No Rerun viewer, no cameras, no robot. What is under test is that every stream
`--display` draws also reaches the file, that a stream the engine cannot supply
is left out rather than invented, and that a recorder which falls behind or
fails says so instead of producing a file that looks complete.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dk1lab.layout import ACTION_KEYS, CAMERA_NAMES
from dk1lab.record import EpisodeRecorder, episode_path, next_index, slug


class FakeStream:
    """A ``rr.RecordingStream``, recording what was logged at which tick."""

    def __init__(self, explode: bool = False):
        self.explode = explode
        self.saved: list = []
        self.blueprints: list = []
        self.flushed = 0
        self.tick = None
        self.elapsed = None
        #: (tick, path, value) for every log call.
        self.logged: list[tuple] = []

    def save(self, path):
        self.saved.append(path)

    def send_blueprint(self, blueprint):
        self.blueprints.append(blueprint)

    def set_time(self, timeline, *, sequence=None, duration=None):
        if sequence is not None:
            self.tick = sequence
        if duration is not None:
            self.elapsed = duration

    def log(self, path, entity, static=False):
        if self.explode:
            raise RuntimeError("no sink")
        self.logged.append((self.tick, path, entity))

    def flush(self):
        self.flushed += 1

    # -- reading it back -------------------------------------------------- #

    def paths(self, prefix: str) -> list[str]:
        return [path for _, path, _ in self.logged if path.startswith(prefix)]

    def at(self, path: str) -> list[tuple]:
        return [(tick, value) for tick, p, value in self.logged if p == path]


class FakeRerun:
    """The handful of ``rerun`` names the recorder touches.

    ``blueprint`` is the **real** submodule: the layout is a contract with
    :mod:`dk1lab.actionview` and faking it would test nothing.
    """

    def __init__(self, stream: FakeStream):
        self.stream = stream
        self.blueprint = pytest.importorskip("rerun.blueprint")
        #: Module-level logging, which is what the *live* view uses. Kept apart
        #: from the stream's so a test can tell the two instruments apart.
        self.logged: list[tuple] = []

    def log(self, path, entity):
        self.logged.append((path, entity))

    def RecordingStream(self, _application_id):  # noqa: N802 - mirroring rerun
        return self.stream

    def Scalars(self, value):  # noqa: N802
        return value

    def EncodedImage(self, *, contents, media_type):  # noqa: N802
        return ("jpeg", len(contents), media_type)

    def TextDocument(self, text, media_type=None):  # noqa: N802
        return text


class FakeEngine:
    def __init__(self, planned=None):
        self.planned = planned

    def get_action(self, obs_frame):
        return "the served action"


class FakeRobot:
    def __init__(self, observation, limited=None):
        self.observation = observation
        self.limited = limited
        self.sent: list = []

    def get_observation(self):
        return self.observation

    def send_action(self, action):
        self.sent.append(action)
        return self.limited


def row(start: float = 0.0) -> list[float]:
    return [start + i for i in range(len(ACTION_KEYS))]


def image() -> np.ndarray:
    return np.zeros((4, 8, 3), dtype=np.uint8)


def observation(*, cameras: bool = True) -> dict:
    obs = dict(zip(ACTION_KEYS, row(10.0), strict=True))
    if cameras:
        obs.update({name: image() for name in CAMERA_NAMES})
    return obs


def context(*, planned=None, limited=None, cameras: bool = True):
    engine = FakeEngine(planned)
    robot = FakeRobot(observation(cameras=cameras), limited)
    ctx = SimpleNamespace(
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(robot_wrapper=robot),
    )
    return ctx, engine, robot


def recorder(tmp_path, **kwargs) -> EpisodeRecorder:
    return EpisodeRecorder(tmp_path / "0001_episode.rrd", task="pick up the dice", **kwargs)


def tick(ctx, *, command=None):
    """One control tick, in the order ``BaseStrategy.run`` does it."""
    robot = ctx.hardware.robot_wrapper
    obs = robot.get_observation()
    ctx.policy.inference.get_action(obs)
    robot.send_action(command if command is not None else dict(zip(ACTION_KEYS, row(), strict=True)))


@pytest.fixture
def rerun(monkeypatch):
    """A fake ``rerun`` module, and the stream the recorder will write to."""
    stream = FakeStream()
    monkeypatch.setitem(sys.modules, "rerun", FakeRerun(stream))
    return stream


# --------------------------------------------------------------------------- #
# Where an episode goes
# --------------------------------------------------------------------------- #


def test_the_filename_carries_the_index_and_the_task():
    assert episode_path("recordings", "Pick up the dice!", 7).name == "0007_pick-up-the-dice.rrd"


def test_two_episodes_of_the_same_task_do_not_collide():
    """A session runs the same instruction over and over; only the index differs."""
    assert episode_path("r", "same", 1) != episode_path("r", "same", 2)


def test_a_task_with_nothing_nameable_in_it_still_gets_a_filename():
    assert episode_path("recordings", "???", 3).name == "0003.rrd"
    assert slug("???") == ""


def test_the_index_counts_up_from_what_is_already_there(tmp_path):
    """Read off the directory, so it keeps rising across sessions."""
    assert next_index(tmp_path) == 1
    (tmp_path / "0001_pick-up-the-dice.rrd").touch()
    (tmp_path / "0007_put-it-in-the-box.rrd").touch()
    (tmp_path / "notes.txt").touch()
    assert next_index(tmp_path) == 8


def test_an_episode_that_was_kept_is_never_overwritten(tmp_path, rerun):
    """The number of a discarded episode comes back; a kept one's never does."""
    kept = tmp_path / "0007_pick-up-the-dice.rrd"
    kept.touch()
    discarded = tmp_path / "0008_pick-up-the-dice.rrd"
    discarded.touch()
    assert next_index(tmp_path) == 9

    ctx, _engine, _robot = context()
    rec = EpisodeRecorder(discarded, task="pick up the dice")
    rec.attach(ctx)
    rec.start()
    tick(ctx)
    assert rec.stop().discard() is True
    assert next_index(tmp_path) == 8, "the discarded number is free again"
    assert kept.exists()


# --------------------------------------------------------------------------- #
# The four streams
# --------------------------------------------------------------------------- #


def test_all_four_streams_reach_the_file(tmp_path, rerun):
    ctx, _engine, _robot = context(planned=row(100.0), limited=dict(zip(ACTION_KEYS, row(50.0), strict=True)))
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    tick(ctx)
    report = rec.stop()
    rec.detach()

    key = ACTION_KEYS[0]
    assert rerun.at(f"observation.{key}") == [(1, 10.0)], "the measured position"
    assert rerun.at(f"policy/{key}") == [(1, 100.0)], "the policy's own plan"
    assert rerun.at(f"command/{key}") == [(1, 50.0)], "what send_action returned"
    images = [path for path in rerun.paths("observation.") if path.endswith(CAMERA_NAMES)]
    assert images == [f"observation.{name}" for name in CAMERA_NAMES]
    assert report.ticks == 1
    assert report.frames == len(CAMERA_NAMES)
    assert report.dropped == 0
    assert report.failed is None


def test_every_stream_lands_on_the_tick_its_observation_came_from(tmp_path, rerun):
    """The four are only comparable if they share an index; that is the whole point."""
    ctx, engine, robot = context(planned=row(0.0))
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    for n in range(3):
        engine.planned = row(float(n))
        robot.observation = dict(zip(ACTION_KEYS, row(float(n)), strict=True))
        tick(ctx, command=dict(zip(ACTION_KEYS, row(float(n)), strict=True)))
    rec.stop()

    key = ACTION_KEYS[0]
    assert rerun.at(f"observation.{key}") == [(1, 0.0), (2, 1.0), (3, 2.0)]
    assert rerun.at(f"policy/{key}") == [(1, 0.0), (2, 1.0), (3, 2.0)]
    assert rerun.at(f"command/{key}") == [(1, 0.0), (2, 1.0), (3, 2.0)]


def test_an_engine_with_no_plan_records_the_other_three(tmp_path, rerun):
    """Under ``--rtc`` or ``--no-fifo`` there is no per-tick plan to record."""
    ctx, _engine, _robot = context(planned=None)
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    tick(ctx)
    rec.stop()
    assert rerun.paths("policy/") == []
    assert rerun.at(f"observation.{ACTION_KEYS[0]}")


def test_the_command_recorded_is_the_limited_one_not_the_request(tmp_path, rerun):
    """``SafeBiDK1Follower.send_action`` returns the rate-limited action."""
    requested = dict(zip(ACTION_KEYS, row(0.0), strict=True))
    limited = dict(zip(ACTION_KEYS, row(50.0), strict=True))
    ctx, _engine, robot = context(limited=limited)
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    tick(ctx, command=requested)
    rec.stop()
    assert robot.sent == [requested], "the follower still got what it was asked"
    assert rerun.at(f"command/{ACTION_KEYS[0]}") == [(1, 50.0)]


def test_the_run_is_described_in_the_file(tmp_path, rerun):
    """A recording that cannot say what settings produced it is evidence of nothing."""
    ctx, _engine, _robot = context()
    rec = EpisodeRecorder(
        tmp_path / "e.rrd", task="pick up the dice", notes={"checkpoint": "bf16", "fps": 30}
    )
    rec.attach(ctx)
    rec.start()
    rec.stop()
    document = next(value for _, path, value in rerun.logged if path == "run")
    assert "pick up the dice" in document
    assert "bf16" in document and "fps" in document


def test_the_layout_written_is_the_one_the_panel_uses(tmp_path, rerun):
    ctx, _engine, _robot = context()
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    rec.stop()
    assert len(rerun.blueprints) == 1, "a replay must lay out like the live view"


# --------------------------------------------------------------------------- #
# Never at the rollout's expense
# --------------------------------------------------------------------------- #


def test_the_wrapped_calls_pass_through_untouched(tmp_path, rerun):
    ctx, _engine, robot = context(limited="whatever send_action returns")
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    assert ctx.policy.inference.get_action(None) == "the served action"
    assert robot.get_observation() is robot.observation
    assert robot.send_action({}) == "whatever send_action returns"
    rec.stop()


def test_detaching_puts_the_original_methods_back(tmp_path, rerun):
    ctx, engine, robot = context()
    inner_get, inner_obs, inner_send = engine.get_action, robot.get_observation, robot.send_action
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.detach()
    assert engine.get_action == inner_get
    assert robot.get_observation == inner_obs
    assert robot.send_action == inner_send


def test_a_failing_sink_disables_recording_instead_of_ending_the_rollout(tmp_path, monkeypatch):
    stream = FakeStream(explode=True)
    monkeypatch.setitem(sys.modules, "rerun", FakeRerun(stream))
    ctx, _engine, _robot = context(planned=row())
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    for _ in range(3):
        tick(ctx)  # must not raise
    report = rec.stop()
    assert report.failed is not None
    assert "stopped early" in report.summary()


def test_frames_that_cannot_be_kept_up_with_are_dropped_and_counted(tmp_path, rerun):
    """A recording with a hole in it has to say so rather than look complete."""
    ctx, _engine, _robot = context()
    rec = recorder(tmp_path, queue_ticks=1)
    rec.attach(ctx)
    rec.start()
    # The encoder is a real thread but the queue is one tick deep; filling it
    # without letting the worker run is what a stalled encoder looks like.
    rec._images.put((0, 0.0, []))
    for _ in range(3):
        tick(ctx)
    report = rec.stop()
    assert report.dropped > 0
    assert "DROPPED" in report.summary()


def test_stopping_twice_reports_the_same_episode(tmp_path, rerun):
    """Two callers ask what was recorded: the rollout, and the operator."""
    ctx, _engine, _robot = context()
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    tick(ctx)
    first = rec.stop()
    assert rec.stop() is first


def test_a_recorder_can_be_removed_from_a_chain_of_instruments(tmp_path, rerun):
    """A session records one episode and not the next, with the trace left attached.

    Three instruments wrap the same two calls. The recorder goes on last and
    comes off first, so what is left behind afterwards is exactly what was there
    before — and the ones that stayed still fire.
    """
    from dk1lab.actionview import ActionView

    ctx, engine, robot = context(planned=row(100.0))
    view = ActionView()
    view.attach(ctx)
    rec = recorder(tmp_path)
    rec.attach(ctx)
    rec.start()
    tick(ctx)
    rec.stop()
    rec.detach()

    live = sys.modules["rerun"]
    recorded_before, displayed_before = len(rerun.logged), len(live.logged)
    tick(ctx)  # the second episode: not recorded, still displayed
    assert len(rerun.logged) == recorded_before, "the recorder is off"
    assert len(live.logged) > displayed_before, "the view that stayed attached still draws"
    assert view.ticks == 2
