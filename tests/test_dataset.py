"""The LeRobot v3.0 recorder: the tick contract, and keeping vs discarding.

The property this file exists for is the one the first version got wrong: an
episode is **not** written until somebody says to keep it, because
``save_episode`` cannot be undone and a ``discard`` that reports success while
changing nothing is worse than no prompt at all.

Everything here drives a fake robot and a fake dataset. Writing a real one costs
a video encode per episode and is exercised by hand — see the module docstring
of :mod:`dk1lab.dataset` for what that path does.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from dk1lab import dataset as ds
from dk1lab.layout import ACTION_KEYS, CAMERA_NAMES


class FakeDataset:
    """Enough of ``LeRobotDataset`` to see what the recorder asked of it."""

    def __init__(self):
        self.frames: list[dict] = []
        self.buffer: list[dict] = []
        self.episodes: list[list[dict]] = []
        self.finalized = False
        #: What the last ``save_episode`` was asked to do about forking.
        self.parallel_encoding: bool | None = None

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def add_frame(self, frame: dict) -> None:
        self.buffer.append(frame)
        self.frames.append(frame)

    def save_episode(self, parallel_encoding: bool = True) -> None:
        self.parallel_encoding = parallel_encoding
        self.episodes.append(list(self.buffer))
        self.buffer.clear()

    def clear_episode_buffer(self) -> None:
        self.buffer.clear()

    def finalize(self) -> None:
        self.finalized = True


class FakeRobot:
    """A robot that reports a moving pose and echoes what it was sent."""

    def __init__(self, height: int = 4, width: int = 6):
        self.tick = 0
        self.height, self.width = height, width
        self.sent: list[dict] = []

    def get_observation(self) -> dict:
        self.tick += 1
        observation = {key: float(self.tick) for key in ACTION_KEYS}
        observation.update(
            {
                name: np.zeros((self.height, self.width, 3), dtype=np.uint8)
                for name in CAMERA_NAMES
            }
        )
        return observation

    def send_action(self, action: dict) -> dict:
        self.sent.append(action)
        # As SafeBiDK1Follower does: what comes back is what the arms were given,
        # which here is deliberately not what was asked for.
        return {key: value * 0.5 for key, value in action.items()}


@pytest.fixture
def session(tmp_path):
    """A :class:`~dk1lab.dataset.DatasetSession` wired to a fake dataset."""
    live = ds.DatasetSession(tmp_path / "run", fps=30, width=6, height=4)
    live.dataset = FakeDataset()
    return live


@pytest.fixture
def robot():
    return FakeRobot()


@pytest.fixture
def ctx(robot):
    return types.SimpleNamespace(hardware=types.SimpleNamespace(robot_wrapper=robot))


def drive(robot, ticks: int) -> None:
    """Run the loop the way a rollout does: observe, then send."""
    for _ in range(ticks):
        observation = robot.get_observation()
        robot.send_action({key: float(observation[key]) for key in ACTION_KEYS})


def record(session, ctx, robot, task="put the dice in the bowl", ticks=5, notes=None):
    recorder = session.episode(task, notes=notes or {})
    recorder.attach(ctx)
    recorder.start()
    drive(robot, ticks)
    report = recorder.stop()
    recorder.detach()
    return report


# --------------------------------------------------------------------------- #
# The tick
# --------------------------------------------------------------------------- #


def test_one_frame_per_tick(session, ctx, robot):
    report = record(session, ctx, robot, ticks=7)
    assert report.ticks == 7
    assert len(session.dataset.frames) == 7


def test_a_frame_carries_the_state_the_action_the_images_and_the_task(session, ctx, robot):
    record(session, ctx, robot, ticks=1)
    frame = session.dataset.frames[0]
    assert frame["task"] == "put the dice in the bowl"
    assert frame["observation.state"].shape == (len(ACTION_KEYS),)
    assert frame["action"].shape == (len(ACTION_KEYS),)
    for name in CAMERA_NAMES:
        assert frame[f"observation.images.{name}"].shape == (4, 6, 3)


def test_the_action_recorded_is_the_one_the_arms_were_given(session, ctx, robot):
    """The rate-limited action, not the one the policy asked for."""
    record(session, ctx, robot, ticks=1)
    frame = session.dataset.frames[0]
    # The fake robot halves what it is sent; tick 1 asks for 1.0.
    assert frame["action"][0] == pytest.approx(0.5)
    assert frame["observation.state"][0] == pytest.approx(1.0)


def test_an_action_with_no_observation_is_dropped_rather_than_paired_with_a_stale_one(
    session, ctx, robot
):
    """A frame pairing this tick's action with the last tick's picture is a lie."""
    recorder = session.episode("t")
    recorder.attach(ctx)
    recorder.start()
    robot.send_action(dict.fromkeys(ACTION_KEYS, 1.0))  # no get_observation first
    assert recorder.stop().ticks == 0


def test_detaching_restores_the_robots_own_methods(session, ctx, robot):
    before = (robot.get_observation, robot.send_action)
    recorder = session.episode("t")
    recorder.attach(ctx)
    recorder.detach()
    assert (robot.get_observation, robot.send_action) == before


# --------------------------------------------------------------------------- #
# Keeping and discarding — the part that was wrong once
# --------------------------------------------------------------------------- #


def test_stopping_does_not_write_the_episode(session, ctx, robot):
    report = record(session, ctx, robot)
    assert report.pending is True
    assert session.dataset.num_episodes == 0


def test_keeping_writes_it(session, ctx, robot):
    report = record(session, ctx, robot)
    assert report.keep() is True
    assert session.dataset.num_episodes == 1
    assert report.pending is False


def test_discarding_actually_removes_it(session, ctx, robot):
    """The bug this file exists for: discard used to report success and do nothing."""
    report = record(session, ctx, robot)
    assert report.discard() is True
    assert session.dataset.num_episodes == 0
    assert session.dataset.buffer == []


def test_a_second_decision_changes_nothing(session, ctx, robot):
    report = record(session, ctx, robot)
    report.keep()
    assert report.keep() is False
    assert report.discard() is False
    assert session.dataset.num_episodes == 1


def test_an_episode_nobody_answered_for_is_kept_when_the_session_closes(session, ctx, robot):
    """An attempt that cannot be repeated must not be lost by default."""
    record(session, ctx, robot)
    fake = session.dataset
    session.close()
    assert fake.num_episodes == 1
    assert fake.finalized is True
    assert session.dataset is None


def test_an_empty_episode_leaves_no_buffer_for_the_next_one(session, ctx, robot):
    recorder = session.episode("nothing happened")
    recorder.attach(ctx)
    recorder.start()
    report = recorder.stop()
    recorder.detach()
    assert report.ticks == 0
    assert report.pending is False
    assert session.pending is None


def test_stopping_twice_reports_the_same_thing(session, ctx, robot):
    recorder = session.episode("t")
    recorder.attach(ctx)
    recorder.start()
    drive(robot, 3)
    first = recorder.stop()
    assert recorder.stop() is first


# --------------------------------------------------------------------------- #
# What the episode is written down as
# --------------------------------------------------------------------------- #


def test_the_notes_say_what_produced_the_episode(session, ctx, robot):
    report = record(session, ctx, robot, notes={"profile": "common", "checkpoint": "pi05"})
    report.keep()
    lines = (session.root / "dk1_notes.jsonl").read_text().strip().splitlines()
    written = json.loads(lines[-1])
    assert written["profile"] == "common"
    assert written["task"] == "put the dice in the bowl"
    assert written["frames"] == 5


def test_a_discarded_episode_leaves_no_note(session, ctx, robot):
    record(session, ctx, robot, notes={"profile": "common"}).discard()
    assert not (session.root / "dk1_notes.jsonl").exists()


def test_the_feature_dict_is_this_cells_layout():
    features = ds.dataset_features(width=640, height=360, use_videos=True)
    assert features["observation.state"]["shape"] == (len(ACTION_KEYS),)
    assert features["action"]["shape"] == (len(ACTION_KEYS),)
    assert features["observation.state"]["names"] == list(ACTION_KEYS)
    for name in CAMERA_NAMES:
        assert features[f"observation.images.{name}"]["dtype"] == "video"


def test_a_repo_id_is_derived_from_the_directory(tmp_path):
    assert ds.default_repo_id(tmp_path / "R0 MolmoAct2") == "dk1/r0-molmoact2"


# --------------------------------------------------------------------------- #
# A recorder must never take a rollout down
# --------------------------------------------------------------------------- #


def test_a_failing_dataset_switches_recording_off_instead_of_raising(session, ctx, robot):
    def explode(_frame):
        raise RuntimeError("disk full")

    session.dataset.add_frame = explode
    report = record(session, ctx, robot, ticks=4)
    assert report.ticks == 0
    assert "disk full" in (report.failed or "")
    assert report.pending is False


def test_a_context_with_no_robot_is_reported_not_raised(session):
    recorder = session.episode("t")
    recorder.attach(types.SimpleNamespace(hardware=None))
    assert recorder.failed is not None
    assert recorder.stop().ticks == 0


# --------------------------------------------------------------------------- #
# Recording two ways at once
# --------------------------------------------------------------------------- #


class Spy:
    """A recorder-shaped object that only records the calls it was given."""

    def __init__(self, name):
        self.name, self.calls = name, []

    def attach(self, ctx):
        self.calls.append("attach")

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")
        return types.SimpleNamespace(
            summary=lambda: f"{self.name} summary",
            discard=lambda: True,
        )

    def detach(self):
        self.calls.append("detach")


def test_one_recorder_is_passed_through_untouched():
    spy = Spy("a")
    assert ds.one(spy) is spy
    assert ds.one(None) is None
    assert ds.one(None, None) is None


def test_two_recorders_are_driven_as_one():
    first, second = Spy("rrd"), Spy("dataset")
    both = ds.one(first, second)
    both.attach(None)
    both.start()
    report = both.stop()
    both.detach()
    assert first.calls == second.calls == ["attach", "start", "stop", "detach"]
    assert "rrd summary" in report.summary()
    assert "dataset summary" in report.summary()


def test_one_decision_covers_every_recorder_on_the_episode():
    """The question is about the attempt, not about the file."""
    both = ds.one(Spy("rrd"), Spy("dataset"))
    both.attach(None)
    both.start()
    assert both.stop().discard() is True


# --------------------------------------------------------------------------- #
# Surviving a crash — a REAL LeRobot dataset, not the fake one
# --------------------------------------------------------------------------- #


@pytest.fixture
def real(tmp_path):
    """A real v3.0 dataset on disk, small and image-only so it is quick."""
    live = ds.DatasetSession(
        tmp_path / "row", fps=30, width=8, height=6, use_videos=False,
        camera_names=("top",),
    )
    live.open()
    yield live
    live.close()


def test_an_episode_is_readable_before_the_session_ends(real, ctx, robot):
    """The crash of 2026-08-25 left seven recorded episodes that nothing can open.

    LeRobot keeps one parquet writer open for the whole session and writes the
    footer on finalize, so a process that dies takes every episode with it. What
    is asserted here is the opposite: one committed episode, and the files on
    disk are complete — no finalize, no close.
    """
    import pyarrow.parquet as pq

    robot.height, robot.width = 6, 8
    report = record(real, ctx, robot, ticks=4)
    report.keep()

    data = sorted((real.root / "data").rglob("*.parquet"))
    assert data, "no data file was written"
    assert pq.read_table(data[-1]).num_rows == 4

    episodes = sorted((real.root / "meta" / "episodes").rglob("*.parquet"))
    assert episodes, "no episode metadata was written"
    assert pq.read_table(episodes[-1]).num_rows == 1


def test_a_second_episode_does_not_overwrite_the_first(real, ctx, robot):
    """Closing a writer per episode is only safe if the next one rotates files."""
    import pyarrow.parquet as pq

    robot.height, robot.width = 6, 8
    record(real, ctx, robot, ticks=4).keep()
    record(real, ctx, robot, ticks=3).keep()

    rows = sum(pq.read_table(p).num_rows for p in (real.root / "data").rglob("*.parquet"))
    assert rows == 7
    episodes = sum(
        pq.read_table(p).num_rows for p in (real.root / "meta" / "episodes").rglob("*.parquet")
    )
    assert episodes == 2


def test_the_frames_of_an_episode_that_was_never_written_are_cleared_on_open(tmp_path):
    """A crash leaves gigabytes of PNGs belonging to an episode nothing knows about."""
    root = tmp_path / "row"
    live = ds.DatasetSession(root, fps=30, width=8, height=6, use_videos=False,
                             camera_names=("top",))
    live.open()
    live.close()

    stale = root / "images" / "observation.images.top" / "episode-000000"
    stale.mkdir(parents=True)
    (stale / "frame-000000.png").write_bytes(b"not really a png")

    ds.DatasetSession(root, fps=30, width=8, height=6, use_videos=False,
                      camera_names=("top",)).open()
    assert not stale.exists()


# -- the encode that could not fork ---------------------------------------- #
#
# LeRobot encodes the three cameras in a ProcessPoolExecutor, which forks. A
# CUDA context cannot survive a fork, and the policy has one, so NVENC in the
# child dies on ``avcodec_open2`` and the episode is lost — which is what
# happened to A0's first attempt on 2026-08-27. Nothing is wrong with the
# codec; only with encoding it behind a fork.


def test_a_gpu_encode_is_not_forked(session, ctx, robot):
    """NVENC cannot start in a child forked from a process holding CUDA."""
    session.vcodec = "h264_nvenc"
    session.resolved_vcodec = "h264_nvenc"
    record(session, ctx, robot).keep()
    assert session.dataset.parallel_encoding is False


def test_a_cpu_encode_still_forks(session, ctx, robot):
    """A CPU codec forks perfectly well and gains most of a 3x by doing so."""
    session.vcodec = "libsvtav1"
    session.resolved_vcodec = "libsvtav1"
    record(session, ctx, robot).keep()
    assert session.dataset.parallel_encoding is True


def test_auto_is_resolved_before_it_is_judged(tmp_path):
    """``auto`` is a request, not a codec.

    Reading ``self.vcodec`` here would see the word "auto", conclude it is not a
    hardware encoder, and fork anyway — losing every episode on this machine,
    where ``auto`` is NVENC.
    """
    live = ds.DatasetSession(tmp_path / "run", vcodec="auto")
    encoder = live._encoder()
    assert live.resolved_vcodec == encoder.vcodec
    assert live.resolved_vcodec != "auto"
    if encoder.vcodec.endswith(ds.NVENC_SUFFIX):
        assert live._parallel_encoding() is False


def test_an_unopened_session_resolves_the_codec_rather_than_guessing(tmp_path):
    """Deciding before ``_encoder`` has run must not default to forking."""
    live = ds.DatasetSession(tmp_path / "run", vcodec="h264_nvenc")
    assert live.resolved_vcodec is None
    assert live._parallel_encoding() is False
