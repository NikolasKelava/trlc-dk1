"""The model's-eye view during teleoperation: right pixels, off the control loop.

No hardware and no checkpoint — the preprocessor is a fake, because what is
being tested here is the wrapping and the threading, not MolmoAct2. The real
pipeline is exercised in ``tests/test_trace.py``'s unpacking tests.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from dk1lab.layout import ACTION_KEYS, CAMERA_NAMES
from dk1lab.modelview import DEFAULT_EVERY, ModelInputProbe

SIDE, PATCH = 27, 14


def observation() -> dict:
    """The shape ``BiDK1Follower.get_observation`` returns: floats plus frames."""
    obs = dict.fromkeys(ACTION_KEYS, 0.0)
    obs.update({name: np.zeros((360, 640, 3), np.uint8) for name in CAMERA_NAMES})
    return obs


class FakePreprocessor:
    """Returns a batch shaped like ``molmoact2_pack_inputs`` emits one."""

    def __init__(self, boom: Exception | None = None) -> None:
        self.calls = 0
        self.boom = boom
        self.seen: list = []

    def __call__(self, frame):
        self.calls += 1
        self.seen.append(frame)
        if self.boom is not None:
            raise self.boom
        return {"pixel_values": np.zeros((3, SIDE * SIDE, PATCH * PATCH * 3), np.float32)}


class ThreadNamingPreprocessor(FakePreprocessor):
    """Records which thread it ran on."""

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[str] = []

    def __call__(self, frame):
        self.threads.append(threading.current_thread().name)
        return super().__call__(frame)


class BlockingPreprocessor(FakePreprocessor):
    """Holds the worker until released, so a backlog has to build up."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def __call__(self, frame):
        self.release.wait(3.0)
        return super().__call__(frame)


def probe(**kw) -> ModelInputProbe:
    inner = kw.pop("inner", lambda obs: obs)
    pre = kw.pop("preprocessor", FakePreprocessor())
    return ModelInputProbe(inner, pre, features={}, **kw)


def wait_for(predicate, timeout: float = 3.0) -> bool:
    """Poll until true. The worker is a thread, so nothing here can assume timing."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# --------------------------------------------------------------------------- #
# It must not disturb the loop it wraps
# --------------------------------------------------------------------------- #


def test_the_observation_passes_through_untouched():
    """It is a probe, not a processor: teleop must behave identically with it on."""
    obs = observation()
    sentinel = object()
    assert probe(inner=lambda _o: sentinel)(obs) is sentinel


def test_the_inner_processor_is_called_once_per_tick():
    calls = []
    watcher = probe(inner=lambda o: calls.append(o) or o)
    for _ in range(5):
        watcher(observation())
    assert len(calls) == 5


def test_nothing_is_sampled_before_start():
    """Constructing must not start a thread — the CLI builds this before connecting."""
    pre = FakePreprocessor()
    watcher = probe(preprocessor=pre)
    for _ in range(3 * DEFAULT_EVERY):
        watcher(observation())
    assert pre.calls == 0


def test_the_work_happens_off_the_calling_thread():
    """A pass costs ~11 ms and the 60 Hz budget is 16.7; inline would drop ticks."""
    pre = ThreadNamingPreprocessor()
    watcher = ModelInputProbe(lambda o: o, pre, features={}, every=1)
    with watcher:
        watcher(observation())
        assert wait_for(lambda: pre.threads)
    assert pre.threads[0] != threading.current_thread().name
    assert pre.threads[0].startswith("dk1-modelview")


def test_stop_is_safe_without_start():
    probe().stop()


def test_start_is_idempotent():
    watcher = probe()
    watcher.start()
    watcher.start()
    watcher.stop()


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_one_tick_in_every_is_sampled():
    pre = FakePreprocessor()
    watcher = ModelInputProbe(lambda o: o, pre, features={}, every=4)
    with watcher:
        for _ in range(12):
            watcher(observation())
            time.sleep(0.02)  # let the worker keep up, so nothing is dropped
        assert wait_for(lambda: pre.calls == 3)
    assert watcher.ticks == 12


def test_a_busy_worker_drops_rather_than_queues():
    """A stale picture is worth less than a loop that keeps time."""
    pre = BlockingPreprocessor()
    watcher = ModelInputProbe(lambda o: o, pre, features={}, every=1)
    with watcher:
        for _ in range(6):
            watcher(observation())
        assert wait_for(lambda: watcher.dropped > 0)
        pre.release.set()
    # six offered, one taken by the worker, so the rest were dropped rather than
    # queued — the backlog never grows past the one slot.
    assert watcher.dropped == 4
    assert pre.calls <= 2


def test_the_frames_are_logged_under_the_camera_names(monkeypatch):
    logged: dict = {}
    fake = type("R", (), {"log": staticmethod(lambda path, image: logged.__setitem__(path, image)),
                          "Image": staticmethod(lambda a: a)})
    monkeypatch.setitem(__import__("sys").modules, "rerun", fake)
    watcher = ModelInputProbe(lambda o: o, FakePreprocessor(), features={}, every=1)
    with watcher:
        watcher(observation())
        assert wait_for(lambda: len(logged) == 3)
    assert set(logged) == {"policy_input/top", "policy_input/left", "policy_input/right"}
    assert logged["policy_input/top"].shape == (378, 378, 3)


def test_an_observation_without_cameras_is_skipped():
    """--no-cameras is legal; there is simply nothing to show."""
    pre = FakePreprocessor()
    watcher = ModelInputProbe(lambda o: o, pre, features={}, every=1)
    with watcher:
        watcher(dict.fromkeys(ACTION_KEYS, 0.0))
        time.sleep(0.1)
    assert pre.calls == 0
    assert watcher.failed is None


# --------------------------------------------------------------------------- #
# Failure must stay inside the display
# --------------------------------------------------------------------------- #


def test_a_failing_pass_disables_the_probe_instead_of_raising():
    pre = FakePreprocessor(boom=RuntimeError("no"))
    watcher = ModelInputProbe(lambda o: o, pre, features={}, every=1)
    with watcher:
        watcher(observation())
        assert wait_for(lambda: watcher.failed is not None)
        for _ in range(5):
            watcher(observation())  # must not raise
        time.sleep(0.1)
    assert watcher.failed == "no"
    assert pre.calls == 1  # switched off, not retried every tick


def test_teleop_still_gets_its_observation_after_a_failure():
    pre = FakePreprocessor(boom=RuntimeError("no"))
    watcher = ModelInputProbe(lambda o: o, pre, features={}, every=1)
    obs = observation()
    with watcher:
        watcher(obs)
        assert wait_for(lambda: watcher.failed is not None)
        assert watcher(obs) is obs


# --------------------------------------------------------------------------- #
# The Rerun layout
# --------------------------------------------------------------------------- #


def test_pin_blueprint_declines_when_upstreams_cache_is_missing(monkeypatch):
    """It fills a cache attribute `init_rerun` creates. If upstream drops it, the
    views still log and Rerun falls back to its own automatic layout — so this
    reports False rather than raising or silently doing nothing."""
    from lerobot.utils import rerun_visualization

    from dk1lab.modelview import pin_blueprint

    monkeypatch.delattr(rerun_visualization.log_rerun_data, "blueprint", raising=False)
    assert pin_blueprint() is False


def test_pin_blueprint_fills_the_cache_so_lerobot_does_not_overwrite_it():
    """LeRobot builds its blueprint from the first observation and omits
    policy_input/, which never passes through it — so those views would have no
    home. Filling the cache first is what stops that."""
    import rerun as rr
    from lerobot.utils import rerun_visualization

    from dk1lab.modelview import pin_blueprint

    rr.init("dk1-modelview-test")  # send_blueprint needs a recording; no viewer
    rerun_visualization.log_rerun_data.blueprint = None  # what init_rerun() does
    try:
        assert pin_blueprint() is True
        pinned = rerun_visualization.log_rerun_data.blueprint
        assert pinned is not None
        # upstream's own builder now returns early rather than replacing ours
        rerun_visualization._ensure_blueprint({"observation.x"}, {"action.y"}, {"observation.top"})
        assert rerun_visualization.log_rerun_data.blueprint is pinned
    finally:
        rerun_visualization.log_rerun_data.blueprint = None
