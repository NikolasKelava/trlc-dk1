"""The action view: three lines per joint, and none of them invented.

No Rerun viewer and no robot. What is under test is that the right *number*
reaches the right *entity path* — the policy's own row under ``policy/``, the
action the follower actually sent under ``command/`` — because that is the whole
claim the panel makes, and a panel that mislabels which of the two it is
showing is worse than no panel at all.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from dk1lab import actionview
from dk1lab.actionview import COMMAND_PREFIX, POLICY_PREFIX, ActionView, pin_blueprint
from dk1lab.layout import ACTION_KEYS


class FakeRerun:
    """``rerun``'s two calls, recording what was logged where."""

    def __init__(self, explode: bool = False):
        self.logged: list[tuple[str, float]] = []
        self.blueprints: list = []
        self.explode = explode

    # -- the rr API surface this module touches -------------------------- #

    def log(self, path, entity):
        if self.explode:
            raise RuntimeError("no viewer")
        self.logged.append((path, entity))

    def Scalars(self, value):  # noqa: N802 - mirroring rerun's own name
        return value

    def send_blueprint(self, blueprint):
        self.blueprints.append(blueprint)

    def paths(self, prefix: str) -> list[str]:
        return [path for path, _ in self.logged if path.startswith(f"{prefix}/")]

    def value(self, path: str) -> float:
        return next(v for p, v in self.logged if p == path)


class FakeEngine:
    """An engine with the one attribute the view reads, plus a served action."""

    def __init__(self, planned=None):
        self.planned = planned
        self.calls = 0

    def get_action(self, obs_frame):
        self.calls += 1
        return "the served action"


class FakeRobot:
    """A follower whose ``send_action`` returns the **limited** action, as ours does."""

    def __init__(self, limited=None):
        self.limited = limited
        self.sent: list = []

    def send_action(self, action):
        self.sent.append(action)
        return self.limited


def context(planned=None, limited=None):
    engine = FakeEngine(planned)
    robot = FakeRobot(limited)
    ctx = SimpleNamespace(
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(robot_wrapper=robot),
    )
    return ctx, engine, robot


def row(start: float = 0.0) -> list[float]:
    """A 14-vector, one distinguishable value per channel."""
    return [start + i for i in range(len(ACTION_KEYS))]


# --------------------------------------------------------------------------- #
# What gets logged, and under which name
# --------------------------------------------------------------------------- #


def test_the_policys_own_row_is_logged_under_policy():
    fake = FakeRerun()
    ctx, engine, _robot = context(planned=row(100.0))
    view = ActionView()
    view.attach(ctx)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "rerun", fake)
        ctx.policy.inference.get_action(None)
    assert fake.paths(POLICY_PREFIX) == [f"{POLICY_PREFIX}/{key}" for key in ACTION_KEYS]
    assert fake.value(f"{POLICY_PREFIX}/{ACTION_KEYS[0]}") == 100.0


def test_the_command_logged_is_the_one_the_follower_sent_not_the_one_asked_for():
    """``SafeBiDK1Follower.send_action`` returns the rate-limited action.

    Logging the argument instead would draw the request and call it the command,
    which hides exactly the thing this panel exists to show.
    """
    fake = FakeRerun()
    requested = dict(zip(ACTION_KEYS, row(0.0), strict=True))
    limited = dict(zip(ACTION_KEYS, row(50.0), strict=True))
    ctx, _engine, robot = context(limited=limited)
    view = ActionView()
    view.attach(ctx)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "rerun", fake)
        ctx.hardware.robot_wrapper.send_action(requested)
    assert robot.sent == [requested]
    assert fake.value(f"{COMMAND_PREFIX}/{ACTION_KEYS[0]}") == 50.0


def test_a_follower_that_returns_nothing_falls_back_to_the_requested_action():
    fake = FakeRerun()
    requested = dict(zip(ACTION_KEYS, row(7.0), strict=True))
    ctx, _engine, _robot = context(limited=None)
    ActionView().attach(ctx)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "rerun", fake)
        ctx.hardware.robot_wrapper.send_action(requested)
    assert fake.value(f"{COMMAND_PREFIX}/{ACTION_KEYS[0]}") == 7.0


def test_an_engine_with_no_plan_logs_nothing_rather_than_inventing_a_line():
    """``--rtc`` and ``--no-fifo`` keep no per-tick plan; two lines beats a fake third."""
    fake = FakeRerun()
    ctx, _engine, _robot = context(planned=None)
    ActionView().attach(ctx)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "rerun", fake)
        ctx.policy.inference.get_action(None)
    assert fake.paths(POLICY_PREFIX) == []


# --------------------------------------------------------------------------- #
# It wraps, it does not replace
# --------------------------------------------------------------------------- #


def test_both_wrapped_calls_forward_untouched():
    ctx, engine, robot = context(planned=row(), limited={"x": 1.0})
    ActionView().attach(ctx)
    assert ctx.policy.inference.get_action(None) == "the served action"
    assert engine.calls == 1
    assert ctx.hardware.robot_wrapper.send_action({"a": 1}) == {"x": 1.0}
    assert robot.sent == [{"a": 1}]


def test_detach_puts_the_original_methods_back():
    ctx, engine, robot = context()
    view = ActionView()
    inner_get, inner_send = ctx.policy.inference.get_action, ctx.hardware.robot_wrapper.send_action
    view.attach(ctx)
    assert ctx.policy.inference.get_action is not inner_get
    view.detach()
    assert ctx.policy.inference.get_action == inner_get
    assert ctx.hardware.robot_wrapper.send_action == inner_send


def test_a_missing_robot_wrapper_still_leaves_the_policy_line():
    fake = FakeRerun()
    engine = FakeEngine(planned=row())
    ctx = SimpleNamespace(policy=SimpleNamespace(inference=engine), hardware=None)
    ActionView().attach(ctx)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "rerun", fake)
        ctx.policy.inference.get_action(None)
    assert fake.paths(POLICY_PREFIX)


# --------------------------------------------------------------------------- #
# A display must never take a rollout down
# --------------------------------------------------------------------------- #


def test_a_logging_failure_disables_the_view_instead_of_raising():
    exploding = FakeRerun(explode=True)
    ctx, _engine, _robot = context(planned=row())
    view = ActionView()
    view.attach(ctx)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "rerun", exploding)
        assert ctx.policy.inference.get_action(None) == "the served action"
        assert view.failed is not None
        # And it does not try again, thirty times a second, for the whole run.
        before = len(exploding.logged)
        ctx.policy.inference.get_action(None)
        assert len(exploding.logged) == before


def test_a_non_numeric_channel_is_skipped_rather_than_crashing():
    fake = FakeRerun()
    limited = dict(zip(ACTION_KEYS, row(), strict=True))
    limited[ACTION_KEYS[3]] = None
    ctx, _engine, _robot = context(limited=limited)
    ActionView().attach(ctx)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "rerun", fake)
        ctx.hardware.robot_wrapper.send_action({})
    assert f"{COMMAND_PREFIX}/{ACTION_KEYS[3]}" not in fake.paths(COMMAND_PREFIX)
    assert len(fake.paths(COMMAND_PREFIX)) == len(ACTION_KEYS) - 1


# --------------------------------------------------------------------------- #
# The layout, which is the whole argument
# --------------------------------------------------------------------------- #


def test_the_blueprint_puts_all_three_series_on_one_axis_per_joint():
    """Built with the real Rerun blueprint API, not a fake — the contents matter."""
    rrb = pytest.importorskip("rerun.blueprint")
    sent: list = []
    stub = SimpleNamespace(blueprint=None)

    with pytest.MonkeyPatch.context() as mp:
        import rerun as rr
        from lerobot.utils import rerun_visualization

        mp.setattr(rr, "send_blueprint", sent.append)
        mp.setattr(rerun_visualization, "log_rerun_data", stub)
        assert pin_blueprint() is True

    assert stub.blueprint is sent[0]
    views = [v for v in _walk(sent[0]) if isinstance(v, rrb.TimeSeriesView)]
    assert len(views) == len(ACTION_KEYS)
    contents = {str(c) for c in views[0].contents}
    for prefix in (POLICY_PREFIX, COMMAND_PREFIX, "observation."):
        assert any(prefix in c for c in contents), contents


def test_pinning_is_skipped_rather_than_raising_when_upstream_has_moved():
    """The cache attribute is an upstream implementation detail; treat it as one."""
    with pytest.MonkeyPatch.context() as mp:
        from lerobot.utils import rerun_visualization

        mp.setattr(rerun_visualization, "log_rerun_data", object())
        assert pin_blueprint() is False


def _walk(node):
    """Every view in a blueprint tree, whatever the container nesting."""
    yield node
    for attr in ("contents", "blueprint", "root_container"):
        children = getattr(node, attr, None)
        if isinstance(children, (list, tuple)):
            for child in children:
                if not isinstance(child, str):
                    yield from _walk(child)
        elif children is not None and not isinstance(children, str):
            yield from _walk(children)


def test_the_module_names_its_paths_once():
    """Entity paths are a contract with the blueprint; derive, do not restate."""
    assert actionview.POLICY_PREFIX == "policy"
    assert actionview.COMMAND_PREFIX == "command"
