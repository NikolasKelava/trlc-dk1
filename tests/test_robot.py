"""SafeBiDK1Follower: the speed limit is actually in the send path.

No hardware: the arms are stubbed, so these test the wiring, not the motors.
"""

from __future__ import annotations

import pytest

from dk1lab.layout import ACTION_KEYS
from dk1lab.limiter import SlewLimiter
from dk1lab.robot import SafeBiDK1Follower, SafeBiDK1FollowerConfig


@pytest.fixture
def robot() -> SafeBiDK1Follower:
    return SafeBiDK1Follower(
        SafeBiDK1FollowerConfig(left_arm_port="/dev/null", right_arm_port="/dev/null")
    )


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --------------------------------------------------------------------------- #
# Contract with upstream
# --------------------------------------------------------------------------- #


def test_upstream_private_readers_still_exist():
    """The one place this fork reaches into upstream internals.

    ``measured_positions`` calls these to read joint state without paying for a
    camera read. If a rebase renames them, fail here — loudly — rather than
    silently losing the speed limit or the anti-windup clamp.
    """
    from lerobot_robot_trlc_dk1.follower import DK1Follower

    assert callable(DK1Follower._get_observation_impedance)
    assert callable(DK1Follower._get_observation_pos_vel)


def test_the_safe_follower_is_registered_under_its_own_type():
    from lerobot.robots import RobotConfig

    assert RobotConfig.get_choice_class("bi_dk1_follower_safe") is SafeBiDK1FollowerConfig


def test_it_does_not_shadow_the_upstream_robot_type():
    """Upstream's unlimited follower must stay reachable and unchanged."""
    from lerobot.robots import RobotConfig

    from lerobot_robot_trlc_dk1.bi_follower import BiDK1FollowerConfig

    assert RobotConfig.get_choice_class("bi_dk1_follower") is BiDK1FollowerConfig


def test_lerobot_can_build_the_follower_from_its_config():
    """Registration is not enough: rollout builds the robot by *class* lookup.

    ``make_robot_from_config`` derives the class name from the config class name
    and imports it from the package holding the config's module — ``dk1lab`` and
    ``dk1lab.safebidk1follower``. Teleoperation never exercises this, because it
    constructs the follower directly; the first policy dry run on the hardware
    died here.
    """
    from lerobot.robots.utils import make_robot_from_config

    built = make_robot_from_config(
        SafeBiDK1FollowerConfig(left_arm_port="/dev/null", right_arm_port="/dev/zero")
    )
    assert isinstance(built, SafeBiDK1Follower)
    assert built.name == "bi_dk1_follower_safe"


def test_the_follower_is_reachable_on_the_package_itself():
    """The attribute LeRobot looks up by name, on the module it looks it up on."""
    import dk1lab

    assert dk1lab.SafeBiDK1Follower is SafeBiDK1Follower


def test_the_package_still_refuses_unknown_attributes():
    """The lazy __getattr__ must not turn typos into silent Nones."""
    import dk1lab

    with pytest.raises(AttributeError):
        dk1lab.NoSuchThing


def test_config_defaults_to_impedance_with_a_limit(robot):
    """Impedance is the mode the bimanual follower actually runs, and the mode
    in which upstream's joint_velocity_scaling does nothing."""
    assert robot.config.control_mode == "impedance"
    assert robot.config.max_joint_rate is not None
    assert robot.limiter.enabled


# --------------------------------------------------------------------------- #
# The send path
# --------------------------------------------------------------------------- #


def test_send_action_limits_before_sending(robot, monkeypatch):
    """A far-away target must reach the arms clamped, not raw."""
    clock = FakeClock()
    robot.limiter = SlewLimiter(max_joint_rate=1.0, max_lag=10.0, clock=clock)

    measured = dict.fromkeys(ACTION_KEYS, 0.0)
    monkeypatch.setattr(robot, "measured_positions", lambda: measured)

    sent: list[dict] = []
    monkeypatch.setattr(
        type(robot).__mro__[1], "send_action", lambda self, action: sent.append(dict(action)) or action
    )

    robot.send_action(dict.fromkeys(ACTION_KEYS, 100.0))  # seeding tick
    clock.advance(0.1)
    robot.send_action(dict.fromkeys(ACTION_KEYS, 100.0))

    assert len(sent) == 2
    assert sent[0]["left_joint_1.pos"] == pytest.approx(0.0)
    assert sent[1]["left_joint_1.pos"] == pytest.approx(0.1)


def test_send_action_returns_what_was_sent_not_what_was_asked(robot, monkeypatch):
    """A recorded dataset must contain the command the arms received."""
    clock = FakeClock()
    robot.limiter = SlewLimiter(max_joint_rate=1.0, max_lag=10.0, clock=clock)
    monkeypatch.setattr(robot, "measured_positions", lambda: dict.fromkeys(ACTION_KEYS, 0.0))
    monkeypatch.setattr(type(robot).__mro__[1], "send_action", lambda self, action: action)

    robot.send_action(dict.fromkeys(ACTION_KEYS, 100.0))
    clock.advance(0.1)
    returned = robot.send_action(dict.fromkeys(ACTION_KEYS, 100.0))

    assert returned["left_joint_1.pos"] == pytest.approx(0.1)
    assert returned["left_joint_1.pos"] != 100.0


def test_every_channel_is_limited_including_both_grippers(robot, monkeypatch):
    clock = FakeClock()
    robot.limiter = SlewLimiter(
        max_joint_rate=1.0, max_gripper_rate=1.0, max_lag=10.0, clock=clock
    )
    monkeypatch.setattr(robot, "measured_positions", lambda: dict.fromkeys(ACTION_KEYS, 0.0))
    monkeypatch.setattr(type(robot).__mro__[1], "send_action", lambda self, action: action)

    robot.send_action(dict.fromkeys(ACTION_KEYS, 100.0))
    clock.advance(0.1)
    result = robot.send_action(dict.fromkeys(ACTION_KEYS, 100.0))

    assert set(result) == set(ACTION_KEYS)
    for key, value in result.items():
        assert value == pytest.approx(0.1), key


def test_disabling_the_limit_sends_the_raw_action(monkeypatch):
    """An explicit opt-out must not silently keep limiting, or vice versa."""
    robot = SafeBiDK1Follower(
        SafeBiDK1FollowerConfig(
            left_arm_port="/dev/null", right_arm_port="/dev/null", max_joint_rate=None
        )
    )
    assert not robot.limiter.enabled
    monkeypatch.setattr(type(robot).__mro__[1], "send_action", lambda self, action: action)
    result = robot.send_action(dict.fromkeys(ACTION_KEYS, 100.0))
    assert result["left_joint_1.pos"] == 100.0


def test_measured_positions_covers_all_fourteen_channels(robot, monkeypatch):
    """Every action channel needs a measurement, or its lag clamp is inert."""
    per_arm = {f"joint_{i}.pos": 0.0 for i in range(1, 7)} | {"gripper.pos": 0.0}
    monkeypatch.setattr(robot.left_arm, "_get_observation_impedance", lambda: dict(per_arm))
    monkeypatch.setattr(robot.right_arm, "_get_observation_impedance", lambda: dict(per_arm))
    assert set(robot.measured_positions()) == set(ACTION_KEYS)


def test_measured_positions_follows_the_control_mode(monkeypatch):
    """pos_vel arms must be read through the pos_vel path, not the impedance one."""
    robot = SafeBiDK1Follower(
        SafeBiDK1FollowerConfig(
            left_arm_port="/dev/null", right_arm_port="/dev/null", control_mode="pos_vel"
        )
    )
    per_arm = {f"joint_{i}.pos": 1.0 for i in range(1, 7)} | {"gripper.pos": 1.0}
    called: list[str] = []

    for arm in (robot.left_arm, robot.right_arm):
        monkeypatch.setattr(
            arm, "_get_observation_pos_vel", lambda: called.append("pos_vel") or dict(per_arm)
        )
        monkeypatch.setattr(
            arm, "_get_observation_impedance", lambda: called.append("impedance") or dict(per_arm)
        )

    robot.measured_positions()
    assert called == ["pos_vel", "pos_vel"]


def test_disconnect_does_not_command_a_pose(robot, monkeypatch):
    """Stopping must never move the arms — no return-to-home, ever, implicitly."""
    sent: list[dict] = []
    monkeypatch.setattr(
        type(robot).__mro__[1], "send_action", lambda self, action: sent.append(action) or action
    )
    monkeypatch.setattr(type(robot).__mro__[1], "disconnect", lambda self: None)
    robot.disconnect()
    assert sent == []
