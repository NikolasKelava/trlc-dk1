"""Teleoperation wiring: the right devices, the right names, the right limits.

The control loop itself is LeRobot's and is not re-tested here. What is tested is
everything this fork decides — which is where a bug would be ours.
"""

from __future__ import annotations

import pytest

from dk1lab.config import load
from dk1lab.layout import CAMERA_NAMES
from dk1lab.limiter import DEFAULT_MAX_JOINT_RATE, DEFAULT_MAX_LAG
from dk1lab.teleop import (
    TELEOP_MAX_JOINT_RATE,
    TELEOP_MAX_LAG,
    TeleopLimits,
    build,
    follower_config,
    leader_config,
    run,
)


@pytest.fixture
def config(config_file):
    return load(config_file)


# --------------------------------------------------------------------------- #
# Devices come from dk1.toml, not from anywhere else
# --------------------------------------------------------------------------- #


def test_the_leader_gets_the_configured_leader_ports(config):
    leader = leader_config(config)
    assert leader.left_arm_port == "/dev/ttyACM0"
    assert leader.right_arm_port == "/dev/ttyACM2"


def test_the_follower_gets_the_configured_follower_ports(config):
    follower = follower_config(config)
    assert follower.left_arm_port == "/dev/ttyACM1"
    assert follower.right_arm_port == "/dev/ttyACM3"


def test_leader_and_follower_never_share_a_port(config):
    leader, follower = leader_config(config), follower_config(config)
    ports = {
        leader.left_arm_port,
        leader.right_arm_port,
        follower.left_arm_port,
        follower.right_arm_port,
    }
    assert len(ports) == 4


# --------------------------------------------------------------------------- #
# The speed limit
# --------------------------------------------------------------------------- #


def test_the_follower_is_the_rate_limited_one(config):
    """Plain bi_dk1_follower would make the speed limit a no-op in impedance mode."""
    assert follower_config(config).type == "bi_dk1_follower_safe"


def test_teleop_raises_the_limit_above_the_policy_crawl():
    """The limiter's own default is sized for a policy nobody trusts yet.

    A human moving a leader arm outruns 0.2 rad/s instantly, and a follower that
    visibly cannot keep up would say nothing about whether the stack works.
    """
    assert TELEOP_MAX_JOINT_RATE > DEFAULT_MAX_JOINT_RATE
    assert TELEOP_MAX_LAG > DEFAULT_MAX_LAG


def test_the_limit_is_on_by_default(config):
    assert follower_config(config).max_joint_rate == TELEOP_MAX_JOINT_RATE


def test_custom_limits_reach_the_follower(config):
    limits = TeleopLimits(max_joint_rate=0.4, max_gripper_rate=0.5, max_lag=0.2, max_dt=0.05)
    follower = follower_config(config, limits=limits)
    assert follower.max_joint_rate == 0.4
    assert follower.max_gripper_rate == 0.5
    assert follower.max_lag == 0.2
    assert follower.max_dt == 0.05


def test_removing_the_limit_takes_saying_so(config):
    """Disabling is possible but never implicit — nothing defaults to None."""
    assert follower_config(config, limits=TeleopLimits().unlimited()).max_joint_rate is None


def test_unlimited_leaves_the_other_caps_alone():
    limits = TeleopLimits().unlimited()
    assert limits.max_lag == TELEOP_MAX_LAG
    assert limits.max_gripper_rate == TeleopLimits().max_gripper_rate


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def test_cameras_are_named_what_the_checkpoint_requires(config):
    """The old repo called these wrist_left / wrist_right, which cannot work."""
    assert tuple(follower_config(config).cameras) == CAMERA_NAMES


def test_camera_order_is_the_trained_order_not_alphabetical(config):
    keys = list(follower_config(config).cameras)
    assert keys == ["top", "left", "right"] != sorted(keys)


def test_cameras_can_be_left_off(config):
    assert follower_config(config, cameras=False).cameras == {}


def test_the_teleop_profile_is_the_default_because_nothing_depends_on_it(config):
    camera = follower_config(config).cameras["top"]
    assert (camera.width, camera.height) == (1280, 720)


def test_the_policy_profile_is_available_for_recording(config):
    camera = follower_config(config, profile="policy").cameras["top"]
    assert (camera.width, camera.height) == (640, 360)


def test_camera_rotation_comes_from_the_config(config):
    cameras = follower_config(config).cameras
    assert int(cameras["top"].rotation) == 180
    assert int(cameras["right"].rotation) == 0


# --------------------------------------------------------------------------- #
# Control mode
# --------------------------------------------------------------------------- #


def test_impedance_is_the_default_matching_upstream(config):
    assert follower_config(config).control_mode == "impedance"


def test_pos_vel_can_be_selected(config):
    assert follower_config(config, control_mode="pos_vel").control_mode == "pos_vel"


# --------------------------------------------------------------------------- #
# build() constructs but connects to nothing
# --------------------------------------------------------------------------- #


def test_build_returns_both_devices_disconnected(config):
    leader, follower = build(config, cameras=False)
    assert not leader.is_connected
    assert not follower.is_connected


# --------------------------------------------------------------------------- #
# run() — connect order, and that stopping never moves the arms
# --------------------------------------------------------------------------- #


class FakeDevice:
    """Records the lifecycle calls made on it."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def connect(self):
        self.log.append(f"connect:{self.name}")

    def disconnect(self):
        self.log.append(f"disconnect:{self.name}")


@pytest.fixture
def fakes(monkeypatch):
    """Fake devices plus a stubbed teleop_loop; returns the call log."""
    import lerobot.scripts.lerobot_teleoperate as script

    log = []
    monkeypatch.setattr(script, "teleop_loop", lambda **kwargs: log.append("loop"))
    return log


def test_the_leader_connects_before_the_follower(fakes):
    """So a hand already resting on a leader cannot command anything yet."""
    leader, follower = FakeDevice(fakes, "leader"), FakeDevice(fakes, "follower")
    run(leader, follower, fps=30)
    assert fakes.index("connect:leader") < fakes.index("connect:follower")


def test_the_loop_runs_between_connect_and_disconnect(fakes):
    run(FakeDevice(fakes, "leader"), FakeDevice(fakes, "follower"), fps=30)
    assert fakes == [
        "connect:leader",
        "connect:follower",
        "loop",
        "disconnect:follower",
        "disconnect:leader",
    ]


def test_stopping_does_not_move_the_arms(fakes):
    """There is no return-to-home step, by design, on any exit path."""
    run(FakeDevice(fakes, "leader"), FakeDevice(fakes, "follower"), fps=30)
    assert not any("home" in call or "initial" in call for call in fakes)


def test_ctrl_c_still_disconnects_both(monkeypatch):
    import lerobot.scripts.lerobot_teleoperate as script

    log = []

    def interrupted(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(script, "teleop_loop", interrupted)
    run(FakeDevice(log, "leader"), FakeDevice(log, "follower"), fps=30)
    assert log == [
        "connect:leader",
        "connect:follower",
        "disconnect:follower",
        "disconnect:leader",
    ]


def test_a_failure_inside_the_loop_still_disconnects_both(monkeypatch):
    """A crash must not leave the motors energised with nothing driving them."""
    import lerobot.scripts.lerobot_teleoperate as script

    log = []

    def explode(**kwargs):
        raise RuntimeError("motor chain died")

    monkeypatch.setattr(script, "teleop_loop", explode)
    with pytest.raises(RuntimeError, match="motor chain died"):
        run(FakeDevice(log, "leader"), FakeDevice(log, "follower"), fps=30)
    assert log[-2:] == ["disconnect:follower", "disconnect:leader"]


def test_the_requested_rate_and_duration_reach_the_loop(monkeypatch):
    import lerobot.scripts.lerobot_teleoperate as script

    seen = {}
    monkeypatch.setattr(script, "teleop_loop", lambda **kwargs: seen.update(kwargs))
    run(FakeDevice([], "leader"), FakeDevice([], "follower"), fps=45, duration_s=12.0)
    assert seen["fps"] == 45
    assert seen["duration"] == 12.0
    assert seen["display_data"] is False
