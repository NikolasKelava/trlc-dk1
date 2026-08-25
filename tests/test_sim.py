"""The MuJoCo cell: the scene it builds, and the robot interface it presents.

These need MuJoCo but no GPU, no display and no hardware — MuJoCo renders on the
CPU when it has to, and the scene is built from a URDF that is checked in.

What is worth holding here is the seam: the 14-D contract belongs to
:mod:`dk1lab.layout`, the joint names belong to the model, and
:meth:`~dk1lab.sim.SimRobot._resolve_indices` is the one place they meet. A
mismatch there is a policy driving joint 4 with joint 5's command, which nothing
downstream would notice.
"""

from __future__ import annotations

import pytest

from dk1lab import scene
from dk1lab.layout import ACTION_KEYS, ARM_KEYS, ARMS, CAMERA_NAMES, DOF
from dk1lab.sim import SimRobot, SimRobotConfig

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def built():
    """The scene, built once — importing the URDF is the slow part."""
    return scene.build()


@pytest.fixture(scope="module")
def model(built):
    return mujoco.MjModel.from_xml_string(built.xml)


@pytest.fixture
def robot():
    """A connected sim, headless and free-running. Nothing is energised."""
    # The cap is the policy's 1.0 rad/s, not the limiter's timid 0.2 default:
    # these tests are about the sim, and a joint that has not arrived because the
    # ramp is slow proves nothing about whether it was driven.
    live = SimRobot(
        SimRobotConfig(view=False, realtime=False, width=64, height=36,
                       max_joint_rate=1.0, id="test")
    )
    live.connect()
    yield live
    live.disconnect()


def names_of(model, kind) -> list[str]:
    count = {mujoco.mjtObj.mjOBJ_CAMERA: model.ncam,
             mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
             mujoco.mjtObj.mjOBJ_JOINT: model.njnt}[kind]
    return [mujoco.mj_id2name(model, kind, index) for index in range(count)]


# --------------------------------------------------------------------------- #
# The scene
# --------------------------------------------------------------------------- #


def test_the_scene_builds_from_this_repos_urdf(built):
    assert built.urdf.is_file()
    assert "dk1-bimanual" in built.xml


def test_both_arms_are_there_and_are_not_the_same_arm(model):
    joints = names_of(model, mujoco.mjtObj.mjOBJ_JOINT)
    for arm in ARMS:
        for name in scene.joint_names(arm):
            assert name in joints
    assert len(set(joints)) == len(joints)


def test_the_cameras_are_this_cells_names_in_this_cells_order(model):
    """The checkpoint's image keys are built from these; the order is not free."""
    cameras = names_of(model, mujoco.mjtObj.mjOBJ_CAMERA)
    assert cameras == list(CAMERA_NAMES)


def test_every_channel_has_an_actuator(model):
    actuators = names_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR)
    for arm in ARMS:
        joint_acts, finger_acts = scene.actuator_names(arm)
        for name in [*joint_acts, *finger_acts]:
            assert name in actuators
    # Six joints and two fingers per arm: the cell's one gripper channel drives
    # both fingers, which is why this is 16 and not 14.
    assert model.nu == len(ARMS) * (len(scene.URDF_JOINTS) + len(scene.URDF_FINGERS))


def test_the_arm_joints_use_the_cells_own_impedance_gains(model):
    """A joint that lags in the sim should lag for a reason the cell shares."""
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint1_act")
    assert model.actuator_gainprm[index][0] == pytest.approx(scene.ARM_KP[0])


def test_the_task_objects_can_be_left_out():
    """`--no-objects` is the kinematic check STUDY.md names as the fallback."""
    with_objects = mujoco.MjModel.from_xml_string(scene.build(objects=True).xml)
    without = mujoco.MjModel.from_xml_string(scene.build(objects=False).xml)
    assert without.nbody < with_objects.nbody
    # The die's free joint is what makes it fall; without it there is nothing to grasp.
    assert without.nq < with_objects.nq


def test_a_missing_urdf_is_refused_by_name(tmp_path):
    with pytest.raises(scene.SceneError) as excinfo:
        scene.build(tmp_path / "nothing.urdf")
    assert "nothing.urdf" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# The gripper, which is one channel and two joints
# --------------------------------------------------------------------------- #


def test_zero_is_open_and_one_is_closed():
    """This cell's convention, not the checkpoint's. See dk1lab.layout."""
    assert scene.gripper_position(0.0) == pytest.approx(scene.GRIPPER_OPEN_M)
    assert scene.gripper_position(1.0) == pytest.approx(scene.GRIPPER_CLOSED_M)
    assert scene.GRIPPER_OPEN_M > scene.GRIPPER_CLOSED_M


def test_a_command_outside_the_range_is_clamped_rather_than_obeyed():
    """A policy that has never seen this cell does ask for these."""
    assert scene.gripper_position(-3.0) == pytest.approx(scene.GRIPPER_OPEN_M)
    assert scene.gripper_position(9.0) == pytest.approx(scene.GRIPPER_CLOSED_M)


def test_the_measurement_maps_back_to_what_was_commanded():
    for value in (0.0, 0.25, 0.5, 1.0):
        assert scene.gripper_normalised(scene.gripper_position(value)) == pytest.approx(value)


# --------------------------------------------------------------------------- #
# The robot interface
# --------------------------------------------------------------------------- #


def test_it_presents_the_same_14_channels_the_cell_does(robot):
    assert list(robot.action_features) == list(ACTION_KEYS)
    assert len(robot.action_features) == DOF


def test_the_observation_is_14_positions_and_three_images(robot):
    observation = robot.get_observation()
    assert [key for key in observation if key.endswith(".pos")] == list(ACTION_KEYS)
    for name in CAMERA_NAMES:
        assert observation[name].shape == (36, 64, 3)


def test_connecting_does_not_move_anything(robot):
    """The one place this robot differs from the real one, and it is the point."""
    before = robot.measured_positions()
    after = robot.measured_positions()
    assert before == after
    assert all(abs(value) < 1e-6 for key, value in before.items() if not key.endswith("gripper.pos"))


def test_each_channel_drives_its_own_joint(robot):
    """The seam: a mix-up here is joint 4 driven by joint 5's command."""
    for key in ACTION_KEYS:
        if key.endswith("gripper.pos"):
            continue
        target = dict.fromkeys(ACTION_KEYS, 0.0)
        target[key] = 0.3
        for _ in range(60):
            robot.send_action(target)
        moved = {
            other: value
            for other, value in robot.measured_positions().items()
            if abs(value) > 0.05 and not other.endswith("gripper.pos")
        }
        assert list(moved) == [key], f"{key} moved {list(moved)}"
        for _ in range(60):
            robot.send_action(dict.fromkeys(ACTION_KEYS, 0.0))


def test_a_channel_left_out_holds_its_last_target(robot):
    """What the real motor chain does between commands."""
    target = dict.fromkeys(ACTION_KEYS, 0.0)
    target["left_joint_2.pos"] = 0.5
    for _ in range(90):
        robot.send_action(target)
    reached = robot.measured_positions()["left_joint_2.pos"]
    for _ in range(30):
        robot.send_action({"left_joint_1.pos": 0.0})
    assert robot.measured_positions()["left_joint_2.pos"] == pytest.approx(reached, abs=0.05)


def test_send_action_returns_what_the_arms_were_given(robot):
    """The recorders write this, so it must be the limited action and not the request."""
    asked = dict.fromkeys(ACTION_KEYS, 2.0)
    sent = robot.send_action(asked)
    assert set(sent) == set(ACTION_KEYS)
    # One tick at 1.0 rad/s from zero cannot reach 2.0 rad, so the limiter is on
    # and what came back is not what was asked for.
    assert sent["left_joint_1.pos"] < 1.0


def test_the_speed_limit_can_be_turned_off_like_the_real_ones():
    live = SimRobot(
        SimRobotConfig(view=False, realtime=False, width=32, height=18,
                       max_joint_rate=None, id="test")
    )
    live.connect()
    try:
        assert live.limiter.enabled is False
        sent = live.send_action(dict.fromkeys(ACTION_KEYS, 0.5))
        assert sent["left_joint_1.pos"] == pytest.approx(0.5)
    finally:
        live.disconnect()


def test_one_tick_is_one_control_period_of_physics(robot):
    """The policy sees 30 Hz whatever the wall clock and the GPU are doing.

    *Exactly* one: the model's timestep is set from the control period rather
    than the substep count being rounded to fit the scene's own. Rounding gave
    17 steps of 2 ms = 34 ms, a 2% clock error nothing downstream would report.
    """
    period = 1.0 / robot.fps
    assert robot._steps_per_tick * robot._model.opt.timestep == pytest.approx(period, rel=1e-9)


def test_nothing_is_touching_anything_at_the_start(robot):
    """An arm that collides with itself at rest has its base yaw pinned.

    The DK1's adjacent collision hulls overlap by a couple of millimetres, so
    imported straight from the URDF each arm starts with four contacts inside
    itself — and the first sim rollout would have found "the policy cannot move
    joint 1", which is a fact about the scene. See ``dk1lab.scene.ARM_CONTACT``.
    """
    assert robot._data.ncon == 0


def test_the_limiter_ramps_on_the_sims_clock_not_the_wall_clock():
    """Otherwise --free-run and --realtime would move the arms differently.

    The limiter steps by ``rate * dt``. On ``time.monotonic`` that ``dt`` is how
    long the last tick took in the room, which free-running is under a
    millisecond — so a 1.0 rad/s cap would behave like 0.03, and the motion would
    depend on how fast the computer is.
    """
    from dk1lab.layout import ACTION_KEYS as keys

    reached = {}
    for realtime in (False, True):
        live = SimRobot(
            SimRobotConfig(view=False, realtime=realtime, width=32, height=18,
                           max_joint_rate=1.0, id="test")
        )
        live.connect()
        try:
            target = dict.fromkeys(keys, 0.0)
            target["left_joint_1.pos"] = 0.3
            for _ in range(30):
                live.send_action(target)
            reached[realtime] = live.measured_positions()["left_joint_1.pos"]
        finally:
            live.disconnect()
    assert reached[False] == pytest.approx(reached[True], abs=1e-6)


def test_free_running_does_not_wait_for_the_wall_clock(robot):
    import time

    started = time.perf_counter()
    for _ in range(30):
        robot.send_action(dict.fromkeys(ACTION_KEYS, 0.0))
    # A realtime second's worth of ticks, well under a realtime second.
    assert time.perf_counter() - started < 0.5


def test_using_it_before_connecting_says_so():
    live = SimRobot(SimRobotConfig(view=False, realtime=False, id="test"))
    with pytest.raises(RuntimeError, match="not connected"):
        live.get_observation()


def test_disconnecting_twice_is_harmless(robot):
    robot.disconnect()
    robot.disconnect()
    assert robot.is_connected is False


def test_it_is_findable_the_way_lerobot_looks_for_it():
    """The trap CLAUDE.md names twice: registration alone does not satisfy this."""
    import dk1lab

    assert dk1lab.SimRobot is SimRobot


def test_the_indices_are_resolved_against_the_layout_not_the_model(robot):
    """The 14 channels come out in ACTION_KEYS order however MuJoCo ordered them."""
    assert len(robot._qpos) == DOF
    assert len(robot._actuators) == DOF
    # The gripper channel drives two actuators; every other channel drives one.
    per_channel = [len(ids) for ids in robot._actuators]
    assert per_channel.count(2) == len(ARMS)
    assert per_channel.count(1) == DOF - len(ARMS)
    assert len(ARM_KEYS) * len(ARMS) == DOF
