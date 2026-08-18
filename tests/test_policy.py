"""Policy deployment wiring, and the one thing LeRobot will not do for us.

The control loop and the model are LeRobot's and are not re-tested here. What is
tested is every decision this fork makes on the way to them — above all that the
gripper inversion is applied to the objects where it actually takes effect,
because the failure mode is silent, symmetric, and lands on the hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from conftest import CHECKPOINT_CONFIG

from dk1lab.config import LimitProfile, load
from dk1lab.layout import ACTION_KEYS, DOF, GRIPPER_INDICES, IMAGE_KEYS
from dk1lab.policy import (
    DEFAULT_FPS,
    POLICY_LIMITS,
    InversionError,
    apply_gripper_inversion,
    follower_config,
    policy_config,
    rollout_config,
)

@pytest.fixture
def checkpoint(checkpoint_dir):
    return str(checkpoint_dir)


@pytest.fixture
def config(config_file):
    return load(config_file)


# --------------------------------------------------------------------------- #
# The gripper inversion — where it is applied, and why there
# --------------------------------------------------------------------------- #


@dataclass
class FakeTransformStep:
    """Stands in for MolmoAct2's state/action frame transform steps."""

    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None


@dataclass
class FakeOtherStep:
    """Any pipeline step that carries no joint transform."""

    name: str = "normalizer"


@dataclass
class FakePipeline:
    steps: list = field(default_factory=list)


def pipelines(pre_steps=None, post_steps=None):
    return (
        FakePipeline(
            pre_steps if pre_steps is not None else [FakeOtherStep(), FakeTransformStep()]
        ),
        FakePipeline(post_steps if post_steps is not None else [FakeTransformStep()]),
    )


def test_the_inversion_is_applied_to_both_pipelines():
    """Both directions, because state goes in and action comes back out."""
    pre, post = pipelines()
    apply_gripper_inversion(pre, post)
    assert pre.steps[1].joint_signs is not None
    assert post.steps[0].joint_signs is not None


def test_only_the_two_gripper_channels_are_touched():
    pre, post = pipelines()
    apply_gripper_inversion(pre, post)
    signs = pre.steps[1].joint_signs
    offsets = pre.steps[1].joint_offsets
    inverted = {i for i, sign in enumerate(signs) if sign != 1.0}
    assert inverted == set(GRIPPER_INDICES)
    assert all(offsets[i] == 0.0 for i in range(DOF) if i not in GRIPPER_INDICES)


def test_the_transform_is_x_to_one_minus_x_on_the_grippers():
    """0 open on the DK1 must arrive at the model as 1 open, and come back 0."""
    pre, post = pipelines()
    apply_gripper_inversion(pre, post)
    signs, offsets = pre.steps[1].joint_signs, pre.steps[1].joint_offsets

    for index in GRIPPER_INDICES:
        to_model = lambda x, i=index: signs[i] * x + offsets[i]  # noqa: E731
        to_robot = lambda x, i=index: signs[i] * (x - offsets[i])  # noqa: E731
        assert to_model(0.0) == pytest.approx(1.0)
        assert to_model(1.0) == pytest.approx(0.0)
        assert to_robot(to_model(0.3)) == pytest.approx(0.3)


def test_the_arm_joints_pass_through_unchanged():
    """The arm joint map is identity; only the gripper convention differs."""
    pre, post = pipelines()
    apply_gripper_inversion(pre, post)
    signs, offsets = pre.steps[1].joint_signs, pre.steps[1].joint_offsets
    for index in range(DOF):
        if index in GRIPPER_INDICES:
            continue
        assert signs[index] * 0.42 + offsets[index] == pytest.approx(0.42)


def test_a_pipeline_with_no_transform_step_is_an_error_not_a_warning():
    """Continuing would deploy an uninverted gripper, which is worse than stopping."""
    pre, post = pipelines(pre_steps=[FakeOtherStep()])
    with pytest.raises(InversionError, match="backwards"):
        apply_gripper_inversion(pre, post)


def test_two_transform_steps_are_an_error_too():
    pre, post = pipelines(post_steps=[FakeTransformStep(), FakeTransformStep()])
    with pytest.raises(InversionError, match="found 2"):
        apply_gripper_inversion(pre, post)


def test_the_result_names_the_channels_it_inverted():
    pre, post = pipelines()
    described = apply_gripper_inversion(pre, post).describe()
    for index in GRIPPER_INDICES:
        assert ACTION_KEYS[index] in described


# --------------------------------------------------------------------------- #
# The policy config: what is overridden, and why
# --------------------------------------------------------------------------- #


def test_the_device_is_forced_off_cpu(checkpoint):
    """The converted checkpoint's config.json says cpu; that loads 7B on the CPU."""
    assert CHECKPOINT_CONFIG["device"] == "cpu"
    assert policy_config(checkpoint).device == "cuda"


def test_the_dtype_is_forced_to_bfloat16(checkpoint):
    assert policy_config(checkpoint).model_dtype == "bfloat16"


def test_inference_runs_the_continuous_head(checkpoint):
    """RTC requires it, and it is what the checkpoint was evaluated with."""
    assert policy_config(checkpoint).inference_action_mode == "continuous"


def test_the_image_keys_are_pinned_in_the_trained_order(checkpoint):
    assert policy_config(checkpoint).image_keys == list(IMAGE_KEYS)


def test_the_config_also_carries_the_inversion_even_though_it_is_not_what_applies_it(checkpoint):
    """Set for anything that rebuilds processors from the config — training, say."""
    config = policy_config(checkpoint)
    assert [config.joint_signs[i] for i in GRIPPER_INDICES] == [-1.0, -1.0]
    assert [config.joint_offsets[i] for i in GRIPPER_INDICES] == [1.0, 1.0]


# --------------------------------------------------------------------------- #
# The follower: capped, and looking at the right cameras
# --------------------------------------------------------------------------- #


def test_the_follower_is_the_rate_limited_one(config):
    assert follower_config(config).type == "bi_dk1_follower_safe"


def test_rollout_is_capped_by_default_unlike_teleoperation(config):
    """The limiter exists for exactly this case: a policy nobody has watched yet."""
    assert follower_config(config).max_joint_rate == POLICY_LIMITS.max_joint_rate
    assert POLICY_LIMITS.max_joint_rate is not None


def test_the_configured_policy_limit_wins_over_the_built_in_one(config_file):
    config_file.write_text(
        config_file.read_text() + "\n[limits.policy]\nmax_joint_rate = 0.05\n"
    )
    assert follower_config(load(config_file)).max_joint_rate == 0.05


def test_an_explicit_limit_wins_over_the_file(config):
    limits = LimitProfile(max_joint_rate=0.9, max_gripper_rate=1.0, max_lag=0.1, max_dt=0.1)
    assert follower_config(config, limits=limits).max_joint_rate == 0.9


def test_the_cameras_run_at_the_resolution_the_policy_was_trained_on(config):
    """640x360 is 16:9; a 4:3 capture would stretch the scene differently."""
    cameras = follower_config(config).cameras
    assert list(cameras) == ["top", "left", "right"]
    assert all((camera.width, camera.height) == (640, 360) for camera in cameras.values())


# --------------------------------------------------------------------------- #
# The rollout config
# --------------------------------------------------------------------------- #


@pytest.fixture
def rollout(config, checkpoint):
    def build(**kwargs):
        return rollout_config(config, checkpoint=checkpoint, task="pick up the pen", **kwargs)

    return build


def test_stopping_never_sweeps_the_arms_home(rollout):
    """LeRobot defaults this to True: pressing stop would move both arms."""
    assert rollout().return_to_initial_position is False


def test_return_home_is_opt_in(rollout):
    assert rollout(return_home=True).return_to_initial_position is True


def test_the_loop_runs_at_the_rate_the_checkpoint_was_trained_at(rollout):
    assert rollout().fps == DEFAULT_FPS == 30


def test_inference_is_synchronous_unless_rtc_is_asked_for(rollout):
    assert rollout().inference.type == "sync"


def test_rtc_carries_the_execution_horizon(rollout):
    inference = rollout(rtc=True, execution_horizon=7).inference
    assert inference.type == "rtc"
    assert inference.rtc.execution_horizon == 7


def test_the_task_reaches_the_config(rollout):
    assert rollout().task == "pick up the pen"


def test_the_robot_is_the_limited_follower_on_the_configured_ports(rollout):
    robot = rollout().robot
    assert robot.type == "bi_dk1_follower_safe"
    assert (robot.left_arm_port, robot.right_arm_port) == ("/dev/ttyACM1", "/dev/ttyACM3")


def test_no_dataset_is_recorded_by_a_plain_rollout(rollout):
    """Phase 4 records; Phase 3 only watches."""
    assert rollout().dataset is None
    assert rollout().strategy.type == "base"
