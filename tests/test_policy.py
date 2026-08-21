"""Policy deployment wiring, and the one thing LeRobot will not do for us.

The control loop and the model are LeRobot's and are not re-tested here. What is
tested is every decision this fork makes on the way to them — above all that the
gripper inversion is applied to the objects where it actually takes effect,
because the failure mode is silent, symmetric, and lands on the hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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


def test_the_config_carries_the_inversion_only_when_it_was_asked_for(checkpoint):
    """Set for anything that rebuilds processors from the config — training, say."""
    config = policy_config(checkpoint, invert_gripper=True)
    assert [config.joint_signs[i] for i in GRIPPER_INDICES] == [-1.0, -1.0]
    assert [config.joint_offsets[i] for i in GRIPPER_INDICES] == [1.0, 1.0]


def test_the_config_leaves_the_gripper_alone_by_default(checkpoint):
    """The inversion is opt-in, so a config nobody asked to invert says nothing."""
    assert policy_config(checkpoint).joint_signs is None


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


# --------------------------------------------------------------------------- #
# RTC headroom — the delay must fit inside the execution horizon
# --------------------------------------------------------------------------- #


def test_the_default_horizon_leaves_room_for_the_measured_latency():
    """The old default of 10 was exactly the degenerate case.

    RTC ramps its prefix weights from the inference delay down to zero at the
    execution horizon. Measured RTC latency on this machine is ~270-330 ms,
    which is 9-10 ticks at 30 Hz, so a horizon of 10 left a ramp of zero width.
    """
    from dk1lab.policy import (
        DEFAULT_EXECUTION_HORIZON,
        DEFAULT_FPS,
        MEASURED_RTC_LATENCY_S,
        rtc_headroom,
    )

    delay, ok = rtc_headroom(
        MEASURED_RTC_LATENCY_S, fps=DEFAULT_FPS, execution_horizon=DEFAULT_EXECUTION_HORIZON
    )
    assert ok, f"delay {delay} does not fit in horizon {DEFAULT_EXECUTION_HORIZON}"
    assert DEFAULT_EXECUTION_HORIZON - delay >= 5, "the blend needs more than a couple of steps"


def test_the_horizon_stays_below_the_chunk_size():
    """A horizon of 30 pins the whole chunk to the previous one, and stops reacting."""
    from dk1lab.policy import DEFAULT_EXECUTION_HORIZON

    assert DEFAULT_EXECUTION_HORIZON < 30


def test_a_slow_inference_is_reported_as_degenerate():
    from dk1lab.policy import rtc_headroom

    delay, ok = rtc_headroom(0.5, fps=30, execution_horizon=10)
    assert delay == 15
    assert not ok


def test_the_delay_matches_rtcs_own_ceiling_arithmetic():
    """``rtc_headroom`` must round the way ``_rtc_loop`` does, or the warning lies."""
    import math

    from dk1lab.policy import rtc_headroom

    for latency in (0.001, 0.171, 0.272, 0.324, 0.333, 0.511, 1.0):
        delay, _ = rtc_headroom(latency, fps=30, execution_horizon=20)
        assert delay == math.ceil(latency / (1 / 30))


def test_the_degenerate_case_really_does_flatten_rtcs_weights():
    """Not a claim about our code — a check on the behaviour we are avoiding.

    With ``delay >= execution_horizon`` the linear schedule has no interior
    points left, so the weights become a step function: consecutive chunks are
    pinned hard for the horizon and then jump. With the delay inside the horizon
    there is a real ramp.
    """
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    processor = RTCProcessor(RTCConfig())

    degenerate = processor.get_prefix_weights(10, 10, 30).tolist()
    assert set(degenerate) == {0.0, 1.0}, "no intermediate weights: the blend is gone"

    healthy = processor.get_prefix_weights(10, 20, 30).tolist()
    assert len([w for w in healthy if 0.0 < w < 1.0]) >= 5


# --------------------------------------------------------------------------- #
# Homing at the end of a rollout
# --------------------------------------------------------------------------- #


@dataclass
class FakeHardware:
    initial_position: dict | None = None
    robot_wrapper: object = None


@dataclass
class FakeContext:
    hardware: FakeHardware = field(default_factory=FakeHardware)


def test_the_configured_home_pose_wins_over_the_start_pose():
    from dk1lab.config import HomePose
    from dk1lab.policy import home_target

    ctx = FakeContext(FakeHardware(initial_position=dict.fromkeys(ACTION_KEYS, 9.0)))
    pose = HomePose(left=(0.1,) * 7, right=(0.2,) * 7)
    assert home_target(ctx, pose)["left_joint_1.pos"] == 0.1


def test_without_a_configured_pose_home_falls_back_to_the_pose_at_connect():
    from dk1lab.policy import home_target

    ctx = FakeContext(FakeHardware(initial_position=dict.fromkeys(ACTION_KEYS, 0.5)))
    assert home_target(ctx, None) == dict.fromkeys(ACTION_KEYS, 0.5)


def test_homing_refuses_when_there_is_no_pose_at_all():
    from dk1lab.home import HomeError
    from dk1lab.policy import home_target

    with pytest.raises(HomeError, match="nothing to home to"):
        home_target(FakeContext(), None)


def test_a_start_pose_that_does_not_cover_the_robot_is_refused():
    """A short initial_position would sweep some joints and silently leave others."""
    from dk1lab.home import HomeError
    from dk1lab.policy import home_target

    partial = dict.fromkeys(ACTION_KEYS, 0.0)
    del partial["right_gripper.pos"]
    with pytest.raises(HomeError):
        home_target(FakeContext(FakeHardware(initial_position=partial)), None)


def test_the_home_sweep_does_not_take_the_policy_cap_as_a_speed(config, checkpoint):
    """The cap bounds a policy nobody trusts; it is not a speed to aim for.

    Reading it as one is why raising [limits.policy] from 0.3 to 1.0 rad/s sped
    the shutdown sweep up as a side effect.
    """
    from dk1lab.home import DEFAULT_HOME_RATE
    from dk1lab.policy import _home_rate

    cfg = rollout_config(config, checkpoint=checkpoint, task="t", limits=POLICY_LIMITS)
    assert POLICY_LIMITS.max_joint_rate > DEFAULT_HOME_RATE
    assert _home_rate(cfg) == DEFAULT_HOME_RATE


def test_a_cap_tighter_than_the_home_rate_still_wins(config, checkpoint):
    """Commanding faster than the limiter allows only means the limiter clamps it."""
    from dk1lab.policy import _home_rate

    limits = replace(POLICY_LIMITS, max_joint_rate=0.1)
    cfg = rollout_config(config, checkpoint=checkpoint, task="t", limits=limits)
    assert _home_rate(cfg) == 0.1


def test_an_uncapped_run_does_not_hand_its_lack_of_a_cap_to_the_home_sweep(config, checkpoint):
    """--no-limit is a deliberate act; a shutdown sweep is not the place to inherit it."""
    from dk1lab.home import DEFAULT_HOME_RATE
    from dk1lab.policy import _home_rate

    cfg = rollout_config(
        config, checkpoint=checkpoint, task="t", limits=POLICY_LIMITS.unlimited()
    )
    assert cfg.robot.max_joint_rate is None
    assert _home_rate(cfg) == DEFAULT_HOME_RATE


def test_lerobots_own_return_to_initial_position_stays_off(config, checkpoint):
    """It fires from teardown on every exit path, sweeps blind for a fixed 3 s,
    and targets the connect-time pose. dk1lab.home replaces it; it must not also
    run, or the arms would be swept twice, the second time without a rate that
    the limiter can satisfy."""
    cfg = rollout_config(config, checkpoint=checkpoint, task="t")
    assert cfg.return_to_initial_position is False


def test_the_sweep_reads_the_motors_and_not_the_cameras():
    """A 30 Hz sweep that called get_observation would grab three camera frames
    per tick and look at none of them."""
    from dk1lab.policy import measure_fn

    class Inner:
        def __init__(self):
            self.reads = 0

        def measured_positions(self):
            self.reads += 1
            return dict.fromkeys(ACTION_KEYS, 0.0)

        def get_observation(self):
            raise AssertionError("get_observation reads the cameras")

    class Wrapper:
        def __init__(self, inner):
            self.inner = inner

        def get_observation(self):
            return self.inner.get_observation()

    inner = Inner()
    assert measure_fn(Wrapper(inner))() == dict.fromkeys(ACTION_KEYS, 0.0)
    assert inner.reads == 1


def test_a_robot_without_measured_positions_still_gets_a_reader():
    from dk1lab.policy import measure_fn

    class Plain:
        def get_observation(self):
            return {**dict.fromkeys(ACTION_KEYS, 0.25), "observation.images.top": object()}

    assert measure_fn(Plain())() == dict.fromkeys(ACTION_KEYS, 0.25)


def test_a_run_that_faulted_is_not_swept_home():
    """Commanding motion into a failure nobody has looked at is not stopping."""
    from dk1lab.policy import ended_cleanly

    assert not ended_cleanly(RuntimeError("camera timed out"))


def test_the_duration_limit_and_ctrl_c_both_count_as_a_clean_end():
    from dk1lab.policy import ended_cleanly

    assert ended_cleanly(None)
    assert ended_cleanly(KeyboardInterrupt())
