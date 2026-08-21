"""The home sweep: every property that decides whether it is safe, tested dry.

The sweep drives both arms with nobody's hand on a leader, so what matters is
not that it moves but *how it stops*: on arrival, on a timeout it derived from
the actual distance, or on an interrupt — and that it reports which of the three
happened rather than raising, because the caller is about to disconnect and
de-energise the motors either way.

``go_home`` takes ``measure``/``send`` callables and a clock, so all of it runs
here with no robot, no LeRobot and no real time.
"""

from __future__ import annotations

import pytest

from dk1lab.home import (
    DEFAULT_EASE_OUT_RAD,
    DEFAULT_HOME_RATE,
    DEFAULT_TOLERANCE,
    HomeError,
    HomeReport,
    MAX_TIMEOUT_S,
    ease_scale,
    estimate_duration,
    farthest,
    go_home,
    home_rate,
    smoothstep,
    step_toward,
    validate_target,
)
from dk1lab.layout import ACTION_KEYS, GRIPPER_INDICES, is_gripper


def k(joint: str) -> str:
    """``left_joint_1`` -> the real action key. The suffix is layout's, not ours."""
    return f"{joint}.pos"


def pose(value: float = 0.0, **overrides: float) -> dict[str, float]:
    """A full 14-D pose at ``value``, with named joints overridden.

    Overrides are given as ``left_joint_1=...`` and mapped onto the real
    ``left_joint_1.pos`` keys, so the tests stay readable while still driving the
    exact key set :mod:`dk1lab.layout` defines.
    """
    out = dict.fromkeys(ACTION_KEYS, value)
    out.update({k(joint): amount for joint, amount in overrides.items()})
    return out


class FakeArms:
    """An arm pair that follows commands perfectly, one tick late.

    Deliberately not instant: a follower that snapped to every command would hide
    the difference between a sweep that ramps and one that steps.
    """

    def __init__(self, start: dict[str, float], *, stuck: set[str] | None = None):
        self.position = dict(start)
        self.stuck = stuck or set()
        self.sent: list[dict[str, float]] = []

    def measure(self) -> dict[str, float]:
        return dict(self.position)

    def send(self, command: dict[str, float]) -> None:
        self.sent.append(dict(command))
        for key, value in command.items():
            if key not in self.stuck:
                self.position[key] = value


class FakeClock:
    """Monotonic time that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def sweep(arms: FakeArms, target: dict[str, float], **kwargs) -> HomeReport:
    """Run a sweep on fake arms and a fake clock, **with the easing off**.

    The easing is a speed profile laid over the ramp; switching it off here keeps
    every test of the ramp itself — rate, timeout, arrival, abort — reading as
    plain distance over speed. :func:`eased_sweep` is the one that leaves it on.
    """
    kwargs.setdefault("ease_in_s", 0.0)
    kwargs.setdefault("ease_out_rad", 0.0)
    return eased_sweep(arms, target, **kwargs)


def eased_sweep(arms: FakeArms, target: dict[str, float], **kwargs) -> HomeReport:
    clock = FakeClock()
    return go_home(
        measure=arms.measure,
        send=arms.send,
        target=target,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def joint_speeds(arms: FakeArms, key: str, fps: float) -> list[float]:
    """Commanded speed of one joint, tick by tick, rad/s."""
    values = [command[key] for command in arms.sent]
    return [abs(b - a) * fps for a, b in zip(values, values[1:])]


# --------------------------------------------------------------------------- #
# The target
# --------------------------------------------------------------------------- #


def test_a_home_pose_must_cover_every_joint():
    partial = pose()
    del partial[k("right_joint_4")]
    with pytest.raises(HomeError, match="missing"):
        validate_target(partial)


def test_a_home_pose_may_not_carry_keys_the_robot_does_not_have():
    with pytest.raises(HomeError, match="unknown"):
        validate_target(pose(elbow=0.0))


def test_a_validated_pose_is_in_layout_order():
    assert list(validate_target(pose())) == list(ACTION_KEYS)


# --------------------------------------------------------------------------- #
# The ramp
# --------------------------------------------------------------------------- #


def test_the_ramp_moves_at_most_one_step_per_tick():
    stepped = step_toward(pose(0.0), pose(1.0), joint_step=0.01, gripper_step=0.05)
    assert stepped[k("left_joint_1")] == pytest.approx(0.01)


def test_the_ramp_does_not_overshoot_a_target_within_one_step():
    stepped = step_toward(pose(0.0), pose(0.004), joint_step=0.01, gripper_step=0.05)
    assert stepped[k("left_joint_1")] == pytest.approx(0.004)


def test_the_ramp_goes_both_ways():
    stepped = step_toward(pose(0.0), pose(-1.0), joint_step=0.01, gripper_step=0.05)
    assert stepped[k("left_joint_1")] == pytest.approx(-0.01)


def test_the_gripper_gets_its_own_step_size():
    gripper = ACTION_KEYS[GRIPPER_INDICES[0]]
    stepped = step_toward(pose(0.0), pose(1.0), joint_step=0.01, gripper_step=0.05)
    assert stepped[gripper] == pytest.approx(0.05)


def test_the_ramp_starts_from_the_previous_command_not_the_measurement():
    # The same reason the limiter does it: clamping to the measured position
    # deadlocks under stiction, because the error never grows large enough to
    # produce a torque that breaks friction.
    first = step_toward(pose(0.0), pose(1.0), joint_step=0.01, gripper_step=0.05)
    second = step_toward(first, pose(1.0), joint_step=0.01, gripper_step=0.05)
    assert second[k("left_joint_1")] == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Arrival
# --------------------------------------------------------------------------- #


def test_arrival_is_judged_on_arm_joints_only():
    gripper = ACTION_KEYS[GRIPPER_INDICES[0]]
    measured = pose(0.0, **{gripper: 1.0})
    key, error = farthest(measured, pose(0.0))
    assert not is_gripper(key)
    assert error == pytest.approx(0.0)


def test_a_gripper_stalled_against_an_object_does_not_hold_the_sweep_open():
    gripper = ACTION_KEYS[GRIPPER_INDICES[0]]
    target = pose(0.0)
    target[gripper] = 1.0
    arms = FakeArms(pose(0.0), stuck={gripper})
    report = sweep(arms, target)
    assert report.reached


def test_the_sweep_reports_the_joint_that_is_furthest_out():
    key, error = farthest(pose(0.0, right_joint_2=0.5), pose(0.0))
    assert key == k("right_joint_2")
    assert error == pytest.approx(0.5)


def test_a_sweep_that_is_already_home_sends_nothing():
    arms = FakeArms(pose(0.0))
    report = sweep(arms, pose(0.0))
    assert report.reached
    assert arms.sent == []


def test_the_sweep_drives_every_joint_to_the_target():
    arms = FakeArms(pose(0.0))
    target = pose(0.0, left_joint_1=0.3, right_joint_5=-0.2)
    report = sweep(arms, target)
    assert report.reached
    for key in ACTION_KEYS:
        assert arms.position[key] == pytest.approx(target[key], abs=DEFAULT_TOLERANCE)


def test_the_sweep_respects_the_rate_it_was_given():
    arms = FakeArms(pose(0.0))
    report = sweep(arms, pose(0.0, left_joint_1=0.6), rate=0.3, fps=30)
    # 0.6 rad at 0.3 rad/s is two seconds of travel — less the tolerance, which
    # the sweep stops inside rather than chasing the last 0.03 rad.
    expected = (0.6 - DEFAULT_TOLERANCE) / 0.3
    assert report.elapsed_s == pytest.approx(expected, abs=0.1)
    assert report.steps == pytest.approx(expected * 30, abs=2)


# --------------------------------------------------------------------------- #
# The speed profile: slow, and eased at both ends
# --------------------------------------------------------------------------- #


def test_the_sweep_speed_is_not_the_policy_cap():
    """A cap is an upper bound on a policy, not a speed for a shutdown sweep."""
    assert home_rate(1.0) == DEFAULT_HOME_RATE
    assert DEFAULT_HOME_RATE <= 0.5


def test_a_tighter_cap_still_wins_over_the_home_rate():
    assert home_rate(0.1) == 0.1


def test_no_cap_at_all_does_not_mean_any_speed():
    # `false` in dk1.toml is a deliberate act for the activity it was set on.
    assert home_rate(None) == DEFAULT_HOME_RATE
    assert home_rate(False) == DEFAULT_HOME_RATE


def test_the_smoothstep_is_flat_at_both_ends_and_clamped_outside():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)
    assert smoothstep(-3.0) == 0.0
    assert smoothstep(4.0) == 1.0
    # Flat at the ends is the whole point: no corner in the velocity.
    assert smoothstep(0.05) < 0.05
    assert smoothstep(0.95) > 0.95


def test_the_easing_starts_slow_speeds_up_and_slows_down_again():
    mid = dict(elapsed=10.0, remaining=10.0)
    assert ease_scale(**mid) == pytest.approx(1.0)
    assert ease_scale(elapsed=0.0, remaining=10.0) < 0.5
    assert ease_scale(elapsed=10.0, remaining=0.0) < 0.5


def test_the_easing_never_returns_zero():
    """A profile that reaches zero never leaves the start and never arrives."""
    assert ease_scale(elapsed=0.0, remaining=0.0) > 0.0


def test_the_tighter_of_the_two_ramps_wins():
    """A sweep too short for both must not have them fight over it."""
    both = ease_scale(elapsed=0.1, remaining=0.01)
    assert both == pytest.approx(
        min(ease_scale(elapsed=0.1, remaining=99.0), ease_scale(elapsed=99.0, remaining=0.01))
    )


def test_the_eased_sweep_ramps_up_from_a_standstill_and_settles_at_the_end():
    arms = FakeArms(pose(0.0))
    key = k("left_joint_1")
    report = eased_sweep(arms, pose(0.0, left_joint_1=2.0), rate=0.3, fps=30)
    assert report.reached
    speeds = joint_speeds(arms, key, 30.0)

    peak = max(speeds)
    assert peak == pytest.approx(0.3, abs=0.01)
    # Starts and ends well under the peak, and reaches it in between.
    assert speeds[0] < 0.4 * peak
    assert speeds[-1] < 0.4 * peak
    assert max(speeds[len(speeds) // 3 : 2 * len(speeds) // 3]) == pytest.approx(peak, abs=0.01)


def test_the_eased_sweep_has_no_step_change_in_speed():
    """Nowhere does the commanded speed jump — that is what "smooth" means here."""
    arms = FakeArms(pose(0.0))
    eased_sweep(arms, pose(0.0, left_joint_1=2.0), rate=0.3, fps=30)
    speeds = joint_speeds(arms, k("left_joint_1"), 30.0)
    assert max(abs(b - a) for a, b in zip(speeds, speeds[1:])) < 0.05


def test_the_easing_makes_the_sweep_slower_not_faster():
    plain = FakeArms(pose(0.0))
    eased = FakeArms(pose(0.0))
    target = pose(0.0, left_joint_1=2.0)
    fast = sweep(plain, target, rate=0.3, fps=30)
    slow = eased_sweep(eased, target, rate=0.3, fps=30)
    assert slow.reached and fast.reached
    assert slow.elapsed_s > fast.elapsed_s


def test_homing_is_at_least_twice_as_slow_as_it_was_at_the_policy_cap():
    """The whole point of this round: 1.0 rad/s flat out was too fast to watch."""
    old = FakeArms(pose(0.0))
    new = FakeArms(pose(0.0))
    target = pose(0.0, left_joint_1=2.0)
    before = sweep(old, target, rate=1.0, fps=30)
    after = eased_sweep(new, target, rate=home_rate(1.0), fps=30)
    assert after.reached
    assert after.elapsed_s >= 2.0 * before.elapsed_s


def test_the_eased_sweep_still_arrives_rather_than_creeping_to_its_timeout():
    # A short sweep is inside the ease-out zone for its whole length, which is
    # where an easing without a floor would stall short of the tolerance.
    arms = FakeArms(pose(0.0))
    report = eased_sweep(
        arms, pose(0.0, left_joint_1=DEFAULT_EASE_OUT_RAD * 0.5), rate=0.3, fps=30
    )
    assert report.reached


def test_the_timeout_leaves_room_for_the_easing():
    """The easing lengthens the sweep, so a timeout blind to it would cut it off."""
    arms = FakeArms(pose(0.0))
    report = eased_sweep(arms, pose(0.0, left_joint_1=2.0), rate=0.3, fps=30)
    assert report.reached


# --------------------------------------------------------------------------- #
# Not arriving — the outcome that matters, because disconnect follows
# --------------------------------------------------------------------------- #


def test_a_blocked_joint_times_out_rather_than_sweeping_forever():
    arms = FakeArms(pose(0.0), stuck={k("left_joint_1")})
    report = sweep(arms, pose(0.0, left_joint_1=0.5))
    assert not report.reached
    assert not report.aborted
    assert report.worst_key == k("left_joint_1")


def test_not_arriving_is_reported_and_not_raised():
    # The caller is about to disconnect, which disables every motor. It needs to
    # print that the arms are not home, not to lose the message to a traceback.
    arms = FakeArms(pose(0.0), stuck={k("left_joint_1")})
    report = sweep(arms, pose(0.0, left_joint_1=0.5))
    assert "NOT reached" in report.summary()


def test_the_timeout_comes_from_the_distance_and_not_from_a_fixed_schedule():
    # LeRobot's return-to-initial sweeps for exactly 3 s whatever the distance,
    # which behind a 0.3 rad/s cap cannot finish anything over ~0.9 rad. A long
    # sweep here gets a proportionally long timeout.
    short = estimate_duration(pose(0.0), pose(0.0, left_joint_1=0.3), rate=0.3)
    long = estimate_duration(pose(0.0), pose(0.0, left_joint_1=3.0), rate=0.3)
    assert short == pytest.approx(1.0)
    assert long == pytest.approx(10.0)


def test_a_long_sweep_still_finishes_under_a_rate_that_would_beat_a_fixed_three_seconds():
    arms = FakeArms(pose(0.0))
    report = sweep(arms, pose(0.0, left_joint_1=2.0), rate=0.3, fps=30)
    assert report.reached
    assert report.elapsed_s > 3.0


def test_no_sweep_may_run_longer_than_the_hard_cap():
    arms = FakeArms(pose(0.0), stuck={k("left_joint_1")})
    report = sweep(arms, pose(0.0, left_joint_1=1000.0), rate=0.3)
    assert not report.reached
    assert report.elapsed_s <= MAX_TIMEOUT_S + 1.0


def test_an_explicit_timeout_wins_over_the_derived_one():
    arms = FakeArms(pose(0.0), stuck={k("left_joint_1")})
    report = sweep(arms, pose(0.0, left_joint_1=5.0), rate=0.3, timeout_s=1.0)
    assert not report.reached
    assert report.elapsed_s == pytest.approx(1.0, abs=0.1)


def test_a_zero_or_negative_rate_is_refused_rather_than_hanging():
    arms = FakeArms(pose(0.0))
    with pytest.raises(HomeError):
        sweep(arms, pose(0.0, left_joint_1=1.0), rate=0.0)


# --------------------------------------------------------------------------- #
# Getting out
# --------------------------------------------------------------------------- #


def test_an_abort_stops_commanding_immediately():
    arms = FakeArms(pose(0.0))
    calls = {"n": 0}

    def abort() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    report = sweep(arms, pose(0.0, left_joint_1=1.0), should_abort=abort)
    assert report.aborted
    assert not report.reached
    assert len(arms.sent) == 3


def test_an_abort_is_reported_differently_from_a_timeout():
    arms = FakeArms(pose(0.0))
    report = sweep(arms, pose(0.0, left_joint_1=1.0), should_abort=lambda: True)
    assert "ABORTED" in report.summary()
    assert arms.sent == []
