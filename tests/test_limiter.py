"""The slew-rate limiter — the safety knob that replaces a dead one.

A fake clock drives every test, so rates are exercised exactly rather than
approximately, and nothing sleeps.
"""

from __future__ import annotations

import pytest

from dk1lab.limiter import SlewLimiter


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make(clock: FakeClock, **kwargs) -> SlewLimiter:
    kwargs.setdefault("max_joint_rate", 1.0)  # rad/s — 1 rad per second
    kwargs.setdefault("max_gripper_rate", 2.0)
    kwargs.setdefault("max_lag", 10.0)  # effectively off unless a test sets it
    return SlewLimiter(clock=clock, **kwargs)


JOINT = "left_joint_1.pos"
GRIPPER = "left_gripper.pos"


def test_the_first_command_holds_the_measured_position(clock):
    """The first action after a connect or reset must never be a step.

    There is no previous timestamp to measure against, so no elapsed time can be
    claimed and the command comes out exactly where the arm already is.
    """
    limiter = make(clock)
    clock.advance(0.1)
    assert limiter.limit({JOINT: 5.0}, {JOINT: 0.0})[JOINT] == pytest.approx(0.0)


def test_motion_begins_on_the_second_tick(clock):
    limiter = make(clock)
    limiter.limit({JOINT: 5.0}, {JOINT: 0.0})  # seeding tick
    clock.advance(0.1)
    assert limiter.limit({JOINT: 5.0}, {JOINT: 0.0})[JOINT] == pytest.approx(0.1)


def test_rate_is_per_second_not_per_command(clock):
    """Same requested move, different loop rate, same resulting speed.

    A per-command step size would mean teleoperation at 200 Hz and a policy at
    30 Hz silently ran at different speeds under the same setting.
    """

    def travel_after(period: float, ticks: int) -> float:
        c = FakeClock()
        limiter = make(c)
        limiter.limit({JOINT: 100.0}, {JOINT: 0.0})  # seeding tick
        for _ in range(ticks):
            c.advance(period)
            command = limiter.limit({JOINT: 100.0}, {JOINT: 0.0})[JOINT]
        return command

    # One second of wall time, at 100 Hz and at 10 Hz.
    assert travel_after(0.01, 100) == pytest.approx(1.0)
    assert travel_after(0.1, 10) == pytest.approx(1.0)


def test_a_far_target_advances_at_the_cap_each_tick(clock):
    limiter = make(clock)
    measured = {JOINT: 0.0}
    commands = []
    for _ in range(6):
        clock.advance(0.1)
        commands.append(limiter.limit({JOINT: 100.0}, measured)[JOINT])
    # First entry is the seeding tick (holds at the measured 0.0).
    assert commands == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_limiting_is_symmetric_for_negative_moves(clock):
    limiter = make(clock)
    limiter.limit({JOINT: 0.0}, {JOINT: 0.0})  # seeding tick
    clock.advance(0.1)
    assert limiter.limit({JOINT: -100.0}, {JOINT: 0.0})[JOINT] == pytest.approx(-0.1)


def test_a_target_within_the_step_passes_through_untouched(clock):
    limiter = make(clock)
    limiter.limit({JOINT: 0.0}, {JOINT: 0.0})  # seeding tick
    clock.advance(0.1)  # 0.1 s at 1 rad/s allows 0.1 rad of travel
    assert limiter.limit({JOINT: 0.05}, {JOINT: 0.0})[JOINT] == pytest.approx(0.05)


def test_the_ramp_advances_from_the_previous_command_not_the_measurement(clock):
    """Ramping from the measurement deadlocks under stiction.

    A joint that has not physically moved must still see its setpoint advance,
    so position error — and therefore torque — can build until it breaks free.
    """
    limiter = make(clock)
    stuck = {JOINT: 0.0}  # the arm never moves
    limiter.limit({JOINT: 100.0}, stuck)  # seeding tick
    commands = []
    for _ in range(4):
        clock.advance(0.1)
        commands.append(limiter.limit({JOINT: 100.0}, stuck)[JOINT])
    assert commands == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert commands[-1] > commands[0]


def test_max_lag_stops_a_blocked_joint_from_winding_up(clock):
    """Without this, a blocked arm stores up travel and lunges when freed."""
    limiter = make(clock, max_lag=0.25)
    stuck = {JOINT: 0.0}
    for _ in range(21):
        clock.advance(0.1)
        command = limiter.limit({JOINT: 100.0}, stuck)[JOINT]
    assert command == pytest.approx(0.25)


def test_max_lag_is_not_applied_to_the_gripper(clock):
    """The gripper is meant to stall against what it holds; clamping fights it."""
    limiter = make(clock, max_lag=0.01)
    stuck = {GRIPPER: 0.0}
    for _ in range(11):
        clock.advance(0.1)
        command = limiter.limit({GRIPPER: 1.0}, stuck)[GRIPPER]
    assert command == pytest.approx(1.0)


def test_gripper_uses_its_own_rate(clock):
    limiter = make(clock, max_joint_rate=1.0, max_gripper_rate=2.0)
    limiter.limit({JOINT: 0.0, GRIPPER: 0.0}, {JOINT: 0.0, GRIPPER: 0.0})  # seeding tick
    clock.advance(0.1)
    out = limiter.limit({JOINT: 10.0, GRIPPER: 10.0}, {JOINT: 0.0, GRIPPER: 0.0})
    assert out[JOINT] == pytest.approx(0.1)
    assert out[GRIPPER] == pytest.approx(0.2)


def test_a_stalled_caller_cannot_buy_a_large_step(clock):
    """A loop that hangs must not come back entitled to a big jump."""
    limiter = make(clock, max_dt=0.1)
    limiter.limit({JOINT: 0.0}, {JOINT: 0.0})  # seeding tick
    clock.advance(5.0)  # a five-second stall
    out = limiter.limit({JOINT: 100.0}, {JOINT: 0.0})
    assert out[JOINT] == pytest.approx(0.1)  # 0.1 s worth, not 5 s worth


def test_all_fourteen_channels_are_limited(clock):
    from dk1lab.layout import ACTION_KEYS

    limiter = make(clock)
    target = dict.fromkeys(ACTION_KEYS, 100.0)
    measured = dict.fromkeys(ACTION_KEYS, 0.0)
    limiter.limit(target, measured)  # seeding tick
    clock.advance(0.1)
    out = limiter.limit(target, measured)
    assert set(out) == set(ACTION_KEYS)
    assert all(value < 1.0 for value in out.values())


def test_reset_reseeds_the_ramp(clock):
    limiter = make(clock)
    limiter.limit({JOINT: 100.0}, {JOINT: 0.0})
    clock.advance(0.1)
    limiter.limit({JOINT: 100.0}, {JOINT: 0.0})
    limiter.reset()
    clock.advance(0.1)
    # After a reset this is a first tick again: it holds at the measurement.
    assert limiter.limit({JOINT: 100.0}, {JOINT: 5.0})[JOINT] == pytest.approx(5.0)


def test_disabling_passes_targets_straight_through(clock):
    limiter = make(clock, max_joint_rate=None)
    assert not limiter.enabled
    clock.advance(0.001)
    assert limiter.limit({JOINT: 100.0}, {JOINT: 0.0})[JOINT] == 100.0


def test_missing_measurement_still_limits_the_ramp(clock):
    """No measurement means no lag clamp, but the rate cap still applies."""
    limiter = make(clock)
    assert limiter.limit({JOINT: 100.0})[JOINT] == pytest.approx(100.0)  # seeds at target
    clock.advance(0.1)
    assert limiter.limit({JOINT: 200.0})[JOINT] == pytest.approx(100.1)


def test_non_numeric_values_pass_through(clock):
    limiter = make(clock)
    limiter.limit({JOINT: 0.0}, {JOINT: 0.0})  # seeding tick
    clock.advance(0.1)
    out = limiter.limit({JOINT: 100.0, "task": "pick up the pen"}, {JOINT: 0.0})
    assert out["task"] == "pick up the pen"


def test_the_input_mapping_is_not_mutated(clock):
    limiter = make(clock)
    limiter.limit({JOINT: 0.0}, {JOINT: 0.0})  # seeding tick
    clock.advance(0.1)
    target = {JOINT: 100.0}
    limiter.limit(target, {JOINT: 0.0})
    assert target == {JOINT: 100.0}
