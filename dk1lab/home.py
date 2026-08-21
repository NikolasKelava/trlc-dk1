"""Drive the followers to a home pose — deliberately, and at a speed you chose.

Why this exists rather than ``return_to_initial_position``
---------------------------------------------------------
LeRobot's rollout has a built-in return-to-home
(``RolloutStrategy._return_to_initial_position``). Three of its properties make
it the wrong thing for this cell:

**Its target is whatever pose the arms were in when the rollout connected.** That
is wherever the last run happened to leave them, not a home.

**It sweeps for a fixed 3 s at 50 Hz and then stops, reached or not.** It
interpolates from the current pose to the target over exactly ``duration_s * fps``
commands and never looks at where the arms actually got to. Behind a rate limiter
— which is the whole point of :class:`~dk1lab.robot.SafeBiDK1Follower` — a sweep
longer than 0.9 rad at ``[limits.policy]``'s 0.3 rad/s cannot finish in 3 s, so
the arms stop partway and ``disconnect()`` then de-energises every motor. A raised
arm sags from wherever it got to. This module drives until the arms *arrive*,
verified against the measurement, with a timeout as the backstop rather than as
the schedule.

**It fires from ``teardown``, on every exit path**, including a crash. Homing is
commanded motion; when it happens is a decision, so it is made by the caller
(:func:`dk1lab.policy.run`) and not by a config flag buried in a shutdown path.

Design
------
The sweep is the same shape as the limiter's: **ramp from the previous command,
not from the measurement**, so stiction cannot deadlock the setpoint, and so the
limiter downstream has nothing left to clamp. Arrival is judged on the *measured*
arm joints.

**The speed is eased at both ends** rather than being one constant rate. A
constant-rate ramp is a step change in velocity at the start and another at the
end — the arms snap into motion and stop dead on the target, which is what made
homing feel fast even before it was. :func:`ease_scale` scales the step by a
smoothstep in time at the start and a smoothstep in *distance to go* at the end,
and takes whichever is smaller, so a sweep too short for both simply never
reaches full speed. The peak rate itself comes from :func:`home_rate` and is no
longer inherited from the policy's speed cap. The grippers are commanded to their home value but excluded from the
arrival test — a gripper is supposed to stall against whatever it is holding, and
waiting for one to reach a number is how a home sweep hangs until its timeout.

:func:`go_home` takes no robot type at all — the caller passes a ``measure`` and
a ``send`` callable, which is why every property of the sweep is tested without
hardware. The two convenience entry points at the bottom do build a follower, and
import LeRobot inside the function bodies so this module stays cheap to import.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .layout import ACTION_KEYS, is_gripper

logger = logging.getLogger(__name__)

#: Peak joint speed of the home sweep, rad/s. **This is a target, not a cap** —
#: :func:`home_rate` takes the smaller of it and whatever ``[limits.policy]``
#: allows, so raising the policy cap never speeds the sweep up.
#:
#: It used to *be* the policy cap, which meant the 2026-08-20 raise from 0.3 to
#: 1.0 rad/s silently tripled the speed of homing as well. Homing is commanded
#: motion with nobody's hand on a leader arm and nothing to react to; the cap
#: exists so a policy cannot lunge, and inheriting it as a *speed* was reading an
#: upper bound as an instruction.
DEFAULT_HOME_RATE: float = 0.3

#: Gripper speed during the sweep, normalised units/s. Neither slowed nor eased:
#: the gripper is already at its home value in the ordinary case, and a slow
#: gripper is not the hazard here.
DEFAULT_HOME_GRIPPER_RATE: float = 1.0

#: Seconds the sweep spends winding up from a standstill to the full rate.
DEFAULT_EASE_IN_S: float = 0.75

#: How far out from home, in rad, the sweep starts slowing down again.
DEFAULT_EASE_OUT_RAD: float = 0.25

#: The slowest the eased sweep ever goes, as a fraction of the full rate. Nonzero
#: on purpose: a profile that reaches zero at either end never leaves the start
#: and never arrives at the end, it only approaches one asymptotically until the
#: timeout fires.
DEFAULT_EASE_FLOOR: float = 0.2

#: How close each arm joint must be to its home value to count as arrived, rad.
#: ~1.7 deg. Tighter than this and stiction alone can hold a joint just outside
#: it until the timeout.
DEFAULT_TOLERANCE: float = 0.03

#: Command rate of the sweep, Hz. The follower is read every tick, so this is
#: also a serial round-trip rate; 30 Hz matches the control loop it follows.
DEFAULT_HOME_FPS: float = 30.0

#: Slack on the estimated sweep time before the timeout fires: the sweep is
#: allowed this multiple of the distance-over-rate estimate, plus
#: :data:`TIMEOUT_MARGIN_S`.
TIMEOUT_SLACK: float = 2.0

#: Constant added to every computed timeout, seconds.
TIMEOUT_MARGIN_S: float = 2.0

#: No home sweep may run longer than this, seconds, whatever the estimate says.
MAX_TIMEOUT_S: float = 60.0


class HomeError(ValueError):
    """Raised for a home pose that is not a usable target."""


@dataclass(frozen=True)
class HomeReport:
    """What the sweep did. Returned rather than logged, so callers can report it."""

    reached: bool
    aborted: bool
    steps: int
    elapsed_s: float
    worst_key: str
    worst_error: float

    def summary(self) -> str:
        worst = f"worst joint {self.worst_key} {self.worst_error:+.3f} rad"
        if self.aborted:
            return f"home sweep ABORTED after {self.elapsed_s:.1f}s — {worst}"
        if self.reached:
            return f"home reached in {self.elapsed_s:.1f}s ({self.steps} commands), {worst}"
        return (
            f"home NOT reached — timed out after {self.elapsed_s:.1f}s "
            f"({self.steps} commands), {worst}"
        )


def validate_target(target: Mapping[str, float]) -> dict[str, float]:
    """Check a home pose covers the 14-D layout exactly, and return it as floats.

    Raises:
        HomeError: naming the missing or unexpected keys. A home pose missing a
            joint would leave that joint wherever the policy left it while the
            other thirteen move, which looks like homing and is not.
    """
    missing = [key for key in ACTION_KEYS if key not in target]
    if missing:
        raise HomeError(f"home pose is missing {len(missing)} keys: {missing}")
    unexpected = [key for key in target if key not in ACTION_KEYS]
    if unexpected:
        raise HomeError(f"home pose has unknown keys: {unexpected}")
    bad = [key for key, value in target.items() if not isinstance(value, (int, float))]
    if bad or any(isinstance(target[key], bool) for key in target):
        raise HomeError(f"home pose values must be numbers; offenders: {bad or list(target)}")
    return {key: float(target[key]) for key in ACTION_KEYS}


def step_toward(
    previous: Mapping[str, float],
    target: Mapping[str, float],
    *,
    joint_step: float,
    gripper_step: float,
) -> dict[str, float]:
    """One tick of the ramp: ``previous`` moved at most one step toward ``target``.

    Ramping from the previous *command* rather than the measured position is the
    same choice :mod:`dk1lab.limiter` makes, and for the same reason: clamping to
    the measurement deadlocks under stiction, because the error never grows large
    enough to produce a torque that breaks friction.
    """
    out: dict[str, float] = {}
    for key, goal in target.items():
        prev = float(previous.get(key, goal))
        step = gripper_step if is_gripper(key) else joint_step
        delta = float(goal) - prev
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        out[key] = prev + delta
    return out


def farthest(
    measured: Mapping[str, float], target: Mapping[str, float]
) -> tuple[str, float]:
    """The arm joint furthest from its home value, and its signed error.

    Grippers are excluded: they are meant to stall, so a gripper holding an
    object would keep a sweep running until its timeout every single time.
    """
    joints = [key for key in target if not is_gripper(key) and key in measured]
    if not joints:
        return ("", 0.0)
    key = max(joints, key=lambda k: abs(float(measured[k]) - float(target[k])))
    return key, float(measured[key]) - float(target[key])


def home_rate(cap: float | None) -> float:
    """The speed to sweep at, given whatever speed limit is in force.

    :data:`DEFAULT_HOME_RATE` unless the limiter is set tighter, in which case
    the limiter wins — commanding faster than the limiter allows only means the
    limiter clamps it, and then the ramp here and the ramp there disagree about
    what was commanded.

    ``cap`` may be ``None`` or ``False``, which is how ``dk1.toml`` spells "no
    limiting at all". That is a deliberate act for the activity it was set on;
    it is not permission for a shutdown sweep to move at any speed it likes.
    """
    if not cap:
        return DEFAULT_HOME_RATE
    return min(DEFAULT_HOME_RATE, float(cap))


def smoothstep(x: float) -> float:
    """The classic ``3x^2 - 2x^3``, clamped to ``[0, 1]``.

    Zero slope at both ends, which is the whole point: the speed profile built
    from it has no corner where the arms would jerk.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def ease_scale(
    *,
    elapsed: float,
    remaining: float,
    ease_in_s: float = DEFAULT_EASE_IN_S,
    ease_out_rad: float = DEFAULT_EASE_OUT_RAD,
    floor: float = DEFAULT_EASE_FLOOR,
) -> float:
    """Fraction of the full joint rate to move at, this tick.

    Two independent ramps, and the tighter of the two wins: one in *time* from
    the start of the sweep, one in *distance* from the target. Taking the minimum
    means a sweep shorter than both simply never reaches full speed, instead of
    the ease-in and the ease-out fighting over it.

    Args:
        elapsed: seconds since the sweep started.
        remaining: how far the joint with furthest to go still has to travel, rad.
            Measured in **command** space, not against the measurement: the ramp
            is driven from the previous command, so that is the quantity that
            actually reaches zero. The measurement lags it and would keep the
            profile pinned at the floor for the whole settling time.
        floor: the smallest fraction returned. Keeps every tick making progress,
            which is what bounds the sweep's duration and lets the timeout
            estimate stay honest.

    Returns:
        A value in ``[floor, 1.0]``.
    """
    scale = 1.0
    if ease_in_s > 0.0:
        scale = min(scale, smoothstep(elapsed / ease_in_s))
    if ease_out_rad > 0.0:
        scale = min(scale, smoothstep(abs(remaining) / ease_out_rad))
    return floor + (1.0 - floor) * scale


def ease_overhead(
    rate: float,
    *,
    ease_in_s: float = DEFAULT_EASE_IN_S,
    ease_out_rad: float = DEFAULT_EASE_OUT_RAD,
) -> float:
    """Seconds the easing adds to a constant-rate sweep, generously rounded up.

    Half of ``ease_in_s`` is lost winding up; the ease-out tail costs a bit over
    twice what its distance would cost at full rate. Both are over-estimated
    here on purpose — this feeds a timeout, and a timeout that fires early on a
    sweep that was going to arrive is the failure mode worth avoiding.
    """
    if rate <= 0.0:
        return 0.0
    return 0.5 * ease_in_s + 3.0 * ease_out_rad / rate


def estimate_duration(
    measured: Mapping[str, float], target: Mapping[str, float], rate: float
) -> float:
    """Seconds the sweep needs at ``rate``, from the joint that has furthest to go."""
    _, error = farthest(measured, target)
    return abs(error) / rate if rate > 0 else 0.0


def go_home(
    *,
    measure: Callable[[], Mapping[str, float]],
    send: Callable[[Mapping[str, float]], object],
    target: Mapping[str, float],
    rate: float = DEFAULT_HOME_RATE,
    gripper_rate: float = DEFAULT_HOME_GRIPPER_RATE,
    fps: float = DEFAULT_HOME_FPS,
    tolerance: float = DEFAULT_TOLERANCE,
    ease_in_s: float = DEFAULT_EASE_IN_S,
    ease_out_rad: float = DEFAULT_EASE_OUT_RAD,
    ease_floor: float = DEFAULT_EASE_FLOOR,
    timeout_s: float | None = None,
    should_abort: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> HomeReport:
    """Ramp the arms to ``target`` and return what happened. **Moves the arms.**

    Args:
        measure: current positions, keyed as in :mod:`dk1lab.layout`. Called every
            tick — pass something that does not read the cameras.
        send: what to command. Whatever rate limiting sits behind this is applied
            on top; the ramp here is sized to stay under it, not to fight it.
        target: the home pose, all 14 keys.
        rate: arm joint *peak* speed, rad/s. The easing means the sweep only
            reaches it in the middle; see ``ease_in_s``.
        gripper_rate: gripper speed, normalised units/s. Not eased.
        fps: command rate.
        tolerance: per-joint arrival threshold, rad.
        ease_in_s: seconds spent winding up to ``rate`` from a standstill, so
            the sweep does not start with a step change in velocity.
        ease_out_rad: distance from home at which the sweep starts slowing
            down again, so it settles onto the target instead of arriving at
            full speed and stopping dead. Set either to 0 for the old
            constant-rate ramp.
        ease_floor: the slowest the easing goes, as a fraction of ``rate``.
        timeout_s: give up after this long. ``None`` derives it from the distance
            actually to travel, so a short sweep does not sit waiting and a long
            one is not cut off halfway — which is exactly what LeRobot's fixed
            3 s does.
        should_abort: polled every tick; return ``True`` to stop commanding
            immediately. This is how Ctrl-C during the sweep gets out.

    Returns:
        A :class:`HomeReport`. **Not reaching home is a normal outcome** and is
        reported, not raised: the caller is about to disconnect, which
        de-energises the motors, and it needs to say so rather than crash.
    """
    if rate <= 0:
        raise HomeError(f"home rate must be positive, got {rate}")
    goal = validate_target(target)

    period = 1.0 / fps
    joint_step = rate * period
    gripper_step = gripper_rate * period

    measured = dict(measure())
    if timeout_s is None:
        travel_s = estimate_duration(measured, goal, rate) + ease_overhead(
            rate, ease_in_s=ease_in_s, ease_out_rad=ease_out_rad
        )
        timeout_s = min(travel_s * TIMEOUT_SLACK + TIMEOUT_MARGIN_S, MAX_TIMEOUT_S)

    key, error = farthest(measured, goal)
    logger.info(
        "home sweep: %.3f rad to travel on %s, peak %.2f rad/s "
        "(ease in %.2fs, ease out %.2f rad), timeout %.1fs",
        abs(error),
        key,
        rate,
        ease_in_s,
        ease_out_rad,
        timeout_s,
    )

    command = dict(measured)
    started = clock()
    steps = 0
    reached = abs(error) <= tolerance
    aborted = False

    while not reached:
        if should_abort is not None and should_abort():
            aborted = True
            break
        elapsed = clock() - started
        if elapsed >= timeout_s:
            break

        # The ease-out reads the distance left in COMMAND space, not against the
        # measurement: the ramp is driven from the previous command, so that is
        # what actually reaches zero. The measured error settles afterwards.
        _, to_go = farthest(command, goal)
        scale = ease_scale(
            elapsed=elapsed,
            remaining=to_go,
            ease_in_s=ease_in_s,
            ease_out_rad=ease_out_rad,
            floor=ease_floor,
        )
        command = step_toward(
            command, goal, joint_step=joint_step * scale, gripper_step=gripper_step
        )
        send(command)
        steps += 1
        sleep(period)

        measured = dict(measure())
        key, error = farthest(measured, goal)
        reached = abs(error) <= tolerance

    return HomeReport(
        reached=reached,
        aborted=aborted,
        steps=steps,
        elapsed_s=clock() - started,
        worst_key=key,
        worst_error=error,
    )


def capture_pose(config: Any, *, control_mode: str = "impedance") -> Any:
    """Connect the followers, read where they are, disconnect. **Energises the arms.**

    Commands no pose — but connecting a DK1 follower is not passive: every motor
    is energised and both grippers self-zero by driving open against their stop.
    Note what that means for a captured home pose: the grippers will read 0
    (open) unless something moved them since, so the captured home is
    "arms where you put them, grippers open".

    Returns:
        A :class:`dk1lab.config.HomePose` ready for
        :func:`dk1lab.config.write_home`.
    """
    from .config import HomePose
    from .teleop import build_follower

    follower = build_follower(config, cameras=False, control_mode=control_mode)
    follower.connect()
    try:
        return HomePose.from_action_dict(dict(follower.measured_positions()))
    finally:
        follower.disconnect()


def sweep_to_home(
    config: Any,
    *,
    target: Mapping[str, float],
    limits: Any = None,
    rate: float | None = None,
    control_mode: str = "impedance",
    fps: float = DEFAULT_HOME_FPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> HomeReport:
    """Connect the followers, drive them to ``target``, disconnect. **Moves the arms.**

    The standalone version of what :func:`dk1lab.policy.run` does at the end of a
    rollout, for putting the cell back in order without loading a 7B model. No
    cameras are opened: nothing here looks at them.

    ``rate`` is the peak speed, reached in the middle of the sweep and eased away
    from at both ends. Passing it is an explicit operator choice and is honoured
    as given; leaving it ``None`` derives it with :func:`home_rate`.

    Ctrl-C during the sweep stops it where the arms are and disconnects — which
    disables the motors, so support anything holding itself up.
    """
    from .teleop import build_follower

    rate = float(rate) if rate else home_rate(getattr(limits, "max_joint_rate", None))
    follower = build_follower(config, cameras=False, control_mode=control_mode, limits=limits)
    follower.connect()
    try:
        with interrupt_aborts() as aborted:
            return go_home(
                measure=follower.measured_positions,
                send=follower.send_action,
                target=target,
                rate=float(rate),
                fps=fps,
                tolerance=tolerance,
                should_abort=aborted,
            )
    finally:
        follower.disconnect()


@contextmanager
def interrupt_aborts() -> Generator[Callable[[], bool], None, None]:
    """Make Ctrl-C set a flag for the duration of the block, then restore.

    The home sweep runs *after* the rollout loop has already exited, which on the
    Ctrl-C path means LeRobot's ``ProcessSignalHandler`` has already counted one
    signal and would ``sys.exit(1)`` on the next — killing the process mid-sweep,
    with the arms holding whatever half-command they last received. Owning SIGINT
    for the length of the sweep turns that second Ctrl-C into "stop commanding and
    disconnect", which is the same thing the rest of this project means by stop.

    Yields a predicate suitable for ``go_home(should_abort=...)``. Outside the
    main thread, where handlers cannot be installed, it yields one that is never
    true and leaves signal handling alone.
    """
    aborted = False

    def handler(_signum: int, _frame: object) -> None:
        nonlocal aborted
        aborted = True
        logger.warning("interrupt during home sweep — stopping where the arms are")

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the main thread
        yield lambda: False
        return
    try:
        yield lambda: aborted
    finally:
        signal.signal(signal.SIGINT, previous)
