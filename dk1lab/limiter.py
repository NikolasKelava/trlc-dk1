"""Joint slew-rate limiting — a real speed cap, in both control modes.

Why this exists
---------------
``DK1FollowerConfig.joint_velocity_scaling`` only does anything in ``pos_vel``
mode: it scales the velocity argument of ``control_Pos_Vel``. In ``impedance``
mode — the mode the bimanual follower runs by default, and therefore the mode
every rollout and evaluation actually used — ``send_action`` calls
``DK1Robot.command_joint_pos``, which writes the target straight into the shared
command buffer. The 250 Hz server loop clamps position limits and torque, but
there is no rate limit anywhere in that path. The knob was silently dead.

This limiter runs in the follower, above both modes, so it cannot be bypassed by
choosing a control mode.

Design notes that are not obvious
---------------------------------
**Limit against the previous command, not the measured position.** Clamping to
the measured position deadlocks under stiction: a small commanded error produces
a torque too small to break friction, the arm does not move, so the next command
is again measured + epsilon, and the setpoint never advances. Ramping the command
lets position error — and therefore torque — accumulate until the joint breaks
free, which is what a slew-rate limiter is supposed to do.

**Cap how far the command may lead the measurement** (``max_lag``). Without it, a
blocked arm lets the setpoint wind up arbitrarily far ahead and then lunge the
moment the obstruction clears.

**Rates are per second, not per command.** The limiter is used at 30 Hz behind a
policy, at 200 Hz behind teleoperation, and at 30 x N Hz when LeRobot
interpolates a chunk. A per-command step would silently mean a different speed in
each case. Each call measures its own elapsed time.

**A stalled caller cannot buy a big step.** ``dt`` is capped at ``max_dt``, so a
loop that hangs for a second does not come back entitled to a second's worth of
motion in one command.

The gripper is rate-limited too, but never lag-clamped: it is *supposed* to stall
against whatever it is holding, and a lag clamp would fight the grasp.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .layout import is_gripper

#: Default joint speed cap in rad/s. Deliberately slow — roughly 11 deg/s, a
#: walking-pace crawl you can watch and react to. Teleoperation raises it.
DEFAULT_MAX_JOINT_RATE = 0.2

#: Default gripper speed cap in normalised units/s (full travel in ~1 s).
DEFAULT_MAX_GRIPPER_RATE = 1.0

#: Default anti-windup cap in radians (~8.6 deg of lead).
DEFAULT_MAX_LAG = 0.15

#: Longest elapsed time a single call may claim, in seconds.
DEFAULT_MAX_DT = 0.1


@dataclass
class SlewLimiter:
    """Rate-limit commanded joint targets. Stateful; one instance per robot.

    Args:
        max_joint_rate: arm joint speed cap, rad/s. ``None`` disables limiting
            for arm joints.
        max_gripper_rate: gripper speed cap, normalised units/s.
        max_lag: how far ahead of the measured position a command may run, rad.
        max_dt: cap on the elapsed time any single call may claim, seconds.
        clock: monotonic time source, injectable for tests.
    """

    max_joint_rate: float | None = DEFAULT_MAX_JOINT_RATE
    max_gripper_rate: float = DEFAULT_MAX_GRIPPER_RATE
    max_lag: float = DEFAULT_MAX_LAG
    max_dt: float = DEFAULT_MAX_DT
    clock: Callable[[], float] = time.monotonic

    _prev_cmd: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_t: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Forget the ramp state. Call at episode boundaries and on connect."""
        self._prev_cmd.clear()
        self._last_t = None

    @property
    def enabled(self) -> bool:
        return self.max_joint_rate is not None

    def limit(
        self,
        target: Mapping[str, float],
        measured: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return ``target`` clamped to the configured rates.

        Args:
            target: desired joint targets, keyed as in :mod:`dk1lab.layout`.
            measured: the arm's current positions, used to seed the ramp on the
                first call and to enforce ``max_lag``. When ``None``, the ramp
                seeds from ``target`` and no lag clamping is applied.

        Returns:
            A new dict with the same keys. Non-numeric values pass through.
        """
        # The first call after construction or reset has no previous timestamp,
        # so there is no elapsed time it can honestly claim: dt is zero and the
        # command comes out equal to the measured position. In other words the
        # first tick holds still and motion begins on the second, one control
        # period later. That is the conservative reading, and it is what makes
        # "connecting never lurches" true by construction rather than by luck.
        now = self.clock()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, self.max_dt)
        dt = max(dt, 0.0)
        self._last_t = now

        if self.max_joint_rate is None:
            return dict(target)

        measured = measured or {}
        joint_step = self.max_joint_rate * dt
        gripper_step = self.max_gripper_rate * dt

        limited: dict[str, float] = {}
        for key, raw in target.items():
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                limited[key] = raw
                continue

            goal = float(raw)
            current = measured.get(key)
            current = float(current) if isinstance(current, (int, float)) else None

            # Seed at the arm's actual position so the very first command after a
            # connect or reset cannot itself be a jump.
            prev = self._prev_cmd.get(key)
            if prev is None:
                prev = current if current is not None else goal

            gripper = is_gripper(key)
            step = gripper_step if gripper else joint_step

            delta = goal - prev
            if delta > step:
                delta = step
            elif delta < -step:
                delta = -step
            cmd = prev + delta

            # Anti-windup. Not applied to the gripper, which is meant to stall.
            if not gripper and current is not None:
                cmd = max(current - self.max_lag, min(current + self.max_lag, cmd))

            self._prev_cmd[key] = cmd
            limited[key] = cmd

        return limited
