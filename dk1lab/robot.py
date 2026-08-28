"""``SafeBiDK1Follower`` — the bimanual follower with a working speed limit.

Registered as ``bi_dk1_follower_safe``. Use it everywhere in place of
``bi_dk1_follower``; it is the same robot with two additions:

1. A joint slew-rate limit that applies in **both** control modes (see
   :mod:`dk1lab.limiter` for why the upstream ``joint_velocity_scaling`` knob does
   not).
2. A loud warning at ``connect()``, because connecting a DK1 follower is not a
   passive act: it energises every motor and self-zeroes both grippers by driving
   them open against their stop.

Implemented as a subclass rather than a patch so this fork stays rebaseable on
robot-learning-co/trlc-dk1 — no upstream file is modified.

Note on plugin registration: LeRobot discovers third-party robots by scanning for
installed *distributions* named ``lerobot_robot_*`` and importing the module of
the same name. ``dk1lab`` does not match that pattern, so ``bi_dk1_follower_safe``
is only registered once this module has been imported. Every ``dk1`` subcommand
imports it; a bare ``lerobot-record --robot.type=bi_dk1_follower_safe`` would not
find it. Go through the ``dk1`` CLI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from lerobot.robots import RobotConfig

from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig

from .layout import GRIPPER_MAX, GRIPPER_MIN, is_gripper
from .limiter import (
    DEFAULT_MAX_DT,
    DEFAULT_MAX_GRIPPER_RATE,
    DEFAULT_MAX_JOINT_RATE,
    DEFAULT_MAX_LAG,
    SlewLimiter,
)

logger = logging.getLogger(__name__)

#: How old a cached position reading may be before ``send_action`` reads the
#: motors again, in seconds.
#:
#: Every control loop in LeRobot calls ``get_observation()`` and then
#: ``send_action()`` within the same tick, so the limiter can use the reading the
#: loop just took instead of paying a second full serial round-trip over the CAN
#: adapters — 12 motor reads per tick rather than 24. At 30 Hz a tick is 33 ms;
#: 15 ms keeps the reuse strictly within one tick, and anything older falls back
#: to reading the motors, so a caller that does not observe first is unaffected.
CACHED_MEASUREMENT_MAX_AGE_S: float = 0.015

CONNECT_WARNING = """\
Connecting the DK1 follower is NOT passive:
  * every arm motor is energised and begins holding position
  * BOTH grippers self-zero by driving OPEN against their stop
Clear the workspace and keep clear of the grippers."""


def _clamp_grippers(action: dict[str, Any]) -> dict[str, Any]:
    """``action`` with every gripper channel clipped the way the robot clips it.

    A new dict: the caller's may be the one the teleoperator's own loop still
    holds, and mutating it would change what the leader is understood to have
    asked for. Non-gripper channels are copied through untouched.
    """
    return {
        key: (
            min(GRIPPER_MAX, max(GRIPPER_MIN, float(value)))
            if is_gripper(key) and isinstance(value, (int, float))
            else value
        )
        for key, value in action.items()
    }


@RobotConfig.register_subclass("bi_dk1_follower_safe")
@dataclass
class SafeBiDK1FollowerConfig(BiDK1FollowerConfig):
    """``BiDK1FollowerConfig`` plus a slew-rate limit that actually applies.

    Args:
        max_joint_rate: arm joint speed cap in rad/s, enforced in both
            ``impedance`` and ``pos_vel`` mode. Set to ``None`` to disable, which
            you should only do with a reason.
        max_gripper_rate: gripper speed cap in normalised units/s.
        max_lag: how far a command may lead the measured position, rad.
        max_dt: longest elapsed time one command may claim, seconds.
    """

    max_joint_rate: float | None = DEFAULT_MAX_JOINT_RATE
    max_gripper_rate: float = DEFAULT_MAX_GRIPPER_RATE
    max_lag: float = DEFAULT_MAX_LAG
    max_dt: float = DEFAULT_MAX_DT


class SafeBiDK1Follower(BiDK1Follower):
    """Bimanual DK1 follower with a real, mode-independent joint speed limit."""

    config_class = SafeBiDK1FollowerConfig
    name = "bi_dk1_follower_safe"

    def __init__(self, config: SafeBiDK1FollowerConfig):
        super().__init__(config)
        self.limiter = SlewLimiter(
            max_joint_rate=config.max_joint_rate,
            max_gripper_rate=config.max_gripper_rate,
            max_lag=config.max_lag,
            max_dt=config.max_dt,
        )
        self._measured: dict[str, float] | None = None
        self._measured_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        for line in CONNECT_WARNING.splitlines():
            logger.warning(line)
        self.limiter.reset()
        self._measured = None
        super().connect()
        # Seed the ramp at the pose the arms are actually in, so the first
        # commanded action cannot be a step away from wherever they came to rest.
        self.limiter.limit(self.measured_positions())

    # ------------------------------------------------------------------ #
    # Motor state without paying for the cameras
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, Any]:
        """The full observation, remembering the joint positions it contains.

        The reading is cached so ``send_action`` does not have to take a second
        one in the same tick — see :data:`CACHED_MEASUREMENT_MAX_AGE_S`. Cameras
        are read here and only here.
        """
        observation = super().get_observation()
        self._measured = {k: v for k, v in observation.items() if k.endswith(".pos")}
        self._measured_at = time.monotonic()
        return observation

    def _fresh_measurement(self) -> dict[str, float]:
        """Positions from this tick's observation, or a fresh read if there is none."""
        if (
            self._measured is not None
            and time.monotonic() - self._measured_at <= CACHED_MEASUREMENT_MAX_AGE_S
        ):
            return self._measured
        return self.measured_positions()

    def measured_positions(self) -> dict[str, float]:
        """Current joint + gripper positions, prefixed ``left_`` / ``right_``.

        ``get_observation()`` would also read all three cameras, which costs far
        more than the control loop can spare on every ``send_action``.

        This is the only place this fork reaches into upstream private methods.
        ``tests/test_robot_layout.py`` asserts they still exist, so a rebase that
        renames them fails a test rather than silently disabling the speed limit.
        """
        out: dict[str, float] = {}
        for prefix, arm in (("left", self.left_arm), ("right", self.right_arm)):
            if arm.config.control_mode == "impedance":
                positions = arm._get_observation_impedance()
            else:
                positions = arm._get_observation_pos_vel()
            out.update({f"{prefix}_{key}": value for key, value in positions.items()})
        return out

    # ------------------------------------------------------------------ #
    # Action
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Rate-limit ``action``, then send it.

        Returns the action that was actually sent, not the one requested — so a
        recorded dataset stores what the arms were told to do.

        **That promise was not true of the gripper until 2026-08-28.**
        ``DK1Robot.command_gripper`` clips its argument to [0, 1] *inside* the
        robot and returns nothing, so a leader trigger squeezed past the
        follower's closed stop was recorded as a command of 1.03 that the robot
        clipped to 1.0 and never executed. Harmless on the arms; not harmless in
        a dataset, because MolmoAct2 passes the gripper channel through its
        normaliser **unnormalised** and refuses anything outside [-1, 1] — so
        those frames stopped a fine-tune dead. ``dk1lab.dataset.clamp_gripper``
        repairs a recording made before this; this is what stops the next one
        needing it. `DIAGNOSTICS §` *The gripper command that was never executed*.
        """
        limited = action if not self.limiter.enabled else self.limiter.limit(
            action, self._fresh_measurement()
        )
        return _clamp_grippers(super().send_action(limited))

    def disconnect(self) -> None:
        """Disconnect. Does not move the arms."""
        self.limiter.reset()
        self._measured = None
        super().disconnect()
