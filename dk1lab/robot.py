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
from dataclasses import dataclass
from typing import Any

from lerobot.robots import RobotConfig

from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig

from .limiter import (
    DEFAULT_MAX_DT,
    DEFAULT_MAX_GRIPPER_RATE,
    DEFAULT_MAX_JOINT_RATE,
    DEFAULT_MAX_LAG,
    SlewLimiter,
)

logger = logging.getLogger(__name__)

CONNECT_WARNING = """\
Connecting the DK1 follower is NOT passive:
  * every arm motor is energised and begins holding position
  * BOTH grippers self-zero by driving OPEN against their stop
Clear the workspace and keep clear of the grippers."""


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

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        for line in CONNECT_WARNING.splitlines():
            logger.warning(line)
        self.limiter.reset()
        super().connect()
        # Seed the ramp at the pose the arms are actually in, so the first
        # commanded action cannot be a step away from wherever they came to rest.
        self.limiter.limit(self.measured_positions())

    # ------------------------------------------------------------------ #
    # Motor state without paying for the cameras
    # ------------------------------------------------------------------ #

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
        """
        if not self.limiter.enabled:
            return super().send_action(action)
        limited = self.limiter.limit(action, self.measured_positions())
        return super().send_action(limited)

    def disconnect(self) -> None:
        """Disconnect. Does not move the arms."""
        self.limiter.reset()
        super().disconnect()
