"""Teleoperation: drive the followers from the leader arms.

One implementation, reached through one entry point (``dk1 teleop``). The
previous iteration of this project had three near-duplicate teleop scripts that
drifted apart on camera names and safety defaults; the point of this module is
that there is nowhere for a second copy to live.

The control loop itself is LeRobot's ``teleop_loop``, imported rather than
reimplemented. That is deliberate: recording and policy rollout run the same
loop, so teleoperation exercising it is what makes this phase a real check that
the ported plugin and motor stack are intact — a bespoke loop here could work
while the one every later phase depends on does not.

What this module owns is everything around that loop:

* device identity comes from ``dk1.toml``, so teleop cannot drift from what the
  policy will later be given;
* cameras are named ``top`` / ``left`` / ``right`` via
  :func:`dk1lab.cameras.camera_configs`, which is the naming the MolmoAct2
  checkpoint requires — the old repo called them ``wrist_left`` / ``wrist_right``
  and would have failed the rollout context's feature check;
* the follower is always :class:`dk1lab.robot.SafeBiDK1Follower`, so the joint
  speed limit applies in impedance mode as well;
* stopping disconnects and nothing else. The arms are never swept home.
"""

from __future__ import annotations

from typing import Any

from lerobot_robot_trlc_dk1.bi_leader import BiDK1Leader, BiDK1LeaderConfig

from .cameras import camera_configs
from .config import DK1Config, LimitProfile
from .limiter import DEFAULT_MAX_DT, DEFAULT_MAX_GRIPPER_RATE, DEFAULT_MAX_LAG
from .robot import SafeBiDK1Follower, SafeBiDK1FollowerConfig

#: The speed limit teleoperation runs under when ``dk1.toml`` says nothing.
#:
#: **No cap.** This matches what the DK1 actually does: upstream's plain
#: ``bi_dk1_follower`` has no slew limit in impedance mode at all, since
#: ``joint_velocity_scaling`` only reaches ``control_Pos_Vel``. It is also the
#: right default for *this* activity — the limiter exists to bound a policy
#: nobody trusts yet, and in teleoperation the commands come from a human hand,
#: so a runaway is already bounded by the person holding the leader arm. A cap
#: tight enough to matter is tight enough to feel, and a follower that visibly
#: lags the leader tells you nothing about whether the stack works.
#:
#: Disabling it also removes a serial round-trip per tick: ``SafeBiDK1Follower``
#: only reads ``measured_positions()`` when the limiter is enabled.
#:
#: Policy rollout is unaffected — Phase 3 sets its own, much tighter, limit.
#: Override per run with ``--max-joint-rate``, or for good with ``[limits.teleop]``
#: in ``dk1.toml``.
TELEOP_LIMITS = LimitProfile(
    max_joint_rate=None,
    max_gripper_rate=DEFAULT_MAX_GRIPPER_RATE,
    max_lag=DEFAULT_MAX_LAG,
    max_dt=DEFAULT_MAX_DT,
)

#: Loop rate. The leader is read over a Dynamixel bus and the follower written
#: over a CAN adapter, and LeRobot's loop additionally reads a full observation
#: every tick, so the achievable rate is an empirical question — the loop prints
#: what it actually gets.
DEFAULT_FPS: int = 60


def leader_config(config: DK1Config) -> BiDK1LeaderConfig:
    """The leader pair, from ``dk1.toml``."""
    return BiDK1LeaderConfig(
        left_arm_port=config.leader.left,
        right_arm_port=config.leader.right,
        id="dk1_leader",
    )


def follower_config(
    config: DK1Config,
    *,
    cameras: bool = True,
    profile: str = "teleop",
    control_mode: str = "impedance",
    limits: LimitProfile | None = None,
) -> SafeBiDK1FollowerConfig:
    """The rate-limited follower pair, from ``dk1.toml``.

    Args:
        cameras: attach the three cameras. Off makes the loop cheaper and is the
            right choice when all you want to know is whether the arms track.
        profile: which ``[capture.*]`` profile the cameras run at. ``teleop`` by
            default — nothing downstream depends on it, so it is sized to be
            looked at. Recording will want ``policy``.
        control_mode: ``impedance`` (upstream's default) or ``pos_vel``.
        limits: speed limit. ``None`` takes ``[limits.teleop]`` from ``dk1.toml``,
            falling back to :data:`TELEOP_LIMITS`.
    """
    limits = limits if limits is not None else config.limit("teleop", TELEOP_LIMITS)
    return SafeBiDK1FollowerConfig(
        left_arm_port=config.follower.left,
        right_arm_port=config.follower.right,
        cameras=camera_configs(config, profile) if cameras else {},
        control_mode=control_mode,
        max_joint_rate=limits.max_joint_rate,
        max_gripper_rate=limits.max_gripper_rate,
        max_lag=limits.max_lag,
        max_dt=limits.max_dt,
        id="dk1_follower",
    )


def build_follower(
    config: DK1Config,
    *,
    cameras: bool = True,
    profile: str = "teleop",
    control_mode: str = "impedance",
    limits: LimitProfile | None = None,
) -> SafeBiDK1Follower:
    """Construct the follower pair alone. Constructing connects to nothing.

    Split out of :func:`build` for the callers with no use for a leader: the home
    sweep in :mod:`dk1lab.home` reads and drives the followers only.
    """
    return SafeBiDK1Follower(
        follower_config(
            config,
            cameras=cameras,
            profile=profile,
            control_mode=control_mode,
            limits=limits,
        )
    )


def build(
    config: DK1Config,
    *,
    cameras: bool = True,
    profile: str = "teleop",
    control_mode: str = "impedance",
    limits: LimitProfile | None = None,
) -> tuple[BiDK1Leader, SafeBiDK1Follower]:
    """Construct both devices. Constructing connects to nothing."""
    leader = BiDK1Leader(leader_config(config))
    follower = build_follower(
        config,
        cameras=cameras,
        profile=profile,
        control_mode=control_mode,
        limits=limits,
    )
    return leader, follower


def run(
    leader: BiDK1Leader,
    follower: SafeBiDK1Follower,
    *,
    fps: int = DEFAULT_FPS,
    display: bool = False,
    duration_s: float | None = None,
    model_view: Any | None = None,
) -> None:
    """Connect, run LeRobot's teleop loop, and disconnect. **Moves the arms.**

    Connecting is itself motion — see :mod:`dk1lab.cli.safety`. Ctrl-C leaves the
    loop and disconnects; it does not move the arms anywhere, and there is
    deliberately no return-to-home.

    Args:
        display: stream observations to Rerun. Needs cameras to be worth much.
        duration_s: stop after this long. ``None`` runs until interrupted.
        model_view: an optional :class:`dk1lab.modelview.ModelInputProbe` to wrap
            the observation processor with, adding the model's-eye view to the
            same Rerun session. Implies ``display``: it logs to Rerun, and the
            loop only calls the observation processor at all when displaying.
    """
    # Imported here, not at module scope: pulling in LeRobot's script module
    # drags in every first-party robot and teleoperator class, which is a slow
    # import to pay for on `dk1 --help`.
    from lerobot.processor import make_default_processors
    from lerobot.scripts.lerobot_teleoperate import teleop_loop
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

    from .modelview import pin_blueprint

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    if display:
        init_visualization("rerun", session_name="dk1-teleop")
    if model_view is not None:
        # Wrap, do not replace: the probe forwards to the processor LeRobot built
        # and only adds a side effect. Pinning the layout has to happen after
        # init_visualization, which clears the blueprint cache the pin fills.
        model_view.inner = robot_observation_processor
        robot_observation_processor = model_view
        pin_blueprint()

    # Leader first: it is the passive half, and connecting it before the
    # followers are live means a hand already resting on a leader arm cannot
    # command anything yet.
    if model_view is not None:
        model_view.start()

    leader.connect()
    follower.connect()
    try:
        teleop_loop(
            teleop=leader,
            robot=follower,
            fps=fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_data=display,
            duration=duration_s,
        )
    except KeyboardInterrupt:
        pass
    finally:
        # Disconnect in both cases. SafeBiDK1Follower.disconnect does not move
        # the arms, and there is no return-to-home here by design: sweeping the
        # arms home is the last thing you want when you stopped because
        # something was wrong.
        # The worker first: it logs to Rerun, so it has to be quiet before the
        # recording stream is shut down.
        if model_view is not None:
            model_view.stop()
        if display:
            shutdown_visualization("rerun")
        follower.disconnect()
        leader.disconnect()
