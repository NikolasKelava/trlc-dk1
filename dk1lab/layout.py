"""The bimanual DK1 vector contract: which value lives at which index.

Everything that turns the robot into a flat 14-D vector — the policy state, the
policy action, the norm statistics, the gripper-inversion transform — depends on
this ordering being right, and gets it wrong silently if it is not. So it is
derived once, here, and asserted against the live LeRobot plugin in the tests
rather than restated as a literal anywhere else.

The order comes from ``BiDK1Follower.action_features``, which prefixes each
arm's own ``action_features`` with ``left_`` / ``right_``:

    index  0..5   left_joint_1.pos .. left_joint_6.pos
    index  6      left_gripper.pos
    index  7..12  right_joint_1.pos .. right_joint_6.pos
    index 13      right_gripper.pos

which is exactly the 7-per-arm, left-block-first layout the MolmoAct2 BimanualYAM
checkpoint was trained on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

# Joint names as declared by ``lerobot_robot_trlc_dk1.follower.JOINT_NAMES``.
ARM_JOINTS: tuple[str, ...] = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")

GRIPPER: str = "gripper"

# Left block first — this is the order the checkpoint expects, not an arbitrary
# choice. See the module docstring.
ARMS: tuple[str, ...] = ("left", "right")

#: The 7 keys one arm contributes, in order: six joints, then the gripper.
ARM_KEYS: tuple[str, ...] = tuple(f"{j}.pos" for j in ARM_JOINTS) + (f"{GRIPPER}.pos",)

#: The full 14-key action/state vector layout.
ACTION_KEYS: tuple[str, ...] = tuple(f"{arm}_{key}" for arm in ARMS for key in ARM_KEYS)

#: State uses the same motor keys as action (the follower reports what it drives).
STATE_KEYS: tuple[str, ...] = ACTION_KEYS

DOF: int = len(ACTION_KEYS)  # 14

#: Indices of the two gripper channels, derived rather than written down.
GRIPPER_INDICES: tuple[int, ...] = tuple(
    i for i, key in enumerate(ACTION_KEYS) if key.endswith(f"{GRIPPER}.pos")
)

#: Camera names, in the order the BimanualYAM checkpoint was trained on. The
#: policy's image keys are ``observation.images.{name}`` for each of these, so
#: the robot's camera keys must use exactly these names.
CAMERA_NAMES: tuple[str, ...] = ("top", "left", "right")

#: The policy's image keys, pinned explicitly. The MolmoAct2 processor falls
#: back to ``sorted()`` when nothing pins the order, and "left" < "right" <
#: "top" is not the trained order.
IMAGE_KEYS: tuple[str, ...] = tuple(f"observation.images.{name}" for name in CAMERA_NAMES)


def is_gripper(key: str) -> bool:
    """True for the gripper channel of either arm."""
    return key.endswith(f"{GRIPPER}.pos")


def vector_from_dict(values: Mapping[str, float], keys: Sequence[str] = ACTION_KEYS) -> list[float]:
    """Flatten a robot action/observation dict into the canonical vector order.

    Raises:
        KeyError: if any expected key is absent, rather than silently producing a
            short or misaligned vector.
    """
    missing = [k for k in keys if k not in values]
    if missing:
        raise KeyError(f"missing {len(missing)} of {len(keys)} vector keys: {missing}")
    return [float(values[k]) for k in keys]


def dict_from_vector(vector: Iterable[float], keys: Sequence[str] = ACTION_KEYS) -> dict[str, float]:
    """Inverse of :func:`vector_from_dict`."""
    values = [float(v) for v in vector]
    if len(values) != len(keys):
        raise ValueError(f"expected {len(keys)} values, got {len(values)}")
    return dict(zip(keys, values, strict=True))


# --------------------------------------------------------------------------- #
# Gripper convention
# --------------------------------------------------------------------------- #
#
# The DK1 normalises its gripper as 0 = open, 1 = closed
# (``DK1Robot.command_gripper``; ``DK1Leader.get_action``).
#
# The MolmoAct2 BimanualYAM checkpoint uses the opposite convention, 1 = open,
# 0 = closed. Two independent sources agree on this:
#   * sai-prasanna/molmoact2 @ sim-eval-dk1, ``sim_eval/inference/common.py`` —
#     both ``yam_state_adapter`` and ``dk1_state_adapter`` document the server as
#     wanting "1=open".
#   * the checkpoint's own statistics: the gripper channels sit at mean 0.64 /
#     median 0.73, i.e. predominantly high, which reads as predominantly open.
#
# MolmoAct2 applies an affine correction in both directions:
#     state_model = signs * state_robot + offsets
#     action_robot = signs * (action_model - offsets)
# so sign=-1, offset=1 on a channel is exactly x -> 1 - x each way.
#
# NOT YET VERIFIED ON HARDWARE. This is a well-supported inference, not an
# observation; ``dk1 policy dryrun`` is what confirms it.

#: The range a DK1 gripper command may take: 0 = open, 1 = closed.
#:
#: Not a convention this file invents — ``DK1Robot.command_gripper`` clips to
#: exactly this before the value reaches the motor, so anything outside it is a
#: command the robot did not execute. Named here because two other places need
#: it: the follower, which must **return** what it really sent, and
#: :func:`dk1lab.dataset.clamp_gripper`, which repairs a dataset recorded before
#: it did.
GRIPPER_MIN: float = 0.0
GRIPPER_MAX: float = 1.0

GRIPPER_INVERSION_SIGN: float = -1.0
GRIPPER_INVERSION_OFFSET: float = 1.0


def yam_joint_signs() -> list[float]:
    """``--policy.joint_signs`` inverting only the two gripper channels."""
    return [GRIPPER_INVERSION_SIGN if i in GRIPPER_INDICES else 1.0 for i in range(DOF)]


def yam_joint_offsets() -> list[float]:
    """``--policy.joint_offsets`` matching :func:`yam_joint_signs`."""
    return [GRIPPER_INVERSION_OFFSET if i in GRIPPER_INDICES else 0.0 for i in range(DOF)]
