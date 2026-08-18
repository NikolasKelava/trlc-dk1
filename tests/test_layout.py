"""The 14-D vector contract, checked against the live LeRobot plugin.

If any of these fail, something silently reorders the policy's state or action
vector, which is the class of bug that produces confident, wrong motion.
"""

from __future__ import annotations

import pytest

from dk1lab import layout


def test_dof_is_fourteen():
    assert layout.DOF == 14
    assert len(layout.ACTION_KEYS) == 14
    assert len(set(layout.ACTION_KEYS)) == 14


def test_layout_is_left_block_then_right_block():
    """7 per arm, left first — the layout BimanualYAM was trained on."""
    assert layout.ACTION_KEYS[:7] == tuple(f"left_{k}" for k in layout.ARM_KEYS)
    assert layout.ACTION_KEYS[7:] == tuple(f"right_{k}" for k in layout.ARM_KEYS)


def test_gripper_is_last_within_each_arm():
    assert layout.GRIPPER_INDICES == (6, 13)
    assert layout.ACTION_KEYS[6] == "left_gripper.pos"
    assert layout.ACTION_KEYS[13] == "right_gripper.pos"


def test_matches_the_live_bimanual_follower():
    """The contract is derived from upstream; assert it still agrees with it.

    This is the canary for an upstream change to the follower's feature
    descriptors — including the ordering, which nothing else would catch.
    """
    from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig

    robot = BiDK1Follower(
        BiDK1FollowerConfig(left_arm_port="/dev/null", right_arm_port="/dev/null")
    )
    assert tuple(robot.action_features) == layout.ACTION_KEYS


def test_matches_the_safe_follower_subclass():
    from dk1lab.robot import SafeBiDK1Follower, SafeBiDK1FollowerConfig

    robot = SafeBiDK1Follower(
        SafeBiDK1FollowerConfig(left_arm_port="/dev/null", right_arm_port="/dev/null")
    )
    assert tuple(robot.action_features) == layout.ACTION_KEYS


def test_camera_names_and_image_keys_are_in_trained_order():
    """top, left, right — NOT alphabetical, which would be left, right, top."""
    assert layout.CAMERA_NAMES == ("top", "left", "right")
    assert layout.IMAGE_KEYS == (
        "observation.images.top",
        "observation.images.left",
        "observation.images.right",
    )
    assert list(layout.IMAGE_KEYS) != sorted(layout.IMAGE_KEYS)


def test_is_gripper_identifies_exactly_the_two_gripper_channels():
    flagged = [i for i, key in enumerate(layout.ACTION_KEYS) if layout.is_gripper(key)]
    assert tuple(flagged) == layout.GRIPPER_INDICES


# --------------------------------------------------------------------------- #
# Vector conversion
# --------------------------------------------------------------------------- #


def test_vector_round_trips():
    values = {key: float(i) for i, key in enumerate(layout.ACTION_KEYS)}
    vector = layout.vector_from_dict(values)
    assert vector == [float(i) for i in range(14)]
    assert layout.dict_from_vector(vector) == values


def test_vector_from_dict_ignores_extra_keys_but_keeps_order():
    values = {key: float(i) for i, key in enumerate(layout.ACTION_KEYS)}
    values["something.else"] = 99.0
    assert layout.vector_from_dict(values) == [float(i) for i in range(14)]


def test_vector_from_dict_refuses_to_silently_shorten():
    values = {key: 0.0 for key in layout.ACTION_KEYS[:-1]}
    with pytest.raises(KeyError, match="right_gripper.pos"):
        layout.vector_from_dict(values)


def test_dict_from_vector_refuses_wrong_length():
    with pytest.raises(ValueError, match="expected 14"):
        layout.dict_from_vector(range(13))


# --------------------------------------------------------------------------- #
# Gripper inversion
# --------------------------------------------------------------------------- #


def test_joint_signs_invert_only_the_grippers():
    signs = layout.yam_joint_signs()
    offsets = layout.yam_joint_offsets()
    assert len(signs) == len(offsets) == 14
    for i in range(14):
        if i in layout.GRIPPER_INDICES:
            assert (signs[i], offsets[i]) == (-1.0, 1.0)
        else:
            assert (signs[i], offsets[i]) == (1.0, 0.0)


@pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 1.0])
def test_gripper_transform_is_exactly_one_minus_x(value):
    """MolmoAct2 applies state_model = sign * state_robot + offset."""
    signs = layout.yam_joint_signs()
    offsets = layout.yam_joint_offsets()
    for i in layout.GRIPPER_INDICES:
        assert signs[i] * value + offsets[i] == pytest.approx(1.0 - value)


def test_gripper_transform_is_its_own_inverse():
    """action_robot = sign * (action_model - offset) must undo the state map."""
    signs = layout.yam_joint_signs()
    offsets = layout.yam_joint_offsets()
    for i in layout.GRIPPER_INDICES:
        for value in (0.0, 0.3, 1.0):
            to_model = signs[i] * value + offsets[i]
            back = signs[i] * (to_model - offsets[i])
            assert back == pytest.approx(value)
