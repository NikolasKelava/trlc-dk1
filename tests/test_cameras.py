"""Camera config construction: right device, right order, right rotation."""

from __future__ import annotations

import pytest

from dk1lab.cameras import camera_configs, cameras_cli_argument, crop_summary
from dk1lab.config import ConfigError, load
from dk1lab.layout import CAMERA_NAMES


def test_configs_are_keyed_by_the_names_the_policy_expects(config_file):
    configs = camera_configs(load(config_file))
    assert tuple(configs) == CAMERA_NAMES


def test_insertion_order_is_the_trained_order_not_alphabetical(config_file):
    """LeRobot builds the image-key order from this dict's insertion order.

    Sorted, these would be left, right, top — which is not what the checkpoint
    was trained on, and would mismatch silently.
    """
    keys = list(camera_configs(load(config_file)))
    assert keys == ["top", "left", "right"]
    assert keys != sorted(keys)


def test_each_camera_gets_its_own_device_path(config_file):
    configs = camera_configs(load(config_file))
    paths = [c.index_or_path for c in configs.values()]
    assert len(set(paths)) == 3
    assert configs["top"].index_or_path.endswith("pci-top-video-index0")
    assert configs["left"].index_or_path.endswith("pci-left-video-index0")


def test_per_camera_rotation_is_applied(config_file):
    configs = camera_configs(load(config_file))
    assert int(configs["top"].rotation) == 180
    assert int(configs["left"].rotation) == 180
    assert int(configs["right"].rotation) == 0


def test_mjpg_is_always_requested(config_file):
    """YUYV at these rates exceeds the UVC bandwidth allocation and reads fail."""
    for config in camera_configs(load(config_file)).values():
        assert config.fourcc == "MJPG"


def test_profiles_change_resolution_but_not_identity(config_file):
    cfg = load(config_file)
    policy = camera_configs(cfg, "policy")
    teleop = camera_configs(cfg, "teleop")
    for name in CAMERA_NAMES:
        assert policy[name].index_or_path == teleop[name].index_or_path
        assert policy[name].rotation == teleop[name].rotation
    assert (policy["top"].width, policy["top"].height, policy["top"].fps) == (640, 360, 30)
    assert (teleop["top"].width, teleop["top"].height, teleop["top"].fps) == (1280, 720, 60)


def test_policy_profile_is_sixteen_by_nine(config_file):
    """Matching the aspect ratio BimanualYAM was trained on (640x360)."""
    top = camera_configs(load(config_file), "policy")["top"]
    assert top.width / top.height == pytest.approx(16 / 9)


def test_unknown_profile_lists_the_available_ones(config_file):
    with pytest.raises(ConfigError, match="policy, teleop"):
        camera_configs(load(config_file), "nonesuch")


def test_cli_argument_lists_cameras_in_order(config_file):
    argument = cameras_cli_argument(load(config_file))
    assert argument.index("top:") < argument.index("left:") < argument.index("right:")


def test_cli_argument_carries_the_same_values_as_the_objects(config_file):
    cfg = load(config_file)
    argument = cameras_cli_argument(cfg, "teleop")
    for name in CAMERA_NAMES:
        assert cfg.camera(name).path in argument
    assert "width: 1280" in argument
    assert "fourcc: MJPG" in argument
    # rotation 0 is the default and is omitted rather than written out
    assert argument.count("rotation: 180") == 2


def test_a_camera_with_a_target_becomes_a_cropping_camera(config_file):
    """And one without stays an ordinary OpenCVCamera — same file, same size."""
    configs = camera_configs(load(config_file))
    assert configs["left"].type == "opencv_cropped"
    assert configs["top"].type == "opencv"
    assert configs["right"].type == "opencv"


def test_the_crop_angles_reach_the_camera_config(config_file):
    left = camera_configs(load(config_file))["left"]
    assert left.source_hfov_deg == 105.0
    assert left.target_hfov_deg == 87.0


def test_cropping_does_not_change_the_frame_size(config_file):
    """Which is what lets it be invisible to every downstream feature shape."""
    configs = camera_configs(load(config_file), "policy")
    assert (configs["left"].width, configs["left"].height) == (640, 360)
    assert (configs["top"].width, configs["top"].height) == (640, 360)


def test_the_crop_follows_the_camera_into_every_profile(config_file):
    """Teleop, recording and rollout see the same field of view or none of this
    is worth doing — an operator checking the view in teleop would be checking a
    picture the policy never gets."""
    cfg = load(config_file)
    for profile in ("policy", "teleop"):
        assert camera_configs(cfg, profile)["left"].type == "opencv_cropped"


def test_crop_summary_names_the_box_and_the_offset(config_file):
    configs = camera_configs(load(config_file), "policy")
    assert crop_summary(configs["left"]) == "crop 455x256 -> 85.6 deg H, offset +0,-20 px"
    assert crop_summary(configs["top"]) is None


def test_crop_summary_is_sized_to_the_profile_but_the_angle_is_not(config_file):
    """1280x720 is the same 16:9, so the box doubles, the angle stays, and the
    offset doubles with it — that is the point of the reference width."""
    teleop = camera_configs(load(config_file), "teleop")["left"]
    assert crop_summary(teleop) == "crop 909x511 -> 85.6 deg H, offset +0,-40 px"


def test_the_crop_adjustments_reach_the_camera_config(config_file):
    left = camera_configs(load(config_file))["left"]
    assert left.crop_inset == 6.0
    assert left.crop_shift_y == -20.0


def test_cli_argument_carries_the_crop_adjustments(config_file):
    argument = cameras_cli_argument(load(config_file))
    assert "crop_inset: 6" in argument
    assert "crop_shift_y: -20" in argument
    # zero adjustments are omitted, like everywhere else
    assert "crop_shift_x" not in argument


def test_cli_argument_carries_the_crop(config_file):
    argument = cameras_cli_argument(load(config_file))
    assert "type: opencv_cropped" in argument
    assert "source_hfov_deg: 105" in argument
    assert "target_hfov_deg: 87" in argument
    # the uncropped cameras stay plain
    assert argument.count("type: opencv,") == 2
