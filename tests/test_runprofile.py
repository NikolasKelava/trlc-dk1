"""Run profiles: what ``--profile common`` gives up, and what it must not touch.

The properties worth holding are the two that would ruin the comparison if they
slipped: that ``common`` really removes the crop from **every** camera, and that
choosing a profile never edits ``dk1.toml``. Everything else here is arithmetic.

Nothing in this file needs a robot, a GPU or LeRobot.
"""

from __future__ import annotations

import pytest

from dk1lab import runprofile
from dk1lab.cameras import camera_configs
from dk1lab.config import load
from dk1lab.crop import CroppedOpenCVCameraConfig
from dk1lab.layout import CAMERA_NAMES


@pytest.fixture
def settings(config_file):
    return load(config_file, require_devices=False)


# --------------------------------------------------------------------------- #
# Choosing one
# --------------------------------------------------------------------------- #


def test_the_default_is_the_configuration_the_cell_already_runs():
    """Nothing that worked before this flag existed moves."""
    assert runprofile.resolve(None).name == runprofile.OPTIMIZED
    assert runprofile.DEFAULT_PROFILE == runprofile.OPTIMIZED


def test_a_misspelled_profile_is_refused_rather_than_defaulted():
    """A run under a configuration nobody chose is evidence about the wrong thing."""
    with pytest.raises(runprofile.ProfileError) as excinfo:
        runprofile.resolve("optimised")
    assert "optimized" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# What each one sees
# --------------------------------------------------------------------------- #


def test_optimized_leaves_the_config_exactly_as_loaded(settings):
    profile = runprofile.resolve(runprofile.OPTIMIZED)
    assert profile.apply(settings) is settings


def test_common_removes_the_crop_from_every_camera(settings):
    """Not just the wrists: the property is that all three agree afterwards."""
    common = runprofile.resolve(runprofile.COMMON).apply(settings)
    for name in CAMERA_NAMES:
        camera = common.camera(name)
        assert camera.cropped is False
        assert camera.target_hfov is None
        assert (camera.crop_inset, camera.crop_shift_x, camera.crop_shift_y) == (0.0, 0.0, 0.0)


def test_common_keeps_the_lens_angle_the_rotation_and_the_device(settings):
    """Only the crop goes. The mount really is upside down, and stays so."""
    common = runprofile.resolve(runprofile.COMMON).apply(settings)
    for name in CAMERA_NAMES:
        before, after = settings.camera(name), common.camera(name)
        assert (after.path, after.rotation, after.hfov) == (before.path, before.rotation, before.hfov)


def test_applying_a_profile_does_not_mutate_the_config_it_was_given(settings):
    """The derived config is a new object; the loaded one is still the file's."""
    runprofile.resolve(runprofile.COMMON).apply(settings)
    assert settings.camera("left").cropped is True


def test_common_builds_plain_cameras_where_optimized_builds_cropped_ones(settings):
    """The end of the chain: what LeRobot is actually handed."""
    optimized = camera_configs(runprofile.resolve(runprofile.OPTIMIZED).apply(settings))
    common = camera_configs(runprofile.resolve(runprofile.COMMON).apply(settings))

    assert isinstance(optimized["left"], CroppedOpenCVCameraConfig)
    assert not any(isinstance(cfg, CroppedOpenCVCameraConfig) for cfg in common.values())
    # The frame size is a capture-profile question and is not what changed.
    for name in CAMERA_NAMES:
        assert (common[name].width, common[name].height) == (
            optimized[name].width,
            optimized[name].height,
        )


# --------------------------------------------------------------------------- #
# How fast each one may move
# --------------------------------------------------------------------------- #


def test_each_profile_reads_its_own_limits_table(settings):
    """The fixture config has neither table, so both fall back — and differ anyway."""
    optimized = runprofile.resolve(runprofile.OPTIMIZED).limits(settings)
    common = runprofile.resolve(runprofile.COMMON).limits(settings)
    assert optimized.max_joint_rate == 1.0
    assert common.max_joint_rate == 0.6


def test_only_the_rate_drops_under_common():
    """max_lag is a torque clamp; lowering it would stall the arms, not calm them."""
    policy, study = runprofile.POLICY_LIMITS, runprofile.STUDY_LIMITS
    assert study.max_joint_rate < policy.max_joint_rate
    assert study.max_lag == policy.max_lag
    assert study.max_gripper_rate == policy.max_gripper_rate
    assert study.max_dt == policy.max_dt


def test_the_file_wins_over_the_fallback(config_file):
    config_file.write_text(
        config_file.read_text() + "\n[limits.study]\nmax_joint_rate = 0.45\n"
    )
    reloaded = load(config_file, require_devices=False)
    assert runprofile.resolve(runprofile.COMMON).limits(reloaded).max_joint_rate == 0.45
    # ... and the other profile is unaffected by it.
    assert runprofile.resolve(runprofile.OPTIMIZED).limits(reloaded).max_joint_rate == 1.0


# --------------------------------------------------------------------------- #
# The file itself
# --------------------------------------------------------------------------- #


def test_choosing_a_profile_never_writes_the_config_file(config_file):
    """`dk1.toml` is the frozen half of the study. Applying a profile is in memory."""
    before = config_file.read_bytes()
    loaded = load(config_file, require_devices=False)
    for name in runprofile.PROFILES:
        profile = runprofile.resolve(name)
        profile.apply(loaded)
        profile.limits(loaded)
    assert config_file.read_bytes() == before


def test_the_repos_own_config_carries_the_study_table(repo_config):
    """STUDY.md's one addition to dk1.toml, and the numbers it names."""
    settings = load(repo_config, require_devices=False)
    study = settings.limits["study"]
    assert study.max_joint_rate == 0.6
    assert study.max_lag == settings.limits["policy"].max_lag


def test_the_repos_own_config_still_crops_only_the_wrists(repo_config):
    """The optimized profile is frozen; this is the assertion that says so."""
    settings = runprofile.resolve(runprofile.OPTIMIZED).apply(
        load(repo_config, require_devices=False)
    )
    assert settings.camera("top").cropped is False
    assert settings.camera("left").cropped is True
    assert settings.camera("right").cropped is True
