"""dk1.toml: validation, round-trip, and — above all — surgical writes."""

from __future__ import annotations

import tomllib

import pytest

from dk1lab.config import (
    ArmPorts,
    CameraDevice,
    ConfigError,
    LimitProfile,
    check_devices,
    load,
    parse,
    write_arms,
    write_cameras,
)


def test_loads_and_exposes_every_field(config_file):
    cfg = load(config_file)
    assert cfg.follower == ArmPorts(left="/dev/ttyACM1", right="/dev/ttyACM3")
    assert cfg.leader == ArmPorts(left="/dev/ttyACM0", right="/dev/ttyACM2")
    assert list(cfg.cameras) == ["top", "left", "right"]
    assert cfg.camera("top").rotation == 180
    assert cfg.camera("right").rotation == 0
    assert cfg.profile("policy").width == 640
    assert cfg.profile("teleop").fps == 60


def test_the_repo_config_is_valid(repo_config):
    """The dk1.toml actually shipped in this repo must parse."""
    cfg = load(repo_config)
    assert set(cfg.cameras) == {"top", "left", "right"}
    assert "policy" in cfg.capture


def test_arm_ports_helper_names_all_four(config_file):
    assert load(config_file).arm_ports() == {
        "follower_left": "/dev/ttyACM1",
        "follower_right": "/dev/ttyACM3",
        "leader_left": "/dev/ttyACM0",
        "leader_right": "/dev/ttyACM2",
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _raw(config_file):
    return tomllib.loads(config_file.read_text())


def test_missing_version_is_rejected(config_file):
    raw = _raw(config_file)
    del raw["version"]
    with pytest.raises(ConfigError, match="version"):
        parse(raw)


def test_wrong_version_is_rejected(config_file):
    raw = _raw(config_file)
    raw["version"] = 99
    with pytest.raises(ConfigError, match="not supported"):
        parse(raw)


def test_missing_arm_section_is_rejected(config_file):
    raw = _raw(config_file)
    del raw["arms"]["leader"]
    with pytest.raises(ConfigError, match=r"\[arms.leader\]"):
        parse(raw)


def test_missing_arm_side_is_rejected(config_file):
    raw = _raw(config_file)
    del raw["arms"]["follower"]["right"]
    with pytest.raises(ConfigError, match=r"\[arms.follower\].right"):
        parse(raw)


def test_duplicate_arm_ports_are_rejected(config_file):
    """Two arms on one port is always a discovery mistake, never a valid setup."""
    raw = _raw(config_file)
    raw["arms"]["leader"]["left"] = raw["arms"]["follower"]["left"]
    with pytest.raises(ConfigError, match="own serial port"):
        parse(raw)


def test_missing_camera_is_rejected(config_file):
    raw = _raw(config_file)
    del raw["cameras"]["top"]
    with pytest.raises(ConfigError, match="missing"):
        parse(raw)


def test_unexpected_camera_is_rejected(config_file):
    """A stray name would silently never be read; say so instead."""
    raw = _raw(config_file)
    raw["cameras"]["wrist_left"] = {"path": "/dev/whatever", "rotation": 0}
    with pytest.raises(ConfigError, match="unexpected"):
        parse(raw)


def test_duplicate_camera_paths_are_rejected(config_file):
    """The failure mode the shared serial 20010101 makes easy to hit."""
    raw = _raw(config_file)
    raw["cameras"]["left"]["path"] = raw["cameras"]["top"]["path"]
    with pytest.raises(ConfigError, match="by-path"):
        parse(raw)


@pytest.mark.parametrize("rotation", [45, 1, -90, "180", None])
def test_bad_rotation_is_rejected(config_file, rotation):
    raw = _raw(config_file)
    raw["cameras"]["top"]["rotation"] = rotation
    with pytest.raises(ConfigError, match="rotation"):
        parse(raw)


@pytest.mark.parametrize("value", [0, -1, 1.5, "640", True])
def test_bad_capture_dimension_is_rejected(config_file, value):
    raw = _raw(config_file)
    raw["capture"]["policy"]["width"] = value
    with pytest.raises(ConfigError, match="width"):
        parse(raw)


def test_bad_fourcc_is_rejected(config_file):
    raw = _raw(config_file)
    raw["capture"]["policy"]["fourcc"] = "MJPEG"
    with pytest.raises(ConfigError, match="fourcc"):
        parse(raw)


def test_invalid_toml_names_the_file(tmp_path):
    path = tmp_path / "dk1.toml"
    path.write_text("version = = 1")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load(path)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load(tmp_path / "nope.toml")


def test_check_devices_reports_every_absentee(config_file):
    # The fixture's /dev/ttyACM* names may well exist on the machine running the
    # tests, so point the arms at nodes that certainly do not.
    raw = _raw(config_file)
    for role in ("follower", "leader"):
        for side in ("left", "right"):
            raw["arms"][role][side] = f"/dev/dk1-absent-{role}-{side}"
    with pytest.raises(ConfigError) as excinfo:
        check_devices(parse(raw))
    message = str(excinfo.value)
    # Four arms and three cameras, all absent — and all of them reported, not
    # just the first, so one run tells you everything to fix.
    assert message.count("not present") == 7


def test_check_devices_passes_when_everything_is_present(tmp_path, config_file):
    """The positive case, built from files that really do exist."""
    nodes = [tmp_path / f"node{i}" for i in range(7)]
    for node in nodes:
        node.touch()
    raw = _raw(config_file)
    raw["arms"]["follower"]["left"] = str(nodes[0])
    raw["arms"]["follower"]["right"] = str(nodes[1])
    raw["arms"]["leader"]["left"] = str(nodes[2])
    raw["arms"]["leader"]["right"] = str(nodes[3])
    for i, name in enumerate(("top", "left", "right")):
        raw["cameras"][name]["path"] = str(nodes[4 + i])
    check_devices(parse(raw))  # must not raise


# --------------------------------------------------------------------------- #
# Surgical writes — the invariant that matters most
# --------------------------------------------------------------------------- #


def test_write_arms_leaves_cameras_byte_identical(config_file):
    """The bug this repo exists to not repeat.

    The previous project's find_ports.py rewrote the whole config file, wiping
    the camera section that every other script depended on.
    """
    before = load(config_file)
    write_arms(
        {
            "follower": ArmPorts(left="/dev/ttyACM7", right="/dev/ttyACM8"),
            "leader": ArmPorts(left="/dev/ttyACM9", right="/dev/ttyACM10"),
        },
        config_file,
    )
    after = load(config_file)

    assert after.follower == ArmPorts(left="/dev/ttyACM7", right="/dev/ttyACM8")
    assert after.leader == ArmPorts(left="/dev/ttyACM9", right="/dev/ttyACM10")
    assert after.cameras == before.cameras
    assert after.capture == before.capture


def test_write_cameras_leaves_arms_byte_identical(config_file):
    """And the mirror image: camera discovery must not clobber the ports."""
    before = load(config_file)
    write_cameras(
        {
            "top": CameraDevice(path="/dev/v4l/by-path/new-top", rotation=0),
            "left": CameraDevice(path="/dev/v4l/by-path/new-left", rotation=90),
            "right": CameraDevice(path="/dev/v4l/by-path/new-right", rotation=180),
        },
        config_file,
    )
    after = load(config_file)

    assert after.camera("left").rotation == 90
    assert after.camera("top").path == "/dev/v4l/by-path/new-top"
    assert after.follower == before.follower
    assert after.leader == before.leader
    assert after.capture == before.capture


def test_writes_preserve_comments(config_file):
    write_arms(
        {
            "follower": ArmPorts(left="/dev/ttyACM7", right="/dev/ttyACM8"),
            "leader": ArmPorts(left="/dev/ttyACM9", right="/dev/ttyACM10"),
        },
        config_file,
    )
    text = config_file.read_text()
    assert "a comment that must survive every surgical write" in text
    assert "camera comment" in text


def test_write_round_trips(config_file):
    """Write what was read, and the parsed result is unchanged."""
    before = load(config_file)
    write_arms({"follower": before.follower, "leader": before.leader}, config_file)
    write_cameras(before.cameras, config_file)
    after = load(config_file)
    assert after.follower == before.follower
    assert after.leader == before.leader
    assert after.cameras == before.cameras
    assert after.capture == before.capture


def test_write_arms_rejects_partial_input(config_file):
    with pytest.raises(ConfigError, match="missing roles"):
        write_arms({"follower": ArmPorts(left="/dev/a", right="/dev/b")}, config_file)


def test_write_cameras_rejects_partial_input(config_file):
    with pytest.raises(ConfigError, match="missing cameras"):
        write_cameras({"top": CameraDevice(path="/dev/x")}, config_file)


# --------------------------------------------------------------------------- #
# [limits.*] — optional, and a typo must never silently disable a cap
# --------------------------------------------------------------------------- #


def _with_limits(config_file, body: str):
    config_file.write_text(config_file.read_text() + f"\n[limits.teleop]\n{body}\n")
    return config_file


def test_the_limits_section_is_optional(config_file):
    assert load(config_file).limits == {}


def test_an_absent_profile_falls_back_to_what_the_caller_supplies(config_file):
    fallback = LimitProfile(0.2, 1.0, 0.15, 0.1)
    assert load(config_file).limit("teleop", fallback) is fallback


def test_a_configured_profile_wins_over_the_fallback(config_file):
    config = load(_with_limits(config_file, "max_joint_rate = 0.5"))
    assert config.limit("teleop", LimitProfile(9.9, 1.0, 0.15, 0.1)).max_joint_rate == 0.5


def test_false_is_how_the_file_spells_no_limit(config_file):
    config = load(_with_limits(config_file, "max_joint_rate = false"))
    assert config.limits["teleop"].max_joint_rate is None


def test_a_missing_rate_key_means_no_limit(config_file):
    assert load(_with_limits(config_file, "max_lag = 0.2")).limits["teleop"].max_joint_rate is None


def test_the_other_keys_have_defaults(config_file):
    limit = load(_with_limits(config_file, "max_joint_rate = 0.5")).limits["teleop"]
    assert (limit.max_gripper_rate, limit.max_lag, limit.max_dt) == (1.0, 0.15, 0.1)


def test_true_is_not_a_rate(config_file):
    """`true` would coerce to 1.0 under a naive isinstance check — a real hazard."""
    with pytest.raises(ConfigError, match="max_joint_rate"):
        load(_with_limits(config_file, "max_joint_rate = true"))


def test_a_zero_rate_is_refused_rather_than_freezing_the_arms(config_file):
    with pytest.raises(ConfigError, match="max_joint_rate"):
        load(_with_limits(config_file, "max_joint_rate = 0"))


def test_a_negative_rate_is_refused(config_file):
    with pytest.raises(ConfigError, match="max_joint_rate"):
        load(_with_limits(config_file, "max_joint_rate = -1.0"))


def test_a_non_numeric_rate_is_refused(config_file):
    with pytest.raises(ConfigError, match="max_joint_rate"):
        load(_with_limits(config_file, 'max_joint_rate = "fast"'))


def test_a_misspelt_key_is_refused_not_ignored(config_file):
    """Silently ignoring `max_joint_rat` would leave the cap at its default while
    the file reads as though it had been set."""
    with pytest.raises(ConfigError, match="unknown keys"):
        load(_with_limits(config_file, "max_joint_rat = 0.5"))


def test_a_bad_supporting_value_is_refused(config_file):
    with pytest.raises(ConfigError, match="max_lag"):
        load(_with_limits(config_file, "max_joint_rate = 0.5\nmax_lag = 0"))


def test_several_profiles_can_coexist(config_file):
    config_file.write_text(
        config_file.read_text()
        + "\n[limits.teleop]\nmax_joint_rate = false\n"
        + "\n[limits.policy]\nmax_joint_rate = 0.2\n"
    )
    limits = load(config_file).limits
    assert limits["teleop"].max_joint_rate is None
    assert limits["policy"].max_joint_rate == 0.2


def test_limits_must_be_a_table(config_file):
    # Inserted before the first table, or it would land inside [capture.teleop].
    config_file.write_text(
        config_file.read_text().replace("version = 1", "version = 1\nlimits = 3", 1)
    )
    with pytest.raises(ConfigError, match=r"\[limits\]"):
        load(config_file)


def test_writing_arms_leaves_the_limits_section_alone(config_file):
    """The surgical-write invariant, extended to the new section."""
    _with_limits(config_file, "max_joint_rate = 0.5")
    write_arms(
        {"follower": ArmPorts("/dev/ttyACM5", "/dev/ttyACM6"),
         "leader": ArmPorts("/dev/ttyACM7", "/dev/ttyACM8")},
        config_file,
    )
    assert load(config_file).limits["teleop"].max_joint_rate == 0.5
