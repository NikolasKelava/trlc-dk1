"""``dk1 teleop`` at the CLI boundary — above all, that --dry-run connects to nothing."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dk1lab.cli import teleop_cmds
from dk1lab.cli.main import app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def no_run(monkeypatch):
    """Replace the thing that actually moves the arms; record if it is reached."""
    calls = []
    monkeypatch.setattr(teleop_cmds, "run", lambda *a, **kw: calls.append(kw))
    return calls


def dry(runner, config_file, *args):
    return runner.invoke(app, ["teleop", "-c", str(config_file), "--dry-run", *args])


def test_dry_run_never_reaches_the_part_that_moves_the_arms(runner, config_file, no_run):
    result = dry(runner, config_file)
    assert result.exit_code == 0
    assert no_run == []
    assert "nothing was connected and nothing moved" in result.output


def test_dry_run_works_with_no_hardware_attached(runner, config_file, no_run):
    """The fixture's /dev paths do not exist, which is the point of --dry-run."""
    assert dry(runner, config_file).exit_code == 0


def test_dry_run_shows_the_ports_it_would_open(runner, config_file, no_run):
    output = dry(runner, config_file).output
    for port in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3"):
        assert port in output


def test_dry_run_shows_the_camera_names_in_the_trained_order(runner, config_file, no_run):
    """Scoped to the cameras block: "left" also appears as an arm side above it."""
    cameras = dry(runner, config_file).output.split("cameras", 1)[1].split("loop", 1)[0]
    names = [line.split()[0] for line in cameras.strip().splitlines()]
    assert names == ["top", "left", "right"]


def test_dry_run_reports_that_teleop_runs_uncapped(runner, config_file, no_run):
    assert "track the leaders at full speed" in dry(runner, config_file).output


def test_a_cap_can_be_imposed_for_one_run(runner, config_file, no_run):
    assert "0.8 rad/s" in dry(runner, config_file, "--max-joint-rate", "0.8").output


def test_no_limit_and_an_explicit_rate_contradict_each_other(runner, config_file, no_run):
    result = dry(runner, config_file, "--no-limit", "--max-joint-rate", "0.8")
    assert result.exit_code != 0
    assert no_run == []


def test_a_cap_in_dk1_toml_is_picked_up(runner, config_file, no_run):
    config_file.write_text(config_file.read_text() + "\n[limits.teleop]\nmax_joint_rate = 0.6\n")
    assert "0.6 rad/s" in dry(runner, config_file).output


def test_no_cameras_says_so(runner, config_file, no_run):
    assert "none (--cameras to attach them)" in dry(runner, config_file, "--no-cameras").output


def test_an_unknown_control_mode_is_refused_before_anything_is_built(runner, config_file, no_run):
    result = dry(runner, config_file, "--control-mode=torque")
    assert result.exit_code != 0
    assert no_run == []


def test_a_nonpositive_rate_is_refused(runner, config_file, no_run):
    assert dry(runner, config_file, "--fps=0").exit_code != 0


def test_display_without_cameras_is_refused(runner, config_file, no_run):
    result = dry(runner, config_file, "--no-cameras", "--display")
    assert result.exit_code != 0
    assert no_run == []


def test_a_real_run_requires_the_devices_to_be_present(runner, config_file, no_run):
    """Without --dry-run the config is loaded with require_devices=True."""
    result = runner.invoke(app, ["teleop", "-c", str(config_file), "--yes"])
    assert result.exit_code != 0
    assert no_run == []


def test_the_help_warns_about_motion_and_about_the_leader_grippers(runner):
    output = runner.invoke(app, ["teleop", "--help"]).output
    assert "CAUSES MOTION" in output
    assert "LEADERS" in output


# --------------------------------------------------------------------------- #
# --record-dataset: the demonstration recorder
# --------------------------------------------------------------------------- #


def test_the_run_profile_defaults_to_the_tuned_one_and_shows_the_crop(runner, config_file, no_run):
    output = dry(runner, config_file).output
    assert "profile optimized" in output
    assert "crop" in output


def test_recording_defaults_to_the_level_playing_field(runner, config_file, no_run):
    """STUDY.md records the demonstrations uncropped and crops at training time."""
    output = dry(runner, config_file, "--record-dataset").output
    assert "profile common" in output
    assert "crop" not in output.split("cameras", 1)[1]


def test_recording_defaults_to_the_policy_capture_and_rate(runner, config_file, no_run):
    """The dataset's frame size and fps are what the rollout will run at."""
    output = dry(runner, config_file, "--record-dataset").output
    assert "capture [policy]" in output
    assert "target 30 Hz" in output


def test_the_capture_profile_is_still_selectable_on_its_own(runner, config_file, no_run):
    assert "capture [policy]" in dry(runner, config_file, "--capture", "policy").output


def test_an_unknown_run_profile_is_refused_before_anything_is_built(runner, config_file, no_run):
    result = dry(runner, config_file, "--profile", "levelled")
    assert result.exit_code != 0
    assert no_run == []


def test_recording_says_where_the_dataset_goes(runner, config_file, no_run):
    output = dry(runner, config_file, "--record-dataset").output
    assert "study/demos" in output
    assert "episodes are appended" in output


def test_recording_streams_the_video_by_default(runner, config_file, no_run):
    """Nowhere else: a human hand on a leader arm is not the experiment."""
    assert "AS THE ARMS MOVE" in dry(runner, config_file, "--record-dataset").output


def test_recording_without_cameras_is_refused(runner, config_file, no_run):
    result = dry(runner, config_file, "--record-dataset", "--no-cameras")
    assert result.exit_code != 0
    assert no_run == []


def test_recording_with_a_duration_is_refused(runner, config_file, no_run):
    """An episode ends on a keypress; a duration would mean two ways to end one."""
    result = dry(runner, config_file, "--record-dataset", "--duration", "60")
    assert result.exit_code != 0
    assert no_run == []


def test_a_scene_below_one_is_refused(runner, config_file, no_run):
    assert dry(runner, config_file, "--record-dataset", "--scene", "0").exit_code != 0


def test_dry_run_opens_no_dataset(runner, config_file, no_run, tmp_path):
    """--dry-run connects to nothing, and that includes not creating a dataset."""
    target = tmp_path / "demos"
    result = dry(runner, config_file, "--record-dataset", "--dataset-dir", str(target))
    assert result.exit_code == 0
    assert not target.exists()
