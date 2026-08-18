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


def test_dry_run_shows_the_speed_limit(runner, config_file, no_run):
    assert "1.5 rad/s" in dry(runner, config_file).output


def test_disabling_the_limit_is_reported_loudly(runner, config_file, no_run):
    assert "DISABLED" in dry(runner, config_file, "--no-limit").output


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
