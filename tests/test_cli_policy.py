"""``dk1 policy`` at the CLI boundary.

The property that matters most here is the same one as for teleoperation, one
step earlier: the commands that would energise or drive the arms must be
reachable only past a confirmation, and the inspect-only paths must not reach
them at all.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from conftest import CHECKPOINT_CONFIG

from dk1lab import policy
from dk1lab.cli.main import app
from dk1lab.layout import ACTION_KEYS
from dk1lab.policy import Inversion, SmokeResult


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def no_motion(monkeypatch):
    """Replace everything that connects to the robot; record if it is reached."""
    calls: dict[str, list] = {"run": [], "dryrun": []}
    monkeypatch.setattr(policy, "run", lambda *a, **kw: calls["run"].append(kw))
    monkeypatch.setattr(policy, "dryrun", lambda *a, **kw: calls["dryrun"].append(kw) or [])
    return calls


def invoke(runner, config_file, *args):
    return runner.invoke(app, ["policy", *args, "-c", str(config_file)])


# --------------------------------------------------------------------------- #
# check — JSON in, verdict out
# --------------------------------------------------------------------------- #


def test_check_accepts_a_checkpoint_that_matches_this_cell(runner, config_file, checkpoint_dir):
    result = invoke(runner, config_file, "check", "--checkpoint", str(checkpoint_dir))
    assert result.exit_code == 0
    assert "usable on this cell" in result.output


def test_check_rejects_the_wrong_normalisation_statistics(runner, config_file, checkpoint_dir):
    raw = dict(CHECKPOINT_CONFIG, norm_tag="something_else")
    (checkpoint_dir / "config.json").write_text(json.dumps(raw))
    result = invoke(runner, config_file, "check", "--checkpoint", str(checkpoint_dir))
    assert result.exit_code == 1


def test_check_on_a_missing_checkpoint_says_so_rather_than_downloading(runner, config_file):
    result = invoke(runner, config_file, "check", "--checkpoint", "/no/such/checkpoint")
    assert result.exit_code == 2


def test_check_reports_that_the_inversion_is_applied_by_us(runner, config_file, checkpoint_dir):
    """The one thing a reader of the output must not get wrong."""
    output = invoke(runner, config_file, "check", "--checkpoint", str(checkpoint_dir)).output
    assert "--policy.joint_signs does nothing" in output


def test_the_checkpoint_defaults_to_the_one_in_dk1_toml(runner, config_file):
    """The fixture points [policy] at a path that does not exist, which must show."""
    result = invoke(runner, config_file, "check")
    assert result.exit_code == 2
    assert "molmoact2_bf16" in result.output


# --------------------------------------------------------------------------- #
# smoke — no robot, and it says so
# --------------------------------------------------------------------------- #


def test_smoke_never_touches_the_robot(runner, config_file, checkpoint_dir, monkeypatch):
    captured = {}

    def fake_smoke(spec, **kwargs):
        captured.update(kwargs)
        return SmokeResult(
            action_keys=tuple(ACTION_KEYS),
            action=tuple(0.1 for _ in ACTION_KEYS),
            chunk_ms=(170.0, 172.0),
            pop_ms=(2.0, 2.0),
            warmup_ms=900.0,
            peak_gpu_gib=12.5,
            inversion=Inversion(("a", "b"), (1.0,), (0.0,)),
        )

    monkeypatch.setattr(policy, "smoke", fake_smoke)
    result = invoke(runner, config_file, "smoke", "--checkpoint", str(checkpoint_dir))

    assert result.exit_code == 0
    assert "Nothing was connected" in result.output
    # The synthetic frame is the size the policy capture profile provides.
    assert (captured["width"], captured["height"]) == (640, 360)


def test_smoke_says_when_inference_is_too_slow_for_the_control_loop(
    runner, config_file, checkpoint_dir, monkeypatch
):
    monkeypatch.setattr(
        policy,
        "smoke",
        lambda spec, **kw: SmokeResult(
            action_keys=tuple(ACTION_KEYS),
            action=tuple(0.0 for _ in ACTION_KEYS),
            chunk_ms=(172.0, 172.0),
            pop_ms=(2.0, 2.0),
            warmup_ms=900.0,
            peak_gpu_gib=12.5,
            inversion=Inversion(("a", "b"), (1.0,), (0.0,)),
        ),
    )
    output = invoke(runner, config_file, "smoke", "--checkpoint", str(checkpoint_dir)).output
    assert "--rtc" in output


# --------------------------------------------------------------------------- #
# dryrun — energises the arms, so --build-only must not
# --------------------------------------------------------------------------- #


def test_build_only_never_reaches_the_part_that_connects(
    runner, config_file, checkpoint_dir, no_motion
):
    result = invoke(
        runner,
        config_file,
        "dryrun",
        "--task",
        "pick up the pen",
        "--checkpoint",
        str(checkpoint_dir),
        "--build-only",
    )
    assert result.exit_code == 0
    assert no_motion["dryrun"] == []
    assert "nothing was connected and nothing moved" in result.output


def test_build_only_works_with_no_hardware_attached(
    runner, config_file, checkpoint_dir, no_motion
):
    """The fixture's /dev paths do not exist — that is the point."""
    result = invoke(
        runner,
        config_file,
        "dryrun",
        "--task",
        "t",
        "--checkpoint",
        str(checkpoint_dir),
        "--build-only",
    )
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# run — the one that drives the arms
# --------------------------------------------------------------------------- #


def dry_run(runner, config_file, checkpoint_dir, *args):
    return invoke(
        runner,
        config_file,
        "run",
        "--task",
        "pick up the pen",
        "--checkpoint",
        str(checkpoint_dir),
        "--dry-run",
        *args,
    )


def test_dry_run_never_reaches_the_rollout(runner, config_file, checkpoint_dir, no_motion):
    result = dry_run(runner, config_file, checkpoint_dir)
    assert result.exit_code == 0
    assert no_motion["run"] == []


def test_dry_run_shows_the_speed_cap_that_would_apply(
    runner, config_file, checkpoint_dir, no_motion
):
    output = dry_run(runner, config_file, checkpoint_dir).output
    assert "0.3 rad/s" in output


def test_dry_run_shows_when_the_cap_has_been_removed(
    runner, config_file, checkpoint_dir, no_motion
):
    output = dry_run(runner, config_file, checkpoint_dir, "--no-limit").output
    assert "NONE" in output


def test_removing_the_cap_and_setting_it_contradict_each_other(
    runner, config_file, checkpoint_dir, no_motion
):
    result = dry_run(runner, config_file, checkpoint_dir, "--no-limit", "--max-joint-rate", "0.5")
    assert result.exit_code != 0


def test_dry_run_shows_that_stopping_does_not_move_the_arms(
    runner, config_file, checkpoint_dir, no_motion
):
    output = dry_run(runner, config_file, checkpoint_dir).output
    assert "return to start pose on stop: no" in output


def test_return_home_shows_up_when_asked_for(runner, config_file, checkpoint_dir, no_motion):
    output = dry_run(runner, config_file, checkpoint_dir, "--return-home").output
    assert "return to start pose on stop: YES" in output


def test_rtc_is_the_default_for_a_seven_billion_parameter_model(
    runner, config_file, checkpoint_dir, no_motion
):
    assert "inference     rtc" in dry_run(runner, config_file, checkpoint_dir).output


def test_an_invalid_control_mode_is_refused_before_anything_is_built(
    runner, config_file, checkpoint_dir, no_motion
):
    result = dry_run(runner, config_file, checkpoint_dir, "--control-mode", "torque")
    assert result.exit_code != 0
    assert no_motion["run"] == []


def test_running_needs_a_task(runner, config_file, checkpoint_dir, no_motion):
    """MolmoAct2 is conditioned on the instruction; there is no default worth having."""
    result = invoke(runner, config_file, "run", "--checkpoint", str(checkpoint_dir), "--dry-run")
    assert result.exit_code != 0
