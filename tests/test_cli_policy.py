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
            rtc_ms=(330.0, 330.0),
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
            rtc_ms=(330.0, 330.0),
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
    assert "disconnect only" in output
    assert "HOME" not in output


def test_home_shows_up_when_asked_for(runner, config_file, checkpoint_dir, no_motion):
    output = dry_run(runner, config_file, checkpoint_dir, "--home").output
    assert "HOME" in output


def test_home_says_which_pose_it_would_use_when_the_file_names_none(
    runner, config_file, checkpoint_dir, no_motion
):
    # The test config has no [home] section, so --home falls back to the pose at
    # connect — and has to say so, because that is not a pose anyone chose.
    output = dry_run(runner, config_file, checkpoint_dir, "--home").output
    assert "captured at connect" in output


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


def test_smoke_reports_the_rtc_latency_and_the_blend_it_leaves(runner, config_file, checkpoint_dir):
    """The sync number is not the deployment number, and reporting only it misled us."""
    from dk1lab import policy
    from dk1lab.policy import Inversion, SmokeResult

    monkeypatch_result = SmokeResult(
        action_keys=tuple(ACTION_KEYS),
        action=tuple(0.0 for _ in ACTION_KEYS),
        chunk_ms=(172.0, 172.0),
        pop_ms=(2.0, 2.0),
        warmup_ms=900.0,
        rtc_ms=(270.0, 270.0),
        peak_gpu_gib=12.5,
        inversion=Inversion(("a", "b"), (1.0,), (0.0,)),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(policy, "smoke", lambda spec, **kw: monkeypatch_result)
        output = invoke(runner, config_file, "smoke", "--checkpoint", str(checkpoint_dir)).output

    assert "RTC call" in output
    assert "270 ms" in output
    assert "9 ticks of inference delay" in output
    assert "blends consecutive chunks over 11 steps" in output


def test_smoke_flags_an_rtc_latency_the_default_horizon_cannot_absorb(
    runner, config_file, checkpoint_dir
):
    from dk1lab import policy
    from dk1lab.policy import Inversion, SmokeResult

    slow = SmokeResult(
        action_keys=tuple(ACTION_KEYS),
        action=tuple(0.0 for _ in ACTION_KEYS),
        chunk_ms=(172.0, 172.0),
        pop_ms=(2.0, 2.0),
        warmup_ms=900.0,
        rtc_ms=(600.0, 600.0),  # 18 ticks: no room inside a horizon of 20
        peak_gpu_gib=12.5,
        inversion=Inversion(("a", "b"), (1.0,), (0.0,)),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(policy, "smoke", lambda spec, **kw: slow)
        output = invoke(runner, config_file, "smoke", "--checkpoint", str(checkpoint_dir)).output

    assert "no blend" in output.lower()
    assert "judder" in output


# --------------------------------------------------------------------------- #
# home — the pose a run ends at
# --------------------------------------------------------------------------- #


HOME_SECTION = """
[home]
left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]
right = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.0]
"""


@pytest.fixture
def no_sweep(monkeypatch):
    """Replace the two functions that touch the arms; record if they are reached."""
    from dk1lab import config as config_module
    from dk1lab import home as home_module

    calls: dict[str, list] = {"sweep": [], "capture": []}
    # The fixture config names /dev nodes that do not exist on a test machine.
    monkeypatch.setattr(config_module, "check_devices", lambda *a, **kw: None)
    monkeypatch.setattr(
        home_module, "sweep_to_home", lambda *a, **kw: calls["sweep"].append(kw) or _reached()
    )
    monkeypatch.setattr(
        home_module, "capture_pose", lambda *a, **kw: calls["capture"].append(kw) or _pose()
    )
    return calls


def _reached():
    from dk1lab.home import HomeReport

    return HomeReport(
        reached=True, aborted=False, steps=10, elapsed_s=1.0, worst_key="left_joint_1.pos",
        worst_error=0.001,
    )


def _pose():
    from dk1lab.config import HomePose

    return HomePose(left=(0.0,) * 7, right=(0.0,) * 7)


def test_showing_the_home_pose_touches_nothing(runner, config_file, no_sweep):
    config_file.write_text(config_file.read_text() + HOME_SECTION)
    result = invoke(runner, config_file, "home", "--show")
    assert result.exit_code == 0
    assert "+0.100" in result.output
    assert no_sweep["sweep"] == [] and no_sweep["capture"] == []


def test_showing_says_so_when_no_home_has_been_captured(runner, config_file, no_sweep):
    result = invoke(runner, config_file, "home", "--show")
    assert result.exit_code == 1
    assert "no [home] section" in result.output


def test_driving_home_without_a_configured_pose_refuses_before_connecting(
    runner, config_file, no_sweep
):
    """The fallback to the connect-time pose exists for a run that is already
    under way. On demand there is nothing to fall back to, so it refuses."""
    result = invoke(runner, config_file, "home", "--yes")
    assert result.exit_code == 2
    assert no_sweep["sweep"] == []


def test_driving_home_needs_a_confirmation(runner, config_file, no_sweep):
    config_file.write_text(config_file.read_text() + HOME_SECTION)
    result = runner.invoke(app, ["policy", "home", "-c", str(config_file)], input="n\n")
    assert result.exit_code != 0
    assert no_sweep["sweep"] == []


def test_driving_home_sweeps_at_the_policy_cap(runner, config_file, no_sweep):
    config_file.write_text(
        config_file.read_text() + HOME_SECTION + "\n[limits.policy]\nmax_joint_rate = 0.25\n"
    )
    result = invoke(runner, config_file, "home", "--yes")
    assert result.exit_code == 0
    assert no_sweep["sweep"][0]["limits"].max_joint_rate == 0.25
    assert no_sweep["sweep"][0]["target"]["left_joint_1.pos"] == 0.1


def test_capturing_writes_the_section_and_drives_nothing(runner, config_file, no_sweep):
    from dk1lab.config import load

    result = invoke(runner, config_file, "home", "--capture", "--yes")
    assert result.exit_code == 0
    assert no_sweep["sweep"] == []
    assert load(config_file).home == _pose()


def test_capturing_needs_a_confirmation_because_connecting_energises_the_arms(
    runner, config_file, no_sweep
):
    result = runner.invoke(
        app, ["policy", "home", "--capture", "-c", str(config_file)], input="n\n"
    )
    assert result.exit_code != 0
    assert no_sweep["capture"] == []


def test_the_run_uses_the_configured_pose_when_there_is_one(
    runner, config_file, checkpoint_dir, no_motion
):
    config_file.write_text(config_file.read_text() + HOME_SECTION)
    output = dry_run(runner, config_file, checkpoint_dir, "--home").output
    assert "+0.100" in output
    assert "captured at connect" not in output
