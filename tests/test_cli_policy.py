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
    # The fixture config carries no [limits.policy], so this is POLICY_LIMITS —
    # raised to 1.0 rad/s on 2026-08-20 from the measured chunk demand.
    output = dry_run(runner, config_file, checkpoint_dir).output
    assert "1.0 rad/s" in output


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


def test_a_run_ends_at_the_home_pose_unless_told_not_to(
    runner, config_file, checkpoint_dir, no_motion
):
    """On by default since 2026-08-21: leaving the arms where the policy stopped wears them."""
    assert "HOME" in dry_run(runner, config_file, checkpoint_dir).output


def test_stopping_can_still_be_told_to_leave_the_arms_alone(
    runner, config_file, checkpoint_dir, no_motion
):
    output = dry_run(runner, config_file, checkpoint_dir, "--no-home").output
    assert "disconnect only" in output
    assert "HOME" not in output


def test_home_says_which_pose_it_would_use_when_the_file_names_none(
    runner, config_file, checkpoint_dir, no_motion
):
    # The test config has no [home] section, so --home falls back to the pose at
    # connect — and has to say so, because that is not a pose anyone chose.
    output = dry_run(runner, config_file, checkpoint_dir, "--home").output
    assert "captured at connect" in output


def test_sync_is_the_default_because_rtc_starved_the_queue_on_the_arms(
    runner, config_file, checkpoint_dir, no_motion
):
    """RTC discarded 27 of every 30 actions in situ; sync cannot discard any.

    ``n_action_steps`` is 30 on this checkpoint, so the sync engine executes the
    whole chunk and then blocks for one call — the same arrangement ``sim_eval``
    uses, which scored 100% in ManiSkill.
    """
    output = dry_run(runner, config_file, checkpoint_dir).output
    assert "inference SYNC" in output
    assert "execution horizon" not in output


def test_rtc_is_still_reachable_and_reports_its_blend(
    runner, config_file, checkpoint_dir, no_motion
):
    assert "inference     rtc" in dry_run(runner, config_file, checkpoint_dir, "--rtc").output


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


# --------------------------------------------------------------------------- #
# session — load once, roll out many times
# --------------------------------------------------------------------------- #


def dry_session(runner, config_file, checkpoint_dir, *args):
    return invoke(
        runner, config_file, "session", "--checkpoint", str(checkpoint_dir), "--dry-run", *args
    )


def test_a_session_dry_run_connects_nothing(runner, config_file, checkpoint_dir, no_motion):
    result = dry_session(runner, config_file, checkpoint_dir)
    assert result.exit_code == 0
    assert "the policy is loaded ONCE" in result.output
    assert no_motion["run"] == []


def test_a_session_needs_no_task_up_front(runner, config_file, checkpoint_dir, no_motion):
    """Unlike `run`: the instruction is what you type at the prompt."""
    assert dry_session(runner, config_file, checkpoint_dir).exit_code == 0


def test_a_session_says_what_happens_after_each_episode_not_after_the_run(
    runner, config_file, checkpoint_dir, no_motion
):
    output = dry_session(runner, config_file, checkpoint_dir, "--home").output
    assert "after each episode" in output
    assert "when the run ends" not in output


def test_a_session_reports_where_recordings_would_go(runner, config_file, checkpoint_dir, no_motion):
    output = dry_session(runner, config_file, checkpoint_dir, "--record").output
    assert "recording ON" in output


class FakeSession:
    """A :class:`~dk1lab.session.PolicySession` that records what it was asked to do."""

    def __init__(self):
        self.task = ""
        self.episodes = 0
        self.record = False
        self.record_dir = "recordings"
        self.duration_s = 180.0
        self.rollouts: list[str] = []
        self.homes = 0

    def rollout(self, task, *, home=None):
        from dk1lab.session import EpisodeOutcome

        self.task = task
        self.episodes += 1
        self.rollouts.append(task)
        return EpisodeOutcome(index=self.episodes, task=task, seconds=1.0, ended="duration")

    def home(self, pose=None):
        self.homes += 1
        from dk1lab.home import HomeReport

        return HomeReport(reached=True, seconds=1.0, commands=30, worst_key="left_joint_1.pos",
                          worst_error=0.001)


def drive(monkeypatch, lines: list[str]) -> FakeSession:
    """Run the prompt loop over ``lines``, as if they had been typed."""
    from dk1lab.cli import policy_cmds

    live = FakeSession()
    supply = iter(lines)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(supply))
    policy_cmds._session_loop(live, None)
    return live


def test_typing_an_instruction_runs_one_episode(monkeypatch):
    live = drive(monkeypatch, ["pick up the dice", ":quit"])
    assert live.rollouts == ["pick up the dice"]


def test_an_empty_line_runs_the_same_instruction_again(monkeypatch):
    live = drive(monkeypatch, ["pick up the dice", "", "", ":quit"])
    assert live.rollouts == ["pick up the dice"] * 3


def test_recording_toggles_without_ending_the_session(monkeypatch):
    live = drive(monkeypatch, [":record on", "pick up the dice", ":record off", ":quit"])
    assert live.record is False
    assert live.rollouts == ["pick up the dice"]


def test_the_prompt_survives_a_line_it_cannot_understand(monkeypatch):
    """Running `:recrod on` as an instruction would be the dangerous reading."""
    live = drive(monkeypatch, [":recrod on", "pick up the dice", ":quit"])
    assert live.rollouts == ["pick up the dice"]


def test_end_of_input_leaves_the_session(monkeypatch):
    from dk1lab.cli import policy_cmds

    live = FakeSession()

    def eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    policy_cmds._session_loop(live, None)
    assert live.rollouts == []


def test_a_failing_episode_does_not_end_the_session(monkeypatch):
    from dk1lab.cli import policy_cmds

    live = FakeSession()

    def explode(task, *, home=None):
        live.rollouts.append(task)
        raise RuntimeError("the cameras went away")

    live.rollout = explode
    supply = iter(["one", "two", ":quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(supply))
    policy_cmds._session_loop(live, None)
    assert live.rollouts == ["one", "two"], "the second episode was still offered"


def test_homing_from_the_prompt_moves_the_arms_without_a_rollout(monkeypatch):
    live = drive(monkeypatch, [":home", ":quit"])
    assert live.homes == 1
    assert live.rollouts == []


class FakeRecording:
    """An :class:`~dk1lab.record.EpisodeRecording` that records being discarded."""

    def __init__(self, path):
        self.path = path
        self.discarded = 0

    def summary(self):
        return f"recorded 100 ticks -> {self.path}"

    def discard(self):
        self.discarded += 1
        return True


def keep_or_not(monkeypatch, answer: str, path="recordings/0001_dice.rrd"):
    from dk1lab.cli import policy_cmds

    recording = FakeRecording(path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(policy_cmds.typer, "confirm", lambda *a, **kw: answer == "y")
    kept = policy_cmds._keep_recording(recording)
    return kept, recording


def test_an_episode_is_kept_when_you_say_so(monkeypatch):
    kept, recording = keep_or_not(monkeypatch, "y")
    assert kept is True
    assert recording.discarded == 0


def test_declining_deletes_the_file_rather_than_leaving_it(monkeypatch):
    kept, recording = keep_or_not(monkeypatch, "n")
    assert kept is False
    assert recording.discarded == 1


def test_a_non_interactive_run_keeps_the_episode(monkeypatch):
    """Nobody is there to answer, and throwing away an attempt is the worse default."""
    from dk1lab.cli import policy_cmds

    recording = FakeRecording("recordings/0001_dice.rrd")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert policy_cmds._keep_recording(recording) is True
    assert recording.discarded == 0


def test_nothing_is_asked_when_the_episode_was_not_recorded(monkeypatch):
    from dk1lab.cli import policy_cmds

    def asked(*_args, **_kwargs):
        pytest.fail("nothing was recorded, so there is nothing to keep")

    monkeypatch.setattr(policy_cmds.typer, "confirm", asked)
    assert policy_cmds._keep_recording(None) is False


def test_the_prompt_names_the_file_the_next_episode_will_write(monkeypatch, tmp_path):
    """The session's own count restarts every time; the file index does not."""
    from dk1lab.cli import policy_cmds

    (tmp_path / "0007_pick-up-the-dice.rrd").touch()
    live = FakeSession()
    live.record_dir = tmp_path
    live.record = True
    assert "episode 8" in policy_cmds._prompt(live)
    live.record = False
    assert "episode 1" in policy_cmds._prompt(live)


def test_the_gripper_inversion_is_on_by_default_now_that_it_is_confirmed(
    runner, config_file, checkpoint_dir, no_motion
):
    """The checkpoint speaks YAM (1=open) and this cell is 0=open. Confirmed on the arms."""
    assert "ON for" in dry_run(runner, config_file, checkpoint_dir).output


def test_turning_the_inversion_off_says_the_grippers_will_work_backwards(
    runner, config_file, checkpoint_dir, no_motion
):
    output = dry_run(runner, config_file, checkpoint_dir, "--no-invert-gripper").output
    assert "BACKWARDS" in output or "backwards" in output


def test_the_sim_server_does_not_invert(runner, config_file, checkpoint_dir, monkeypatch):
    """`sim_eval` already speaks the checkpoint's own convention on the wire.

    Inverting there would test the policy through a sign error the simulator
    does not have — which is precisely what makes the sim a control.
    """
    served: dict = {}
    monkeypatch.setattr("dk1lab.serve.serve", lambda *a, **kw: served.update(kw))
    invoke(runner, config_file, "serve", "--checkpoint", str(checkpoint_dir), "--no-warmup")
    assert served["invert_gripper"] is False
