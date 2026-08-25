"""The scored session at the CLI boundary, and `dk1 study`.

The robot half of a scored session is already covered by `test_session.py` and
`test_cli_policy.py`. What is new here is the paperwork: that a scored row lands
in the right file under the right scene, that its recordings go somewhere of
their own and are not offered for deletion, and that an operator who mistypes a
score is asked again rather than losing the attempt.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dk1lab import study
from dk1lab.cli import policy_cmds
from dk1lab.cli.main import app
from dk1lab.study import Attempt


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def scored(tmp_path):
    """A scored row of the default shape, writing into a temporary directory."""
    return policy_cmds._study(
        "A0",
        scenes=3,
        attempts=3,
        scores_dir=tmp_path / "scores",
        scene_dir=Path("study/scene"),
    )


def outcome(recording=None, seconds=42.0):
    return SimpleNamespace(recording=recording, seconds=seconds)


def answers(monkeypatch, *lines):
    """Feed the score prompt a script, and record what it asked for."""
    asked: list[str] = []
    queue = list(lines)

    def fake_input(prompt: str = "") -> str:
        asked.append(prompt)
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(policy_cmds.sys.stdin, "isatty", lambda: True, raising=False)
    return asked


# --------------------------------------------------------------------------- #
# Building and resuming a scored row
# --------------------------------------------------------------------------- #


def test_no_study_flag_is_no_scoring():
    assert policy_cmds._study(
        None, scenes=3, attempts=3, scores_dir=Path("x"), scene_dir=Path("y")
    ) is None


def test_a_scored_row_resumes_from_its_own_csv(tmp_path):
    path = study.scores_path("A0", tmp_path)
    for n in (1, 2, 3):
        study.append(path, Attempt(scene=1, attempt=n, score=0))
    scored = policy_cmds._study(
        "A0", scenes=3, attempts=3, scores_dir=tmp_path, scene_dir=Path("study/scene")
    )
    assert scored.plan.scene == 2
    assert scored.plan.total == 3


def test_an_impossible_study_shape_is_refused_at_the_command_line(tmp_path):
    import typer

    with pytest.raises(typer.BadParameter):
        policy_cmds._study(
            "A0", scenes=0, attempts=3, scores_dir=tmp_path, scene_dir=Path("study/scene")
        )


# --------------------------------------------------------------------------- #
# Scoring one attempt
# --------------------------------------------------------------------------- #


def test_an_attempt_is_written_under_the_live_scene(monkeypatch, scored):
    answers(monkeypatch, "3 left dropped it")
    policy_cmds._score_attempt(scored, outcome())

    rows = study.read(scored.path)
    assert len(rows) == 1
    assert (rows[0].scene, rows[0].attempt, rows[0].score) == (1, 1, 3)
    assert rows[0].arm == "left"
    assert rows[0].note == "dropped it"
    assert scored.plan.next_attempt == 2


def test_a_success_takes_its_time_from_the_episode(monkeypatch, scored):
    """One line, no second prompt: the episode ends at the success, so it knows."""
    asked = answers(monkeypatch, "5 left")
    policy_cmds._score_attempt(scored, outcome(seconds=42.0))

    assert asked == ["  score> "]
    assert study.read(scored.path)[0].seconds == pytest.approx(42.0)


def test_a_time_typed_inline_overrides_the_episode_length(monkeypatch, scored):
    """For the attempt that succeeded early and the arms went on moving."""
    answers(monkeypatch, "5 right 21.4")
    policy_cmds._score_attempt(scored, outcome(seconds=42.0))
    assert study.read(scored.path)[0].seconds == pytest.approx(21.4)


def test_only_a_success_carries_a_time(monkeypatch, scored):
    answers(monkeypatch, "3 left nearly")
    policy_cmds._score_attempt(scored, outcome(seconds=42.0))
    assert study.read(scored.path)[0].seconds is None


def test_a_line_that_cannot_be_understood_is_asked_again(monkeypatch, scored):
    """The attempt has already happened. Losing it to a typo is the failure."""
    answers(monkeypatch, "three", "3 left")
    policy_cmds._score_attempt(scored, outcome())
    assert study.read(scored.path)[0].score == 3


def test_skip_records_nothing_and_costs_no_attempt(monkeypatch, scored):
    answers(monkeypatch, "skip")
    policy_cmds._score_attempt(scored, outcome())
    assert study.read(scored.path) == []
    assert scored.plan.next_attempt == 1


def test_the_episode_reference_is_written_with_the_score(monkeypatch, scored):
    answers(monkeypatch, "0")
    policy_cmds._score_attempt(scored, outcome(recording=SimpleNamespace(index=4)))
    assert study.read(scored.path)[0].episode == "4"


def test_a_non_interactive_session_says_the_attempt_was_not_scored(monkeypatch, scored):
    monkeypatch.setattr(policy_cmds.sys.stdin, "isatty", lambda: False, raising=False)
    policy_cmds._score_attempt(scored, outcome())
    assert study.read(scored.path) == []


def test_the_scenes_are_walked_in_order(monkeypatch, scored):
    answers(monkeypatch, *["0"] * 4)
    for _ in range(3):
        policy_cmds._score_attempt(scored, outcome())
    assert scored.plan.scene == 2
    policy_cmds._score_attempt(scored, outcome())
    assert [r.scene for r in study.read(scored.path)] == [1, 1, 1, 2]


# --------------------------------------------------------------------------- #
# The prompt, the banner and the recordings
# --------------------------------------------------------------------------- #


def test_the_prompt_names_the_row_the_scene_and_the_attempt(scored):
    live = SimpleNamespace(
        duration_s=60.0, record=False, record_dataset=True, episodes=0, record_dir=Path("x")
    )
    line = policy_cmds._prompt(live, scored)
    assert "A0 scene 1/3, attempt 1/3" in line
    assert policy_cmds._prompt(live) == policy_cmds._prompt(live, None)


def test_the_scene_banner_is_printed_once_per_scene(scored, capsys):
    policy_cmds._study_banner(scored)
    policy_cmds._study_banner(scored)
    printed = capsys.readouterr().out
    assert printed.count("scene 1 of 3") == 1

    scored.plan.record(Attempt(scene=1, attempt=1, score=0))
    policy_cmds._study_banner(scored)
    assert "scene 1 of 3" not in capsys.readouterr().out


def test_a_scored_row_keeps_every_recording_without_asking(monkeypatch):
    """A failure is evidence. The only prompt after an attempt is the score."""
    asked: list[str] = []
    monkeypatch.setattr(policy_cmds.typer, "confirm", lambda *a, **kw: asked.append(a) or False)
    monkeypatch.setattr(policy_cmds.sys.stdin, "isatty", lambda: True, raising=False)

    recording = SimpleNamespace(
        summary=lambda: "recorded", discard=lambda: True, keep=lambda: True
    )
    assert policy_cmds._keep_recording(recording, ask=False) is True
    assert asked == []


def test_a_scored_rows_recordings_go_to_a_directory_of_their_own(
    runner, config_file, checkpoint_dir
):
    """Never into recordings/, which holds six unscored tasks from before the study."""
    result = runner.invoke(
        app,
        ["policy", "session", "-c", str(config_file), "--checkpoint", str(checkpoint_dir),
         "--study", "A0", "--profile", "common", "--record", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert str(Path("study/rrd/A0")) in result.output
    assert "A0: 3 scene(s) x 3 attempts = 9" in result.output


def test_a_scored_row_that_records_nothing_says_so(runner, config_file, checkpoint_dir):
    result = runner.invoke(
        app,
        ["policy", "session", "-c", str(config_file), "--checkpoint", str(checkpoint_dir),
         "--study", "A0", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "nothing is being recorded" in result.output


def test_the_prompt_loop_jumps_scenes_and_refuses_an_impossible_one(monkeypatch, scored, capsys):
    """`:scene` is for a VOID attempt, and a scene outside the study is a typo."""
    live = SimpleNamespace(
        duration_s=60.0, record=False, record_dataset=False, episodes=0,
        record_dir=Path("x"), task="",
    )
    answers(monkeypatch, ":scene 3", ":scene 9", ":quit")
    policy_cmds._session_loop(live, None, study=scored)

    assert scored.plan.scene == 3
    assert "scenes 1-3, not 9" in capsys.readouterr().out


def test_scene_without_a_study_says_there_is_no_row(monkeypatch, capsys):
    live = SimpleNamespace(
        duration_s=0.0, record=False, record_dataset=False, episodes=0,
        record_dir=Path("x"), task="",
    )
    answers(monkeypatch, ":scene 2", ":quit")
    policy_cmds._session_loop(live, None)
    assert "not scoring a row" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# dk1 study scores
# --------------------------------------------------------------------------- #


def test_study_scores_prints_the_grid(runner, tmp_path):
    path = study.scores_path("A0", tmp_path)
    study.append(path, Attempt(scene=1, attempt=1, score=5, arm="left", seconds=20.0))
    study.append(path, Attempt(scene=1, attempt=2, score=0))

    result = runner.invoke(app, ["study", "scores", "A0", "--scores-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "scene 1: 5 0" in result.output
    assert "overall: 1/2 success" in result.output


def test_study_scores_on_an_unscored_row_is_an_error(runner, tmp_path):
    result = runner.invoke(app, ["study", "scores", "B1", "--scores-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "no attempts scored yet" in result.output


def test_study_photo_needs_to_know_which_scene(runner, config_file):
    result = runner.invoke(app, ["study", "photo", "-c", str(config_file)])
    assert result.exit_code != 0
    assert "which scene" in result.output


def test_study_photo_opens_a_camera_and_nothing_else(runner, config_file, tmp_path, monkeypatch):
    """No motor, no arm: one still off the top camera, written where it was asked for."""
    grabbed: dict[str, object] = {}

    def fake_capture(device, **kwargs):
        grabbed.update(device=device, **kwargs)
        return "an image"

    monkeypatch.setattr("dk1lab.discovery.preview.capture_still", fake_capture)
    monkeypatch.setattr(
        "dk1lab.discovery.preview.save_still",
        lambda image, path: (Path(path).write_bytes(b"jpeg"), Path(path))[1],
    )
    result = runner.invoke(
        app,
        ["study", "photo", "-c", str(config_file), "--scene", "2",
         "--scene-dir", str(tmp_path), "--no-open"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "2.jpg").is_file()
    assert grabbed["rotation"] == 180
