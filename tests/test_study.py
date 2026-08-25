"""The scored study's bookkeeping: the score grammar, the CSV, the scene walk.

None of this touches a robot, a camera or a checkpoint. It is the part of
`STUDY.md` that can be wrong quietly — a score line understood as something else,
an attempt appended under the wrong scene, a resumed session starting at the
wrong place — so it is the part with tests.
"""

from __future__ import annotations

import pytest

from dk1lab import study
from dk1lab.study import Attempt, ScenePlan, ScoreError


# --------------------------------------------------------------------------- #
# parse_score
# --------------------------------------------------------------------------- #


def test_score_zero_needs_no_arm():
    parsed = study.parse_score("0")
    assert (parsed.score, parsed.arm, parsed.note, parsed.seconds) == (0, "none", "", None)


def test_anything_that_happened_needs_an_arm():
    """The arm column exists to settle an open question; a blank is a lost row."""
    with pytest.raises(ScoreError, match="which arm"):
        study.parse_score("3")


def test_arm_and_note():
    parsed = study.parse_score("3 left dropped it short of the bowl")
    assert parsed.score == 3
    assert parsed.arm == "left"
    assert parsed.note == "dropped it short of the bowl"
    assert parsed.seconds is None


@pytest.mark.parametrize(
    ("text", "arm"), [("2 l", "left"), ("2 R", "right"), ("2 both", "both"), ("0 n", "none")]
)
def test_arm_aliases(text, arm):
    assert study.parse_score(text).arm == arm


def test_time_on_a_success():
    parsed = study.parse_score("5 right 21.4")
    assert parsed.score == 5
    assert parsed.arm == "right"
    assert parsed.seconds == pytest.approx(21.4)
    assert parsed.note == ""


def test_time_is_refused_on_anything_but_a_success():
    """`4 left 30` almost certainly means the episode length, not time-to-success."""
    with pytest.raises(ScoreError, match="only on a 5"):
        study.parse_score("4 left 30")


def test_a_number_that_is_not_a_time_stays_in_the_note():
    parsed = study.parse_score("2 left knocked it 3 cm sideways")
    assert parsed.note == "knocked it 3 cm sideways"


@pytest.mark.parametrize("text", ["", "   ", "yes", "-1 left", "6 left"])
def test_bad_score_lines(text):
    with pytest.raises(ScoreError):
        study.parse_score(text)


# --------------------------------------------------------------------------- #
# The CSV
# --------------------------------------------------------------------------- #


def test_append_writes_a_header_once_and_round_trips(tmp_path):
    path = tmp_path / "A0.csv"
    first = Attempt(scene=1, attempt=1, score=5, arm="left", episode="0", seconds=21.4)
    second = Attempt(scene=1, attempt=2, score=0, arm="none", note="no motion", episode="1")
    study.append(path, first)
    study.append(path, second)

    assert path.read_text().count("scene,attempt") == 1
    rows = study.read(path)
    assert [r.scene for r in rows] == [1, 1]
    assert rows[0].seconds == pytest.approx(21.4)
    assert rows[1].seconds is None
    assert rows[1].note == "no motion"
    assert rows[0].episode == "0"


def test_a_note_with_a_comma_survives(tmp_path):
    """The note is free text an operator types in a hurry; CSV quoting is the job."""
    path = tmp_path / "A0.csv"
    study.append(path, Attempt(scene=2, attempt=1, score=2, arm="right", note="hit it, spun"))
    assert study.read(path)[0].note == "hit it, spun"


def test_read_of_a_missing_file_is_no_attempts(tmp_path):
    assert study.read(tmp_path / "nothing.csv") == []


def test_a_corrupt_row_is_a_loud_error(tmp_path):
    path = tmp_path / "A0.csv"
    path.write_text("scene,attempt,episode,score,seconds,arm,note\nx,1,,5,,left,\n")
    with pytest.raises(ScoreError, match="not a score row"):
        study.read(path)


def test_scores_path():
    assert study.scores_path("R0").name == "R0.csv"


# --------------------------------------------------------------------------- #
# ScenePlan
# --------------------------------------------------------------------------- #


def test_a_fresh_plan_starts_at_scene_one_attempt_one():
    plan = ScenePlan()
    assert (plan.scene, plan.next_attempt) == (1, 1)
    assert plan.wanted == 9
    assert plan.scene_starting
    assert not plan.complete


def test_the_scene_advances_only_when_it_is_full():
    plan = ScenePlan(scenes=3, attempts=3)
    for n in range(1, 4):
        assert plan.scene == 1
        plan.record(Attempt(scene=1, attempt=n, score=0))
    assert plan.scene == 2
    assert plan.next_attempt == 1
    assert plan.scene_starting


def test_a_full_row_is_complete():
    plan = ScenePlan(scenes=2, attempts=2)
    for scene in (1, 2):
        for attempt in (1, 2):
            plan.record(Attempt(scene=scene, attempt=attempt, score=1, arm="left"))
    assert plan.complete
    assert plan.total == plan.wanted == 4


def test_resuming_reads_the_position_off_the_attempts_already_scored():
    done = [
        Attempt(scene=1, attempt=1, score=0),
        Attempt(scene=1, attempt=2, score=0),
        Attempt(scene=1, attempt=3, score=0),
        Attempt(scene=2, attempt=1, score=5, arm="left"),
    ]
    plan = ScenePlan(done=done)
    assert (plan.scene, plan.next_attempt) == (2, 2)
    assert plan.total == 4
    assert not plan.scene_starting


def test_resuming_a_complete_row_stays_complete():
    done = [Attempt(scene=s, attempt=a, score=0) for s in (1, 2, 3) for a in (1, 2, 3)]
    plan = ScenePlan(done=done)
    assert plan.complete


def test_jumping_back_numbers_the_extra_attempt_onward():
    """A void attempt is re-run; the CSV records that it was the fourth."""
    plan = ScenePlan()
    for n in (1, 2, 3):
        plan.record(Attempt(scene=1, attempt=n, score=0))
    assert plan.scene == 2
    plan.jump(1)
    assert plan.next_attempt == 4


def test_jumping_outside_the_study_is_refused():
    plan = ScenePlan(scenes=3)
    with pytest.raises(ValueError, match="scenes 1-3"):
        plan.jump(4)
    with pytest.raises(ValueError):
        plan.jump(0)


@pytest.mark.parametrize(("scenes", "attempts"), [(0, 3), (3, 0), (-1, 1)])
def test_a_study_needs_scenes_and_attempts(scenes, attempts):
    with pytest.raises(ValueError):
        ScenePlan(scenes=scenes, attempts=attempts)


def test_the_label_and_the_banner_name_the_layout():
    plan = ScenePlan()
    assert plan.label() == "scene 1/3, attempt 1/3"
    assert "scene 1 of 3" in plan.banner()
    assert "study/scene/1.jpg" in plan.banner()


# --------------------------------------------------------------------------- #
# Reading a row back
# --------------------------------------------------------------------------- #


def test_the_grid_separates_two_rows_with_the_same_success_rate():
    clustered = [Attempt(scene=1, attempt=n, score=5, arm="left") for n in (1, 2, 3)]
    clustered += [Attempt(scene=s, attempt=n, score=0) for s in (2, 3) for n in (1, 2, 3)]
    spread = [
        Attempt(scene=s, attempt=n, score=5 if n == 1 else 0, arm="left" if n == 1 else "none")
        for s in (1, 2, 3)
        for n in (1, 2, 3)
    ]
    assert sum(a.success for a in clustered) == sum(a.success for a in spread) == 3
    assert study.grid(clustered) != study.grid(spread)
    assert "overall: 3/9 success" in study.grid(clustered)[-2]


def test_the_grid_counts_the_arms():
    rows = [
        Attempt(scene=1, attempt=1, score=5, arm="left"),
        Attempt(scene=1, attempt=2, score=2, arm="right"),
        Attempt(scene=1, attempt=3, score=0, arm="none"),
    ]
    lines = study.grid(rows)
    assert any("left 1, right 1" in line for line in lines)


# --------------------------------------------------------------------------- #
# The episode reference — the only join from a score to its frames
# --------------------------------------------------------------------------- #


class _DatasetEpisode:
    index = 7


class _Rrd:
    path = "study/rrd/R0/0003_put-the-dice-in-the-bowl.rrd"


class _Combined:
    def __init__(self, *reports):
        self.reports = reports


def test_a_dataset_episode_is_referenced_by_its_index():
    assert study.episode_reference(_DatasetEpisode()) == "7"


def test_an_rrd_is_referenced_by_its_file_stem():
    assert study.episode_reference(_Rrd()) == "0003_put-the-dice-in-the-bowl"


def test_the_dataset_index_wins_when_an_attempt_produced_both():
    assert study.episode_reference(_Combined(_Rrd(), _DatasetEpisode())) == "7"


def test_no_recording_is_an_empty_reference():
    assert study.episode_reference(None) == ""
