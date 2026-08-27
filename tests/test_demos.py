"""The demonstration recorder: the grammar, and the one-episode-late commit.

The property this file exists for is that ``again`` is a real deletion. LeRobot
v3.0 cannot take a written episode back out, so an episode is held until the next
one starts — and what has to hold is that the *previous* episodes are on disk by
then, that the held one is written when it should be, and that nothing is written
when it should not.

Everything here drives the fakes from :mod:`tests.test_dataset` through a
scripted console. Nothing connects to anything and nothing imports LeRobot.
"""

from __future__ import annotations

import pytest

from dk1lab import demos
from dk1lab.dataset import DatasetSession
from dk1lab.layout import ACTION_KEYS

from test_dataset import FakeDataset, FakeRobot


# --------------------------------------------------------------------------- #
# The grammar
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("line", ["", "   ", "\n"])
def test_an_empty_line_starts_or_stops(line):
    """The common case is one keystroke: the operator has a hand on a leader arm."""
    assert demos.parse_command(line).kind == demos.RECORD


@pytest.mark.parametrize("line", ["again", "AGAIN", " a ", "redo", ":again"])
def test_again_is_understood(line):
    assert demos.parse_command(line).kind == demos.AGAIN


@pytest.mark.parametrize("line", ["done", "quit", "q", ":quit", "exit"])
def test_done_is_understood(line):
    assert demos.parse_command(line).kind == demos.DONE


def test_a_scene_carries_its_number():
    command = demos.parse_command("scene 3")
    assert (command.kind, command.value, command.error) == (demos.SCENE, 3, None)


@pytest.mark.parametrize("line", ["scene", "scene two", "scene 0", "scene -1"])
def test_a_scene_that_is_not_a_number_from_one_is_refused(line):
    command = demos.parse_command(line)
    assert command.kind == demos.SCENE and command.error


def test_an_unknown_word_is_a_complaint_not_a_recording():
    """It must not fall through to RECORD: a typo would start an episode."""
    command = demos.parse_command("agian")
    assert command.kind == demos.NOTHING and command.error


# --------------------------------------------------------------------------- #
# The fixtures
# --------------------------------------------------------------------------- #


class ScriptedConsole:
    """Hands the loop one line at a time, and remembers everything printed."""

    def __init__(self, lines=()):
        self.lines = list(lines)
        self.said: list[str] = []
        self.muted = False

    def say(self, text: str = "") -> None:
        self.said.append(text)

    def status(self, text: str) -> None:
        pass

    def end_status(self) -> None:
        pass

    def interactive(self) -> bool:
        return True

    def poll(self) -> str | None:
        return self.lines.pop(0) if self.lines else None

    def mute(self) -> None:
        self.muted = True

    def unmute(self) -> None:
        self.muted = False

    def output(self) -> str:
        return "\n".join(self.said)


@pytest.fixture
def live(tmp_path):
    session = DatasetSession(tmp_path / "demos", fps=30, width=6, height=4)
    session.dataset = FakeDataset()
    return session


@pytest.fixture
def robot():
    return FakeRobot()


def session(live, robot, console=None, **kwargs):
    return demos.DemoSession(
        leader=None,
        follower=robot,
        dataset=live,
        task="put the dice in the bowl",
        console=console or ScriptedConsole(),
        **kwargs,
    )


def drive(robot, ticks: int) -> None:
    """One teleoperation tick's worth of robot calls, without a leader."""
    for _ in range(ticks):
        observation = robot.get_observation()
        robot.send_action({key: float(observation[key]) for key in ACTION_KEYS})


# --------------------------------------------------------------------------- #
# The episode, and where it lives between stop and commit
# --------------------------------------------------------------------------- #


def test_an_episode_is_not_written_when_it_is_stopped(live, robot):
    demo = session(live, robot)
    demo.start_episode()
    drive(robot, 5)
    demo.stop_episode()
    assert live.dataset.episodes == []
    assert live.pending is not None


def test_the_held_episode_is_written_when_the_next_one_starts(live, robot):
    demo = session(live, robot)
    demo.start_episode()
    drive(robot, 5)
    demo.stop_episode()
    demo.start_episode()
    assert len(live.dataset.episodes) == 1
    assert live.dataset.episodes[0][0]["task"] == "put the dice in the bowl"


def test_again_deletes_the_held_episode_and_writes_nothing(live, robot):
    demo = session(live, robot)
    demo.start_episode()
    drive(robot, 5)
    demo.stop_episode()
    assert demo.drop()
    demo.start_episode()
    assert live.dataset.episodes == []


def test_again_during_an_episode_stops_it_and_throws_it_away(live, robot):
    demo = session(live, robot)
    demo.start_episode()
    drive(robot, 5)
    demo.handle("again")
    assert demo.recorder is None
    assert live.pending is None
    demo.start_episode()
    assert live.dataset.episodes == []


def test_again_cannot_reach_an_episode_already_on_disk(live, robot):
    """The whole reason the commit is one episode late — and its limit."""
    demo = session(live, robot)
    demo.start_episode()
    drive(robot, 5)
    demo.stop_episode()
    demo.start_episode()  # commits the first
    drive(robot, 5)
    demo.stop_episode()
    demo.drop()  # only reaches the second
    assert len(live.dataset.episodes) == 1


def test_a_recorded_episode_carries_its_scene_in_the_notes(live, robot):
    import json

    demo = session(live, robot, scene=2)
    demo.start_episode()
    drive(robot, 3)
    demo.stop_episode()
    demo.commit_held()
    notes = json.loads((live.root / "dk1_notes.jsonl").read_text().splitlines()[0])
    assert notes["scene"] == 2
    assert notes["task"] == "put the dice in the bowl"


def test_the_scene_never_reaches_the_task_string(live, robot):
    demo = session(live, robot, scene=3)
    demo.start_episode()
    drive(robot, 2)
    demo.stop_episode()
    demo.commit_held()
    assert live.dataset.episodes[0][0]["task"] == "put the dice in the bowl"


def test_the_recorder_is_detached_when_the_episode_stops(live, robot):
    """Otherwise every episode would leave a wrapper on the robot."""
    original = robot.get_observation
    demo = session(live, robot)
    demo.start_episode()
    assert robot.get_observation != original
    demo.stop_episode()
    # `==`, not `is`: every attribute access builds a fresh bound method, so what
    # detach put back is equal to the original rather than the same object.
    assert robot.get_observation == original


def test_ticks_between_episodes_are_not_recorded(live, robot):
    demo = session(live, robot)
    drive(robot, 4)
    demo.start_episode()
    drive(robot, 3)
    demo.stop_episode()
    drive(robot, 4)
    demo.commit_held()
    assert len(live.dataset.episodes[0]) == 3


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #


def test_the_prompt_names_the_scene_and_the_next_episode_index(live, robot):
    demo = session(live, robot, scene=2)
    assert "scene 2" in demo.prompt()
    assert "episode 0" in demo.prompt()


def test_the_held_episode_counts_towards_the_next_index(live, robot):
    """It is not in the dataset yet, but it will be by the time the next is."""
    demo = session(live, robot)
    demo.start_episode()
    drive(robot, 2)
    demo.stop_episode()
    assert demo.next_index() == 1


def test_scene_changes_only_between_episodes(live, robot):
    console = ScriptedConsole()
    demo = session(live, robot, console=console)
    demo.start_episode()
    demo.handle("scene 3")
    assert demo.scene == 1
    assert "stop the episode first" in console.output()


def test_done_ends_the_session(live, robot):
    demo = session(live, robot)
    assert demo.handle("done") is False
    assert demo.handle("") is True


# --------------------------------------------------------------------------- #
# The tick
# --------------------------------------------------------------------------- #


class Leader:
    def get_action(self):
        return {key: 1.0 for key in ACTION_KEYS}


def identity(pair):
    return pair[0] if isinstance(pair, tuple) else pair


def test_a_tick_sends_what_the_leader_asked_for(robot):
    sent = demos.tick(Leader(), robot, (identity, identity, identity))
    # SafeBiDK1Follower returns what the arms were GIVEN; the fake halves it, and
    # that — not the request — is what a dataset has to record.
    assert sent == {key: 0.5 for key in ACTION_KEYS}
    assert robot.sent == [{key: 1.0 for key in ACTION_KEYS}]


def test_a_tick_observes_before_it_sends(robot):
    """The recorder holds the observation and closes the tick on the action."""
    demos.tick(Leader(), robot, (identity, identity, identity))
    assert robot.tick == 1


# --------------------------------------------------------------------------- #
# The loop, end to end
# --------------------------------------------------------------------------- #


class Follower(FakeRobot):
    """A follower that stops the loop after a fixed number of ticks."""

    def __init__(self, ticks: int):
        super().__init__()
        self.budget = ticks

    def get_observation(self) -> dict:
        if self.tick >= self.budget:
            raise KeyboardInterrupt
        return super().get_observation()


def test_the_loop_records_what_it_was_told_to_and_keeps_it(live):
    """Enter, ticks, Enter, done — and the episode is on disk at the end."""
    robot = Follower(60)
    console = ScriptedConsole()
    demo = demos.DemoSession(
        leader=Leader(),
        follower=robot,
        dataset=live,
        task="put the dice in the bowl",
        fps=200,
        console=console,
    )
    # Stop the episode part way, then finish; the rest of the ticks are idle.
    console.lines = [""] + [None] * 20 + [""] + [None] * 5 + ["done"]
    demo.loop()
    assert len(live.dataset.episodes) == 1
    assert demo.attempts == 1


def test_an_interrupt_keeps_what_was_recorded(live):
    """A day of demonstrations must not be lost to the way the session ended."""
    robot = Follower(10)
    console = ScriptedConsole([""])
    demo = demos.DemoSession(
        leader=Leader(),
        follower=robot,
        dataset=live,
        task="put the dice in the bowl",
        fps=200,
        console=console,
    )
    demo.loop()
    assert len(live.dataset.episodes) == 1
    assert live.pending is None
