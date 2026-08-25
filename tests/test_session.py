"""A session: one loaded policy, many rollouts, and the task actually changing.

No GPU, no robot, no LeRobot — the strategy, the engine and the context are
fakes. What is under test is the part that is ours: that the instruction reaches
the engine, that a second rollout does not reload or reconnect anything, that
Ctrl-C ends an episode rather than the session, and that the prompt grammar
means what it says.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dk1lab.session import (
    DURATION,
    HELP,
    HOME,
    NOTHING,
    QUIT,
    RECORD,
    RUN,
    SCENE,
    PolicySession,
    parse_command,
)


# --------------------------------------------------------------------------- #
# The prompt grammar
# --------------------------------------------------------------------------- #


def test_anything_that_is_not_a_colon_word_is_the_instruction():
    command = parse_command("  pick up the dice  ")
    assert command.kind == RUN
    assert command.task == "pick up the dice"


def test_an_empty_line_repeats_the_last_task():
    """Scoring a policy is the same instruction, over and over."""
    assert parse_command("", last_task="pick up the dice").task == "pick up the dice"


def test_an_empty_line_with_nothing_to_repeat_does_nothing():
    assert parse_command("", last_task="").kind == NOTHING


@pytest.mark.parametrize("line", [":q", ":quit", ":exit", ":QUIT"])
def test_the_ways_out(line):
    assert parse_command(line).kind == QUIT


def test_help_and_home():
    assert parse_command(":help").kind == HELP
    assert parse_command(":home").kind == HOME


@pytest.mark.parametrize(("line", "value"), [(":record on", True), (":record off", False)])
def test_recording_is_toggled_at_the_prompt(line, value):
    command = parse_command(line)
    assert command.kind == RECORD
    assert command.value is value


def test_a_command_that_cannot_be_understood_complains_instead_of_running_it():
    """The dangerous failure would be treating `:recrod on` as an instruction."""
    assert parse_command(":recrod on").error is not None
    assert parse_command(":record maybe").error is not None
    assert parse_command(":duration soon").error is not None
    assert parse_command(":duration -5").error is not None


def test_a_scene_is_a_number_the_scored_session_jumps_to():
    """The grammar knows nothing about a study; the prompt loop refuses if there is none."""
    assert parse_command(":scene 2").kind == SCENE
    assert parse_command(":scene 2").value == 2
    assert parse_command(":scene two").error is not None
    assert parse_command(":scene").error is not None


def test_a_duration_is_seconds_and_zero_means_until_stopped():
    assert parse_command(":duration 60").kind == DURATION
    assert parse_command(":duration 60").value == 60.0
    assert parse_command(":duration 0").value == 0.0


# --------------------------------------------------------------------------- #
# Fakes: everything a session touches that we do not own
# --------------------------------------------------------------------------- #


class FakeEngine:
    """An inference engine with the one attribute the instruction lives in."""

    def __init__(self, task: str = ""):
        self._task = task
        self.started = 0
        self.stopped = 0
        self.resets = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def reset(self):
        self.resets += 1

    def get_action(self, obs_frame):
        return None


class FakeStrategy:
    """``BaseStrategy``: setup starts the engine, run is the loop, teardown disconnects."""

    def __init__(self, engine, *, loop=None):
        self._engine = engine
        self.setups = 0
        self.runs: list[str] = []
        self.teardowns = 0
        self._loop = loop

    def setup(self, ctx):
        self.setups += 1
        self._engine.reset()
        self._engine.start()

    def run(self, ctx):
        self.runs.append(ctx.policy.inference._task)
        if self._loop is not None:
            self._loop(ctx)

    def teardown(self, ctx):
        self.teardowns += 1


def session(*, loop=None, **kwargs) -> tuple[PolicySession, FakeStrategy, FakeEngine]:
    """A session with its context already built — i.e. as if ``open()`` had run."""
    engine = FakeEngine()
    cfg = SimpleNamespace(fps=30, duration=0.0, task="", strategy=None)
    ctx = SimpleNamespace(
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(robot_wrapper=None, initial_position={}),
        runtime=SimpleNamespace(cfg=cfg),
    )
    live = PolicySession(cfg, **kwargs)
    live.ctx = ctx
    strategy = FakeStrategy(engine, loop=loop)
    live._strategy = strategy
    return live, strategy, engine


# --------------------------------------------------------------------------- #
# The point of the whole module: the task changes, the policy does not reload
# --------------------------------------------------------------------------- #


def test_the_instruction_reaches_the_engine_that_will_run_it():
    live, _strategy, engine = session()
    live.set_task("pick up the dice")
    assert engine._task == "pick up the dice"
    assert live.cfg.task == "pick up the dice", "the banner has to agree with the engine"


def test_a_session_refuses_an_engine_whose_task_it_cannot_change():
    """Silently running the previous instruction would produce evidence about the wrong thing."""
    live, _strategy, _engine = session()
    live.ctx.policy.inference = SimpleNamespace(get_action=lambda f: None)
    with pytest.raises(AttributeError, match="_task"):
        live.set_task("pick up the dice")


def test_consecutive_rollouts_reuse_one_context_and_one_strategy():
    live, strategy, engine = session()
    live.rollout("pick up the dice")
    live.rollout("put it in the box")
    assert strategy.runs == ["pick up the dice", "put it in the box"]
    assert strategy.teardowns == 0, "nothing is disconnected between episodes"
    assert live.episodes == 2


def test_every_episode_restarts_the_engine_and_stops_it_afterwards():
    """A worker thread left running would keep burning the GPU while you type."""
    live, _strategy, engine = session()
    live.rollout("one")
    assert (engine.started, engine.stopped, engine.resets) == (1, 1, 1)
    live.rollout("two")
    assert (engine.started, engine.stopped, engine.resets) == (2, 2, 2)


def test_the_duration_is_per_episode_and_can_be_changed_between_them():
    live, _strategy, _engine = session(duration_s=180.0)
    live.rollout("one")
    assert live.cfg.duration == 180.0
    live.duration_s = 60.0
    live.rollout("two")
    assert live.cfg.duration == 60.0
    live.rollout("three", duration_s=0.0)
    assert live.cfg.duration == 0.0


def test_a_rollout_with_no_instruction_at_all_is_refused():
    live, _strategy, _engine = session()
    with pytest.raises(ValueError, match="instruction"):
        live.rollout()


# --------------------------------------------------------------------------- #
# Ending one episode is not ending the session
# --------------------------------------------------------------------------- #


def test_the_shutdown_event_is_cleared_before_each_episode():
    """Set by the previous Ctrl-C. Left set, the next loop would exit on its first tick."""
    seen: list[bool] = []

    def interrupt(ctx):
        seen.append(live.shutdown_event.is_set())
        live.shutdown_event.set()

    live, strategy, _engine = session(loop=interrupt)
    first = live.rollout("one")
    second = live.rollout("two")
    assert seen == [False, False], "each episode started with a clear event"
    assert first.ended == second.ended == "interrupted"
    assert strategy.runs == ["one", "two"], "the session survived the interrupt"


def test_a_keyboard_interrupt_inside_the_loop_ends_the_episode_cleanly():
    def interrupt(ctx):
        raise KeyboardInterrupt

    live, _strategy, engine = session(loop=interrupt)
    outcome = live.rollout("one")
    assert outcome.ended == "interrupted"
    assert engine.stopped == 1, "the engine is stopped even on the interrupt path"


def test_an_episode_that_faults_still_stops_the_engine_and_re_raises():
    def explode(ctx):
        raise RuntimeError("the cameras went away")

    live, _strategy, engine = session(loop=explode)
    with pytest.raises(RuntimeError, match="cameras"):
        live.rollout("one")
    assert engine.stopped == 1


def test_a_faulted_episode_does_not_sweep_the_arms_home():
    """Commanding more motion into a fault is the opposite of stopping."""

    def explode(ctx):
        raise RuntimeError("boom")

    live, _strategy, _engine = session(loop=explode)
    live.home = lambda pose=None: pytest.fail("homing must not run after a fault")
    with pytest.raises(RuntimeError):
        live.rollout("one", home=object())


# --------------------------------------------------------------------------- #
# The instruments
# --------------------------------------------------------------------------- #


class FakeTrace:
    display = False

    def __init__(self):
        self.attached = 0
        self.resets = 0
        self.closes = 0

    def attach(self, ctx):
        self.attached += 1

    def attach_queue(self, strategy):
        pass

    def reset(self):
        self.resets += 1

    def close(self):
        self.closes += 1


def test_the_trace_is_reset_per_episode_rather_than_re_attached():
    """Attaching again per episode would stack another wrapper on every call."""
    trace = FakeTrace()
    live, _strategy, _engine = session(trace=trace)
    live.rollout("one")
    live.rollout("two")
    assert trace.attached == 0, "attaching happens once, in open()"
    assert (trace.resets, trace.closes) == (2, 2)


def test_recording_is_per_episode_and_can_be_switched_between_them(tmp_path, monkeypatch):
    made: list = []

    class FakeRecorder:
        def __init__(self, path, **kwargs):
            self.path = path
            self.attached = self.started = self.stopped = self.detached = 0
            made.append(self)

        def attach(self, ctx):
            self.attached += 1

        def start(self):
            self.started += 1

        def stop(self):
            self.stopped += 1
            return "the recording"

        def detach(self):
            self.detached += 1

    monkeypatch.setattr("dk1lab.record.EpisodeRecorder", FakeRecorder)
    live, _strategy, _engine = session(record_dir=tmp_path, record=True)
    first = live.rollout("one")
    live.record = False
    second = live.rollout("two")

    assert first.recording == "the recording"
    assert second.recording is None, "the second episode was not recorded"
    assert len(made) == 1
    assert (made[0].attached, made[0].started, made[0].stopped, made[0].detached) == (1, 1, 1, 1)
    assert made[0].path.parent == tmp_path
