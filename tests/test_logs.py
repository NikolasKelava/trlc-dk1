"""The session log file, and the one thing it must not do to the terminal.

Opening the log means lowering the **root** logger to DEBUG, because the root
level gates every child before any handler sees a record. Anything already
attached to root then starts emitting DEBUG from every library in the process —
and something always is attached, because importing ``lerobot`` calls the
module-level ``logging.debug()`` while probing for optional packages, which
calls ``logging.basicConfig()``. On 2026-08-27 that put thousands of lines of
``DEBUG:PIL.PngImagePlugin:STREAM`` across the operator's screen while an
episode was being saved.
"""

from __future__ import annotations

import logging

import pytest

from dk1lab import logs


@pytest.fixture
def root(tmp_path):
    """A root logger restored afterwards, whatever the test does to it."""
    logger = logging.getLogger()
    saved_handlers, saved_level = list(logger.handlers), logger.level
    saved_children = {
        name: logging.getLogger(name).level for name in ("dk1lab", "lerobot", "PIL")
    }
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    yield logger
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)
    for name, level in saved_children.items():
        logging.getLogger(name).setLevel(level)


class Collector(logging.Handler):
    """Stands in for the console handler ``basicConfig`` leaves on root."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def emit_one(name: str, level: int) -> None:
    logging.getLogger(name).log(level, "x")


def test_a_third_partys_debug_does_not_reach_a_console_we_did_not_open(root, tmp_path):
    """The bug: PIL at DEBUG, on the screen, because we lowered root for our file."""
    console = Collector()
    root.addHandler(console)

    logs.start("t", directory=tmp_path)
    console.records.clear()  # `start` announces the file, and should
    emit_one("PIL.PngImagePlugin", logging.DEBUG)

    assert [r.name for r in console.records] == []


def test_the_lines_the_operator_reads_still_reach_it(root, tmp_path):
    """Taming the console must not silence it — INFO from ours and LeRobot's stays."""
    console = Collector()
    root.addHandler(console)

    logs.start("t", directory=tmp_path)
    console.records.clear()
    emit_one("dk1lab.dataset", logging.INFO)
    emit_one("lerobot.rollout.strategies.base", logging.INFO)
    emit_one("PIL.PngImagePlugin", logging.WARNING)

    assert [r.name for r in console.records] == [
        "dk1lab.dataset",
        "lerobot.rollout.strategies.base",
        "PIL.PngImagePlugin",
    ]


def test_our_own_debug_still_reaches_the_file(root, tmp_path):
    """The file is the whole point: `dk1lab` at DEBUG, and a third party's not."""
    path = logs.start("t", directory=tmp_path)
    emit_one("dk1lab.dataset", logging.DEBUG)
    emit_one("PIL.PngImagePlugin", logging.DEBUG)

    written = path.read_text()
    assert "dk1lab.dataset" in written
    assert "PngImagePlugin" not in written


def test_starting_twice_does_not_stack_filters(root, tmp_path):
    """A second run in one process must not add a second copy of the policy."""
    console = Collector()
    root.addHandler(console)

    logs.start("t", directory=tmp_path)
    before = len(console.filters)
    logs.start("t2", directory=tmp_path)

    assert len(console.filters) == before


# --------------------------------------------------------------------------- #
# Surviving a library that clears the root logger
# --------------------------------------------------------------------------- #


def test_a_bare_logging_info_reaches_the_file():
    """`lerobot/scripts/lerobot_train.py` logs with bare `logging.info(...)`.

    A bare call goes to the ROOT logger, so its records arrive named `root` — not
    `lerobot.scripts.lerobot_train`. Treating that as third-party noise cost the
    whole of a fine-tune's log on 2026-08-28, including the
    `step N: eval_loss=...` line that `dk1 policy curve` reads to pick a
    checkpoint from. A run whose log is empty cannot be selected from.
    """
    keep = logs.Interesting(logging.DEBUG)
    assert keep.filter(logging.LogRecord("root", logging.INFO, "f", 1, "step 20", (), None))
    assert not keep.filter(logging.LogRecord("root", logging.DEBUG, "f", 1, "x", (), None))


def test_a_chatty_library_is_still_held_at_warning():
    """The reason the filter exists: PIL's DEBUG once buried an operator's screen.

    A library that spams uses `logging.getLogger(__name__)`; `root` is where a
    program talks about itself, which is the distinction this rests on.
    """
    keep = logs.Interesting(logging.DEBUG)
    noisy = logging.LogRecord("PIL.PngImagePlugin", logging.INFO, "f", 1, "STREAM", (), None)
    assert not keep.filter(noisy)


def test_handlers_cleared_by_a_third_party_are_put_back(tmp_path):
    """`lerobot.utils.utils.init_logging` does `root.handlers.clear()`.

    Everything we opened before it is gone without a word — which is how a
    fine-tune's train.log came to hold exactly one line.
    """
    path = logs.start("probe", path=tmp_path / "train.log")
    ours = logs.handlers()
    root = logging.getLogger()
    try:
        root.handlers.clear()
        root.addHandler(logging.StreamHandler())  # what init_logging attaches
        logs.restore(ours)
        assert all(handler in root.handlers for handler in ours)

        logging.getLogger().info("step 20: eval_loss=0.0685")
        assert "eval_loss=0.0685" in path.read_text()
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)


def test_restoring_twice_does_not_double_every_line(tmp_path):
    path = logs.start("probe", path=tmp_path / "train.log")
    ours = logs.handlers()
    root = logging.getLogger()
    try:
        logs.restore(ours)
        logs.restore(ours)
        logging.getLogger("dk1lab.probe").info("once")
        assert path.read_text().count("once") == 1
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
