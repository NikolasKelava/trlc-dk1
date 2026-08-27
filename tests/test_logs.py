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
