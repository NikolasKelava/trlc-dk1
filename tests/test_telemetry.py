"""The two instruments that exist because the machine froze twice.

Neither has anything to do with the robot. What they must do is survive the
thing they are watching: a log line and a telemetry sample have to be **on the
disk** when they are written, not in a buffer that a hard reset throws away.
"""

from __future__ import annotations

import json
import logging

import pytest

from dk1lab import logs, telemetry


# --------------------------------------------------------------------------- #
# The session log
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_root():
    """Restore the root logger, so a test's handler does not follow the suite."""
    root = logging.getLogger()
    before = list(root.handlers), root.level
    yield root
    root.handlers, root.level = before[0], before[1]


def test_a_warning_reaches_the_file_immediately(tmp_path, clean_root):
    """Immediately is the whole point: a freeze does not flush anything."""
    path = logs.start("session", directory=tmp_path)
    logging.getLogger("dk1lab.something").warning("the encode failed")

    assert path.exists()
    assert "the encode failed" in path.read_text()


def test_the_traceback_is_in_the_file_even_though_the_terminal_never_showed_it(
    tmp_path, clean_root
):
    path = logs.start("session", directory=tmp_path)
    try:
        raise RuntimeError("avcodec_open2")
    except RuntimeError:
        logging.getLogger("dk1lab.dataset").exception("could not write episode 0")

    written = path.read_text()
    assert "avcodec_open2" in written
    assert "Traceback" in written


def test_a_chatty_dependency_does_not_bury_the_two_lines_that_matter(tmp_path, clean_root):
    path = logs.start("session", directory=tmp_path)
    logging.getLogger("torch.somewhere").debug("a tensor moved")
    logging.getLogger("dk1lab.policy").debug("chunk 4 queued")

    written = path.read_text()
    assert "chunk 4 queued" in written
    assert "a tensor moved" not in written


def test_the_file_is_named_by_when_and_what(tmp_path):
    path = logs.log_path("session-A0", tmp_path)
    assert path.name.endswith("-session-A0.log")
    assert path.parent == tmp_path


# --------------------------------------------------------------------------- #
# The machine telemetry
# --------------------------------------------------------------------------- #


def test_a_sample_is_stamped_and_never_raises():
    row = telemetry.Sources.detect().sample()
    assert "t" in row and "clock" in row
    # Everything else is machine-dependent — no hwmon in a container, no GPU on
    # a laptop — and the sampler's contract is that it degrades rather than fails.


def test_an_unknown_sensor_is_absent_rather_than_an_error():
    assert telemetry.find_hwmon("no-such-sensor-here") is None


def test_every_sample_is_on_the_disk_when_it_is_written(tmp_path):
    monitor = telemetry.Telemetry(tmp_path / "t.jsonl", interval_s=0.01)
    monitor.start()
    try:
        deadline = 200
        while monitor.samples < 2 and deadline:
            deadline -= 1
            import time

            time.sleep(0.01)
        # Read it while the thread is still running: this is what a machine that
        # freezes mid-session leaves behind.
        rows = telemetry.read(monitor.path)
        assert len(rows) >= 2
        assert rows[0]["event"] == "start"
    finally:
        monitor.stop()

    rows = telemetry.read(monitor.path)
    assert rows[-1]["event"] == "stop"


def test_an_event_line_says_what_the_operator_was_doing(tmp_path):
    monitor = telemetry.Telemetry(tmp_path / "t.jsonl", interval_s=10)
    monitor.start()
    monitor.mark(event="episode_start", episode=3, task="put the dice in the bowl")
    monitor.stop()

    events = [r for r in telemetry.read(monitor.path) if r.get("event") == "episode_start"]
    assert events and events[0]["episode"] == 3


def test_a_truncated_last_line_is_dropped_rather_than_fatal(tmp_path):
    """The file's purpose is to be interrupted, so half a line is the normal case."""
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"t": 1.0, "clock": "10:00:00.000", "psu_w": 300}) + "\n{\"t\": 2.0")
    rows = telemetry.read(path)
    assert len(rows) == 1
    assert rows[0]["psu_w"] == 300


def test_the_summary_names_the_extremes_and_says_it_ended_with_the_machine(tmp_path):
    rows = [
        {"t": 1.0, "clock": "10:00:00.000", "psu_w": 300.0, "gpu_c": 60},
        {"t": 2.0, "clock": "10:00:01.000", "psu_w": 780.0, "gpu_c": 71},
    ]
    lines = telemetry.summary(rows)
    assert any("PSU total" in line and "780" in line for line in lines)
    assert not any("stop" in line for line in lines)


def test_no_samples_is_said_plainly():
    assert telemetry.summary([]) == ["no samples"]
