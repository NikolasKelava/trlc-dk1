"""A log file per session, because the terminal is gone after a hard reset.

Twice now this cell has taken the machine down mid-session, and both times the
one thing that would have explained what the software was doing scrolled past in
a terminal that no longer exists. Worse, on 2026-08-26 a *recoverable* failure —
the first episode's video encode — was reported as a single log line, the run
carried on for five more attempts, and nothing was recorded. The line was seen by
nobody.

So every command that touches the arms writes a file:

* **everything** from :mod:`dk1lab` at DEBUG, and from ``lerobot`` at INFO,
* full tracebacks, which the terminal never showed,
* **flushed and ``fsync``-ed on every record**, because a machine that freezes
  does not get to flush its buffers. A log that loses its last ten lines loses
  exactly the ten that matter.

The cost of the fsync is real and is why this is a *file* handler on its own and
not the whole logging config: at a few dozen records a minute it is nothing, and
nothing in here is on the control path.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

#: Where logs go unless asked otherwise. Not tracked: they are machine-local.
DEFAULT_LOG_DIR = Path("logs")

#: What is written on every line. The thread name is there because the
#: instruments, the inference worker and the control loop are different threads
#: and a fault usually belongs to exactly one of them.
FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(threadName)-14s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"


class Interesting(logging.Filter):
    """Keep the file readable: ours in full, LeRobot's at INFO, the rest at WARNING.

    The root logger has to be set to the lowest level wanted, which lets a
    DEBUG-level torch or matplotlib through to every handler. Filtering here
    rather than by logger level is what keeps a chatty dependency from burying
    the two lines that matter — and burying them is the same as losing them.
    """

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("dk1lab"):
            return record.levelno >= self.level
        if record.name.startswith("lerobot"):
            return record.levelno >= logging.INFO
        return record.levelno >= logging.WARNING


class SyncingFileHandler(logging.FileHandler):
    """A file handler that reaches the disk before it returns.

    ``flush()`` alone hands the bytes to the kernel, which is enough for a
    process that dies and not enough for a machine that stops. ``fsync`` is the
    difference, and the difference is the last few seconds before a freeze.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            if self.stream is not None:
                self.stream.flush()
                os.fsync(self.stream.fileno())
        except (OSError, ValueError):  # pragma: no cover - a closed stream at exit
            pass


def log_path(what: str, directory: Path | str = DEFAULT_LOG_DIR) -> Path:
    """``<dir>/<date>-<time>-<what>.log`` — one file per run, named by when."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(directory) / f"{stamp}-{what}.log"


def start(
    what: str,
    *,
    directory: Path | str = DEFAULT_LOG_DIR,
    path: Path | str | None = None,
    level: int = logging.DEBUG,
) -> Path:
    """Attach the file handler to the root logger. Returns the file's path.

    Args:
        what: names the file — ``session``, ``teleop``, the study row.
        path: an explicit file, overriding ``directory`` and the naming.
        level: what ``dk1lab`` logs at. ``lerobot`` is held at INFO and
            everything else at WARNING, because a DEBUG-level torch is noise
            that would bury the two lines that matter.
    """
    target = Path(path) if path is not None else log_path(what, directory)
    target.parent.mkdir(parents=True, exist_ok=True)

    handler = SyncingFileHandler(target, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
    handler.setLevel(logging.DEBUG)
    handler.addFilter(Interesting(level))

    root = logging.getLogger()
    # The root logger's own level gates every child, so it has to be the lowest
    # of the levels wanted; the per-logger levels below do the actual filtering.
    root.setLevel(min(root.level or logging.WARNING, level))
    root.addHandler(handler)
    logging.getLogger("dk1lab").setLevel(level)
    logging.getLogger("lerobot").setLevel(logging.INFO)

    logging.getLogger(__name__).info("logging to %s", target)
    return target


__all__ = ["DEFAULT_LOG_DIR", "Interesting", "SyncingFileHandler", "log_path", "start"]
