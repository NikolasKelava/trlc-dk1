"""The scored study: three scene configurations, three attempts each, on paper.

`STUDY.md` is the protocol. This is the bookkeeping it needs, and the whole of
it is decisions about text and numbers — no lerobot, no camera, no robot — so it
tests without any of them.

**What a scored session is.** One row of the results table (R0, A0, A1, B0, B1)
is nine attempts: scene 1 three times, scene 2 three times, scene 3 three times.
The task string never changes — it is the prompt, byte-identical everywhere — so
the scene cannot live in it. It lives in :class:`Attempt`, which is one line of
``study/scores/<row>.csv``, written **as the attempt ends** rather than
reconstructed from memory afterwards.

**Why the scenes are grouped rather than interleaved.** The dice and the bowl
are put on marks drawn on the desk; changing the layout is a physical act and
doing it nine times instead of three is nine chances to put it back wrong. Three
attempts at one layout, then the next, is also what makes the per-scene grid
readable at a glance.

**Resuming is reading.** :class:`ScenePlan` is built from the rows already in
the CSV, so a session interrupted after five attempts starts at scene 2 attempt
3 — the file is the state, and there is nothing else to keep in step with it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Three layouts, three attempts each. `STUDY.md` § *Three scene configurations*.
DEFAULT_SCENES = 3
DEFAULT_ATTEMPTS = 3

#: Where the tracked evidence goes. The datasets and the `.rrd` are far too big
#: for git; these three are text and photographs, and they are the study.
DEFAULT_SCORES_DIR = Path("study/scores")
DEFAULT_SCENE_DIR = Path("study/scene")

#: R0 runs under `optimized`, so its frames carry a different lens from every
#: other row's and must not join their dataset. It still gets recorded — to its
#: own directory, kept apart from `recordings/` and from every other row.
DEFAULT_STUDY_RRD_DIR = Path("study/rrd")

#: Which arm did the work. `none` is a real answer: it is what a score of 0 gets.
ARMS: tuple[str, ...] = ("left", "right", "both", "none")

_ARM_ALIASES = {"l": "left", "r": "right", "b": "both", "n": "none", "-": "none"}

#: The header, and the field order. `episode` is the only join from a score to
#: the frames it was scored from — the task string is identical in every episode
#: and the scene is not in it.
COLUMNS: tuple[str, ...] = ("scene", "attempt", "episode", "score", "seconds", "arm", "note")

#: The rubric, for the prompt. `STUDY.md` § *Scoring* is the authority; this is
#: the one-line reminder shown while the operator is deciding.
RUBRIC: tuple[str, ...] = (
    "0 no purposeful motion   1 approach (~5 cm)   2 contact",
    "3 grasp (lifted, >=1 s)   4 transport (over the bowl)   5 success (in, stays)",
)

MAX_SCORE = 5


class ScoreError(ValueError):
    """A score line that could not be understood. The message is shown as typed."""


@dataclass(frozen=True)
class Attempt:
    """One scored attempt — one line of ``study/scores/<row>.csv``.

    ``episode`` is whatever identifies the frames: the LeRobot dataset's episode
    index for a recorded row, the ``.rrd`` file's stem for R0, and empty when an
    attempt produced no recording at all.
    """

    scene: int
    attempt: int
    score: int
    arm: str = "none"
    note: str = ""
    episode: str = ""
    seconds: float | None = None

    @property
    def success(self) -> bool:
        return self.score >= MAX_SCORE

    def row(self) -> dict[str, str]:
        """The CSV record. Every value is text, because that is what a CSV holds."""
        return {
            "scene": str(self.scene),
            "attempt": str(self.attempt),
            "episode": self.episode,
            "score": str(self.score),
            "seconds": "" if self.seconds is None else f"{self.seconds:.1f}",
            "arm": self.arm,
            "note": self.note,
        }

    def line(self) -> str:
        """One line for the operator, straight after it is written."""
        time = f" in {self.seconds:.1f} s" if self.seconds is not None else ""
        note = f" — {self.note}" if self.note else ""
        return (
            f"scene {self.scene} attempt {self.attempt}: "
            f"score {self.score}{time}, {self.arm}{note}"
        )


def parse_score(text: str) -> Attempt:
    """One typed line as a partial :class:`Attempt` — scene and attempt unset.

    The grammar, in the order the tokens appear::

        <0-5> [arm] [seconds] [note ...]

    Only the score is required, and it is first because it is the thing the
    operator has just decided. ``arm`` is one of ``left`` / ``right`` / ``both``
    / ``none`` or its first letter, and is **required whenever anything happened**
    (score 1 or more) — the arm column exists to settle an open question about
    this cell, and a blank in it is a lost data point rather than a tidy one.
    ``seconds`` is a bare number, accepted only on a 5, since time-to-success is
    what it means.

    Raises:
        ScoreError: with a message written to be read at the prompt.
    """
    tokens = text.split()
    if not tokens:
        raise ScoreError("type a score 0-5, then the arm, then a note")

    head, *rest = tokens
    try:
        score = int(head)
    except ValueError:
        raise ScoreError(f"the score comes first and is 0-5, not {head!r}") from None
    if not 0 <= score <= MAX_SCORE:
        raise ScoreError(f"the score is 0-5, not {score}")

    arm = ""
    if rest:
        candidate = rest[0].lower()
        candidate = _ARM_ALIASES.get(candidate, candidate)
        if candidate in ARMS:
            arm, rest = candidate, rest[1:]
    if not arm:
        if score >= 1:
            raise ScoreError(
                f"say which arm did it — one of {', '.join(ARMS)} — after the score"
            )
        arm = "none"

    seconds: float | None = None
    if rest:
        try:
            seconds = float(rest[0])
        except ValueError:
            pass
        else:
            rest = rest[1:]
    if seconds is not None and score < MAX_SCORE:
        raise ScoreError(
            "the time is time-to-SUCCESS and belongs only on a 5; "
            "put anything else in the note"
        )

    return Attempt(scene=0, attempt=0, score=score, arm=arm, note=" ".join(rest), seconds=seconds)


# --------------------------------------------------------------------------- #
# The CSV
# --------------------------------------------------------------------------- #


def scores_path(row: str, directory: Path | str = DEFAULT_SCORES_DIR) -> Path:
    """Where one configuration's scores live: ``<dir>/<row>.csv``."""
    return Path(directory) / f"{row}.csv"


def append(path: Path | str, attempt: Attempt) -> Path:
    """Add one attempt to the CSV, writing the header if the file is new.

    Appended per attempt rather than dumped at the end for the reason the whole
    module exists: a session that crashes, or an operator who has to stop, must
    not lose the attempts already run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if fresh:
            writer.writeheader()
        writer.writerow(attempt.row())
    return path


def read(path: Path | str) -> list[Attempt]:
    """Every attempt already in the CSV, in file order. Missing file: no attempts."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[Attempt] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            try:
                rows.append(
                    Attempt(
                        scene=int(record["scene"]),
                        attempt=int(record["attempt"]),
                        score=int(record["score"]),
                        arm=(record.get("arm") or "none").strip(),
                        note=(record.get("note") or "").strip(),
                        episode=(record.get("episode") or "").strip(),
                        seconds=float(record["seconds"]) if record.get("seconds") else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ScoreError(f"{path}: line {len(rows) + 2} is not a score row: {exc}") from exc
    return rows


# --------------------------------------------------------------------------- #
# The walk through the scenes
# --------------------------------------------------------------------------- #


class ScenePlan:
    """Which scene is live, which attempt is next, and when the row is done.

    Built from the attempts already recorded, so resuming a session needs no
    state beyond the CSV. Scenes are walked **in order** and only advance when
    one is full; :meth:`jump` is the deliberate exception.
    """

    def __init__(
        self,
        *,
        scenes: int = DEFAULT_SCENES,
        attempts: int = DEFAULT_ATTEMPTS,
        done: Iterable[Attempt] = (),
    ) -> None:
        if scenes < 1:
            raise ValueError(f"a study needs at least one scene, got {scenes}")
        if attempts < 1:
            raise ValueError(f"a scene needs at least one attempt, got {attempts}")
        self.scenes = scenes
        self.attempts = attempts
        self.counts: dict[int, int] = {n: 0 for n in range(1, scenes + 1)}
        for attempt in done:
            if attempt.scene in self.counts:
                self.counts[attempt.scene] += 1
        self.scene = self._first_unfilled() or 1

    def _first_unfilled(self) -> int | None:
        return next(
            (n for n in range(1, self.scenes + 1) if self.counts[n] < self.attempts), None
        )

    @property
    def next_attempt(self) -> int:
        """The number the next attempt at the live scene will carry, 1-based."""
        return self.counts[self.scene] + 1

    @property
    def total(self) -> int:
        """Attempts recorded so far, across every scene."""
        return sum(self.counts.values())

    @property
    def wanted(self) -> int:
        """Attempts the row asks for."""
        return self.scenes * self.attempts

    @property
    def complete(self) -> bool:
        """True once every scene has its attempts. Extra attempts do not undo it."""
        return self._first_unfilled() is None

    @property
    def scene_starting(self) -> bool:
        """True when the next attempt is the first one at the live scene."""
        return self.counts[self.scene] == 0

    def record(self, attempt: Attempt) -> None:
        """Count an attempt, then advance if the live scene is now full."""
        if attempt.scene in self.counts:
            self.counts[attempt.scene] += 1
        if self.counts[self.scene] >= self.attempts:
            self.scene = self._first_unfilled() or self.scene

    def jump(self, scene: int) -> None:
        """Go to a scene by number, full or not.

        Going back is for an attempt that was **void** — the dice knocked off the
        table before the policy moved, a camera that had dropped out — not for a
        result the operator dislikes. The extra attempt is numbered onward (4, 5)
        and every row counts; the note is where the reason goes.
        """
        if not 1 <= scene <= self.scenes:
            raise ValueError(f"this study has scenes 1-{self.scenes}, not {scene}")
        self.scene = scene

    def label(self) -> str:
        """The prompt fragment: ``scene 2/3, attempt 1/3``."""
        return (
            f"scene {self.scene}/{self.scenes}, "
            f"attempt {self.next_attempt}/{self.attempts}"
        )

    def banner(self, scene_dir: Path | str = DEFAULT_SCENE_DIR) -> str:
        """What to print before the first attempt at a scene: set the desk up."""
        photo = Path(scene_dir) / f"{self.scene}.jpg"
        return (
            f"=== scene {self.scene} of {self.scenes} — put the dice and the bowl "
            f"on their marks for scene {self.scene} ({photo}) ==="
        )


# --------------------------------------------------------------------------- #
# Reading a row back
# --------------------------------------------------------------------------- #


def grid(rows: Sequence[Attempt], *, scenes: int = DEFAULT_SCENES) -> list[str]:
    """The per-scene scores, one line per scene, plus the row's success rate.

    The grid is the point of running three layouts: a row that reads ``5 5 5 /
    0 0 0 / 0 0 0`` and one that reads ``0 5 0 / 5 0 0 / 0 0 5`` have the same
    success rate and are not the same result.
    """
    lines: list[str] = []
    for scene in range(1, scenes + 1):
        scored = [a for a in rows if a.scene == scene]
        marks = " ".join(str(a.score) for a in scored) or "-"
        wins = sum(1 for a in scored if a.success)
        lines.append(f"  scene {scene}: {marks:<12}  {wins}/{len(scored) or 0} success")
    wins = sum(1 for a in rows if a.success)
    lines.append(f"  overall: {wins}/{len(rows)} success")
    arms = sorted({a.arm for a in rows if a.arm != "none"})
    if arms:
        counts = ", ".join(f"{arm} {sum(1 for a in rows if a.arm == arm)}" for arm in arms)
        lines.append(f"  arms used: {counts}")
    return lines


def episode_reference(recording: object) -> str:
    """What to put in the ``episode`` column for whatever the attempt recorded.

    A LeRobot episode is an index; an ``.rrd`` is a file, and its stem carries
    the index and the task. A combined recording is asked in that order, because
    the dataset index is the one a fine-tune can look up.
    """
    if recording is None:
        return ""
    reports = getattr(recording, "reports", (recording,))
    for report in reports:
        index = getattr(report, "index", None)
        if index is not None:
            return str(index)
    for report in reports:
        path = getattr(report, "path", None)
        if path is not None:
            return Path(path).stem
    return ""


__all__ = [
    "ARMS",
    "COLUMNS",
    "DEFAULT_ATTEMPTS",
    "DEFAULT_SCENES",
    "DEFAULT_SCENE_DIR",
    "DEFAULT_SCORES_DIR",
    "DEFAULT_STUDY_RRD_DIR",
    "MAX_SCORE",
    "RUBRIC",
    "Attempt",
    "ScenePlan",
    "ScoreError",
    "append",
    "episode_reference",
    "grid",
    "parse_score",
    "read",
    "scores_path",
]
