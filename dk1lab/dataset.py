"""Record episodes as a **LeRobot dataset v3.0** — the study's own evidence format.

This sits *alongside* :mod:`dk1lab.record`, which writes Rerun ``.rrd`` files and
is not touched by any of it. The two answer different questions and both are
kept:

===================  ========================================================
``dk1lab.record``    one ``.rrd`` per episode. Four streams, including **the
                     policy's own plan** — the stream a rough rollout has to be
                     diagnosed against, and the one no dataset format has a slot
                     for. The eight legacy episodes and every future debugging
                     run are these.
``dk1lab.dataset``   a LeRobot v3.0 dataset, appended to across a session.
                     Readable by ``lerobot-dataset-viz`` and the Hub's viewer,
                     trainable by ``lerobot-train``, and the format ``STUDY.md``
                     names for both the ~100 demonstrations and every scored
                     rollout.
===================  ========================================================

A LeRobot dataset holds what a *policy* needs — observations and the actions the
arms were given — and nothing about what the model was thinking. That is exactly
why the ``.rrd`` path stays: recording the plan is diagnosis, recording the
episode is evidence, and neither substitutes for the other.

**The two share their instrument shape deliberately.** Both attach by wrapping
``get_observation`` and ``send_action`` on the live robot, both are per-episode,
both restore what they wrapped on :meth:`detach`, and both return a report with
``summary()`` and ``discard()`` — so ``dk1 policy run`` and ``dk1 policy session``
drive either through the same four calls, and an episode can be recorded to both
at once.

**One dataset, many episodes.** A ``.rrd`` is one file per episode; a LeRobot
dataset is one directory that episodes are appended to. :class:`DatasetSession`
owns the directory and hands out a :class:`DatasetEpisodeRecorder` per episode,
so a scored run of five attempts is five episodes of one dataset rather than
five datasets — which is what makes it viewable, trainable and quotable as a
single thing.

**A tick is closed by ``send_action``, not by ``get_observation``.** A LeRobot
frame carries the observation *and* the action taken from it, so nothing can be
written until both exist. The observation is held from the top of the tick and
the frame is added when the arms have been told what to do — with **what
``send_action`` returned**, which on :class:`~dk1lab.robot.SafeBiDK1Follower` is
the rate-limited action the arms were actually given rather than the one the
policy asked for. A dataset that recorded the request would teach a fine-tune to
ask for motion the cell will not perform.

**A recorder must never take a rollout down.** Every call is guarded and a
failure switches recording off for the rest of the episode, exactly as in
:mod:`dk1lab.record`. Losing a recording is bad; losing the arms mid-motion
because a disk filled up is worse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .layout import ACTION_KEYS, CAMERA_NAMES
from .record import slug

logger = logging.getLogger(__name__)

#: Where datasets go when nothing else is said. ``STUDY.md`` puts the scored
#: rollouts under ``study/rollouts/<config>/`` and the demonstrations under
#: ``study/demos/``; neither is in git — they are LeRobot datasets and belong on
#: the Hugging Face Hub.
DEFAULT_DATASET_DIR = Path("study")

#: Where the demonstrations go. ``STUDY.md`` names this directory. Defined here
#: rather than in the command that writes it, because the command that *reads*
#: it is a different one — ``dk1 policy finetune`` — and two spellings of one
#: directory is how a fine-tune ends up trained on yesterday's recording.
DEFAULT_DEMO_DIR = DEFAULT_DATASET_DIR / "demos"

#: The video codec. ``auto`` picks a hardware encoder when one is present, which
#: on this machine is NVENC on the 5090 — and that is the whole point: LeRobot's
#: default is SVT-AV1 on the CPU, which took **minutes** per episode for three
#: 1280x720 streams, all of it after the arms had stopped, with the operator
#: waiting. DIAGNOSTICS § *The episode that took minutes to save*.
DEFAULT_VCODEC = "auto"

#: NVENC refuses a GOP below 4 — `avcodec_open2` fails outright with LeRobot's
#: default of 2, which is how a hardware encoder turns into a silent fall back
#: to no video at all. Four keyframes' distance still seeks cheaply.
NVENC_MIN_GOP = 4

#: The marker of a codec that runs on the GPU, and therefore needs CUDA.
NVENC_SUFFIX = "_nvenc"

#: What one data file may hold before the writer rotates to the next one. Tiny,
#: deliberately: LeRobot keeps ONE parquet writer open across episodes and the
#: footer is written only when it closes, so a crash mid-session leaves every
#: episode already recorded unreadable — which is exactly what happened on
#: 2026-08-25. A rotation per episode means each episode's file is closed, and
#: therefore readable, the moment the next one starts.
PER_EPISODE_FILE_MB = 0.001

#: The codebase version this module writes. Asserted at open rather than assumed:
#: a v2.1 directory has a different layout on disk and reading it back with a
#: v3.0 tool is the kind of failure that is discovered a week later.
CODEBASE_VERSION = "v3.0"

#: Image-writing threads per camera. The frames are PNG-encoded and, for a video
#: dataset, encoded again at ``save_episode``; neither may happen on the control
#: thread. Three cameras at 30 Hz is the load this is sized for.
DEFAULT_IMAGE_WRITER_THREADS = 4


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be opened, with the reason."""


@dataclass(frozen=True)
class EpisodeDataset:
    """What one recorded episode turned out to be, and whether to keep it.

    **An episode is not committed by :meth:`DatasetEpisodeRecorder.stop`.** It is
    held in the dataset's episode buffer until :meth:`keep` writes it or
    :meth:`discard` drops it, because that is the only order in which declining
    an attempt actually removes it: ``save_episode`` encodes the video and
    appends the metadata, and LeRobot v3.0 has no way to take an episode back out
    afterwards. The first version of this called ``save_episode`` in ``stop`` and
    a ``discard`` that reported success while changing nothing.

    Committing is still the **default** everywhere it is asked for, and
    :meth:`DatasetSession.close` commits anything left pending — an attempt that
    cannot be repeated must not be lost to a stray keypress or a forgotten call.
    """

    root: Path
    repo_id: str
    index: int
    ticks: int
    seconds: float
    task: str
    #: Why recording stopped early, or ``None`` if it ran to the end.
    failed: str | None = None
    #: The session this episode is pending in. Held so :meth:`keep` and
    #: :meth:`discard` can act on it without the report owning a dataset.
    _session: Any = field(default=None, repr=False, compare=False)

    @property
    def pending(self) -> bool:
        """True while the episode is buffered and neither kept nor discarded."""
        return self._session is not None and self._session.pending is self

    def keep(self) -> bool:
        """Write the episode into the dataset. Returns whether anything was written.

        Encodes the episode's video, which for three 720p cameras over a minute
        is seconds of work — the caller is expected to say so before calling.
        """
        if not self.pending:
            return False
        return self._session.commit()

    def discard(self) -> bool:
        """Drop the episode without writing it. Returns whether there was one.

        Cheap, unlike declining an ``.rrd``: the frames went to a temporary image
        cache rather than into the dataset, so nothing has to be deleted from it.
        """
        if not self.pending:
            return False
        return self._session.drop()

    def summary(self) -> str:
        """One line for the operator, with anything lost stated rather than implied."""
        if self.ticks == 0:
            nothing = f"recorded no frames to {self.root}"
            return f"{nothing}\n  recording stopped early: {self.failed}" if self.failed else nothing
        rate = self.ticks / self.seconds if self.seconds > 0 else 0.0
        line = (
            f"episode {self.index} — {self.ticks} frames "
            f"({self.seconds:.1f} s at {rate:.1f} Hz) for {self.repo_id} in {self.root}"
        )
        if self.failed:
            line += f"\n  recording stopped early: {self.failed}"
        return line


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #


def dataset_features(
    *,
    width: int,
    height: int,
    use_videos: bool = True,
    keys: tuple[str, ...] = ACTION_KEYS,
    camera_names: tuple[str, ...] = CAMERA_NAMES,
) -> dict:
    """The v3.0 feature dict for this cell: 14 state, 14 action, three views.

    Derived from :mod:`dk1lab.layout` for the same reason
    :func:`dk1lab.policy.dataset_features` is — the channel order is the contract
    everything else in this project reads, and restating it here is how the two
    would drift. The only difference from that function is ``use_video``: a
    rollout's synthetic frame is an array, a recorded episode is an mp4.
    """
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    observation_hw: dict[str, Any] = {key: float for key in keys}
    observation_hw.update({name: (height, width, 3) for name in camera_names})
    features = hw_to_dataset_features(observation_hw, OBS_STR, use_video=use_videos)
    features.update(
        hw_to_dataset_features({key: float for key in keys}, ACTION, use_video=use_videos)
    )
    return features


class DatasetSession:
    """One LeRobot v3.0 dataset directory, appended to episode by episode.

    Args:
        root: the directory the dataset lives in. Created if absent; an existing
            one is **resumed**, so a session interrupted after three attempts
            continues at the fourth rather than starting a second dataset.
        repo_id: the ``<owner>/<name>`` this dataset will carry to the Hub.
            Defaults to a name derived from ``root``, because the directory is
            what the operator typed and the id is bookkeeping.
        fps: the control rate. Written into the metadata and used to compute
            every frame's timestamp, so it has to be the rate the loop actually
            ran at.
        width, height: the frame size, for the feature declaration.
        use_videos: encode the camera streams as mp4 (the v3.0 default) rather
            than keeping PNG frames. Off is for tests.
        robot_type: recorded in the metadata. This cell's is
            ``bi_dk1_follower_safe``; the borrowed statistics in
            :mod:`dk1lab.pi05` come from a ``bi_dk1_follower`` dataset with the
            same 14 channel names, which is the fact that matters.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        repo_id: str | None = None,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        use_videos: bool = True,
        robot_type: str = "bi_dk1_follower_safe",
        keys: tuple[str, ...] = ACTION_KEYS,
        camera_names: tuple[str, ...] = CAMERA_NAMES,
        image_writer_threads: int = DEFAULT_IMAGE_WRITER_THREADS,
        vcodec: str = DEFAULT_VCODEC,
        streaming: bool = True,
    ) -> None:
        self.root = Path(root)
        self.repo_id = repo_id or default_repo_id(self.root)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.use_videos = use_videos
        self.robot_type = robot_type
        self.keys = tuple(keys)
        self.camera_names = tuple(camera_names)
        self.image_writer_threads = int(image_writer_threads)
        self.vcodec = vcodec
        #: What ``vcodec`` resolved to once LeRobot had a look at the machine —
        #: ``auto`` is a request, not a codec. Set by :meth:`_encoder`.
        self.resolved_vcodec: str | None = None
        self.streaming = streaming
        self.dataset: Any = None
        #: The episode that has been recorded but not yet written. At most one:
        #: a rollout is finished before the next is asked for.
        self.pending: EpisodeDataset | None = None
        self._pending_notes: dict[str, Any] = {}
        #: Every episode that could not be written, in order. Read by the CLI,
        #: which says so loudly: a scored row that records nothing is worthless,
        #: and the operator has to find that out during the session rather than
        #: the next morning.
        self.failures: list[str] = []

    # -- lifecycle ---------------------------------------------------------- #

    def open(self) -> Any:
        """Create the dataset, or resume the one already in ``root``. Idempotent.

        Resuming rather than creating is the behaviour that matters: an
        interrupted scoring session must add its remaining attempts to the same
        dataset. A directory that exists but holds a different codebase version,
        or a different robot, is an error — silently appending 14-D DK1 episodes
        to somebody else's dataset is not recoverable.
        """
        if self.dataset is not None:
            return self.dataset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = dataset_features(
            width=self.width,
            height=self.height,
            use_videos=self.use_videos,
            keys=self.keys,
            camera_names=self.camera_names,
        )
        threads = self.image_writer_threads * len(self.camera_names)
        encoder = self._encoder()
        if self._exists():
            self.dataset = self._resume(LeRobotDataset, threads, encoder)
        else:
            # The parent only: LeRobotDatasetMetadata.create makes `root` itself
            # with exist_ok=False, so pre-creating it fails with EEXIST — which is
            # a confusing way to be told the directory was already there.
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.dataset = LeRobotDataset.create(
                self.repo_id,
                self.fps,
                root=self.root,
                robot_type=self.robot_type,
                features=features,
                use_videos=self.use_videos,
                image_writer_threads=threads,
                rgb_encoder=encoder,
                streaming_encoding=self.streaming,
                # One episode's metadata, written when that episode is written.
                # The default buffers ten and flushes in batches.
                metadata_buffer_size=1,
            )
            logger.info("created %s dataset at %s", CODEBASE_VERSION, self.root)
        self._check_version()
        self._make_durable()
        self._clear_stale_frames()
        return self.dataset

    def _encoder(self) -> Any:
        """The video encoder settings, or ``None`` to take LeRobot's."""
        if not self.use_videos:
            return None
        from lerobot.configs.video import RGBEncoderConfig

        encoder = RGBEncoderConfig(vcodec=self.vcodec)
        if encoder.vcodec.endswith(NVENC_SUFFIX) and (encoder.g or 0) < NVENC_MIN_GOP:
            encoder = RGBEncoderConfig(vcodec=encoder.vcodec, g=NVENC_MIN_GOP)
        # What ``auto`` turned into. Every later decision about the codec has to
        # read this rather than ``self.vcodec``, which is still the word "auto".
        self.resolved_vcodec = encoder.vcodec
        logger.info("encoding video with %s (gop %s)", encoder.vcodec, encoder.g)
        return encoder

    def _parallel_encoding(self) -> bool:
        """Whether ``save_episode`` may encode the three cameras in parallel.

        **It may not, when the codec runs on the GPU.** LeRobot encodes the
        cameras concurrently in a :class:`~concurrent.futures.ProcessPoolExecutor`,
        which on Linux **forks** — and a CUDA context cannot survive a fork. The
        policy has one the moment its weights reach the GPU, so every child
        process inherits an unusable driver state and NVENC cannot start in it:
        ``avcodec_open2(h264_nvenc)`` fails with a bare `UNKNOWN`, once per
        camera, and the episode is lost. That is what ate the first attempt of
        A0 on 2026-08-27.

        Nothing about the *codec* is wrong — the same encoder works in this
        process, before and after. Only the fork is, so the fix is not to fork:
        encoding the three streams one after another keeps NVENC and costs
        about a third more wall clock at the end of an episode, because the
        wait is dominated by staging frames through PNG rather than by the
        encode. ``--stream-video`` is the lever that removes *that*.

        A CPU codec forks perfectly well and gains most of a 3x from doing so,
        so it keeps the parallel path.
        DIAGNOSTICS § *Recording: the encode that could not fork*.
        """
        vcodec = self.resolved_vcodec
        if vcodec is None:
            # No episode has been opened, so nothing has resolved ``auto`` yet.
            # Assume the hardware encoder: serialising a CPU encode is slow,
            # losing a GPU one is fatal.
            vcodec = self._encoder().vcodec if self.use_videos else ""
        parallel = not str(vcodec).endswith(NVENC_SUFFIX)
        if not parallel:
            logger.debug("encoding cameras serially: %s cannot start after a fork", vcodec)
        return parallel

    def _make_durable(self) -> None:
        """Arrange for every episode to be readable the moment it is written.

        Three knobs, all of them working around the same thing: v3.0 keeps its
        parquet writers open across episodes and writes the footer only on
        ``finalize``. A dataset whose process dies — the machine froze on
        2026-08-25, mid-session, with seven episodes recorded — is then missing
        the footer on the data file AND the whole of ``meta/episodes/``, and
        cannot be opened at all. The frames are there; nothing can read them.

        So: one file per episode (the size limit is what triggers a rotation,
        and a rotation is what closes the previous writer), a metadata buffer of
        one, and :meth:`_seal` after every commit.
        """
        meta = getattr(self.dataset, "meta", None)
        if meta is None:  # pragma: no cover - a fake dataset in the tests
            return
        try:
            # `update_chunk_settings` is the supported way in; it also writes the
            # number into info.json, so a session that resumes this directory
            # rotates per episode too, without being told again.
            meta.update_chunk_settings(data_files_size_in_mb=PER_EPISODE_FILE_MB)
            meta._metadata_buffer_size = 1
        except Exception as exc:  # noqa: BLE001 - a changed internal must not stop a run
            logger.warning("could not make %s crash-durable: %s", self.root, exc)

    def _seal(self) -> None:
        """Close both parquet writers so what is on disk is a readable file.

        Called after every committed episode. The next episode opens a new file
        rather than reopening this one — that is what :meth:`_make_durable`
        arranges, and without it reopening would truncate the file it appends to.
        """
        for owner, name in ((getattr(self.dataset, "writer", None), "close_writer"),
                            (getattr(self.dataset, "meta", None), "_close_writer")):
            close = getattr(owner, name, None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - the episode is already written
                logger.warning("could not close a writer on %s: %s", self.root, exc)

    def _clear_stale_frames(self) -> None:
        """Delete the PNG cache of an episode that was never written.

        A crash leaves one episode's frames — 4.6 GB, for three 720p cameras
        over two minutes — under ``images/``, belonging to an episode the
        metadata does not know about. Nothing will ever read them.
        """
        import shutil

        images = self.root / "images"
        if not images.is_dir():
            return
        known = self.episodes
        for key_dir in images.iterdir():
            for episode_dir in key_dir.iterdir() if key_dir.is_dir() else []:
                if not episode_dir.name.startswith("episode-"):
                    continue
                try:
                    index = int(episode_dir.name.removeprefix("episode-"))
                except ValueError:  # pragma: no cover - not ours to judge
                    continue
                if index < known:
                    continue
                logger.warning("removing frames of unwritten episode %d in %s", index, key_dir)
                shutil.rmtree(episode_dir, ignore_errors=True)

    def _exists(self) -> bool:
        return (self.root / "meta" / "info.json").is_file()

    def _resume(self, cls: Any, threads: int, encoder: Any = None) -> Any:
        try:
            dataset = cls.resume(
                self.repo_id,
                root=self.root,
                image_writer_threads=threads,
                rgb_encoder=encoder,
                streaming_encoding=self.streaming,
            )
        except Exception as exc:  # noqa: BLE001 - the reason is what the operator needs
            raise DatasetError(
                f"{self.root} already holds a dataset that could not be resumed: {exc}. "
                f"Point --dataset-dir somewhere else rather than mixing two runs into "
                f"one dataset."
            ) from exc
        logger.info("resuming %s at episode %d", self.root, dataset.num_episodes)
        return dataset

    def _check_version(self) -> None:
        """Assert the directory really is v3.0. Matters only when resuming.

        The leading ``v`` is stripped from both sides: the file writes
        ``codebase_version: "v3.0"`` and LeRobot parses it into a version object
        that renders as ``3.0``, so comparing the two verbatim rejects every
        dataset including the ones this module just wrote.
        """
        version = str(getattr(getattr(self.dataset, "meta", None), "_version", "") or "")
        if version and version.lstrip("v") != CODEBASE_VERSION.lstrip("v"):
            raise DatasetError(
                f"{self.root} is a v{version.lstrip('v')} dataset; this writes "
                f"{CODEBASE_VERSION}. The two lay episodes out differently on disk."
            )

    def commit(self) -> bool:
        """Write the pending episode into the dataset. Returns whether one went.

        This is where ``save_episode`` happens — the video encode and the
        metadata append — rather than at the end of the rollout, so that
        declining an attempt is a decision the operator can still make.
        """
        episode = self.pending
        if episode is None or self.dataset is None:
            return False
        logger.info("encoding episode %d (%d frames) ...", episode.index, episode.ticks)
        try:
            self.dataset.save_episode(parallel_encoding=self._parallel_encoding())
        except Exception as exc:  # noqa: BLE001 - a failed encode must not raise here
            # The whole traceback, to the log file: on 2026-08-26 this failure
            # happened once, printed one line nobody read, and the session went
            # on to score five more attempts that recorded nothing.
            logger.exception("could not write episode %d", episode.index)
            self.failures.append(f"episode {episode.index}: {type(exc).__name__}: {exc}")
            self._recover()
            return False
        finally:
            self.pending = None
        self._seal()
        self._write_notes(episode)
        return True

    def _recover(self) -> None:
        """Put the writer back in a state the next episode can use.

        A ``save_episode`` that raises part-way leaves the episode buffer
        populated, and the next episode's first frame then fails on a frame
        index that does not follow — which is how one failed encode turned into
        a session that recorded nothing at all. Clearing the buffer costs the
        episode that already failed and saves every one after it.
        """
        try:
            self.dataset.clear_episode_buffer()
        except Exception as exc:  # noqa: BLE001 - recovery must not raise
            logger.warning("could not clear the buffer after a failed write: %s", exc)

    def drop(self) -> bool:
        """Throw the pending episode away. Returns whether there was one."""
        if self.pending is None:
            return False
        self.pending = None
        self._pending_notes = {}
        if self.dataset is None:
            return True
        try:
            self.dataset.clear_episode_buffer()
        except Exception as exc:  # noqa: BLE001 - discarding must not raise
            logger.warning("could not clear the episode buffer: %s", exc)
        return True

    def _write_notes(self, episode: EpisodeDataset) -> None:
        """Append the episode's settings to ``dk1_notes.jsonl``.

        Its own file rather than the dataset's metadata: v3.0 has no per-episode
        free-form slot, and inventing one inside ``meta/`` would make the
        directory something the standard tools no longer recognise. An episode
        that cannot say what produced it is evidence of nothing.
        """
        import json

        notes, self._pending_notes = self._pending_notes, {}
        record = {
            "episode": episode.index,
            "task": episode.task,
            "frames": episode.ticks,
            "recorded": datetime.now().isoformat(timespec="seconds"),
            **notes,
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "dk1_notes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # noqa: BLE001 - notes are not worth a failed run
            logger.warning("could not write the episode notes: %s", exc)

    def close(self) -> None:
        """Commit anything pending, finalise the dataset, and let it go.

        **A pending episode is committed, not dropped.** Reaching here with one
        means nobody answered the keep prompt — the process ended, or a caller
        forgot — and the rule is the same one the ``.rrd`` recorder follows: an
        attempt that cannot be repeated must not be lost by default.

        Finalising matters on its own. v3.0 buffers per-episode metadata and
        flushes it in batches, so a dataset that is never finalised can be
        missing its last episodes' metadata even though their frames are on disk.
        """
        if self.dataset is None:
            return
        if self.pending is not None:
            logger.warning(
                "episode %d was never kept or discarded; keeping it", self.pending.index
            )
            self.commit()
        try:
            self.dataset.finalize()
        except Exception as exc:  # noqa: BLE001 - closing must not raise
            logger.warning("could not finalise the dataset: %s", exc)
        finally:
            self.dataset = None

    def __enter__(self) -> DatasetSession:
        self.open()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @property
    def episodes(self) -> int:
        """How many episodes the dataset already holds."""
        if self.dataset is None:
            return 0
        return int(getattr(self.dataset, "num_episodes", 0))

    # -- one episode -------------------------------------------------------- #

    def episode(self, task: str, *, notes: dict[str, Any] | None = None) -> DatasetEpisodeRecorder:
        """A recorder for the next episode. Does not start it."""
        return DatasetEpisodeRecorder(self, task=task, notes=notes)


def default_repo_id(root: Path | str) -> str:
    """A ``<owner>/<name>`` derived from the directory, for a dataset with no id.

    The directory is what the operator typed; the id is bookkeeping that only
    matters once the dataset goes to the Hub, which is where these belong (they
    exceed GitHub's 100 MB file limit several times over). ``dk1`` is the owner
    stand-in and is replaced at upload.
    """
    name = slug(Path(root).name) or "dk1"
    return f"dk1/{name}"


# --------------------------------------------------------------------------- #
# The per-episode instrument
# --------------------------------------------------------------------------- #


class DatasetEpisodeRecorder:
    """Writes one episode into a :class:`DatasetSession`. **Read-only.**

    Lifecycle, the same four calls :class:`dk1lab.record.EpisodeRecorder` takes::

        recorder = session.episode(task)
        recorder.attach(ctx)      # wrap the robot's two calls
        recorder.start()          # open the dataset, begin the episode
        ...                       # the rollout runs
        report = recorder.stop()  # save the episode and report
        recorder.detach()

    Args:
        session: the dataset this episode is appended to.
        task: the instruction, written onto **every frame**. That is v3.0's own
            arrangement, not a choice here, and it is what makes an episode
            searchable in the Hub viewer.
        notes: anything else worth keeping — the checkpoint, the profile, the
            speed cap. Written to ``<root>/dk1_notes.jsonl``, one line per
            episode, because the v3.0 metadata has no per-episode slot for it and
            an episode that cannot say what produced it is evidence of nothing.
    """

    def __init__(
        self,
        session: DatasetSession,
        *,
        task: str,
        notes: dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.task = task
        self.notes = dict(notes or {})
        self.ticks = 0
        self.failed: str | None = None

        self._restore: list[tuple[Any, str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._started: float = 0.0
        self._report: EpisodeDataset | None = None
        self._index: int = 0

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Open the dataset and begin an episode. Idempotent.

        Guarded like everything else here: an unwritable directory must leave the
        rollout running and unrecorded rather than stop it, and the operator is
        told rather than left to notice that no dataset appeared.
        """
        import time

        if self._started or self.failed is not None:
            return
        self._started = time.perf_counter()
        try:
            self.session.open()
            self._index = self.session.episodes
        except Exception as exc:  # noqa: BLE001 - a recorder must never stop a rollout
            self._fail(exc)
            return
        logger.info("recording this episode into %s", self.session.root)

    def stop(self) -> EpisodeDataset:
        """End the episode and report. **Does not write it.** Safe without :meth:`start`.

        The episode stays in the dataset's buffer until the report's
        :meth:`~EpisodeDataset.keep` or :meth:`~EpisodeDataset.discard` is called,
        or until :meth:`DatasetSession.close` keeps it. That deferral is what
        makes declining an attempt actually remove it — ``save_episode`` encodes
        the video and appends the metadata, and v3.0 has no way to take an
        episode back out afterwards.

        Idempotent, and it returns the *same* report every time: the rollout that
        ends the episode and the operator who is told about it are two callers
        wanting one answer, and the second must not be handed a longer duration
        measured from a stopped clock.
        """
        import time

        if self._report is not None:
            return self._report
        seconds = time.perf_counter() - self._started if self._started else 0.0
        writable = bool(self.ticks) and self.failed is None and self.session.dataset is not None
        report = EpisodeDataset(
            root=self.session.root,
            repo_id=self.session.repo_id,
            index=self._index,
            ticks=self.ticks,
            seconds=seconds,
            task=self.task,
            failed=self.failed,
            _session=self.session if writable else None,
        )
        if writable:
            self.session.pending = report
            self.session._pending_notes = dict(self.notes)
        elif self.session.dataset is not None:
            # An episode that produced nothing, or that failed part way, must not
            # leave a buffer behind for the next one to inherit.
            self.session.drop()
        self._report = report
        return report

    # -- attach ------------------------------------------------------------- #

    def attach(self, ctx: Any) -> None:
        """Wrap the robot's two calls, in tick order.

        ``get_observation`` opens the tick and its result is held; ``send_action``
        closes it, and the frame is written from the pair. Only the robot is
        wrapped — unlike the ``.rrd`` recorder there is no engine call to hook,
        because a LeRobot dataset has no slot for the policy's own plan. That
        absence is the reason both recorders exist.
        """
        robot = getattr(getattr(ctx, "hardware", None), "robot_wrapper", None)
        if robot is None:
            logger.warning("no robot wrapper: nothing to record into a dataset")
            self._fail(RuntimeError("no robot wrapper on the rollout context"))
            return
        self.attach_robot(robot)

    def attach_robot(self, robot: Any) -> None:
        """Wrap a robot directly, with no rollout context around it.

        :meth:`attach` is the policy path, where the robot is reached through the
        context LeRobot's rollout builds. **Teleoperation has no such context** —
        :mod:`dk1lab.demos` drives the leader and the follower itself — and the
        two calls this recorder needs are the robot's own, so the context was
        never the thing actually being depended on. Same wrapping, same
        restoration, one object less to have to fake.
        """
        inner_obs = robot.get_observation

        def get_observation(*args: Any, **kwargs: Any) -> Any:
            observation = inner_obs(*args, **kwargs)
            self._hold(observation)
            return observation

        robot.get_observation = get_observation
        self._restore.append((robot, "get_observation", inner_obs))

        inner_send = robot.send_action

        def send_action(action: Any) -> Any:
            sent = inner_send(action)
            self._close_tick(sent if sent is not None else action)
            return sent

        robot.send_action = send_action
        self._restore.append((robot, "send_action", inner_send))

    def detach(self) -> None:
        """Put the wrapped methods back, in reverse order of wrapping."""
        for owner, name, inner in reversed(self._restore):
            try:
                setattr(owner, name, inner)
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("could not restore %s.%s", type(owner).__name__, name)
        self._restore.clear()
        self._pending = None

    # -- the control loop's half -------------------------------------------- #

    def _hold(self, observation: Any) -> None:
        """Keep this tick's observation until the action that goes with it exists."""
        if isinstance(observation, dict):
            self._pending = observation

    def _close_tick(self, action: Any) -> None:
        """Write one frame: the held observation and the action the arms were given.

        A tick with no held observation is dropped rather than written with a
        stale one — a frame pairing this tick's action with the previous tick's
        picture would teach a fine-tune a lag that is not there.
        """
        observation, self._pending = self._pending, None
        if observation is None or not isinstance(action, dict):
            return
        if self.session.dataset is None or self.failed is not None or not self._started:
            return
        try:
            frame = self._frame(observation, action)
        except Exception as exc:  # noqa: BLE001 - a recorder must never stop a rollout
            # Every exception, not only the KeyError a missing channel raises:
            # anything escaping here leaves through ``send_action`` and takes the
            # teleoperation loop with it, with the arms live.
            self._fail(exc)
            return
        try:
            self.session.dataset.add_frame(frame)
        except Exception as exc:  # noqa: BLE001 - a recorder must never stop a rollout
            self._fail(exc)
            return
        self.ticks += 1

    def _frame(self, observation: dict, action: dict) -> dict:
        """One v3.0 frame: the state vector, the three images, the action, the task."""
        import numpy as np

        from .layout import vector_from_dict

        keys = self.session.keys
        frame: dict[str, Any] = {
            "observation.state": np.asarray(vector_from_dict(observation, keys), dtype=np.float32),
            "action": np.asarray(vector_from_dict(action, keys), dtype=np.float32),
            "task": self.task,
        }
        for name in self.session.camera_names:
            image = observation.get(name)
            if image is not None:
                frame[f"observation.images.{name}"] = image
        return frame

    def _fail(self, exc: BaseException) -> None:
        """Switch recording off, once, and say why."""
        if self.failed is None:
            self.failed = str(exc)
            logger.warning("dataset recording disabled after an error: %s", exc)


# --------------------------------------------------------------------------- #
# Recording an episode two ways at once
# --------------------------------------------------------------------------- #


class Recorders:
    """Several per-episode recorders driven as one. **Read-only, like each of them.**

    ``STUDY.md`` puts the LeRobot dataset in the evidence path and keeps the
    ``.rrd`` for when a rollout has to be diagnosed against the policy's own
    plan — which means an attempt is sometimes worth recording both ways, and
    the operator should still be asked about it **once**, because the question is
    about the attempt rather than about the file.

    Both recorders take the same four calls, so this is a fan-out and nothing
    more. Order is preserved: they attach in the order given and detach in
    reverse, so the wrapping chain is restored exactly as it was found.
    """

    def __init__(self, *recorders: Any) -> None:
        self.recorders = tuple(r for r in recorders if r is not None)

    def __bool__(self) -> bool:
        return bool(self.recorders)

    def attach(self, ctx: Any) -> None:
        for recorder in self.recorders:
            recorder.attach(ctx)

    def start(self) -> None:
        for recorder in self.recorders:
            recorder.start()

    def stop(self) -> Any:
        reports = [recorder.stop() for recorder in self.recorders]
        return reports[0] if len(reports) == 1 else CombinedRecording(tuple(reports))

    def detach(self) -> None:
        for recorder in reversed(self.recorders):
            recorder.detach()


@dataclass(frozen=True)
class CombinedRecording:
    """What one episode produced, across every recorder that was on it.

    Carries the same three methods a single report does, so the operator prompt
    does not have to know how many files an attempt turned into.
    """

    reports: tuple[Any, ...]

    def summary(self) -> str:
        return "\n".join(report.summary() for report in self.reports)

    def keep(self) -> bool:
        """Commit every report that needs committing. An ``.rrd`` already is one."""
        return any(report.keep() for report in self.reports if hasattr(report, "keep"))

    def discard(self) -> bool:
        """Drop all of them. One attempt, one decision."""
        return all(report.discard() for report in self.reports)


def one(*recorders: Any) -> Any:
    """The single recorder among ``recorders``, a :class:`Recorders`, or ``None``.

    So a caller can hand ``run`` or ``PolicySession`` whatever was asked for
    without either of them growing a notion of how many recorders there are.
    """
    present = [r for r in recorders if r is not None]
    if not present:
        return None
    return present[0] if len(present) == 1 else Recorders(*present)


# --------------------------------------------------------------------------- #
# Reading one back
# --------------------------------------------------------------------------- #
#
# A recording session is the one thing in this study that a day cannot buy back,
# so the question "is what is on disk the dataset we meant to record?" has to be
# answerable before anything is trained on it — and answerable *cheaply*, with
# nothing decoded and no GPU touched. Everything below reads `meta/` and this
# fork's own `dk1_notes.jsonl`: JSON and parquet, no torch, no LeRobot, no video.
#
# It is deliberately not `LeRobotDatasetMetadata`. That class downloads from the
# Hub when local metadata looks incomplete, which for a half-written local
# dataset named `dk1/demos` means a network error instead of the diagnosis.


@dataclass(frozen=True)
class DatasetSummary:
    """What a recorded dataset says about itself, and what looks wrong with it.

    ``problems`` is the part that matters: a dataset that reads back clean is
    worth one line, and one that does not has to say so before an overnight run
    is spent on it.
    """

    root: Path
    repo_id: str = ""
    codebase_version: str = ""
    robot_type: str = ""
    fps: int = 0
    episodes: int = 0
    frames: int = 0
    lengths: tuple[int, ...] = ()
    tasks: tuple[str, ...] = ()
    cameras: dict[str, tuple[int, int]] = field(default_factory=dict)
    video_bytes: int = 0
    #: The per-episode records of ``dk1_notes.jsonl``, in the order written.
    notes: list[dict[str, Any]] = field(default_factory=list)
    #: ``{episode: scene}``, from the notes. ``None`` for an episode with no note.
    scenes: dict[int, Any] = field(default_factory=dict)
    #: The distinct value of each recording setting across the episodes. More
    #: than one value in any of them means the dataset is not one experiment.
    settings: dict[str, list[Any]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def by_scene(self) -> dict[Any, int]:
        counts: dict[Any, int] = {}
        for scene in self.scenes.values():
            counts[scene] = counts.get(scene, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (item[0] is None, item[0])))


#: Settings whose value must be the same in every episode of one dataset. The
#: profile and the capture decide what the policy will see; the rate decides what
#: an action chunk's time scale means. A dataset mixing two of any of them is two
#: datasets, and a fine-tune cannot tell them apart.
ONE_VALUE_SETTINGS = ("profile", "capture", "fps", "vcodec")


def summarise(root: Path | str) -> DatasetSummary:
    """Read a recorded dataset's metadata and say whether it looks right.

    Decodes no video and loads no model: ``meta/info.json``, the episode
    parquet, and ``dk1_notes.jsonl``. Seconds, not minutes.

    Raises:
        DatasetError: if the directory is not a LeRobot dataset at all. Every
            other fault is a line in :attr:`DatasetSummary.problems`, because
            the operator wants the whole list at once rather than the first one.
    """
    import json

    from .finetune import episode_scenes, read_notes

    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise DatasetError(
            f"{root} is not a LeRobot dataset — no meta/info.json. Point at the "
            f"dataset directory itself, not its parent."
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{info_path}: invalid JSON — {exc}") from exc

    problems: list[str] = []
    lengths, tasks = _episode_lengths(root, problems)
    features = info.get("features") or {}
    cameras = {}
    for key, feature in features.items():
        if not key.startswith("observation.images."):
            continue
        shape = feature.get("shape") or []
        if len(shape) == 3:
            cameras[key] = (int(shape[1]), int(shape[0]))

    notes = read_notes(root)
    episodes = len(lengths)
    scenes = episode_scenes(notes, range(episodes))
    settings = {
        name: sorted(
            {record[name] for record in notes if name in record}, key=lambda value: str(value)
        )
        for name in ONE_VALUE_SETTINGS
    }

    summary = DatasetSummary(
        root=root,
        repo_id=str(info.get("repo_id") or ""),
        codebase_version=str(info.get("codebase_version") or ""),
        robot_type=str(info.get("robot_type") or ""),
        fps=int(info.get("fps") or 0),
        episodes=episodes,
        frames=sum(lengths),
        lengths=tuple(lengths),
        tasks=tuple(sorted(tasks)),
        cameras=cameras,
        video_bytes=_tree_bytes(root / "videos"),
        notes=notes,
        scenes=scenes,
        settings=settings,
        problems=problems,
    )
    problems.extend(_faults(summary, info))
    return summary


def _episode_lengths(root: Path, problems: list[str]) -> tuple[list[int], set[str]]:
    """Every episode's frame count and task, from ``meta/episodes/``.

    The episode metadata is what the 2026-08-25 freeze destroyed while leaving
    the videos intact, so its absence is the fault this function exists to find:
    a dataset whose ``videos/`` is full and whose ``meta/episodes/`` is empty has
    no per-frame state and cannot be trained on.
    """
    import pyarrow.parquet as pq

    directory = root / "meta" / "episodes"
    if not directory.is_dir():
        problems.append(
            "meta/episodes/ is missing entirely — the videos may be intact but the "
            "per-episode metadata is not, and nothing can read the frames back"
        )
        return [], set()

    lengths: list[int] = []
    tasks: set[str] = set()
    for path in sorted(directory.rglob("*.parquet")):
        try:
            table = pq.read_table(path)
        except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
            problems.append(f"{path.relative_to(root)} cannot be read: {exc}")
            continue
        columns = table.column_names
        if "length" in columns:
            lengths.extend(int(value) for value in table.column("length").to_pylist())
        if "tasks" in columns:
            for entry in table.column("tasks").to_pylist():
                if isinstance(entry, list):
                    tasks.update(str(task) for task in entry)
                elif entry is not None:
                    tasks.add(str(entry))
    return lengths, tasks


def _tree_bytes(directory: Path) -> int:
    """Total size of a directory tree, or 0 if it is not there."""
    if not directory.is_dir():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _faults(summary: DatasetSummary, info: dict[str, Any]) -> list[str]:
    """Everything wrong with a dataset that its own metadata can reveal."""
    found: list[str] = []

    if summary.codebase_version != CODEBASE_VERSION:
        found.append(
            f"codebase_version is {summary.codebase_version!r}, not {CODEBASE_VERSION!r} — "
            f"a v2.1 directory has a different layout and reading it with a v3.0 tool "
            f"fails later rather than here"
        )
    if summary.episodes == 0:
        found.append("no episodes: meta/episodes/ holds nothing")
    if len(summary.tasks) > 1:
        found.append(
            f"{len(summary.tasks)} different task strings: {sorted(summary.tasks)}. The "
            f"prompt has to be byte-identical on every frame — it is what both "
            f"fine-tunes condition on"
        )
    if missing := [name for name in CAMERA_NAMES if f"observation.images.{name}" not in summary.cameras]:
        found.append(f"missing camera stream(s): {missing}; this cell records {list(CAMERA_NAMES)}")
    sizes = set(summary.cameras.values())
    if len(sizes) > 1:
        found.append(f"the cameras disagree about frame size: {sorted(sizes)}")
    if summary.frames and summary.video_bytes == 0:
        found.append("videos/ is empty although episodes were recorded — the encodes failed")

    short = [index for index, length in enumerate(summary.lengths) if length < 2]
    if short:
        found.append(f"episode(s) {short} hold fewer than two frames")

    for name in ONE_VALUE_SETTINGS:
        values = summary.settings.get(name) or []
        if len(values) > 1:
            found.append(
                f"the episodes disagree about {name}: {values}. One dataset is one "
                f"observation path; two is two experiments mixed together"
            )

    if not summary.notes:
        found.append(
            "no dk1_notes.jsonl — the scene of each episode is unknown, so the "
            "validation hold-out cannot be spread across the three layouts"
        )
    else:
        unlabelled = [index for index, scene in summary.scenes.items() if scene is None]
        if unlabelled:
            found.append(f"episode(s) {unlabelled} carry no scene label")
        noted = {record.get("episode") for record in summary.notes}
        if summary.episodes and len(noted) != summary.episodes:
            found.append(
                f"{len(noted)} episode(s) have notes but {summary.episodes} are on disk"
            )

    if (info.get("fps") or 0) <= 0:
        found.append("fps is missing or zero in meta/info.json")
    return found
