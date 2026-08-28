"""Apply the ``optimized`` wrist crop to a dataset that was recorded without it.

The demonstrations for ``STUDY.md`` Phase 3 are recorded under ``--profile
common`` — the full 105 degree lens, no crop — because that way **one** day of
hands serves two rows: A1 trains on those bytes as they are, and R1 trains on
them with the crop applied. Recording the other way round cannot be undone: a
cropped frame does not contain the pixels the uncropped row needs.

So the crop has to reach the dataset. This module is how, and the choice it
makes is to **materialise** it: copy the dataset and rewrite the two wrist video
streams through the same box :class:`dk1lab.crop.CroppedOpenCVCamera` uses, so
what comes out is an ordinary LeRobot v3.0 dataset that ``lerobot-train``, the
dataset viewer and the Hub all read with no help from us.

**Why materialise rather than transform at load.** Three reasons, in the order
they mattered:

* ``make_train_eval_datasets`` passes ``image_transforms`` to the training
  dataset and **``None`` to the evaluation one** — it is LeRobot's augmentation
  hook, not a lens. A crop applied to only the training half would make the
  held-out loss measure a different camera;
* the hook is called per camera key with no key, so it cannot crop the wrists
  and leave the top view alone, which is exactly what this cell's configuration
  says to do;
* a materialised dataset can be *looked at*. The crop is a geometry change on the
  input of a 5.44 B model trained overnight; being able to open the result in
  ``lerobot-dataset-viz`` and see the box is worth a disk copy.

The alternative that was **not** taken: LeRobot registers an
``image_crop_resize_processor`` step, which would put the crop in the fine-tuned
checkpoint's own preprocessor. That works, and it makes R1 the one row whose
crop is not in the camera — so its rollout would need ``--profile common``
cameras plus a pipeline that crops, disagreeing with R0 about where the lens
lives and with ``CLAUDE.md`` about why the crop is in the camera at all. Not
worth the disk it saves.

**Nothing but pixels changes.** The box is cropped and then stretched back to the
frame's own size, exactly as the camera does it, so every video keeps its width,
height, frame count and frame rate. That is what lets the copy reuse the source's
``meta/`` untouched: the parquet timestamps, the episode ranges, the feature
shapes and the per-frame state all still describe it. A crop that changed the
frame size would need every one of those rewritten, and a v3.0 video file holds
several episodes, so getting it wrong is not visible until training reads the
wrong frames.

**Generation loss is real and is the accepted price.** The recorded frame was
encoded once; this decodes it, crops, resizes and encodes again. The camera path
at rollout crops and resizes *before* its only encode. So R1's training frames
are one encode generation behind what its cameras will deliver. Measurable,
small at the CRF this cell records at, and cheaper than a second day of
teleoperation. Stated here so nobody rediscovers it as a mystery.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import DK1Config
from .fov import CropBox, crop_box, describe
from .layout import CAMERA_NAMES, IMAGE_KEYS

__all__ = [
    "CropPlan",
    "LENS_FILE",
    "RecropError",
    "Report",
    "crop_dataset",
    "image_key",
    "plan",
    "read_lens",
    "video_files",
]

#: Written into the cropped copy, so a dataset can say what was done to it.
#: ``dk1 dataset check`` reads it, and ``dk1 policy finetune`` refuses to train a
#: row whose lens does not match its profile.
LENS_FILE = "dk1_lens.json"


class RecropError(Exception):
    """Raised when a dataset cannot be cropped. Nothing has been written."""


def image_key(name: str) -> str:
    """``top`` -> ``observation.images.top``.

    Derived from :data:`dk1lab.layout.IMAGE_KEYS` rather than formatted here, so
    the one place that decides what a camera is called stays the one place.
    """
    try:
        return IMAGE_KEYS[CAMERA_NAMES.index(name)]
    except ValueError:
        raise RecropError(
            f"no camera named {name!r} — this cell has {list(CAMERA_NAMES)}"
        ) from None


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CropPlan:
    """Which video streams get which box, at one frame size.

    Attributes:
        boxes: ``{video key: CropBox}``, holding **only** the streams that
            change. A camera with no ``target_hfov`` in ``dk1.toml`` — the top
            view on this cell — is absent, and its video file is copied
            byte-for-byte rather than re-encoded.
        width: the frame width the boxes were computed against.
        height: the frame height.
        lines: one human-readable line per cropped stream, from
            :func:`dk1lab.fov.describe`. Printed before anything is written and
            kept in :data:`LENS_FILE`, because the box has gone stale in prose
            three times and a copy of the arithmetic is not the arithmetic.
    """

    boxes: dict[str, CropBox]
    width: int
    height: int
    lines: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.boxes)


def plan(config: DK1Config, *, width: int, height: int) -> CropPlan:
    """The crop this ``dk1.toml`` asks for, at a ``width x height`` frame.

    The same call :mod:`dk1lab.cameras` makes when it builds a
    :class:`~dk1lab.crop.CroppedOpenCVCameraConfig`, against the same
    :func:`dk1lab.fov.crop_box` — which is the whole point. A second
    implementation of the box here would be a second lens, and the row it trains
    would be for a camera that does not exist.

    Note it reads the **unmodified** config: ``--profile common`` is what strips
    the crop out for a rollout, and this function's job is the opposite one.
    """
    boxes: dict[str, CropBox] = {}
    lines: dict[str, str] = {}
    for name in CAMERA_NAMES:
        device = config.camera(name)
        if not device.cropped:
            continue
        # `cropped` is exactly "both angles are set"; the locals are what tells
        # a type checker that, and they cost nothing.
        source_hfov, target_hfov = float(device.hfov), float(device.target_hfov)
        box = crop_box(
            width,
            height,
            source_hfov,
            target_hfov,
            inset=device.crop_inset,
            shift_x=device.crop_shift_x,
            shift_y=device.crop_shift_y,
        )
        if box.is_full_frame:
            continue
        key = image_key(name)
        boxes[key] = box
        lines[key] = describe(
            width,
            height,
            source_hfov,
            target_hfov,
            inset=device.crop_inset,
            shift_x=device.crop_shift_x,
            shift_y=device.crop_shift_y,
        )
    return CropPlan(boxes=boxes, width=width, height=height, lines=lines)


# --------------------------------------------------------------------------- #
# Reading the dataset
# --------------------------------------------------------------------------- #


def read_info(root: Path | str) -> dict[str, Any]:
    """A dataset's ``meta/info.json``.

    Raises:
        RecropError: if the directory is not a LeRobot dataset. Said here rather
            than letting a ``FileNotFoundError`` out, because the usual cause is
            pointing at ``study/`` instead of ``study/demos``.
    """
    path = Path(root) / "meta" / "info.json"
    if not path.is_file():
        raise RecropError(
            f"{Path(root)} is not a LeRobot dataset — no meta/info.json. "
            f"Point at the dataset directory itself, not its parent."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecropError(f"{path}: invalid JSON — {exc}") from exc


def frame_size(info: dict[str, Any], key: str) -> tuple[int, int]:
    """``(width, height)`` of one video stream, off the dataset's own metadata.

    Read rather than assumed: the boxes are computed for the size the frames
    really are, and a dataset recorded at ``[capture.teleop]`` instead of
    ``[capture.policy]`` would otherwise be cropped to a box meant for a
    different picture.
    """
    feature = (info.get("features") or {}).get(key) or {}
    shape = feature.get("shape") or []
    names = [str(name) for name in (feature.get("names") or [])]
    if len(shape) != 3:
        raise RecropError(f"{key} has shape {shape}, expected three dimensions")
    if names[:3] == ["channels", "height", "width"]:
        return int(shape[2]), int(shape[1])
    # LeRobot records camera features as (height, width, channels).
    return int(shape[1]), int(shape[0])


def video_files(root: Path | str, info: dict[str, Any], key: str) -> list[Path]:
    """Every video file holding ``key``'s frames, in path order.

    A v3.0 dataset stores several episodes per file under
    ``videos/<key>/chunk-XXX/file-XXX.mp4``, so this is a glob rather than one
    path per episode — and it is why cropping cannot be done per episode.
    """
    template = info.get("video_path")
    if not template:
        raise RecropError(
            f"{Path(root)} has no video_path in its metadata, so its images are not "
            f"videos. Only a video dataset can be re-encoded."
        )
    directory = Path(root) / "videos" / key
    if not directory.is_dir():
        raise RecropError(f"{directory} does not exist — {key} has no video files")
    return sorted(directory.rglob("*.mp4"))


# --------------------------------------------------------------------------- #
# Doing it
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    """What :func:`crop_dataset` did, for the operator and for the lens file."""

    destination: Path
    streams: list[str] = field(default_factory=list)
    files: int = 0
    frames: int = 0
    seconds: float = 0.0
    copied: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.files} video file(s), {self.frames} frames across "
            f"{len(self.streams)} stream(s) in {self.seconds:.0f} s; "
            f"{len(self.copied)} stream(s) copied unchanged"
        )


def crop_dataset(
    source: Path | str,
    destination: Path | str,
    crop: CropPlan,
    *,
    vcodec: str | None = None,
    overwrite: bool = False,
    say: Callable[[str], None] | None = None,
    notes: dict[str, Any] | None = None,
) -> Report:
    """Copy ``source`` to ``destination`` and rewrite its cropped video streams.

    Args:
        source: the recorded dataset. **Read only** — nothing here writes to it,
            because it is the one artefact of this study that a day of hands
            cannot buy back.
        destination: where the cropped copy goes. Must not exist unless
            ``overwrite``.
        crop: what :func:`plan` decided, at this dataset's own frame size.
        vcodec: the codec for the rewritten streams. ``None`` reuses **the
            source's own encoder settings**, read off ``meta/info.json`` — which
            is what keeps the copy differing from the original in pixels and
            nothing else.
        overwrite: replace ``destination`` if it is there. Off by default: an
            hour of transcoding is cheap next to deleting the wrong directory.
        say: called with progress lines, or ``None`` for silence.
        notes: extra fields for :data:`LENS_FILE`.

    Raises:
        RecropError: for anything that would produce a dataset that is wrong
            rather than absent — a missing stream, a frame size the plan was not
            computed for, a re-encode that lost frames.
    """
    import time

    source, destination = Path(source), Path(destination)
    talk = say or (lambda line: None)

    if not crop:
        raise RecropError(
            "this dk1.toml asks for no crop on any camera, so there is nothing to "
            "apply. Under --profile common that is expected; R1 needs the "
            "optimized profile's [cameras.*].target_hfov."
        )
    info = read_info(source)
    for key in crop.boxes:
        if key not in (info.get("features") or {}):
            raise RecropError(
                f"{source} has no {key} stream. Its cameras are "
                f"{sorted(k for k in (info.get('features') or {}) if k.startswith('observation.images.'))}"
            )
        width, height = frame_size(info, key)
        if (width, height) != (crop.width, crop.height):
            raise RecropError(
                f"{key} is {width}x{height} but the crop was computed for "
                f"{crop.width}x{crop.height}. Re-plan at the dataset's own size — "
                f"the box is in pixels and a box for the wrong frame is a different lens."
            )

    if destination.exists():
        if not overwrite:
            raise RecropError(
                f"{destination} already exists. Pass --overwrite to replace it, or "
                f"choose another directory — a half-cropped dataset is not detectable "
                f"from the outside."
            )
        talk(f"removing the existing {destination}")
        shutil.rmtree(destination)

    talk(f"copying {source} -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)

    report = Report(destination=destination)
    report.copied = [
        key
        for key in (info.get("features") or {})
        if key.startswith("observation.images.") and key not in crop.boxes
    ]

    started = time.perf_counter()
    for key, box in crop.boxes.items():
        encoder = _encoder(info, key, vcodec)
        files = video_files(destination, info, key)
        talk(f"{key}: {crop.lines.get(key, '')}")
        talk(f"  {len(files)} file(s), re-encoding with {encoder.vcodec}")
        for path in files:
            frames = _crop_video(path, box, encoder)
            report.frames += frames
            report.files += 1
            talk(f"  {path.relative_to(destination)}: {frames} frames")
        report.streams.append(key)
    report.seconds = time.perf_counter() - started

    _write_lens(destination, source, crop, report, info, notes)
    return report


def _encoder(info: dict[str, Any], key: str, vcodec: str | None):
    """The encoder settings for one stream — the source's own unless overridden.

    Two things about NVENC, both of which present as ``avcodec_open2`` failing
    with a bare ``UNKNOWN``, i.e. as no video at all: it refuses a GOP below 4,
    and it cannot start in a forked child. Nothing here forks, and the GOP is
    raised the same way :mod:`dk1lab.dataset` raises it — same reason, same
    number, and both would otherwise be rediscovered here.
    """
    from lerobot.configs.video import RGBEncoderConfig

    from .dataset import NVENC_MIN_GOP, NVENC_SUFFIX

    feature_info = ((info.get("features") or {}).get(key) or {}).get("info")
    encoder = RGBEncoderConfig.from_video_info(feature_info)
    if vcodec:
        encoder = RGBEncoderConfig(
            vcodec=vcodec,
            pix_fmt=encoder.pix_fmt,
            g=encoder.g,
            crf=encoder.crf,
            preset=None if vcodec != encoder.vcodec else encoder.preset,
        )
    if encoder.vcodec.endswith(NVENC_SUFFIX) and (encoder.g or 0) < NVENC_MIN_GOP:
        encoder = RGBEncoderConfig(
            vcodec=encoder.vcodec, pix_fmt=encoder.pix_fmt, g=NVENC_MIN_GOP, crf=encoder.crf
        )
    return encoder


def _crop_video(path: Path, box: CropBox, encoder) -> int:
    """Rewrite one video file through ``box``. Returns the frame count.

    Decode, crop, stretch back to the frame's own size, encode — the same four
    steps, in the same order, that :meth:`dk1lab.crop.CroppedOpenCVCamera._postprocess_image`
    performs on a live frame. The rotation is **not** repeated: it was applied by
    the camera before the frame was ever recorded, so these pixels are already
    the right way up and rotating again would turn the picture over.

    Written to a temporary file beside the target and moved into place, so an
    interrupted transcode leaves the original file rather than half of a new one.
    """
    import av
    import cv2

    frames = 0
    temporary = path.with_suffix(".dk1crop.mp4")
    options = encoder.get_codec_options(as_strings=True)
    try:
        with av.open(str(path), mode="r") as source:
            stream = source.streams.video[0]
            width, height = int(stream.width), int(stream.height)
            if (width, height) != (box.frame_width, box.frame_height):
                raise RecropError(
                    f"{path} is {width}x{height} but the crop box is for "
                    f"{box.frame_width}x{box.frame_height}"
                )
            with av.open(str(temporary), mode="w", options={"movflags": "faststart"}) as target:
                out = target.add_stream(encoder.vcodec, stream.base_rate, options=options)
                out.pix_fmt = encoder.pix_fmt
                out.width, out.height = width, height
                for frame in source.decode(stream):
                    image = frame.to_ndarray(format="rgb24")
                    cropped = box.apply(image)
                    # Always a magnification: the box is smaller than the frame
                    # it came from, and it is being stretched back to it.
                    resized = cv2.resize(
                        cropped, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                    new = av.VideoFrame.from_ndarray(resized, format="rgb24")
                    new.pts = frame.pts
                    new.time_base = frame.time_base
                    packet = out.encode(new)
                    if packet:
                        target.mux(packet)
                    frames += 1
                packet = out.encode()
                if packet:
                    target.mux(packet)
        _verify(temporary, frames, width, height)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return frames


def _verify(path: Path, frames: int, width: int, height: int) -> None:
    """Read the new file back and insist it is the same shape and length.

    The metadata of the copy — every timestamp, every episode's frame range — was
    inherited from the source and is only still true if this holds. A video
    silently one frame short would shift every observation after it against its
    action, which is a fine-tune trained on a lie and no error anywhere.
    """
    import av

    with av.open(str(path), mode="r") as handle:
        stream = handle.streams.video[0]
        got = sum(1 for _ in handle.decode(stream))
        shape = (int(stream.width), int(stream.height))
    if got != frames or shape != (width, height):
        raise RecropError(
            f"{path}: re-encode produced {got} frames at {shape[0]}x{shape[1]}, "
            f"expected {frames} at {width}x{height}"
        )


def _write_lens(
    destination: Path,
    source: Path,
    crop: CropPlan,
    report: Report,
    info: dict[str, Any],
    notes: dict[str, Any] | None,
) -> None:
    """Record what lens this dataset now carries, in the dataset itself."""
    record: dict[str, Any] = {
        "profile": "optimized",
        "cropped": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "frame": {"width": crop.width, "height": crop.height},
        "streams": {
            key: {
                "x": box.x,
                "y": box.y,
                "width": box.width,
                "height": box.height,
                "shift_x": box.shift_x,
                "shift_y": box.shift_y,
                "describe": crop.lines.get(key, ""),
            }
            for key, box in crop.boxes.items()
        },
        "unchanged": report.copied,
        "frames": report.frames,
        "files": report.files,
        "codebase_version": info.get("codebase_version"),
        **(notes or {}),
    }
    (destination / LENS_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def read_lens(root: Path | str) -> dict[str, Any] | None:
    """The :data:`LENS_FILE` of a dataset, or ``None`` if it has none.

    Absent means *as recorded*, which for this study is ``common`` — the full
    lens. Present means somebody derived it, and the file says from what.
    """
    path = Path(root) / LENS_FILE
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def lens_profile(root: Path | str, notes: Iterable[dict[str, Any]] = ()) -> str | None:
    """Which run profile a dataset's frames carry: ``optimized``, ``common``, or unknown.

    Two sources, in order of authority: :data:`LENS_FILE`, written by whoever
    derived the dataset, and the ``profile`` field of the episodes' own notes,
    written by whoever recorded it. They answer different questions — what was
    done to it, and what it was captured as — and the first wins because it is
    the later fact.

    ``None`` when neither says, which is a dataset that cannot be matched to a
    row and must not be trained on without somebody looking at it.
    """
    lens = read_lens(root)
    if lens and lens.get("profile"):
        return str(lens["profile"])
    profiles = {
        str(record["profile"]) for record in notes if record.get("profile")
    }
    if len(profiles) == 1:
        return profiles.pop()
    return None
