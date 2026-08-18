"""Capture a still from a camera and put it in front of the operator.

Which by-path node is ``top``, which is ``left`` and which is ``right`` cannot be
derived from anything the system knows — the three cameras are the same model and
all report serial ``20010101``. The only way to assign the labels is to look at
the pictures, so this grabs one frame per candidate and shows it.

Display goes through an external image viewer rather than ``cv2.imshow``: the
``opencv-python`` wheel installed here is built with ``GUI: NONE``, so
``imshow`` raises. Writing a PNG and handing it to ``xdg-open`` also leaves the
operator a file they can re-open, which is what you want when a label turns out
to be wrong three candidates later.

Stills are captured *with* the configured rotation applied, because the question
being answered is "what does this camera see", and all three DK1 cameras are
mounted upside down.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

#: Tried in order; the first one present wins.
VIEWERS: tuple[str, ...] = ("xdg-open", "eog", "eom", "feh", "gwenview", "display")

#: UVC cameras need a few frames before auto-exposure and white balance settle;
#: the first frame off a cold device is routinely black or wildly dark, which
#: would make the labelling prompt unanswerable.
WARMUP_FRAMES = 12


class CaptureError(Exception):
    """Raised when a camera could not be opened or produced no frame."""


_ROTATION_FLAGS = {90: 0, 180: 1, 270: 2}  # cv2.ROTATE_* enum values, in order


def rotate(image: Any, degrees: int) -> Any:
    """Rotate a frame clockwise by 0/90/180/270 degrees."""
    if degrees == 0:
        return image
    if degrees not in _ROTATION_FLAGS:
        raise ValueError(f"rotation must be one of 0, 90, 180, 270 — got {degrees!r}")
    import cv2

    return cv2.rotate(image, _ROTATION_FLAGS[degrees])


def capture_still(
    device: str,
    *,
    width: int,
    height: int,
    fourcc: str = "MJPG",
    rotation: int = 0,
    warmup_frames: int = WARMUP_FRAMES,
) -> Any:
    """Grab one frame from ``device``, rotated as configured.

    Args:
        device: a ``/dev/v4l/by-path`` node or ``/dev/videoN``.

    Raises:
        CaptureError: if the device will not open or yields no frame.
    """
    import cv2

    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise CaptureError(
            f"{device}: could not be opened. Is it in use by another process "
            f"(a running teleop or a preview window)?"
        )
    try:
        # Order matters: FOURCC before the size. Setting the size first makes the
        # driver pick a size valid for the *current* (YUYV) format, and the later
        # MJPG switch then silently keeps that size.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, latest = capture.read()
            if ok and latest is not None:
                frame = latest
        if frame is None:
            raise CaptureError(
                f"{device}: opened, but returned no frame at {width}x{height} {fourcc}. "
                f"On these cameras MJPG is mandatory — YUYV at high resolutions "
                f"exceeds the USB bandwidth the uvc driver can allocate."
            )
        return rotate(frame, rotation)
    finally:
        capture.release()


def save_still(image: Any, path: Path) -> Path:
    """Write a captured frame to ``path`` as PNG."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise CaptureError(f"could not write the preview image to {path}")
    return path


def viewer_command() -> str | None:
    """The first available image viewer, or ``None`` if there is none."""
    return next((name for name in VIEWERS if shutil.which(name)), None)


def open_viewer(path: Path) -> str | None:
    """Open ``path`` in an image viewer, detached. Returns the viewer used.

    Returns ``None`` when no viewer is installed — the caller then falls back to
    printing the path, which is still workable over SSH.
    """
    viewer = viewer_command()
    if viewer is None:
        return None
    try:
        subprocess.Popen(
            [viewer, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return None
    return viewer
