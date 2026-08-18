"""Enumerate the cameras and let the operator say which physical view is which.

The three DK1 cameras are indistinguishable by identity: they are the same model
and they all report serial ``20010101``, so ``/dev/v4l/by-id`` collapses to a
single entry and cannot address them. ``/dev/v4l/by-path`` encodes the USB hub
port instead, which is stable as long as a camera stays in its socket.

Each camera also exposes two ``video`` nodes; only ``-index0`` carries frames
(``-index1`` is a metadata node), so candidates are filtered to ``-index0``.

Labelling — which by-path node is ``top``, which is ``left``, which is ``right``
— cannot be derived and has to be seen, so the CLI captures a still from each
candidate and asks. That part lives in the CLI; everything here is pure enough to
test against a fake directory listing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

BY_PATH_DIR = Path("/dev/v4l/by-path")

#: Only ``-index0`` streams video; ``-index1`` is the UVC metadata node.
VIDEO_NODE_SUFFIX = "-video-index0"

#: The kernel exposes both ``usb-`` and ``usbv3-`` spellings of the same device.
#: They resolve identically; prefer one so the same camera is never offered twice.
_PREFERRED_INFIX = "-usb-"
_DUPLICATE_INFIX = "-usbv3-"


@dataclass(frozen=True)
class CameraCandidate:
    """One addressable camera node."""

    by_path: str
    """Stable ``/dev/v4l/by-path/...`` node — what goes in the config."""

    device: str | None = None
    """The ``/dev/videoN`` it currently resolves to. Informational only."""

    @property
    def hub_port(self) -> str:
        """The USB hub port from the by-path name, e.g. ``4.3``, for display."""
        # Node names look like "...-usb-0:10.1:1.0-video-index0", i.e.
        # "<bus>:<hub port path>:<config>.<interface>". The middle field is the
        # physical port chain, which is what identifies the socket.
        name = Path(self.by_path).name
        for infix in (_PREFERRED_INFIX, _DUPLICATE_INFIX):
            if infix in name:
                fields = name.split(infix, 1)[1].split(":")
                if len(fields) >= 2:
                    return fields[1]
        return "?"


def candidates_from_names(names: Iterable[str], base: Path = BY_PATH_DIR) -> list[CameraCandidate]:
    """Filter a raw ``by-path`` directory listing down to addressable cameras.

    Pure: takes filenames, returns candidates. Tested against a fake listing.
    """
    video0 = [n for n in names if n.endswith(VIDEO_NODE_SUFFIX)]
    preferred = [n for n in video0 if _DUPLICATE_INFIX not in n]
    # Fall back to the usbv3 spelling only for devices with no usb- equivalent.
    seen_ports = {n.split(_PREFERRED_INFIX, 1)[-1] for n in preferred}
    for name in video0:
        if _DUPLICATE_INFIX in name and name.split(_DUPLICATE_INFIX, 1)[-1] not in seen_ports:
            preferred.append(name)
    return [CameraCandidate(by_path=str(base / n)) for n in sorted(preferred)]


def list_candidates(base: Path = BY_PATH_DIR) -> list[CameraCandidate]:
    """Enumerate the cameras attached right now, resolving each to ``/dev/videoN``."""
    if not base.exists():
        return []
    candidates = candidates_from_names([p.name for p in base.iterdir()], base=base)
    resolved = []
    for candidate in candidates:
        path = Path(candidate.by_path)
        device = str(path.resolve()) if path.exists() else None
        resolved.append(CameraCandidate(by_path=candidate.by_path, device=device))
    return resolved
