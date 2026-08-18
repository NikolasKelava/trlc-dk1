"""Enumerate the cameras and let the operator say which physical view is which.

The three DK1 cameras are indistinguishable by identity: they are the same model
and they all report serial ``20010101``, so ``/dev/v4l/by-id`` collapses to a
single entry and cannot address them. ``/dev/v4l/by-path`` encodes the USB hub
port instead, which is stable as long as a camera stays in its socket.

Each camera also exposes two ``video`` nodes; only ``-index0`` carries frames
(``-index1`` is a metadata node), so candidates are filtered to ``-index0``.

Labelling — which by-path node is ``top``, which is ``left``, which is ``right``
— cannot be derived and has to be seen, so :func:`assign_labels` shows a still
from each candidate and asks. Showing and asking are injected, so both halves of
this module stay testable with no camera: enumeration against a fake directory
listing, labelling against scripted answers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
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


class LabelError(Exception):
    """Raised when the three camera labels could not be assigned unambiguously."""


def assign_labels(
    candidates: Sequence[CameraCandidate],
    *,
    names: Sequence[str],
    show: Callable[[CameraCandidate], None],
    ask: Callable[[CameraCandidate, Sequence[str]], str],
    announce: Callable[[str], None] = print,
) -> dict[str, CameraCandidate]:
    """Walk the operator through naming each attached camera.

    The pure half of the preview-and-label loop: ``show`` puts a still in front of
    the operator and ``ask`` returns the name they chose, so the whole flow can be
    driven by a test with a scripted set of answers and no camera.

    A name already taken is offered again only if the operator frees it, which
    they do by relabelling — so instead of unwinding, the loop keeps asking until
    every name is claimed exactly once.

    Raises:
        LabelError: if there are not exactly as many candidates as names, or a
            candidate is left unnamed.
    """
    if len(candidates) != len(names):
        raise LabelError(
            f"{len(candidates)} camera(s) attached but {len(names)} names to assign "
            f"({', '.join(names)}). All three DK1 cameras must be plugged in and "
            f"nothing else that presents a video node. Nothing has been written."
        )

    assigned: dict[str, CameraCandidate] = {}
    for candidate in candidates:
        remaining = [name for name in names if name not in assigned]
        show(candidate)
        choice = ask(candidate, remaining)
        if choice not in remaining:
            raise LabelError(
                f"{choice!r} is not one of the remaining names ({', '.join(remaining)}). "
                f"Nothing has been written."
            )
        assigned[choice] = candidate
        announce(f"  {choice:6s} <- hub port {candidate.hub_port}")

    unnamed = [name for name in names if name not in assigned]
    if unnamed:
        raise LabelError(f"no camera was labelled {unnamed}. Nothing has been written.")
    return assigned
