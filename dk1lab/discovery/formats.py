"""Ask a camera which capture modes it actually offers.

``dk1.toml``'s ``[capture.*]`` profiles are assertions about the hardware:
``policy`` claims the cameras can deliver 640x360 MJPG (16:9, the aspect ratio
the MolmoAct2 BimanualYAM checkpoint was trained on) and ``teleop`` claims
1280x720 MJPG at 60. Neither is guaranteed by the datasheet, and a profile the
camera cannot serve fails at rollout time rather than at configuration time.

So the probe runs at discovery time and says so out loud. Parsing is separated
from the subprocess call: :func:`parse_formats_ext` is pure and tested against
recorded ``v4l2-ctl`` output, so the interesting logic needs no camera.

Why ``v4l2-ctl`` rather than OpenCV: OpenCV will happily accept a resolution the
device does not offer and silently hand back the nearest one it does, which is
exactly the failure this is meant to catch. ``VIDIOC_ENUM_FMT`` enumerates what
the device really advertises.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

#: ``v4l2-ctl`` lives in the ``v4l-utils`` package; probing degrades to a
#: skipped check rather than an error when it is absent.
V4L2_CTL = "v4l2-ctl"

_FORMAT_RE = re.compile(r"^\s*\[\d+\]:\s*'(\w{4})'")
_SIZE_RE = re.compile(r"^\s*Size:\s*Discrete\s*(\d+)x(\d+)")
_INTERVAL_RE = re.compile(r"\(([\d.]+)\s*fps\)")


class ProbeError(Exception):
    """Raised when a camera's formats could not be read at all."""


@dataclass(frozen=True)
class Mode:
    """One advertised capture mode: a pixel format at a size, with its rates."""

    fourcc: str
    width: int
    height: int
    fps: tuple[float, ...]

    def supports_fps(self, fps: float, *, tolerance: float = 0.5) -> bool:
        """Whether this mode advertises (close enough to) ``fps``."""
        return any(abs(rate - fps) <= tolerance for rate in self.fps)

    def __str__(self) -> str:
        rates = ", ".join(f"{rate:g}" for rate in sorted(self.fps, reverse=True))
        return f"{self.fourcc} {self.width}x{self.height} @ {rates} fps"


def parse_formats_ext(text: str) -> list[Mode]:
    """Parse ``v4l2-ctl --list-formats-ext`` output into modes.

    Pure. The device repeats a size when it advertises it under two different
    frame-rate sets (the 4K entries do this), so rates for the same
    ``(fourcc, width, height)`` are merged rather than duplicated.
    """
    merged: dict[tuple[str, int, int], set[float]] = {}
    order: list[tuple[str, int, int]] = []
    fourcc: str | None = None
    key: tuple[str, int, int] | None = None

    for line in text.splitlines():
        if match := _FORMAT_RE.match(line):
            fourcc, key = match.group(1), None
            continue
        if match := _SIZE_RE.match(line):
            if fourcc is None:
                continue  # a size before any format header: malformed, skip it
            key = (fourcc, int(match.group(1)), int(match.group(2)))
            if key not in merged:
                merged[key] = set()
                order.append(key)
            continue
        if key is not None and (match := _INTERVAL_RE.search(line)):
            merged[key].add(float(match.group(1)))

    return [
        Mode(fourcc=f, width=w, height=h, fps=tuple(sorted(merged[(f, w, h)], reverse=True)))
        for (f, w, h) in order
    ]


def probe(device: str, *, timeout_s: float = 5.0) -> list[Mode]:
    """Enumerate the capture modes ``device`` advertises right now.

    Args:
        device: any node the driver accepts — a ``/dev/video*`` or the
            ``/dev/v4l/by-path`` symlink pointing at one.

    Raises:
        ProbeError: if ``v4l2-ctl`` is missing, or the device cannot be read.
    """
    if shutil.which(V4L2_CTL) is None:
        raise ProbeError(
            f"{V4L2_CTL} not found. Install v4l-utils to check that a camera "
            f"really offers the configured capture profile."
        )
    try:
        result = subprocess.run(
            [V4L2_CTL, "-d", device, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"{device}: {V4L2_CTL} timed out after {timeout_s}s") from exc
    # v4l2-ctl prints the "ioctl: VIDIOC_ENUM_FMT" banner to stdout and still
    # exits 0 on some drivers, so the modes found matter more than the code.
    modes = parse_formats_ext(result.stdout)
    if not modes:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise ProbeError(f"{device}: no capture formats reported. {detail[-1] if detail else ''}")
    return modes


def find_mode(
    modes: Iterable[Mode], *, fourcc: str, width: int, height: int, fps: float
) -> Mode | None:
    """The advertised mode matching a capture profile exactly, or ``None``."""
    for mode in modes:
        if (
            mode.fourcc == fourcc
            and mode.width == width
            and mode.height == height
            and mode.supports_fps(fps)
        ):
            return mode
    return None


def nearest_aspect(modes: Iterable[Mode], *, fourcc: str, aspect: float) -> list[Mode]:
    """Modes with the given pixel format, closest to ``aspect`` first.

    Used to suggest a fallback when a profile's exact size is unavailable: what
    matters for the policy is the aspect ratio, since MolmoAct2 resizes every
    view to 378x378 regardless.
    """
    matching = [m for m in modes if m.fourcc == fourcc and m.height]
    return sorted(matching, key=lambda m: (abs(m.width / m.height - aspect), m.width * m.height))


@dataclass(frozen=True)
class ProfileCheck:
    """Whether one ``[capture.*]`` profile survives contact with a real camera."""

    profile: str
    """The profile's name, e.g. ``policy``."""

    wanted: str
    """The requested mode, rendered for display."""

    matched: Mode | None
    """The advertised mode that satisfies it, or ``None``."""

    alternatives: tuple[Mode, ...] = ()
    """Closest advertised modes by aspect ratio, best first. Only when unmatched."""

    @property
    def ok(self) -> bool:
        return self.matched is not None

    @property
    def aspect(self) -> float:
        width, height = (int(n) for n in self.wanted.split()[0].split("x"))
        return width / height

    def aspect_gap(self, *, tolerance: float = 0.01) -> float | None:
        """How far the best fallback's aspect ratio is from the one wanted.

        ``None`` when the profile matched, when there is no fallback, or when the
        fallback's aspect ratio is close enough to make no visual difference.
        MolmoAct2 resizes every view to 378x378, so the aspect ratio is what
        survives the resize — a 4:3 capture stretches the scene differently than
        the 16:9 the checkpoint was trained on.
        """
        if self.ok or not self.alternatives:
            return None
        best = self.alternatives[0]
        gap = abs(best.width / best.height - self.aspect)
        return gap if gap > tolerance else None


def check_profiles(modes: Iterable[Mode], profiles: dict[str, Any]) -> list[ProfileCheck]:
    """Check every configured capture profile against what a camera advertises.

    Args:
        modes: what :func:`probe` returned for this camera.
        profiles: ``{name: CaptureProfile}`` from ``dk1.toml``. Taken
            structurally rather than by import so this module keeps no
            dependency on the config layer.
    """
    modes = list(modes)
    checks = []
    for name, profile in profiles.items():
        matched = find_mode(
            modes,
            fourcc=profile.fourcc,
            width=profile.width,
            height=profile.height,
            fps=profile.fps,
        )
        alternatives: tuple[Mode, ...] = ()
        if matched is None:
            aspect = profile.width / profile.height
            alternatives = tuple(nearest_aspect(modes, fourcc=profile.fourcc, aspect=aspect)[:3])
        checks.append(
            ProfileCheck(
                profile=name,
                wanted=f"{profile.width}x{profile.height} {profile.fourcc}@{profile.fps}",
                matched=matched,
                alternatives=alternatives,
            )
        )
    return checks
