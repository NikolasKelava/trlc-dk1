"""Field-of-view arithmetic: how much of a frame to keep to hit a target HFOV.

Pure geometry, no imports beyond the standard library, so it stays testable on a
machine with no robot stack — the same split every other decision-making module
in this package keeps. :mod:`dk1lab.crop` is the half that touches pixels.

**Why this exists.** The MolmoAct2 BimanualYAM checkpoint was trained in
simulation against pinhole cameras whose intrinsics are built straight from a
horizontal field of view (``sim_eval/robots/bimanual_yam.py``)::

    fx = (width / 2) / tan(hfov / 2)      # fy = fx, so the pixels are square
    top = 69.4 deg (RealSense D435i), wrists = 87.0 deg (RealSense D405)

Our three Innomaker U30CAM-4K-S1 are **much wider**: the user manual gives
``Fov(D) = 116``, ``Fov(H) = 105`` degrees for its 2.25 mm f/2.0 M12 lens on a
1/2.8" IMX415. A policy that learned "the gripper is aligned when the object
appears *here*, at *this* size" reads a 105-degree frame as a scene further away
and further off-axis than it is, which is exactly a consistent spatial
misalignment sitting on top of correct visual tracking.

A wider lens is the fixable direction: crop the middle out and the remaining
frame spans a narrower angle. Narrower than the target would be unfixable, which
is why :func:`crop_box` refuses it rather than silently doing nothing.

**The model is a pinhole**, matching what the checkpoint was trained against. On
a rectilinear lens the image height on the sensor is ``f * tan(theta)``, so the
fraction of the frame width that spans ``target`` is

    scale = tan(target / 2) / tan(source / 2)

and that is the whole calculation. Two caveats worth stating rather than hiding:

* our lens has real barrel distortion (the manual says TV distortion < -6.2%),
  so the true mapping is not quite rectilinear and the crop matches the trained
  geometry at the frame centre better than at its corners. Correcting that needs
  a calibration this cell does not have;
* the 105-degree figure is the manufacturer's, not a measurement.

Both push the same way: treat the crop as bringing us close to the trained
geometry, not onto it.

**Rounding is always outward.** Every rounding here takes the larger crop, i.e.
the wider field of view, because a policy shown slightly too much of the scene
degrades more gracefully than one shown too little.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class FOVError(ValueError):
    """Raised for a field of view that cannot be reached by cropping."""


@dataclass(frozen=True)
class CropBox:
    """A centred crop rectangle, in pixels of the frame it applies to."""

    x: int
    y: int
    width: int
    height: int

    @property
    def is_full_frame(self) -> bool:
        """True when the box keeps everything — nothing to do."""
        return self.x == 0 and self.y == 0

    def apply(self, frame):
        """Slice ``frame`` (an ``H x W x C`` array) to this box."""
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


def check_hfov(degrees: float, what: str) -> float:
    """Validate one horizontal field of view, in degrees."""
    if not isinstance(degrees, (int, float)) or isinstance(degrees, bool):
        raise FOVError(f"{what} must be a number of degrees, got {degrees!r}")
    if not 0.0 < float(degrees) < 180.0:
        raise FOVError(f"{what} must be in (0, 180) degrees, got {degrees!r}")
    return float(degrees)


def hfov_scale(source_hfov_deg: float, target_hfov_deg: float) -> float:
    """The fraction of a frame's width that spans ``target_hfov_deg``.

    ``1.0`` means the whole frame; smaller means a tighter crop. Pinhole model —
    see the module docstring.
    """
    source = check_hfov(source_hfov_deg, "source hfov")
    target = check_hfov(target_hfov_deg, "target hfov")
    return math.tan(math.radians(target) / 2.0) / math.tan(math.radians(source) / 2.0)


def hfov_from_scale(source_hfov_deg: float, scale: float) -> float:
    """Inverse of :func:`hfov_scale`: what a crop of ``scale`` actually spans."""
    source = check_hfov(source_hfov_deg, "source hfov")
    if scale <= 0.0:
        raise FOVError(f"crop scale must be positive, got {scale!r}")
    return math.degrees(2.0 * math.atan(scale * math.tan(math.radians(source) / 2.0)))


def vfov(hfov_deg: float, width: int, height: int) -> float:
    """The vertical field of view implied by a square-pixel pinhole frame.

    The checkpoint's simulated cameras set ``fy = fx``, so this is the same
    relation they were rendered under.
    """
    hfov_deg = check_hfov(hfov_deg, "hfov")
    if width <= 0 or height <= 0:
        raise FOVError(f"frame must be positive, got {width}x{height}")
    return math.degrees(
        2.0 * math.atan((height / width) * math.tan(math.radians(hfov_deg) / 2.0))
    )


def crop_box(
    width: int, height: int, source_hfov_deg: float, target_hfov_deg: float
) -> CropBox:
    """The centred box of a ``width x height`` frame that spans ``target_hfov_deg``.

    Raises:
        FOVError: if the target is wider than the source. Cropping only ever
            narrows, so a too-narrow lens is a hardware problem, not a software
            one, and saying so beats returning the full frame and letting the
            caller believe the field of view was corrected.
    """
    if width <= 0 or height <= 0:
        raise FOVError(f"frame must be positive, got {width}x{height}")
    source = check_hfov(source_hfov_deg, "source hfov")
    target = check_hfov(target_hfov_deg, "target hfov")
    if target > source:
        raise FOVError(
            f"cannot crop a {source:g} degree lens up to {target:g} degrees — cropping "
            f"only narrows the field of view. This camera is too narrow for the "
            f"target and no image processing can fix that."
        )

    scale = hfov_scale(source, target)
    # ceil throughout: every rounding takes the wider field of view.
    crop_w = min(width, math.ceil(width * scale))
    crop_h = min(height, math.ceil(crop_w * height / width))
    # Centred, and biased left/up by the odd pixel so the box stays in frame.
    return CropBox(
        x=(width - crop_w) // 2, y=(height - crop_h) // 2, width=crop_w, height=crop_h
    )


def describe(width: int, height: int, source_hfov_deg: float, target_hfov_deg: float) -> str:
    """One line naming the box and the field of view it really achieves.

    The achieved figure is not the requested one — the ceil rounding above leaves
    it a little wider — and printing the requested number instead would hide
    exactly the discrepancy an operator wants to see.
    """
    box = crop_box(width, height, source_hfov_deg, target_hfov_deg)
    got_h = hfov_from_scale(source_hfov_deg, box.width / width)
    got_v = vfov(got_h, box.width, box.height)
    return (
        f"{width}x{height} @ {source_hfov_deg:g} deg -> crop {box.width}x{box.height} "
        f"at ({box.x},{box.y}) -> {got_h:.1f} deg H / {got_v:.1f} deg V "
        f"(asked {target_hfov_deg:g})"
    )
