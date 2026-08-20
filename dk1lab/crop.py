"""A LeRobot camera that crops to the field of view the checkpoint was trained on.

:class:`CroppedOpenCVCamera` is :class:`~lerobot.cameras.opencv.OpenCVCamera`
with one extra step at the end of the read path: take the centre of every frame
and stretch it back to the configured size. :mod:`dk1lab.fov` decides how big
that centre is; this module only moves pixels.

**Why a camera subclass rather than a processor step.** The crop has to be true
of *every* image this cell produces — what teleoperation displays, what a
recording stores, what the policy is fed — and there is exactly one place all
three go through. Putting it in a LeRobot processor would cover rollout and miss
teleop; putting it in :mod:`dk1lab.policy` would cover rollout and miss
recording, which is the one that would quietly poison a fine-tune. Downstream is
therefore unchanged: the camera still advertises, and still returns, frames of
the configured ``width x height``.

**Why it resizes back instead of returning the smaller crop.** Keeping the
output shape fixed is what makes this seamless — every feature shape, every
recorded video, every ``dk1.toml`` capture profile and every test stays as it
was. The cost is one resample, and it is only free while the crop stays larger
than what the model asks for: MolmoAct2's HF image processor resizes each view
to **378x378** (``crop_mode: "resize"``, ``size: 378``, ``patch_size: 14``), so
the crop has to clear 378 in both axes for the round trip through the configured
frame size to be a no-op. At ``[capture.policy]`` 1280x720 the wrist crop is
909x511 and it does. At 640x360 it was 455x256 and it did not — 256 rows were
being upsampled to fill 378.

**Rotation.** The crop is applied *after* the rotation the base class does, so
the box is centred on the picture as it is finally seen. That is right for 0 and
180 degrees. At 90 or 270 the output's horizontal axis is the sensor's
*vertical* one, so a horizontal field of view no longer describes it — that
combination is rejected at construction rather than cropped wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from numpy.typing import NDArray

from .fov import CropBox, FOVError, check_hfov, crop_box, describe, hfov_from_scale, vfov

__all__ = ["CroppedOpenCVCamera", "CroppedOpenCVCameraConfig"]


@CameraConfig.register_subclass("opencv_cropped")
@dataclass
class CroppedOpenCVCameraConfig(OpenCVCameraConfig):
    """``opencv``, plus the two angles that define the crop.

    Both default to ``None`` so the dataclass stays constructible the way the
    base one is; a config with only one of them set is a mistake and is rejected
    here rather than at connect time.

    Attributes:
        source_hfov_deg: the lens's own horizontal field of view, in degrees.
            105.0 for the Innomaker U30CAM-4K-S1, per its user manual.
        target_hfov_deg: what to crop to. 87.0 for the BimanualYAM wrist views
            (RealSense D405), 69.4 for its top view (D435i).
        crop_inset: extra pixels off the left and right edges, beyond the field
            of view, with the top and bottom following to keep the aspect ratio.
        crop_shift_x: move the box right (+) or left (-).
        crop_shift_y: move the box down (+) or **up** (-). Up shows more of what
            is above the lens's centre line.

    The three pixel offsets are quoted at :data:`dk1lab.fov.REFERENCE_WIDTH` and
    scaled to whatever frame the camera delivers, so the geometry is the same at
    every capture resolution. That is what lets the teleop view be evidence about
    the policy view when the two run at different sizes.
    """

    source_hfov_deg: float | None = None
    target_hfov_deg: float | None = None
    crop_inset: float = 0.0
    crop_shift_x: float = 0.0
    crop_shift_y: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source_hfov_deg is None or self.target_hfov_deg is None:
            raise FOVError(
                "CroppedOpenCVCameraConfig needs both source_hfov_deg and "
                f"target_hfov_deg; got {self.source_hfov_deg!r} and "
                f"{self.target_hfov_deg!r}. Use a plain OpenCVCameraConfig for an "
                "uncropped camera."
            )
        self.source_hfov_deg = check_hfov(self.source_hfov_deg, "source_hfov_deg")
        self.target_hfov_deg = check_hfov(self.target_hfov_deg, "target_hfov_deg")
        if self.rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            raise FOVError(
                f"rotation {int(self.rotation)} turns the frame on its side, so a "
                "horizontal field of view no longer describes its width. Cropping to "
                "an HFOV is only defined for rotation 0 or 180."
            )


class CroppedOpenCVCamera(OpenCVCamera):
    """An ``OpenCVCamera`` narrowed to ``target_hfov_deg`` by a centred crop."""

    def __init__(self, config: CroppedOpenCVCameraConfig) -> None:
        super().__init__(config)
        self.config: CroppedOpenCVCameraConfig = config
        self.source_hfov_deg: float = float(config.source_hfov_deg)
        self.target_hfov_deg: float = float(config.target_hfov_deg)
        self.crop_inset: float = float(config.crop_inset)
        self.crop_shift_x: float = float(config.crop_shift_x)
        self.crop_shift_y: float = float(config.crop_shift_y)
        self._box: CropBox | None = None

    # ----------------------------------------------------------------- report

    @property
    def crop(self) -> CropBox | None:
        """The box in use, or ``None`` until the frame size is known.

        Frame size is only settled after :meth:`connect`, because LeRobot lets a
        camera config leave ``width``/``height`` unset and adopt whatever the
        device reports.
        """
        if self._box is None and self.width and self.height:
            self._box = crop_box(
                int(self.width),
                int(self.height),
                self.source_hfov_deg,
                self.target_hfov_deg,
                inset=self.crop_inset,
                shift_x=self.crop_shift_x,
                shift_y=self.crop_shift_y,
            )
        return self._box

    @property
    def achieved_hfov_deg(self) -> float | None:
        """The field of view the crop really spans — a little wider than asked.

        :func:`dk1lab.fov.crop_box` rounds outward, so this is the number to
        report. ``None`` while the frame size is still unknown.
        """
        box = self.crop
        if box is None:
            return None
        return hfov_from_scale(self.source_hfov_deg, box.width / int(self.width))

    @property
    def achieved_vfov_deg(self) -> float | None:
        """The vertical counterpart of :attr:`achieved_hfov_deg`."""
        box, hfov = self.crop, self.achieved_hfov_deg
        if box is None or hfov is None:
            return None
        return vfov(hfov, box.width, box.height)

    def describe_crop(self) -> str:
        """One line an operator can read, naming the box and what it achieves."""
        if not self.width or not self.height:
            return (
                f"crop to {self.target_hfov_deg:g} deg from {self.source_hfov_deg:g} deg "
                "(frame size not known until connect)"
            )
        return describe(
            int(self.width),
            int(self.height),
            self.source_hfov_deg,
            self.target_hfov_deg,
            inset=self.crop_inset,
            shift_x=self.crop_shift_x,
            shift_y=self.crop_shift_y,
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    # ------------------------------------------------------------- the pixels

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """Base class's colour/validate/rotate, then crop and resize back."""
        frame = super()._postprocess_image(image)
        box = self.crop
        if box is None or box.is_full_frame:
            return frame
        cropped = box.apply(frame)
        height, width = frame.shape[:2]
        if (cropped.shape[1], cropped.shape[0]) == (width, height):
            return cropped
        # INTER_AREA is the right filter when shrinking, INTER_LINEAR when
        # growing; a crop-then-restore always grows, but the branch keeps this
        # honest if a future config ever crops a frame larger than its output.
        interpolation = cv2.INTER_AREA if cropped.shape[1] > width else cv2.INTER_LINEAR
        # cv2.resize allocates its own output, so the returned array is a fresh
        # contiguous buffer rather than a view into `frame` — which downstream
        # encoders and `torch.from_numpy` both assume.
        return cv2.resize(cropped, (width, height), interpolation=interpolation)
