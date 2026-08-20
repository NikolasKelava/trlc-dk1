"""Build LeRobot camera configs from ``dk1.toml``.

The only module that turns configured device identity into
``OpenCVCameraConfig`` objects, so teleop, recording and policy rollout cannot
drift apart on which physical camera is ``top``.

Camera *names* are load-bearing. The MolmoAct2 BimanualYAM checkpoint's image
keys are ``observation.images.{top,left,right}``, and LeRobot derives those from
the robot's camera keys, so a camera named ``wrist_left`` would fail the rollout
context's visual-feature check outright.

A camera with ``target_hfov`` in ``dk1.toml`` is built as a
:class:`dk1lab.crop.CroppedOpenCVCamera` instead, so its frames arrive narrowed
to the field of view the checkpoint was trained on. That choice is made here,
once, for exactly the same reason the names are: teleoperation, recording and
rollout all come through this function, and a crop that applied to only some of
them would be worse than no crop at all.
"""

from __future__ import annotations

from lerobot.cameras.opencv import OpenCVCameraConfig

from .config import CameraDevice, CaptureProfile, DK1Config
from . import fov
from .crop import CroppedOpenCVCameraConfig
from .layout import CAMERA_NAMES


def _camera_config(
    device: CameraDevice, capture: CaptureProfile
) -> OpenCVCameraConfig | CroppedOpenCVCameraConfig:
    """One camera's LeRobot config — cropped iff ``dk1.toml`` gives it a target."""
    common = dict(
        index_or_path=device.path,
        width=capture.width,
        height=capture.height,
        fps=capture.fps,
        rotation=device.rotation,
        fourcc=capture.fourcc,
    )
    if not device.cropped:
        return OpenCVCameraConfig(**common)
    return CroppedOpenCVCameraConfig(
        **common,
        source_hfov_deg=device.hfov,
        target_hfov_deg=device.target_hfov,
        crop_inset=device.crop_inset,
        crop_shift_x=device.crop_shift_x,
        crop_shift_y=device.crop_shift_y,
    )


def camera_configs(
    config: DK1Config, profile: str = "policy"
) -> dict[str, OpenCVCameraConfig]:
    """One ``OpenCVCameraConfig`` per camera, keyed by its canonical name.

    Args:
        config: loaded ``dk1.toml``.
        profile: which ``[capture.*]`` profile to use — ``"policy"`` or ``"teleop"``.

    Returns:
        A dict in ``top, left, right`` order. Python preserves insertion order and
        LeRobot's feature builders use it verbatim, so this ordering is what pins
        the camera order end to end.
    """
    capture = config.profile(profile)
    return {
        name: _camera_config(config.camera(name), capture) for name in CAMERA_NAMES
    }


def crop_summary(camera) -> str | None:
    """A short phrase describing a camera's crop, or ``None`` if it has none.

    Sized to sit at the end of a one-line banner entry. ``dk1 config show`` uses
    :func:`dk1lab.fov.describe` instead, which spells the whole thing out.

    Duck-typed on the two angles rather than on a class, because the banners that
    want this line hold a :class:`CroppedOpenCVCameraConfig` in one command and a
    live :class:`~dk1lab.crop.CroppedOpenCVCamera` in another, and both carry the
    same four attributes. Either way it describes what will actually happen to
    the pixels rather than what ``dk1.toml`` asked for.
    """
    source = getattr(camera, "source_hfov_deg", None)
    target = getattr(camera, "target_hfov_deg", None)
    if source is None or target is None or not camera.width or not camera.height:
        return None
    box = fov.crop_box(
        camera.width,
        camera.height,
        source,
        target,
        inset=getattr(camera, "crop_inset", 0.0),
        shift_x=getattr(camera, "crop_shift_x", 0.0),
        shift_y=getattr(camera, "crop_shift_y", 0.0),
    )
    got = fov.hfov_from_scale(source, box.width / camera.width)
    line = f"crop {box.width}x{box.height} -> {got:.1f} deg H"
    if box.shift_x or box.shift_y:
        line += f", offset {box.shift_x:+d},{box.shift_y:+d} px"
    return line


def cameras_cli_argument(config: DK1Config, profile: str = "policy") -> str:
    """The same configuration as a ``--robot.cameras=...`` argument.

    For the LeRobot CLIs (``lerobot-record``, ``lerobot-rollout``) which take the
    camera set as a draccus-parsed inline dict rather than Python objects.
    """
    capture = config.profile(profile)
    entries = []
    for name in CAMERA_NAMES:
        device = config.camera(name)
        kind = "opencv_cropped" if device.cropped else "opencv"
        entry = (
            f"{name}: {{type: {kind}, index_or_path: {device.path}, "
            f"width: {capture.width}, height: {capture.height}, fps: {capture.fps}, "
            f"fourcc: {capture.fourcc}"
        )
        if device.rotation:
            entry += f", rotation: {device.rotation}"
        if device.cropped:
            entry += (
                f", source_hfov_deg: {device.hfov:g}, target_hfov_deg: {device.target_hfov:g}"
            )
            for key in ("crop_inset", "crop_shift_x", "crop_shift_y"):
                value = getattr(device, key)
                if value:
                    entry += f", {key}: {value:g}"
        entries.append(entry + "}")
    return "{ " + ", ".join(entries) + " }"
