"""Build LeRobot camera configs from ``dk1.toml``.

The only module that turns configured device identity into
``OpenCVCameraConfig`` objects, so teleop, recording and policy rollout cannot
drift apart on which physical camera is ``top``.

Camera *names* are load-bearing. The MolmoAct2 BimanualYAM checkpoint's image
keys are ``observation.images.{top,left,right}``, and LeRobot derives those from
the robot's camera keys, so a camera named ``wrist_left`` would fail the rollout
context's visual-feature check outright.
"""

from __future__ import annotations

from lerobot.cameras.opencv import OpenCVCameraConfig

from .config import DK1Config
from .layout import CAMERA_NAMES


def camera_configs(config: DK1Config, profile: str = "policy") -> dict[str, OpenCVCameraConfig]:
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
        name: OpenCVCameraConfig(
            index_or_path=config.camera(name).path,
            width=capture.width,
            height=capture.height,
            fps=capture.fps,
            rotation=config.camera(name).rotation,
            fourcc=capture.fourcc,
        )
        for name in CAMERA_NAMES
    }


def cameras_cli_argument(config: DK1Config, profile: str = "policy") -> str:
    """The same configuration as a ``--robot.cameras=...`` argument.

    For the LeRobot CLIs (``lerobot-record``, ``lerobot-rollout``) which take the
    camera set as a draccus-parsed inline dict rather than Python objects.
    """
    capture = config.profile(profile)
    entries = []
    for name in CAMERA_NAMES:
        device = config.camera(name)
        entry = (
            f"{name}: {{type: opencv, index_or_path: {device.path}, "
            f"width: {capture.width}, height: {capture.height}, fps: {capture.fps}, "
            f"fourcc: {capture.fourcc}"
        )
        if device.rotation:
            entry += f", rotation: {device.rotation}"
        entries.append(entry + "}")
    return "{ " + ", ".join(entries) + " }"
