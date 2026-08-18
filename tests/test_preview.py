"""Preview capture: rotation, and getting a still in front of the operator.

The capture itself needs a camera, so what is tested here is everything around
it — rotation validation, viewer selection, and that a failure to open reports
the device rather than a bare OpenCV false.
"""

from __future__ import annotations

import numpy as np
import pytest

from dk1lab.discovery import preview
from dk1lab.discovery.preview import (
    CaptureError,
    capture_still,
    open_viewer,
    rotate,
    viewer_command,
)


@pytest.fixture
def frame():
    """A frame whose corners are all distinguishable, so rotation is visible."""
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[0, 0] = (1, 1, 1)
    image[0, -1] = (2, 2, 2)
    image[-1, -1] = (3, 3, 3)
    return image


def test_no_rotation_returns_the_frame_unchanged(frame):
    assert rotate(frame, 0) is frame


def test_180_is_a_half_turn_which_is_how_the_dk1_cameras_are_mounted(frame):
    turned = rotate(frame, 180)
    assert turned.shape == frame.shape
    assert tuple(turned[-1, -1]) == (1, 1, 1)
    assert tuple(turned[0, 0]) == (3, 3, 3)


def test_90_swaps_the_axes(frame):
    assert rotate(frame, 90).shape == (6, 4, 3)


def test_two_half_turns_are_the_identity(frame):
    assert np.array_equal(rotate(rotate(frame, 180), 180), frame)


def test_an_unsupported_rotation_is_refused(frame):
    with pytest.raises(ValueError, match="0, 90, 180, 270"):
        rotate(frame, 45)


def test_a_device_that_will_not_open_names_the_device():
    with pytest.raises(CaptureError, match="/dev/video-nope"):
        capture_still("/dev/video-nope", width=640, height=360)


def test_the_first_available_viewer_wins(monkeypatch):
    monkeypatch.setattr(preview, "VIEWERS", ("nonexistent-viewer", "feh", "eog"))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/feh" if name == "feh" else None)
    assert viewer_command() == "feh"


def test_no_viewer_installed_is_reported_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert viewer_command() is None
    assert open_viewer(tmp_path / "still.png") is None
