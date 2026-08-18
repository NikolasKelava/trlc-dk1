"""Shared fixtures. Nothing here touches hardware."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_TOML = """\
version = 1

# a comment that must survive every surgical write
[arms.follower]
left = "/dev/ttyACM1"
right = "/dev/ttyACM3"

[arms.leader]
left = "/dev/ttyACM0"
right = "/dev/ttyACM2"

# camera comment
[cameras.top]
path = "/dev/v4l/by-path/pci-top-video-index0"
rotation = 180

[cameras.left]
path = "/dev/v4l/by-path/pci-left-video-index0"
rotation = 180

[cameras.right]
path = "/dev/v4l/by-path/pci-right-video-index0"
rotation = 0

[capture.policy]
width = 640
height = 360
fps = 30
fourcc = "MJPG"

[capture.teleop]
width = 1280
height = 720
fps = 60
fourcc = "MJPG"
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A valid dk1.toml in a temp dir, with comments to prove writes preserve them."""
    path = tmp_path / "dk1.toml"
    path.write_text(VALID_TOML)
    return path


@pytest.fixture
def repo_config() -> Path:
    """The real dk1.toml tracked in this repo."""
    return REPO_ROOT / "dk1.toml"
